# Fused convolution: the numbers against a host reference, the kernel-lifetime
# and plan-independence guarantees the design rests on, and the refusals that
# need a device array. The pure layout and region arithmetic is in planner.jl.
#
# The host reference is FFTW's own circular convolution,
# ifft(fft(x) .* fft(kernel)), rather than a direct summation. It is O(n log n),
# so it covers Bluestein-sized axes and 4096-point transforms that a quadratic
# reference could not, and it is the same convolution theorem VkFFT implements,
# which would make it a weak check of the index convention on its own. The
# "direct summation" testset anchors that convention at one small size, in both
# the convolution and the correlation direction.

"""
    _conv_ref(h, k, region)

Returns the circular convolution of two host arrays over `region`, through FFTW.
"""
_conv_ref(h, k, region) = ifft(fft(h, region) .* fft(k, region), region)

@testset verbose = true "conv" begin
    wide_rtol = _rtol(WIDE_COMPLEX)

    @testset "$T $(join(dims, 'x')) region=$region features=$stack" for T in COMPLEX_TYPES,
                                                                       (dims, region, stack) in CONV_CASES
        rtol = _rtol(T)
        h = _noise(T, dims)
        k = _kernel_noise(T, dims)
        reference = _conv_ref(h, k, region)

        plan = VkFFT.plan_conv(_upload(h), _upload(k), region)
        @test plan isa VkFFTConvPlan{T, length(dims)}
        @test size(plan) == dims
        @test AbstractFFTs.fftdims(plan) == region
        @test plan.features == stack

        x = _upload(h)
        got = plan * x
        @test size(got) == dims
        @test _relmax(got, reference) < rtol

        # The plain circular convolution, at a scale of exactly one. Without the
        # normalize VkFFT applies over everything but its last axis, this would
        # come back scaled by the product of the other transformed lengths.
        scale = prod(i -> dims[i], region; init=1) ÷ dims[region[end]]
        if scale != 1
            @test _relmax(got, scale .* reference) > 1e-3
        end

        # The input is only read, and mul! into a fresh buffer agrees with `*`.
        @test Array(x) == h
        @test Array(mul!(similar(x), plan, x)) == Array(got)
    end

    @testset "correlate $T $(join(dims, 'x')) region=$region features=$stack" for T in COMPLEX_TYPES,
                                                                  (dims, region, stack) in CONV_CASES
        rtol = _rtol(T)
        h = _noise(T, dims)
        k = _kernel_noise(T, dims)

        plan = VkFFT.plan_conv(_upload(h), _upload(k), region; correlate=true)
        @test plan.correlate
        got = plan * _upload(h)
        @test _relmax(got, ifft(conj.(fft(h, region)) .* fft(k, region), region)) < rtol

        # A correlation is not the convolution of the same pair, so this is not
        # a test that passes whatever the flag did.
        @test _relmax(got, _conv_ref(h, k, region)) > 1e-3
    end

    @testset "direct summation" begin
        # The index convention, anchored without the convolution theorem.
        for n in (8, 9)
            h = _noise(WIDE_COMPLEX, (n,))
            k = _kernel_noise(WIDE_COMPLEX, (n,))
            by_sum = [sum(h[j] * k[mod(m - j, n) + 1] for j in 1:n) for m in 1:n]
            correlation_by_sum = [sum(conj(h[j]) * k[mod(j + m - 2, n) + 1] for j in 1:n) for m in 1:n]
            @test _relmax(VkFFT.plan_conv(_upload(h), _upload(k), 1) * _upload(h), by_sum) < wide_rtol
            @test _relmax(VkFFT.plan_conv(_upload(h), _upload(k), 1; correlate=true) * _upload(h), correlation_by_sum) < wide_rtol
        end
    end

    @testset "Bluestein axes above one upload" begin
        # Pinned rather than refused. VkFFT takes its Bluestein path for any
        # length with a prime factor above 13, and splits a long Bluestein axis
        # over several uploads, which computes something that is no convolution
        # and reports nothing. The length where the split starts moves with the
        # device's shared memory, so the planner has no boundary to test
        # against, and so a runner only pins the failure when its device is
        # known to reach it.
        if BACKEND.pin_conv_bluestein
            for n in (4093,)
                h = _noise(WIDE_COMPLEX, (n,))
                k = zeros(WIDE_COMPLEX, n)
                k[1] = 1 # a delta, so the whole pipeline has to return its input
                got = VkFFT.plan_conv(_upload(h), _upload(k), 1) * _upload(h)
                @test_broken _relmax(got, h) < wide_rtol
            end
        else
            _skip("the long-Bluestein convolution pin", "the upload boundary moves with the device, so it is pinned only where it was measured")
        end

        # The control: a length whose prime factors are all at most 13 is
        # correct at the same order of magnitude, so this is the Bluestein path
        # and not the length.
        h = _noise(WIDE_COMPLEX, (4096,))
        k = zeros(WIDE_COMPLEX, 4096)
        k[1] = 1
        @test _relmax(VkFFT.plan_conv(_upload(h), _upload(k), 1) * _upload(h), h) < wide_rtol
    end

    @testset "a delta kernel is the identity" begin
        # The one convolution whose answer needs no reference at all.
        h = _noise(WIDE_COMPLEX, (16, 8))
        k = zeros(WIDE_COMPLEX, 16, 8)
        k[1, 1] = 1
        @test _relmax(VkFFT.plan_conv(_upload(h), _upload(k), (1, 2)) * _upload(h), h) < wide_rtol

        # A shifted delta is a circular shift.
        shifted = zeros(WIDE_COMPLEX, 16, 8)
        shifted[4, 3] = 1
        @test _relmax(VkFFT.plan_conv(_upload(h), _upload(shifted), (1, 2)) * _upload(h), circshift(h, (3, 2))) < wide_rtol
    end

    @testset "plan reuse" begin
        h = _noise(WIDE_COMPLEX, (16, 8))
        k = _kernel_noise(WIDE_COMPLEX, (16, 8))
        plan = VkFFT.plan_conv(_upload(h), _upload(k), (1, 2))
        x = _upload(h)

        first_run = Array(plan * x)
        @test Array(plan * x) == first_run
        @test Array(plan * x) == first_run
        @test Array(mul!(similar(x), plan, x)) == first_run
        @test _relmax(first_run, _conv_ref(h, k, (1, 2))) < wide_rtol

        # A plan is reusable across buffers, not just across calls.
        @test Array(plan * _upload(h)) == first_run
    end

    @testset "the plan owns its kernel" begin
        # The user's array is read once, at plan time, and nothing afterwards
        # depends on it. This is the guarantee that lets the caller free it.
        h = _noise(WIDE_COMPLEX, (16,))
        k = _kernel_noise(WIDE_COMPLEX, (16,))
        kernel = _upload(k)
        plan = VkFFT.plan_conv(_upload(h), kernel, 1)
        expected = Array(plan * _upload(h))

        @test Array(kernel) == k # the kernel plan is a forward transform, not an inverse
        copyto!(kernel, fill(WIDE_COMPLEX(1234), 16))
        @test Array(plan * _upload(h)) == expected
        @test _relmax(expected, _conv_ref(h, k, (1,))) < wide_rtol
    end

    @testset "two kernels, two plans, one shape" begin
        # The cache-aliasing regression. Two plans of identical shape and
        # different kernels must be two applications with two kernel buffers,
        # both live at once and neither answering with the other's kernel.
        VkFFT.clear_cache!()
        h = _noise(WIDE_COMPLEX, (16, 8))
        k1 = _kernel_noise(WIDE_COMPLEX, (16, 8))
        k2 = _noise(WIDE_COMPLEX, (16, 8)) .+ 1
        x = _upload(h)

        p1 = VkFFT.plan_conv(x, _upload(k1), (1, 2))
        p2 = VkFFT.plan_conv(x, _upload(k2), (1, 2))
        @test p1 !== p2
        @test p1.app != p2.app

        got1 = Array(p1 * x)
        got2 = Array(p2 * x)
        @test _relmax(got1, _conv_ref(h, k1, (1, 2))) < wide_rtol
        @test _relmax(got2, _conv_ref(h, k2, (1, 2))) < wide_rtol
        @test _relmax(got1, _conv_ref(h, k2, (1, 2))) > 1e-3

        # Interleaved, so neither plan can be answering out of state the other
        # one last left behind.
        @test Array(p1 * x) == got1
        @test Array(p2 * x) == got2
        @test Array(p1 * x) == got1

        # A third plan of the same shape as the first, same kernel: still its
        # own application, because the cache never sees a convolution plan.
        p3 = VkFFT.plan_conv(x, _upload(k1), (1, 2))
        @test p3 !== p1
        @test Array(p3 * x) == got1
        @test VkFFT.cache_size() == 0
    end

    @testset "unsafe_free!" begin
        h = _noise(WIDE_COMPLEX, (16,))
        plan = VkFFT.plan_conv(_upload(h), _upload(_kernel_noise(WIDE_COMPLEX, (16,))), 1)
        VkFFT.unsafe_free!(plan)
        @test plan.destroyed
        VkFFT.unsafe_free!(plan) # a second free is a no-op
        err = _thrown(() -> plan * _upload(h))
        @test err isa ArgumentError
        @test occursin("already been freed", err.msg)
    end

    @testset "plan traits and inference" begin
        h = _noise(WIDE_COMPLEX, (16, 8))
        x = _upload(h)
        plan = VkFFT.plan_conv(x, _upload(_kernel_noise(WIDE_COMPLEX, (16, 8))), (1, 2))

        @test plan isa AbstractFFTs.Plan{WIDE_COMPLEX}
        @test eltype(plan) === WIDE_COMPLEX
        @test size(plan) == (16, 8)
        @test size(plan, 2) == 8
        @test ndims(plan) == 2
        @test length(plan) == 128
        @test AbstractFFTs.output_size(plan) == (16, 8)

        @test typeof(plan) === VkFFTConvPlan{WIDE_COMPLEX, 2, BACKEND.name, 2}
        @test isconcretetype(typeof(plan))
        @test all(isconcretetype, fieldtypes(typeof(plan)))

        @test (@inferred plan * x) isa DeviceArray{WIDE_COMPLEX, 2}
        @test (@inferred mul!(similar(x), plan, x)) isa DeviceArray{WIDE_COMPLEX, 2}
    end

    @testset "finalizers" begin
        # Two apps are built per plan and one is freed at plan time, so a
        # double free would show up here rather than in the c2c suite.
        plan = VkFFT.plan_conv(_upload(_noise(ComplexF32, (256,))), _upload(_kernel_noise(ComplexF32, (256,))), 1)
        @test !plan.destroyed
        plan = nothing
        GC.gc(true)
        GC.gc(true)
        @test true # a double free or a destroy without a live context would have crashed here
    end

    @testset "show" begin
        x = _upload(_noise(ComplexF32, (16, 8)))
        k = _upload(_kernel_noise(ComplexF32, (16, 8)))
        @test sprint(show, VkFFT.plan_conv(x, k, (1, 2))) == "2D out-of-place circular convolution VkFFT plan for 16×8 ComplexF32 on dims (1, 2) [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_conv(x, k, (1, 2); correlate=true)) == "2D out-of-place circular cross-correlation VkFFT plan for 16×8 ComplexF32 on dims (1, 2) [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_conv(x, k, 1)) == "1D out-of-place circular convolution VkFFT plan for 16×8 ComplexF32 on dim (1,) ×8 features [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_conv(x, k, 1; correlate=true)) == "1D out-of-place circular cross-correlation VkFFT plan for 16×8 ComplexF32 on dim (1,) ×8 features [$(BACKEND.name)]"
    end

    @testset "no inverse and no adjoint" begin
        h = _noise(WIDE_COMPLEX, (16,))
        x = _upload(h)
        plan = VkFFT.plan_conv(x, _upload(_kernel_noise(WIDE_COMPLEX, (16,))), 1)

        @test_throws ArgumentError inv(plan)
        @test_throws ArgumentError AbstractFFTs.plan_inv(plan)
        @test_throws ArgumentError plan'
        @test_throws ArgumentError plan \ x

        @test occursin("no inverse", _thrown(() -> inv(plan)).msg)
        @test occursin("adjoint", _thrown(() -> plan').msg)
    end

    @testset "errors" begin
        h = _noise(WIDE_COMPLEX, (16, 8))
        x = _upload(h)
        k = _upload(_kernel_noise(WIDE_COMPLEX, (16, 8)))

        @testset "regions a convolution cannot take" begin
            for region in ((2,), ())
                err = _thrown(() -> VkFFT.plan_conv(x, k, region))
                @test err isa ArgumentError
                @test !occursin("VKFFT_ERROR", err.msg)
            end
            @test_throws ArgumentError VkFFT.plan_conv(_upload(_noise(WIDE_COMPLEX, (8, 5, 4))), _upload(_noise(WIDE_COMPLEX, (8, 5, 4))), (1, 3))

            cube = _upload(_noise(WIDE_COMPLEX, (8, 5, 4)))
            err = _thrown(() -> VkFFT.plan_conv(cube, cube, (1, 2, 3)))
            @test err isa ArgumentError
            @test occursin("at most two transformed dimensions", err.msg)
            @test VkFFT.plan_conv(cube, _upload(_kernel_noise(WIDE_COMPLEX, (8, 5, 4))), (1, 2)) isa VkFFTConvPlan

            # Positive control on the same array.
            @test VkFFT.plan_conv(x, k, (1, 2)) isa VkFFTConvPlan
        end

        @testset "zeropad is not supported yet" begin
            err = _thrown(() -> VkFFT.plan_conv(x, k, (1, 2); zeropad=3:6))
            @test err isa ArgumentError
            @test occursin("does not take zeropad", err.msg)
        end

        @testset "kernel has to match the transform layout" begin
            for bad in (_upload(_noise(WIDE_COMPLEX, (16,))), _upload(_noise(WIDE_COMPLEX, (8, 8))),
                        _upload(_noise(WIDE_COMPLEX, (16, 8, 2))), _upload(_noise(WIDE_COMPLEX, (8, 16))))
                err = _thrown(() -> VkFFT.plan_conv(x, bad, (1, 2)))
                @test err isa ArgumentError
                @test occursin("same size as the input", err.msg)
            end

            if BACKEND.fp64
                mixed = _upload(_noise(ComplexF32, (16, 8)))
                err = _thrown(() -> VkFFT.plan_conv(x, mixed, (1, 2)))
                @test err isa ArgumentError
                @test occursin("same element type", err.msg)
            end

            host = _noise(WIDE_COMPLEX, (16, 8))
            err = _thrown(() -> VkFFT.plan_conv(x, host, (1, 2)))
            @test err isa ArgumentError
            @test occursin("no backend for arrays of type", err.msg)
        end

        @testset "element types" begin
            reals = BACKEND.fp64 ? (_upload(_noise(Float64, (16, 8))), _upload(_noise(Float32, (16, 8)))) : (_upload(_noise(Float32, (16, 8))),)
            for bad in reals
                err = _thrown(() -> VkFFT.plan_conv(bad, bad, (1, 2)))
                @test err isa ArgumentError
                @test occursin("convolves complex arrays", err.msg)
            end
        end

        @testset "application" begin
            plan = VkFFT.plan_conv(x, k, (1, 2))

            err = _thrown(() -> mul!(x, plan, x))
            @test err isa ArgumentError
            @test occursin("no in-place convolution plan", err.msg)

            @test_throws ArgumentError plan * _upload(_noise(WIDE_COMPLEX, (8, 8)))
            @test_throws ArgumentError mul!(_upload(_noise(WIDE_COMPLEX, (8, 8))), plan, x)
            @test_throws ArgumentError plan * h
            if BACKEND.fp64
                @test_throws ArgumentError plan * _upload(_noise(ComplexF32, (16, 8)))
            end
        end
    end
end
