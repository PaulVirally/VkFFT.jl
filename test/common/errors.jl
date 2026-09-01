# What the complex-to-complex planner and plan refuse once a device array is
# involved, plus one positive control per refusal so a blanket rejection cannot
# pass by accident. The refusals that need no device are in planner.jl.

@testset verbose = true "errors" begin
    x8 = _upload(_noise(ComplexF32, (8,)))
    h345 = _noise(ComplexF32, (3, 4, 5))
    x345 = _upload(h345)

    @testset "unsupported array types" begin
        # A host array has no backend at all, and neither does a wrapper with no
        # dense layout: a genuinely strided view is a SubArray, and an adjoint
        # or a permuted view is its own type.
        @test_throws ArgumentError VkFFT.plan_fft(_noise(ComplexF32, (8,)))
        @test_throws ArgumentError VkFFT.plan_fft(view(x345, :, 2, :))
        @test_throws ArgumentError VkFFT.plan_fft(view(x345, 2:3, :, :))
        @test_throws ArgumentError VkFFT.plan_fft(adjoint(_upload(_noise(ComplexF32, (4, 4)))))
        @test_throws ArgumentError VkFFT.plan_fft(PermutedDimsArray(_upload(_noise(ComplexF32, (4, 4))), (2, 1)))
    end

    if BACKEND.offset_views
        _skip("the c2c offset-view refusals", "this backend's arrays carry their own offset, so the positive case is in its runner")
    else
        @testset "offset arrays" begin
            # A contiguous range view is a device array in its own right sharing
            # the parent's allocation. With a nonzero offset the buffer handle
            # still points at the base of that allocation, so this has to be
            # refused rather than silently transforming the wrong elements.
            offset_view = view(x8, 3:6)
            @test offset_view isa DeviceArray
            @test _offset(offset_view) != 0
            @test_throws ArgumentError VkFFT.plan_fft(offset_view)

            # Positive control: a zero-offset contiguous view is a device array
            # in its own right, and transforming it must give the transform of
            # the view, not of the parent.
            slice = view(x345, :, :, 1)
            @test slice isa DeviceArray
            @test _offset(slice) == 0
            @test _relmax(VkFFT.plan_fft(slice) * slice, fft(h345[:, :, 1])) < BACKEND.rtol_f32
            @test _relmax(mul!(similar(slice), VkFFT.plan_fft(slice), slice), fft(h345[:, :, 1])) < BACKEND.rtol_f32
        end

        @testset "offset arrays at apply time" begin
            # A plan is reusable across buffers, so the array reaching mul! is
            # not the one the plan was made from. An offset view whose size
            # matches the plan clears every size check, and its buffer handle is
            # the base of the parent's allocation: without a layout check per
            # application this transforms the wrong elements and reports
            # success. Chunked processing over a larger buffer is exactly how a
            # caller arrives here.
            host = ComplexF32[i for i in 1:8]
            big = _upload(host)
            tail = view(big, 5:8)
            @test tail isa DeviceArray
            @test _offset(tail) != 0
            @test size(tail) == (4,)

            plan = VkFFT.plan_fft(_upload(_noise(ComplexF32, (4,))))
            out = _upload(_noise(ComplexF32, (4,)))

            @test_throws ArgumentError mul!(out, plan, tail)   # offset input
            @test_throws ArgumentError mul!(tail, plan, _upload(_noise(ComplexF32, (4,)))) # offset output
            @test_throws ArgumentError plan * tail
            @test_throws ArgumentError VkFFT.plan_fft!(_upload(_noise(ComplexF32, (4,)))) * tail

            # The refusal must not be a blanket one: the same plan still applies
            # to the head of the same buffer, which is a zero-offset device
            # array, and gives the head's transform.
            head = view(big, 1:4)
            @test _offset(head) == 0
            @test _relmax(mul!(out, plan, head), fft(host[1:4])) < BACKEND.rtol_f32

            # An array with no backend at all reaches mul! whenever its eltype
            # and ndims match the plan, so it has to be refused there too rather
            # than at the ccall.
            @test_throws ArgumentError mul!(Array{ComplexF32}(undef, 4), plan, Array{ComplexF32}(undef, 4))
        end
    end

    @testset "bad regions" begin
        @test_throws ArgumentError VkFFT.plan_fft(x8, (2,))
        @test_throws ArgumentError VkFFT.plan_fft(x8, (0,))
        @test_throws ArgumentError VkFFT.plan_fft(x345, (1, 1))
        @test_throws ArgumentError VkFFT.plan_fft(x345, (4,))
        @test_throws ArgumentError VkFFT.plan_fft(x345, 1:4)
    end

    @testset "too interleaved" begin
        deep = _upload(_noise(ComplexF32, ntuple(_ -> 2, 13)))
        err = _thrown(() -> VkFFT.plan_fft(deep, ntuple(i -> 2i - 1, 7)))
        @test err isa ArgumentError
        @test occursin("Reshape or permute", err.msg)
        @test !occursin("VKFFT_ERROR", err.msg) # a raw code would be a planner bug
    end

    @testset "identity semantics instead of a silent no-op" begin
        # VkFFT would build both of these, report VKFFT_SUCCESS and dispatch
        # nothing, leaving an out-of-place caller with uninitialized memory.
        # Julia applies them as a copy instead, so the results are exact.
        h = _noise(WIDE_COMPLEX, (4, 1))
        x = _upload(h)

        for region in ((), (2,))
            @testset "region=$region" begin
                plan = VkFFT.plan_fft(x, region)
                @test Array(plan * x) == h
                @test Array(inv(plan) * (plan * x)) == h
                @test Array(VkFFT.plan_bfft(x, region) * x) == h
                @test Array(VkFFT.plan_ifft(x, region) * x) == h
            end
        end

        xi = _upload(h)
        in_place = VkFFT.plan_fft!(xi, (2,))
        @test (in_place * xi) === xi
        @test Array(xi) == h # in place, so leaving the array alone is the right answer
    end

    @testset "mismatched arrays" begin
        p = VkFFT.plan_fft(x8)
        @test_throws ArgumentError mul!(_upload(_noise(ComplexF32, (4,))), p, _upload(_noise(ComplexF32, (4,))))
        @test_throws ArgumentError mul!(_upload(_noise(ComplexF32, (8,))), p, _upload(_noise(ComplexF32, (4,))))
        @test_throws ArgumentError mul!(x8, p, x8)                                  # out-of-place needs distinct arrays
        @test_throws ArgumentError p * _upload(_noise(ComplexF32, (4,)))
        @test_throws ArgumentError p * _upload(_noise(ComplexF32, (2, 4)))
        @test_throws ArgumentError mul!(Array{ComplexF32}(undef, 8), p, Array{ComplexF32}(undef, 8))

        pip = VkFFT.plan_fft!(x8)
        @test_throws ArgumentError mul!(_upload(_noise(ComplexF32, (8,))), pip, x8) # in-place needs y === x

        wrong = Any[_upload(_noise(ComplexF32, (2, 4)))]
        if BACKEND.fp64
            @test_throws ArgumentError mul!(_upload(_noise(ComplexF64, (8,))), p, _upload(_noise(ComplexF64, (8,))))
            push!(wrong, _upload(_noise(ComplexF64, (8,))))
        end

        # The wrong-eltype and wrong-ndims fallbacks report the plan's own
        # shape, which means they interpolate ndims(plan). That resolves to
        # AbstractFFTs' Base.ndims(::Plan), so assert the rendered text: a
        # missing method would mask these behind a MethodError.
        for bad in wrong
            err = _thrown(() -> p * bad)
            @test err isa ArgumentError
            @test occursin("this VkFFT plan transforms 1-dimensional ComplexF32 arrays", err.msg)
        end
    end

    @testset "freed plans" begin
        VkFFT.clear_cache!()
        p = VkFFT.plan_fft(_upload(_noise(ComplexF32, (32,))))
        y = _upload(_noise(ComplexF32, (32,)))
        VkFFT.unsafe_free!(p)
        @test p.destroyed
        @test p.app == C_NULL
        @test_throws ArgumentError mul!(y, p, _upload(_noise(ComplexF32, (32,))))
        @test VkFFT.unsafe_free!(p) === nothing # idempotent, as the finalizer needs it to be

        # Plans dropped on the floor are freed by the garbage collector, from
        # outside any pool or context of ours.
        VkFFT.clear_cache!()
        for _ in 1:16
            VkFFT.plan_fft(_upload(_noise(ComplexF32, (64,))), 1)
            VkFFT.clear_cache!()
        end
        GC.gc(true)
        GC.gc(true)
        @test true # a double free or a release into a dead queue would have crashed here
    end
end
