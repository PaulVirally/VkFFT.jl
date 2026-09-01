# Plan algebra: the parts of the AbstractFFTs interface that are not "does the
# transform come out right", plus the plan cache.

@testset verbose = true "interface" begin
    rtol = _rtol(WIDE_COMPLEX)
    h = _noise(WIDE_COMPLEX, (8, 6, 4))
    x = _upload(h)

    @testset "plan traits" begin
        plan = VkFFT.plan_fft(x, (1, 3))
        @test plan isa AbstractFFTs.Plan{WIDE_COMPLEX}
        @test eltype(plan) === WIDE_COMPLEX
        @test size(plan) == (8, 6, 4)
        @test size(plan, 2) == 6
        @test ndims(plan) == 3
        @test length(plan) == 8 * 6 * 4
        @test AbstractFFTs.fftdims(plan) == (1, 3)
        @test AbstractFFTs.output_size(plan) == (8, 6, 4)

        @test isconcretetype(typeof(plan))
        @test typeof(plan) === VkFFTPlan{WIDE_COMPLEX, 3, false, BACKEND.name, 2}
        @test typeof(VkFFT.plan_fft!(x, (1, 3))) === VkFFTPlan{WIDE_COMPLEX, 3, true, BACKEND.name, 2}
        @test all(isconcretetype, fieldtypes(typeof(plan))[1:(end - 1)])
        @test fieldtype(typeof(plan), :pinv) === Union{Nothing, typeof(plan)}
    end

    @testset "inference" begin
        # A tuple, integer or default region carries its length in the type, so
        # the whole plan type is inferable and applying it needs no dynamic
        # dispatch. A range region only gets its length at run time, which costs
        # the region-length parameter and nothing else.
        @test (@inferred VkFFT.plan_fft(x, (1, 3))) isa VkFFTPlan{WIDE_COMPLEX, 3, false, BACKEND.name, 2}
        @test (@inferred VkFFT.plan_fft(x, 2)) isa VkFFTPlan{WIDE_COMPLEX, 3, false, BACKEND.name, 1}
        @test (@inferred VkFFT.plan_fft(x)) isa VkFFTPlan{WIDE_COMPLEX, 3, false, BACKEND.name, 3}
        @test Base.return_types(VkFFT.plan_fft, (typeof(x), UnitRange{Int}))[1] == VkFFTPlan{WIDE_COMPLEX, 3, false, BACKEND.name}

        plan = VkFFT.plan_fft(x, (1, 3))
        @test (@inferred plan * x) isa DeviceArray{WIDE_COMPLEX, 3}
        @test (@inferred mul!(similar(x), plan, x)) isa DeviceArray{WIDE_COMPLEX, 3}
    end

    @testset "show" begin
        @test sprint(show, VkFFT.plan_fft(x, (1, 3))) == "2D out-of-place complex forward VkFFT plan for 8×6×4 $WIDE_COMPLEX on dims (1, 3) [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_fft!(x, 2)) == "1D in-place complex forward VkFFT plan for 8×6×4 $WIDE_COMPLEX on dim (2,) [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_bfft(x, 2)) == "1D out-of-place complex unnormalized inverse VkFFT plan for 8×6×4 $WIDE_COMPLEX on dim (2,) [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_ifft(x, 2)) == "1D out-of-place complex normalized inverse VkFFT plan for 8×6×4 $WIDE_COMPLEX on dim (2,) [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_fft(x, ())) == "trivial out-of-place complex forward VkFFT plan for 8×6×4 $WIDE_COMPLEX on dims () [$(BACKEND.name)]"
    end

    @testset "inv, plan_inv and \\" begin
        for region in ((1, 2, 3), (2,), (1, 3))
            @testset "region=$region" begin
                forward = VkFFT.plan_fft(x, region)
                y = forward * x

                # A forward plan inverts to VkFFT's in-kernel normalized
                # inverse, so no ScaledPlan and no extra scaling kernel.
                inverse = inv(forward)
                @test inverse isa VkFFTPlan
                @test inverse.normalize
                @test inverse.direction == VkFFT.INVERSE
                @test inv(forward) === inverse                  # cached in the pinv field
                @test _relmax(inverse * y, h) < rtol
                @test _relmax(forward \ y, h) < rtol
                @test _relmax(AbstractFFTs.plan_inv(forward) * y, h) < rtol

                # ... and the normalized inverse inverts straight back to the
                # forward plan.
                @test inv(inverse) === forward
                @test _relmax(inv(inverse) * x, fft(h, region)) < rtol

                # The unnormalized inverse is the one case that needs a
                # ScaledPlan, because VkFFT's normalize only exists on the
                # inverse direction.
                backward = VkFFT.plan_bfft(x, region)
                scaled = inv(backward)
                @test scaled isa AbstractFFTs.ScaledPlan
                @test scaled.p isa VkFFTPlan
                @test scaled.scale ≈ 1 / prod(size(h)[collect(region)])
                @test _relmax(scaled * (backward * x), h) < rtol
                @test _relmax(backward \ (backward * x), h) < rtol

                out = similar(x)
                ldiv!(out, forward, y)
                @test _relmax(out, h) < rtol
            end
        end
    end

    @testset "adjoints" begin
        v = _noise(WIDE_COMPLEX, (8, 6, 4))
        w = _upload(v)

        for region in ((1, 2, 3), (2,), (1, 3), ())
            @testset "region=$region" begin
                for plan in (VkFFT.plan_fft(x, region), VkFFT.plan_bfft(x, region), VkFFT.plan_ifft(x, region))
                    @test (plan')' === plan
                    @test size(plan') == AbstractFFTs.output_size(plan)
                    @test AbstractFFTs.fftdims(plan') == AbstractFFTs.fftdims(plan)

                    # The identity AD needs: <v, P x> == <P' v, x>.
                    left = dot(v, Array(plan * x))
                    right = dot(Array(plan' * w), h)
                    @test abs(left - right) <= rtol * abs(left)
                end
            end
        end

        # The normalized inverse needs its own adjoint style: FFTAdjointStyle
        # assumes an unnormalized plan and would be off by N^2 here.
        @test AbstractFFTs.AdjointStyle(VkFFT.plan_fft(x, 2)) === AbstractFFTs.FFTAdjointStyle()
        @test AbstractFFTs.AdjointStyle(VkFFT.plan_bfft(x, 2)) === AbstractFFTs.FFTAdjointStyle()
        @test AbstractFFTs.AdjointStyle(VkFFT.plan_ifft(x, 2)) === VkFFT.VkFFTNormalizedAdjointStyle()
    end

    @testset "ScaledPlan arithmetic" begin
        # A ScaledPlan's scale reaches the device as the scalar of an rmul!, so
        # it has to be a type the device has arithmetic for. An Int scale is
        # fine on its own (measured on Metal: rmul! of an MtlArray{ComplexF32}
        # by an Int64 works, by a Float64 fails to compile for want of a
        # double). What is not fine is the Float64 reciprocal inv takes of it,
        # which is why the plan carries the scale in its own real precision.
        reference = fft(h, (1, 2))
        plan = VkFFT.plan_fft(x, (1, 2))

        for three in (3, 3.0, Float32(3))
            @testset "scale $three::$(typeof(three))" begin
                scaled = three * plan
                @test scaled.scale === WIDE_REAL(3)
                @test (plan * three).scale === WIDE_REAL(3)
                @test _relmax(scaled * x, 3 .* reference) < rtol
                @test _relmax((plan * three) * x, 3 .* reference) < rtol

                # The case the coercion exists for: inv of a ScaledPlan is the
                # reciprocal of its scale, and 1/3 in the plan's precision is
                # what a device without double can apply.
                inverse = inv(scaled)
                @test inverse.scale === inv(WIDE_REAL(3))
                @test _relmax(inverse * (scaled * x), h) < rtol
                @test _relmax(AbstractFFTs.plan_inv(scaled) * (scaled * x), h) < rtol
            end
        end

        # A complex scale stays complex, in the plan's precision, since a phase
        # on a complex plan is meaningful.
        rotated = im * plan
        @test rotated.scale === Complex{WIDE_REAL}(im)
        @test _relmax(rotated * x, im .* reference) < rtol
    end

    @testset "AbstractFFTs.TestUtils.test_plan" begin
        # TestUtils.test_complex_ffts is deliberately not called: it drives the
        # global fft / plan_fft, which would require extending
        # AbstractFFTs.plan_fft on the device array type. That is the type
        # piracy this package exists to avoid, and the backend package is free
        # to claim that dispatch itself. test_plan takes a plan directly, so it
        # applies unchanged.
        #
        # TestUtils.test_plan_adjoint is also skipped: it builds its cotangent
        # as a host Array and then calls dot(::Array, ::device array), which
        # LinearAlgebra sends to BLAS.dotc and which fails inside the backend
        # package, not here. The dot test above covers the same identity.
        hh = _noise(WIDE_COMPLEX, (8, 4))
        for region in (1, 2, (1, 2))
            AbstractFFTs.TestUtils.test_plan(VkFFT.plan_fft(_upload(hh), region), _upload(hh), _upload(fft(hh, region)))
            AbstractFFTs.TestUtils.test_plan(VkFFT.plan_fft!(_upload(hh), region), _upload(hh), _upload(fft(hh, region)); inplace_plan=true)
        end
    end

    @testset "plan cache" begin
        VkFFT.clear_cache!()
        @test VkFFT.cache_size() == 0

        first_plan = VkFFT.plan_fft(x, (1, 2))
        @test VkFFT.cache_size() == 1
        @test VkFFT.plan_fft(x, (1, 2)) === first_plan
        @test VkFFT.plan_fft(_upload(h), (1, 2)) === first_plan      # keyed on shape, not identity
        @test VkFFT.plan_fft(x, (2, 1)) === first_plan               # region is canonicalized first

        # Everything in the key has to separate plans.
        @test VkFFT.plan_fft!(x, (1, 2)) !== first_plan
        @test VkFFT.plan_bfft(x, (1, 2)) !== first_plan
        @test VkFFT.plan_ifft(x, (1, 2)) !== first_plan
        @test VkFFT.plan_fft(x, (1, 3)) !== first_plan
        @test VkFFT.plan_fft(_upload(_noise(WIDE_COMPLEX, (8, 6, 2))), (1, 2)) !== first_plan

        # The precision is in the key too, on a backend that has a second one.
        # Half precision gets the same check in fp16.jl.
        precisions = 0
        if BACKEND.fp64
            @test VkFFT.plan_fft(_upload(_noise(ComplexF32, (8, 6, 4))), (1, 2)) !== first_plan
            precisions = 1
        end
        @test VkFFT.cache_size() == 6 + precisions

        # A cache hit gives back a plan whose application still matches FFTW,
        # which is the point of caching at all.
        @test _relmax(VkFFT.plan_fft(x, (1, 2)) * x, fft(h, (1, 2))) < rtol

        VkFFT.clear_cache!()
        @test VkFFT.cache_size() == 0
        @test VkFFT.plan_fft(x, (1, 2)) !== first_plan

        # Concurrent planning must not race or hand back a half-built plan.
        VkFFT.clear_cache!()
        tasks = [Threads.@spawn VkFFT.plan_fft(_upload(_noise(WIDE_COMPLEX, (16, 4))), (1, 2)) for _ in 1:8]
        plans = fetch.(tasks)
        @test all(p -> p === plans[1], plans)
        @test VkFFT.cache_size() == 1
    end

    @testset "finalizers" begin
        VkFFT.clear_cache!()
        plan = VkFFT.plan_fft(_upload(_noise(ComplexF32, (256,))), 1)
        @test !plan.destroyed
        VkFFT.clear_cache!()
        plan = nothing
        GC.gc(true)
        GC.gc(true)
        @test true # a double free or a destroy without a live context would have crashed here

        # An explicit free is idempotent, which is what makes the finalizer safe
        # to run twice.
        early = VkFFT.plan_fft(_upload(_noise(ComplexF32, (128,))), 1)
        VkFFT.unsafe_free!(early)
        @test early.destroyed
        @test early.app == C_NULL
        VkFFT.unsafe_free!(early)
        @test early.destroyed
    end
end
