# The DCT/DST family: the numerical matrix against FFTW's r2r kinds, both round
# trips, the adjoint algebra, the plan algebra, and the refusals that need a
# device array. The refusals with no C library involved are in planner.jl.

# One size per codegen path: 16 power of two, 13 Bluestein prime, 15 odd
# composite, 120 mixed radix (whose DCT-I also maps to a 238-point internal
# transform). The per-type matrix stays complete across all 8 (kind, type)
# pairs, since each type is its own codegen.
const R2R_SIZES = (16, 13, 15, 120)

"""
    _test_r2r(::Type{T}, kind::Symbol, type::Int, dims, region, inplace::Bool)

Runs the forward transform, the inverse and the round trip on one configuration.
"""
function _test_r2r(::Type{T}, kind::Symbol, type::Int, dims, region, inplace::Bool) where T
    rtol = _rtol(T)
    n = length(dims)
    h = _noise(T, dims)
    reference = _r2r_ref(h, kind, type, region)

    forward = _r2r_plan(_upload(h), kind, region; type=type, inplace=inplace)
    x = _upload(h)
    @test _relmax(forward * x, reference) < rtol
    if inplace
        reused = _upload(h) # the same plan on a second buffer, which is the whole point of caching them
        @test _relmax(mul!(reused, forward, reused), reference) < rtol
    else
        @test _relmax(mul!(DeviceArray{T, n}(undef, dims), forward, _upload(h)), reference) < rtol
    end

    # The inverse is FFTW's inverse kind divided by the exact per-type factor,
    # which VkFFT applies in-kernel.
    scale = prod(i -> _r2r_scale(kind, type, dims[i]), region)
    inverse = _r2r_plan(_upload(reference), kind, region; type=type, inverse=true, inplace=inplace)
    @test _relmax(inverse * _upload(h), _r2r_ref(h, kind, _r2r_inverse_type(type), region) ./ scale) < rtol

    # ... so the round trip lands on 1.0, not on the factor.
    @test _relmax(inverse * _upload(reference), h) < rtol

    return nothing
end

@testset verbose = true "r2r" begin
    rtol = _rtol(WIDE_REAL)
    h864 = _noise(WIDE_REAL, (8, 6, 4))
    x864 = _upload(h864)

    @testset "1d $T $kind$type n=$n" for T in REAL_TYPES,
                                         kind in R2R_KINDS,
                                         type in 1:4,
                                         n in R2R_SIZES
        _test_r2r(T, kind, type, (n,), (1,), false)
    end

    # In-place is its own codegen (one buffer, not two) but not its own
    # per-size story, so one size covers it per type and precision.
    @testset "1d in-place $T $kind$type" for T in REAL_TYPES, kind in R2R_KINDS, type in 1:4
        _test_r2r(T, kind, type, (16,), (1,), true)
    end

    @testset "shapes and regions $kind$type" for kind in R2R_KINDS, type in 1:4
        # One full 2d region (out-of-place and in-place), a middle omit, a
        # trailing batch fold and a leading omit, which with the 1d matrix
        # covers every shape class _map_region can produce around a transformed
        # axis. The scale factor must come from the transformed axes only.
        _test_r2r(WIDE_REAL, kind, type, (16, 4), (1, 2), false)
        _test_r2r(Float32, kind, type, (16, 4), (1, 2), true)
        _test_r2r(WIDE_REAL, kind, type, (8, 5, 3), (1, 3), false)
        _test_r2r(Float32, kind, type, (16, 4), (1,), false)
        _test_r2r(Float32, kind, type, (4, 16), (2,), false)
    end

    @testset "region as any iterable" begin
        reference = _r2r_ref(h864, :dct, 2, (1, 3))
        for region in ((1, 3), (3, 1), [1, 3], (i for i in (1, 3)))
            @test _relmax(VkFFT.plan_dct(x864, region) * x864, reference) < rtol
        end
        @test _relmax(VkFFT.plan_dst(x864, 1) * x864, _r2r_ref(h864, :dst, 2, 1)) < rtol
        @test _relmax(VkFFT.plan_dct(x864, 1:2) * x864, _r2r_ref(h864, :dct, 2, 1:2)) < rtol
    end

    @testset "default type is 2" begin
        @test VkFFT.plan_dct(x864, 1).type == 2
        @test VkFFT.plan_dst(x864, 1).type == 2
        @test VkFFT.plan_idct(x864, 1).type == 2
        @test VkFFT.plan_idst(x864, 1).type == 2
        @test _relmax(VkFFT.plan_dct(x864, 1) * x864, _r2r_ref(h864, :dct, 2, 1)) < rtol
    end

    @testset "plan traits" begin
        plan = VkFFT.plan_dct(x864, (1, 3); type=3)
        @test plan isa AbstractFFTs.Plan{WIDE_REAL}
        @test eltype(plan) === WIDE_REAL
        @test size(plan) == (8, 6, 4)
        @test size(plan, 2) == 6
        @test ndims(plan) == 3
        @test AbstractFFTs.fftdims(plan) == (1, 3)
        @test AbstractFFTs.output_size(plan) == (8, 6, 4)
        @test plan.kind === :dct
        @test plan.type == 3
        @test plan.zeropad == (0, 0)

        @test isconcretetype(typeof(plan))
        @test typeof(plan) === VkFFTR2RPlan{WIDE_REAL, 3, false, BACKEND.name, 2}
        @test typeof(VkFFT.plan_dst!(x864, (1, 3))) === VkFFTR2RPlan{WIDE_REAL, 3, true, BACKEND.name, 2}
        @test all(isconcretetype, fieldtypes(typeof(plan))[1:(end - 1)])
        @test fieldtype(typeof(plan), :pinv) === Union{Nothing, typeof(plan)}
    end

    @testset "inference" begin
        @test (@inferred VkFFT.plan_dct(x864, (1, 3))) isa VkFFTR2RPlan{WIDE_REAL, 3, false, BACKEND.name, 2}
        @test (@inferred VkFFT.plan_dst(x864, 2)) isa VkFFTR2RPlan{WIDE_REAL, 3, false, BACKEND.name, 1}
        @test (@inferred VkFFT.plan_idct(x864)) isa VkFFTR2RPlan{WIDE_REAL, 3, false, BACKEND.name, 3}
        @test (@inferred VkFFT.plan_idst!(x864, (1, 3))) isa VkFFTR2RPlan{WIDE_REAL, 3, true, BACKEND.name, 2}

        plan = VkFFT.plan_dct(x864, (1, 3))
        @test (@inferred plan * x864) isa DeviceArray{WIDE_REAL, 3}
        @test (@inferred mul!(DeviceArray{WIDE_REAL, 3}(undef, (8, 6, 4)), plan, x864)) isa DeviceArray{WIDE_REAL, 3}
    end

    @testset "show" begin
        @test sprint(show, VkFFT.plan_dct(x864, (1, 3))) == "2D out-of-place DCT-II VkFFT plan for 8×6×4 $WIDE_REAL on dims (1, 3) [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_dct!(x864, 2; type=1)) == "1D in-place DCT-I VkFFT plan for 8×6×4 $WIDE_REAL on dim (2,) [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_dst(x864, 2; type=3)) == "1D out-of-place DST-III VkFFT plan for 8×6×4 $WIDE_REAL on dim (2,) [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_idst(x864, 2; type=4)) == "1D out-of-place normalized inverse DST-IV VkFFT plan for 8×6×4 $WIDE_REAL on dim (2,) [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_idst!(x864, 2)) == "1D in-place normalized inverse DST-II VkFFT plan for 8×6×4 $WIDE_REAL on dim (2,) [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_idct(x864, (1, 3))) == "2D out-of-place normalized inverse DCT-II VkFFT plan for 8×6×4 $WIDE_REAL on dims (1, 3) [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_dct(x864, ())) == "trivial out-of-place DCT-II VkFFT plan for 8×6×4 $WIDE_REAL on dims () [$(BACKEND.name)]"
    end

    @testset "inv, plan_inv and \\" begin
        for kind in R2R_KINDS, type in 1:4
            @testset "$kind$type" begin
                forward = _r2r_plan(x864, kind, (1, 3); type=type)
                spectrum = Array(forward * x864)

                # VkFFT's normalize carries the exact per-type factor, so a
                # forward plan inverts to a true inverse with no ScaledPlan and
                # no scaling kernel anywhere in this family.
                inverse = inv(forward)
                @test inverse isa VkFFTR2RPlan
                @test inverse.normalize
                @test inverse.direction == VkFFT.INVERSE
                @test inverse.kind === kind
                @test inverse.type == type
                @test inv(forward) === inverse                  # cached in the pinv field
                @test _relmax(inverse * _upload(spectrum), h864) < rtol
                @test _relmax(forward \ _upload(spectrum), h864) < rtol
                @test _relmax(AbstractFFTs.plan_inv(forward) * _upload(spectrum), h864) < rtol

                # ... and the inverse inverts straight back to the forward plan.
                @test inv(inverse) === forward
                @test _relmax(inv(inverse) * x864, spectrum) < rtol

                out = DeviceArray{WIDE_REAL, 3}(undef, (8, 6, 4))
                ldiv!(out, forward, _upload(spectrum))
                @test _relmax(out, h864) < rtol
            end
        end
    end

    @testset "adjoints" begin
        # The eight transposes, in FFTW's normalization: types 4 and RODFT00 are
        # symmetric, types 2 and 3 of a kind are each other's transpose, and
        # REDFT00 is its own transpose conjugated by a diagonal. The dot
        # identity is what actually checks the algebra.
        for T in REAL_TYPES, kind in R2R_KINDS, type in 1:4,
            (dims, region) in (((16,), (1,)), ((13,), (1,)), ((16, 8), (1, 2)),
                               ((16, 8), (1,)), ((8, 5, 3), (1, 3)), ((4, 16), (2,)))
            @testset "$T $kind$type $(join(dims, 'x')) region=$region" begin
                tol = _rtol(T)
                h = _noise(T, dims)
                cotangent = _noise(T, dims)

                for plan in (_r2r_plan(_upload(h), kind, region; type=type),
                             _r2r_plan(_upload(h), kind, region; type=type, inverse=true))
                    @test (plan')' === plan
                    @test size(plan') == AbstractFFTs.output_size(plan)
                    @test AbstractFFTs.fftdims(plan') == AbstractFFTs.fftdims(plan)

                    # <v, P x> == <P' v, x>, which is what AD needs.
                    left = dot(cotangent, Array(plan * _upload(h)))
                    right = dot(Array(plan' * _upload(cotangent)), h)
                    @test abs(left - right) <= tol * abs(left)
                end
            end
        end

        @test AbstractFFTs.AdjointStyle(VkFFT.plan_dct(x864, 1)) === VkFFT.VkFFTR2RAdjointStyle()
        @test AbstractFFTs.AdjointStyle(VkFFT.plan_idst(x864, 1)) === VkFFT.VkFFTR2RAdjointStyle()

        # The partner type the algebra needs, spelled out so a table edit cannot
        # pass the dot identity by accident on a symmetric case.
        @test VkFFT._r2r_transpose(:dct, 1) == (1, (0.5, 0.5), (2.0, 2.0))
        @test VkFFT._r2r_transpose(:dct, 2) == (3, (1.0, 1.0), (2.0, 1.0))
        @test VkFFT._r2r_transpose(:dct, 3) == (2, (0.5, 1.0), (1.0, 1.0))
        @test VkFFT._r2r_transpose(:dct, 4) == (4, (1.0, 1.0), (1.0, 1.0))
        @test VkFFT._r2r_transpose(:dst, 1) == (1, (1.0, 1.0), (1.0, 1.0))
        @test VkFFT._r2r_transpose(:dst, 2) == (3, (1.0, 1.0), (1.0, 2.0))
        @test VkFFT._r2r_transpose(:dst, 3) == (2, (1.0, 0.5), (1.0, 1.0))
        @test VkFFT._r2r_transpose(:dst, 4) == (4, (1.0, 1.0), (1.0, 1.0))

        # An inverse plan's recipe is the forward one with the weightings
        # swapped and reciprocated, because transposing an inverse inverts a
        # transpose.
        @test VkFFT._r2r_adjoint_recipe(VkFFT.plan_idct(x864, 1; type=2)) == (3, (0.5, 1.0), (1.0, 1.0))
        @test VkFFT._r2r_adjoint_recipe(VkFFT.plan_idst(x864, 1; type=3)) == (2, (1.0, 1.0), (1.0, 2.0))
        @test VkFFT._r2r_adjoint_recipe(VkFFT.plan_idct(x864, 1; type=1)) == (1, (0.5, 0.5), (2.0, 2.0))
    end

    @testset "ScaledPlan arithmetic" begin
        # A plan carries its scale in its own real precision, whatever the
        # literal was, so that the Float64 reciprocal inv would otherwise take
        # never reaches a device without double. The whole coercion is covered
        # in interface.jl. Here it is a real-to-real plan's turn to hold one.
        three = 3
        plan = VkFFT.plan_dct(x864, (1, 2))
        @test (three * plan).scale === WIDE_REAL(3)
        reference = _r2r_ref(h864, :dct, 2, (1, 2))
        @test _relmax((three * plan) * x864, 3 .* reference) < rtol
        @test _relmax((plan * three) * x864, 3 .* reference) < rtol
        @test _relmax(inv(three * plan) * ((three * plan) * x864), h864) < rtol
    end

    @testset "AbstractFFTs.TestUtils.test_plan" begin
        for kind in R2R_KINDS, (dims, region) in (((16,), (1,)), ((16, 4), (1, 2)), ((13, 3), (1,)))
            h = _noise(WIDE_REAL, dims)
            reference = _r2r_ref(h, kind, 2, region)
            AbstractFFTs.TestUtils.test_plan(_r2r_plan(_upload(h), kind, region; type=2), _upload(h), _upload(reference))
            AbstractFFTs.TestUtils.test_plan(_r2r_plan(_upload(h), kind, region; type=2, inplace=true), _upload(h), _upload(reference); inplace_plan=true)
        end
    end

    @testset "identity semantics instead of a silent no-op" begin
        # An empty region asks for no transform at all, which is the identity
        # along every axis and exact. A length-1 axis inside the region is the
        # different case, and is refused rather than quietly made an identity.
        # That refusal has to be asserted in every process: a DCT-I of a
        # length-1 axis is the one configuration that hangs inside vkfft_create
        # rather than failing.
        h = _noise(Float32, (4, 3))
        x = _upload(h)
        for kind in R2R_KINDS, type in 1:4
            @test Array(_r2r_plan(x, kind, (); type=type) * x) == h
            @test Array(_r2r_plan(x, kind, (); type=type, inverse=true) * x) == h
            @test_throws ArgumentError _r2r_plan(_upload(Float32[2.5;;]), kind, (1, 2); type=type)
        end
    end

    @testset "errors" begin
        x16 = _upload(_noise(Float32, (16,)))

        @testset "unsupported array types" begin
            @test_throws ArgumentError VkFFT.plan_dct(_noise(Float32, (16,)))
            @test_throws ArgumentError VkFFT.plan_dst(view(_upload(_noise(Float32, (3, 4, 5))), :, 2, :))
        end

        if BACKEND.offset_views
            _skip("the r2r offset-view refusals", "this backend's arrays carry their own offset, so the positive case is in its runner")
        else
            @testset "offset arrays at apply time" begin
                host = Float32[i for i in 1:16]
                big = _upload(host)
                tail = view(big, 9:16)
                @test _offset(tail) != 0

                forward = VkFFT.plan_dct(_upload(_noise(Float32, (8,))))
                out = DeviceArray{Float32, 1}(undef, (8,))
                @test_throws ArgumentError mul!(out, forward, tail)
                @test_throws ArgumentError mul!(tail, forward, _upload(_noise(Float32, (8,))))
                @test_throws ArgumentError forward * tail

                head = view(big, 1:8)
                @test _offset(head) == 0
                @test _relmax(mul!(out, forward, head), _r2r_ref(host[1:8], :dct, 2, 1)) < BACKEND.rtol_f32

                @test_throws ArgumentError mul!(Array{Float32}(undef, 8), forward, Array{Float32}(undef, 8))
            end
        end

        @testset "mismatched arrays" begin
            forward = VkFFT.plan_dct(x16)
            @test_throws ArgumentError mul!(DeviceArray{Float32, 1}(undef, (8,)), forward, x16)
            @test_throws ArgumentError mul!(x16, forward, x16)                       # out-of-place needs two arrays
            @test_throws ArgumentError forward * _upload(_noise(Float32, (8,)))
            @test_throws ArgumentError forward * _upload(_noise(ComplexF32, (16,)))

            in_place = VkFFT.plan_dct!(x16)
            @test_throws ArgumentError mul!(DeviceArray{Float32, 1}(undef, (16,)), in_place, x16)

            if BACKEND.fp64
                @test_throws ArgumentError forward * _upload(_noise(Float64, (16,)))
            end

            for bad in (_upload(_noise(ComplexF32, (16,))), _upload(_noise(Float32, (4, 4))))
                err = _thrown(() -> forward * bad)
                @test err isa ArgumentError
                @test occursin("this VkFFT plan transforms 1-dimensional Float32 arrays", err.msg)
            end
        end

        @testset "freed plans" begin
            forward = VkFFT.plan_dst(_upload(_noise(Float32, (32,))))
            VkFFT.unsafe_free!(forward)
            @test forward.destroyed
            @test forward.app == C_NULL
            @test_throws ArgumentError mul!(DeviceArray{Float32, 1}(undef, (32,)), forward, _upload(_noise(Float32, (32,))))
            @test VkFFT.unsafe_free!(forward) === nothing
        end
    end

    @testset "plan cache" begin
        VkFFT.clear_cache!()
        @test VkFFT.cache_size() == 0

        hr = _noise(WIDE_REAL, (16, 4))
        xr = _upload(hr)
        first_plan = VkFFT.plan_dct(xr, (1, 2))
        @test VkFFT.cache_size() == 1
        @test VkFFT.plan_dct(xr, (1, 2)) === first_plan
        @test VkFFT.plan_dct(_upload(hr), (2, 1)) === first_plan       # keyed on shape, region canonicalized

        # Everything the key holds has to separate plans, and the kind and the
        # type are the two this family adds.
        @test VkFFT.plan_dct(xr, (1, 2); type=3) !== first_plan
        @test VkFFT.plan_dst(xr, (1, 2)) !== first_plan
        @test VkFFT.plan_idct(xr, (1, 2)) !== first_plan
        @test VkFFT.plan_dct!(xr, (1, 2)) !== first_plan
        @test VkFFT.plan_fft(_upload(_noise(WIDE_COMPLEX, (16, 4))), (1, 2)) !== first_plan
        @test VkFFT.plan_rfft(xr, (1, 2)) !== first_plan
        @test VkFFT.cache_size() == 7

        VkFFT.clear_cache!()
        @test VkFFT.cache_size() == 0
        @test VkFFT.plan_dct(xr, (1, 2)) !== first_plan
    end

    @testset "finalizers" begin
        VkFFT.clear_cache!()
        plan = VkFFT.plan_dct(_upload(_noise(Float32, (256,))), 1)
        @test !plan.destroyed
        VkFFT.clear_cache!()
        plan = nothing
        GC.gc(true)
        GC.gc(true)
        @test true # a double free or a destroy without a live context would have crashed here
    end
end
