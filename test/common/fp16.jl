# Half precision, on the backends that run it. VkFFT stores 16-bit values and
# computes every butterfly in fp32, so what these cases measure is the accuracy
# of the storage format rather than of half-precision arithmetic. Every
# reference is FFTW's in fp32 over the widened input, which leaves the
# transform's own error in the comparison and not the error of rounding the
# input.

# One full region, one middle omit, one interleaved 5d case. The full
# shape-and-region matrix belongs to the fp32 and fp64 sets, since region
# mapping is precision-independent.
const FP16_SHAPE_CASES = (((32, 17), ((1, 2),)),
                          ((16, 8, 4), ((1, 3),)),
                          ((2, 3, 4, 5, 6), ((2, 4),)))

# Power of two, even mixed radix, Bluestein prime.
const FP16_R2R_SIZES = (16, 12, 17)

"""
    _term_scale(a, b)

Returns `sum(abs.(a) .* abs.(b))`, the scale an adjoint identity's gap is scored against.

`<v, P x> == <P' v, x>` puts both sides through a sum whose terms cancel, and how
much they cancel is a property of the draw rather than of the plan: one case here
cancels by a factor of 400, which multiplies a gap scored against the sum by the
same 400 while the rounding that produced it is unchanged. The rounding is
proportional to the magnitude of the terms, so that is what the gap is divided
by.
"""
_term_scale(a, b) = sum(abs.(a) .* abs.(b))

"""
    _test_fp16_c2c(dims, region, inplace::Bool)

Runs the three complex-to-complex transforms, plan reuse and both round trips at fp16.
"""
function _test_fp16_c2c(dims, region, inplace::Bool)
    h = _noise16(ComplexF16, dims)
    wide = ComplexF32.(h)

    for op in (:fft, :bfft, :ifft)
        reference = op === :fft ? fft(wide, region) : (op === :bfft ? bfft(wide, region) : ifft(wide, region))
        x = _upload(h)
        plan = _plan_for(op, x, region, inplace)
        @test eltype(plan) === ComplexF16
        got = inplace ? plan * x : mul!(similar(x), plan, x)
        @test eltype(got) === ComplexF16
        @test _relmax(got, reference) < BACKEND.rtol_f16
    end

    # Plan reuse has to be bit-identical, which catches a plan that quietly
    # depends on the buffer it was created with.
    x = _upload(h)
    plan = _plan_for(:fft, x, region, false)
    first_run = Array(plan * x)
    @test Array(plan * x) == first_run
    @test Array(mul!(similar(x), plan, x)) == first_run

    # Both round trips: through VkFFT's in-kernel 1/N, and through the
    # ScaledPlan an unnormalized inverse needs, whose scale is itself a Float16.
    x = _upload(h)
    forward = _plan_for(:fft, x, region, inplace)
    @test _relmax(inv(forward) * (forward * x), wide) < BACKEND.rtol_f16

    xb = _upload(h)
    backward = _plan_for(:bfft, xb, region, inplace)
    @test _relmax(inv(backward) * (backward * xb), wide) < BACKEND.rtol_f16

    return nothing
end

"""
    _test_fp16_rfft(dims, region)

Runs the three real transforms and both round trips at fp16.
"""
function _test_fp16_rfft(dims, region)
    h = _noise16(Float16, dims)
    wide = Float32.(h)
    d = dims[first(region)]
    spectrum = rfft(wide, region)
    s16 = ComplexF16.(spectrum)

    x = _upload(h)
    forward = VkFFT.plan_rfft(x, region)
    @test typeof(forward) <: VkFFTRealPlan{Float16, ComplexF16}
    @test _relmax(forward * x, spectrum) < BACKEND.rtol_f16

    # Each inverse gets its own upload of the spectrum, because applying one
    # clobbers it.
    backward = VkFFT.plan_brfft(_upload(s16), d, region)
    normalized = VkFFT.plan_irfft(_upload(s16), d, region)
    @test _relmax(backward * _upload(s16), brfft(spectrum, d, region)) < BACKEND.rtol_f16
    @test _relmax(normalized * _upload(s16), irfft(spectrum, d, region)) < BACKEND.rtol_f16

    @test _relmax(normalized * (forward * x), wide) < BACKEND.rtol_f16
    @test _relmax(inv(forward) * (forward * x), wide) < BACKEND.rtol_f16

    return nothing
end

@testset verbose = true "fp16" begin
    @testset "c2c 1d n=$n" for n in BACKEND.sizes_1d
        _test_fp16_c2c((n,), (1,), false)
    end

    @testset "c2c 1d in-place" begin
        _test_fp16_c2c((64,), (1,), true)
    end

    @testset "c2c $(join(dims, 'x')) region=$region" for (dims, regions) in FP16_SHAPE_CASES,
                                                         region in regions
        _test_fp16_c2c(dims, region, false)
    end

    @testset "rfft 1d n=$n" for n in (16, 13, 1021)
        _test_fp16_rfft((n,), (1,))
    end

    @testset "rfft $(join(dims, 'x')) region=$region" for (dims, region) in (((16, 8), (1, 2)), ((1, 16, 4), (2,)))
        _test_fp16_rfft(dims, region)
    end

    @testset "$kind-$type n=$n" for kind in R2R_KINDS, type in 1:4, n in FP16_R2R_SIZES
        h = _noise16(Float16, (n,))
        wide = Float32.(h)
        reference = _r2r_ref(wide, kind, type, (1,))

        x = _upload(h)
        forward = _r2r_plan(x, kind, (1,); type=type)
        got = forward * x
        @test eltype(got) === Float16
        @test _relmax(got, reference) < BACKEND.rtol_f16

        # VkFFT applies the exact per-type factor in-kernel, so the round trip
        # lands on the input at a scale of one and needs no ScaledPlan.
        @test _relmax(inv(forward) * got, wide) < BACKEND.rtol_f16

        inverse = _r2r_plan(x, kind, (1,); type=type, inverse=true)
        @test _relmax(inv(inverse) * (inverse * x), wide) < BACKEND.rtol_f16
    end

    @testset "$kind-$type $(join(dims, 'x')) region=$region" for kind in R2R_KINDS, type in 1:4,
                                                                 (dims, region) in (((16, 8), (1, 2)),)
        h = _noise16(Float16, dims)
        wide = Float32.(h)
        x = _upload(h)
        forward = _r2r_plan(x, kind, region; type=type)
        got = forward * x
        @test _relmax(got, _r2r_ref(wide, kind, type, region)) < BACKEND.rtol_f16
        @test _relmax(inv(forward) * got, wide) < BACKEND.rtol_f16
    end

    @testset "plan traits and show" begin
        h = _noise16(ComplexF16, (8, 6, 4))
        x = _upload(h)
        plan = VkFFT.plan_fft(x, (1, 3))
        @test plan isa AbstractFFTs.Plan{ComplexF16}
        @test typeof(plan) === VkFFTPlan{ComplexF16, 3, false, BACKEND.name, 2}
        @test (@inferred VkFFT.plan_fft(x, (1, 3))) isa VkFFTPlan{ComplexF16, 3, false, BACKEND.name, 2}
        @test (@inferred plan * x) isa DeviceArray{ComplexF16, 3}
        @test sprint(show, plan) == "2D out-of-place complex forward VkFFT plan for 8×6×4 ComplexF16 on dims (1, 3) [$(BACKEND.name)]"

        r = _noise16(Float16, (16, 4))
        rplan = VkFFT.plan_rfft(_upload(r), (1, 2))
        @test typeof(rplan) === VkFFTRealPlan{Float16, ComplexF16, 2, BACKEND.name, 2}
        @test sprint(show, rplan) == "2D out-of-place real-to-complex forward VkFFT plan for 16×4 Float16 to 9×4 ComplexF16 on dims (1, 2) [$(BACKEND.name)]"

        dplan = VkFFT.plan_dct(_upload(r), (1, 2))
        @test typeof(dplan) === VkFFTR2RPlan{Float16, 2, false, BACKEND.name, 2}
        @test sprint(show, dplan) == "2D out-of-place DCT-II VkFFT plan for 16×4 Float16 on dims (1, 2) [$(BACKEND.name)]"
    end

    @testset "precision is part of the cache key" begin
        VkFFT.clear_cache!()
        h16 = _noise16(ComplexF16, (64,))
        h32 = _noise(ComplexF32, (64,))
        p16 = VkFFT.plan_fft(_upload(h16), 1)
        p32 = VkFFT.plan_fft(_upload(h32), 1)
        @test p16 !== p32
        @test VkFFT.plan_fft(_upload(h16), 1) === p16
        @test VkFFT.cache_size() == 2
        VkFFT.clear_cache!()
    end

    @testset "adjoints" begin
        # Scored against the magnitude of the terms the inner products sum, for
        # the reason _term_scale documents.
        for region in ((1, 2, 3), (2,), (1, 3))
            h = _noise16(ComplexF16, (8, 6, 4))
            v = ComplexF16.(_noise(ComplexF32, (8, 6, 4)) .- ComplexF32(0.7, 0.3))
            x = _upload(h)
            w = _upload(v)

            for plan in (VkFFT.plan_fft(x, region), VkFFT.plan_bfft(x, region), VkFFT.plan_ifft(x, region))
                @test (plan')' === plan
                out = _wide16(plan * x)
                left = _component_dot(ComplexF32.(v), out)
                right = _component_dot(_wide16(plan' * w), ComplexF32.(h))
                @test abs(left - right) <= BACKEND.rtol_f16 * _term_scale(ComplexF32.(v), out)
            end
        end

        for (dims, region) in (((16,), (1,)), ((120,), (1,)), ((16, 8), (1, 2)), ((16, 8, 4), (1, 3)))
            r = _noise16(Float16, dims)
            d = dims[first(region)]
            spectrum = ComplexF16.(rfft(Float32.(r), region))
            cotangent_c = ComplexF16.(_noise(ComplexF32, size(spectrum)) .- ComplexF32(0.7, 0.3))
            cotangent_r = Float16.(_noise(Float32, dims) .- 0.3f0)

            for plan in (VkFFT.plan_rfft(_upload(r), region),
                         VkFFT.plan_brfft(_upload(spectrum), d, region),
                         VkFFT.plan_irfft(_upload(spectrum), d, region))
                forward = plan.direction == VkFFT.FORWARD
                input = forward ? r : spectrum
                cotangent = forward ? cotangent_c : cotangent_r
                out = _wide16(plan * _upload(input))
                left = _component_dot(_wide16(cotangent), out)
                right = _component_dot(_wide16(plan' * _upload(cotangent)), _wide16(input))
                @test abs(left - right) <= BACKEND.rtol_f16 * _term_scale(_wide16(cotangent), out)
            end
        end

        for kind in R2R_KINDS, type in 1:4
            h = _noise16(Float16, (16, 8))
            v = Float16.(_noise(Float32, (16, 8)) .- 0.3f0)
            x = _upload(h)
            w = _upload(v)
            for plan in (_r2r_plan(x, kind, (1, 2); type=type), _r2r_plan(x, kind, (1, 2); type=type, inverse=true))
                out = _wide16(plan * x)
                left = _component_dot(Float32.(v), out)
                right = _component_dot(_wide16(plan' * w), Float32.(h))
                @test abs(left - right) <= BACKEND.rtol_f16 * _term_scale(Float32.(v), out)
            end
        end
    end

    @testset "zero-padding" begin
        for n in (64,)
            lo, hi = n ÷ 2 + 1, n

            h = _noise16(ComplexF16, (n,))
            x = _upload(h)
            plan = VkFFT.plan_fft(x, 1; zeropad=lo:hi)
            @test _relmax(plan * x, fft(_zeroed(ComplexF32.(h), 1, lo, hi))) < BACKEND.rtol_f16

            # The padded block is never read, so filling it with something large
            # has to leave the result alone.
            @test Array(plan * _upload(_filled(h, 1, lo, hi, 1000))) == Array(plan * x)

            r = _noise16(Float16, (n,))
            xr = _upload(r)
            @test _relmax(VkFFT.plan_rfft(xr, 1; zeropad=lo:hi) * xr, rfft(_zeroed(Float32.(r), 1, lo, hi))) < BACKEND.rtol_f16
        end
    end

    @testset "convolution refuses half precision" begin
        # VkFFT builds the plan, reports success at every step and computes
        # something that is no convolution: 16 points come back O(1) away from
        # the answer and 64 points come back holding infinities, where the same
        # shapes in ComplexF32 are correct to 2e-7.
        h = _noise16(ComplexF16, (16,))
        x = _upload(h)
        k = _upload(_noise16(ComplexF16, (16,)))

        err = _thrown(() -> VkFFT.plan_conv(x, k))
        @test err isa ArgumentError
        @test occursin("cannot convolve in half precision", err.msg)
        @test !occursin("VKFFT_ERROR", err.msg)

        # Positive control: the same shape in ComplexF32 plans.
        x32 = _upload(_noise(ComplexF32, (16,)))
        @test VkFFT.plan_conv(x32, _upload(_noise(ComplexF32, (16,)))) isa VkFFTConvPlan
    end
end
