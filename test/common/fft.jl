# The complex-to-complex numerical matrix: every plan is scored against FFTW on
# the host copy, at the measured per-precision tolerance, never at a bare ≈. The
# 1d size list comes from the runner, because one backend's codegen paths are
# not another's.

"""
    _test_transform(::Type{T}, dims, region, inplace::Bool)

Runs the three c2c transforms plus a plan-reuse and a round-trip check on one configuration.
"""
function _test_transform(::Type{T}, dims, region, inplace::Bool) where T
    h = _noise(T, dims)
    rtol = _rtol(T)

    for op in (:fft, :bfft, :ifft)
        reference = op === :fft ? fft(h, region) : (op === :bfft ? bfft(h, region) : ifft(h, region))
        x = _upload(h)
        plan = _plan_for(op, x, region, inplace)
        got = inplace ? plan * x : mul!(similar(x), plan, x)
        inplace && @test got === x
        @test _relmax(got, reference) < rtol
    end

    # Plan reuse: the same plan applied three times to the same input has to
    # give bit-identical results, which also catches a plan that quietly depends
    # on the buffer it was created with, and a command buffer whose work is read
    # before it has run.
    x = _upload(h)
    plan = _plan_for(:fft, x, region, false)
    first_run = Array(plan * x)
    @test Array(plan * x) == first_run
    @test Array(mul!(similar(x), plan, x)) == first_run
    @test _relmax(first_run, fft(h, region)) < rtol

    # Round trip through the normalized inverse, which is where VkFFT's
    # in-kernel 1/N is exercised over the transformed axes only.
    x = _upload(h)
    forward = _plan_for(:fft, x, region, inplace)
    y = forward * x
    back = inv(forward) * y
    @test _relmax(back, h) < rtol

    return nothing
end

@testset verbose = true "fft" begin
    @testset "1d $T n=$n inplace=$inplace" for T in COMPLEX_TYPES,
                                               n in (T === ComplexF64 ? (BACKEND.sizes_1d..., BACKEND.sizes_1d_fp64_only...) : BACKEND.sizes_1d),
                                               inplace in (false, true)
        _test_transform(T, (n,), (1,), inplace)
    end

    @testset "$(join(dims, 'x')) region=$region $T inplace=$inplace" for (dims, regions) in SHAPE_CASES,
                                                                         region in regions,
                                                                         T in COMPLEX_TYPES,
                                                                         inplace in (false, true)
        _test_transform(T, dims, region, inplace)
    end

    @testset "batched along a longer axis" begin
        # A batch big enough that VkFFT dispatches more than one threadgroup per
        # axis, over each of the three positions a transformed axis can take.
        h = _noise(ComplexF32, (256, 64))
        x = _upload(h)
        @test _relmax(VkFFT.plan_fft(x, 1) * x, fft(h, 1)) < BACKEND.rtol_f32
        @test _relmax(VkFFT.plan_fft(x, 2) * x, fft(h, 2)) < BACKEND.rtol_f32

        v = _noise(ComplexF32, (8, 128, 6))
        w = _upload(v)
        @test _relmax(VkFFT.plan_fft(w, 2) * w, fft(v, 2)) < BACKEND.rtol_f32
        @test _relmax(VkFFT.plan_fft(w, (1, 3)) * w, fft(v, (1, 3))) < BACKEND.rtol_f32
    end

    @testset "region as any iterable" begin
        h = _noise(WIDE_COMPLEX, (8, 6, 4))
        x = _upload(h)
        rtol = _rtol(WIDE_COMPLEX)
        reference = fft(h, (1, 3))
        for region in ((1, 3), (3, 1), [1, 3], (i for i in (1, 3)))
            @test _relmax(VkFFT.plan_fft(x, region) * x, reference) < rtol
        end
        @test _relmax(VkFFT.plan_fft(x, 2) * x, fft(h, 2)) < rtol
        @test _relmax(VkFFT.plan_fft(x, 1:2) * x, fft(h, 1:2)) < rtol
    end

    @testset "batched region needs no extra allocation" begin
        # number_batches must count only the dimensions absent from size[],
        # never the omitted ones: double-counting makes VkFFT read prod(omitted
        # sizes) times past the end of the buffer. A wrong count here shows up
        # as garbage or a crash.
        h = _noise(WIDE_COMPLEX, (8, 5, 4))
        x = _upload(h)
        rtol = _rtol(WIDE_COMPLEX)
        for region in ((1,), (2,), (3,), (1, 3))
            layout = VkFFT._map_region(size(h), region)
            @test prod(layout.size[1:layout.fft_dim]) * layout.number_batches == length(h)
            @test _relmax(VkFFT.plan_fft(x, region) * x, fft(h, region)) < rtol
        end
    end
end
