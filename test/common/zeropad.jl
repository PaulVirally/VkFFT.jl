# Zero-padding: the numbers against FFTW of the manually zeroed signal for every
# family that offers it, the invariance the whole feature exists for, and
# everything the planner refuses. The pure mapping is in regions.jl.

# dims, region, and the 1-based inclusive padded range on the first transformed
# dimension. Covers a block in the middle, one at the start, one running to the
# end, and a single index, plus batches, an omit and a full region.
const ZEROPAD_CASES = (((16,), (1,), 6, 11),
                       ((16,), (1,), 1, 8),
                       ((16,), (1,), 9, 16),
                       ((16,), (1,), 2, 2),
                       ((13,), (1,), 4, 9),
                       ((120,), (1,), 33, 96),
                       ((16, 4), (1,), 6, 11),
                       ((16, 4), (1, 2), 6, 11),
                       ((16, 3, 2), (1, 3), 6, 11))

@testset verbose = true "zeropad" begin
    wide_rtol = _rtol(WIDE_COMPLEX)

    @testset "c2c forward $T $(join(dims, 'x')) region=$region pad=$lo:$hi" for T in COMPLEX_TYPES,
                                                                               (dims, region, lo, hi) in ZEROPAD_CASES
        rtol = _rtol(T)
        h = _noise(T, dims)
        zeroed = _zeroed(h, 1, lo, hi)
        reference = fft(zeroed, region)

        plan = VkFFT.plan_fft(_upload(h), region; zeropad=lo:hi)
        @test plan.zeropad == (lo, hi)
        @test _relmax(plan * _upload(h), reference) < rtol

        # The control: a plan that quietly ignored zeropad would give the
        # transform of the signal as given, which this is not.
        @test _relmax(plan * _upload(h), fft(h, region)) > 1e-3

        # Garbage in the block does not reach the result, bit for bit, which is
        # the whole reason the feature exists.
        @test Array(plan * _upload(_filled(h, 1, lo, hi, 1000))) == Array(plan * _upload(h))
        @test _relmax(plan * _upload(zeroed), reference) < rtol

        # In-place, same answer.
        in_place = VkFFT.plan_fft!(_upload(h), region; zeropad=lo:hi)
        @test _relmax(in_place * _upload(_filled(h, 1, lo, hi, -7)), reference) < rtol
    end

    # An even and an odd halved axis, plus one 2d region. The batch and omit
    # crossings of the padding logic are the c2c cases above.
    @testset "r2c forward $T $(join(dims, 'x')) region=$region" for T in REAL_TYPES,
                                                                    (dims, region) in (((16,), (1,)), ((13,), (1,)),
                                                                                       ((16, 4), (1, 2)))
        lo, hi = 4, 9
        rtol = _rtol(T)
        h = _noise(T, dims)
        reference = rfft(_zeroed(h, 1, lo, hi), region)

        plan = VkFFT.plan_rfft(_upload(h), region; zeropad=lo:hi)
        @test _relmax(plan * _upload(h), reference) < rtol
        @test _relmax(plan * _upload(h), rfft(h, region)) > 1e-3
        @test Array(plan * _upload(_filled(h, 1, lo, hi, 1000))) == Array(plan * _upload(h))
    end

    # Padding is per-type codegen, so the (kind, type) matrix stays complete and
    # the shapes reduce to one 1d and one 2d case.
    @testset "r2r forward $kind$type $(join(dims, 'x')) region=$region" for kind in R2R_KINDS,
                                                                           type in 1:4,
                                                                           (dims, region) in (((16,), (1,)), ((16, 4), (1, 2)))
        lo, hi = 6, 11
        h = _noise(WIDE_REAL, dims)
        reference = _r2r_ref(_zeroed(h, 1, lo, hi), kind, type, region)

        plan = _r2r_plan(_upload(h), kind, region; type=type, zeropad=lo:hi)
        @test _relmax(plan * _upload(h), reference) < _rtol(WIDE_REAL)
        @test _relmax(plan * _upload(h), _r2r_ref(h, kind, type, region)) > 1e-3
        @test Array(plan * _upload(_filled(h, 1, lo, hi, 1000))) == Array(plan * _upload(h))

        in_place = _r2r_plan(_upload(h), kind, region; type=type, inplace=true, zeropad=lo:hi)
        @test _relmax(in_place * _upload(_filled(h, 1, lo, hi, -7)), reference) < _rtol(WIDE_REAL)
    end

    @testset "a whole padded axis transforms zeros" begin
        h = _noise(WIDE_COMPLEX, (16,))
        @test maximum(abs.(Array(VkFFT.plan_fft(_upload(h), 1; zeropad=1:16) * _upload(h)))) < 1e-12
    end

    @testset "an inverse skips writes rather than writing zeros" begin
        # The documented behaviour, and the reason the padded range of an
        # inverse is undefined: the destination keeps whatever it held.
        h = _noise(WIDE_COMPLEX, (16,))
        plan = VkFFT.plan_ifft(_upload(h), 1; zeropad=6:11)
        sentinel = fill(WIDE_COMPLEX(-987654), 16)
        out = _upload(sentinel)
        mul!(out, plan, _upload(h))
        got = Array(out)
        @test got[6:11] == sentinel[6:11]
        @test all(got[1:5] .!= sentinel[1:5])
        @test all(got[12:16] .!= sentinel[12:16])
    end

    @testset "zeropad joins the cache key" begin
        VkFFT.clear_cache!()
        h = _noise(WIDE_COMPLEX, (16, 4))
        x = _upload(h)

        plain = VkFFT.plan_fft(x, (1, 2))
        padded = VkFFT.plan_fft(x, (1, 2); zeropad=6:11)
        @test padded !== plain
        @test VkFFT.plan_fft(x, (1, 2); zeropad=6:11) === padded
        @test VkFFT.plan_fft(x, (1, 2); zeropad=(6:11,)) === padded   # a one-element tuple is the same range
        @test VkFFT.plan_fft(x, (1, 2); zeropad=7:11) !== padded
        @test VkFFT.plan_fft(x, (1, 2); zeropad=6:12) !== padded
        @test VkFFT.plan_fft(x, (1, 2); zeropad=nothing) === plain
        @test VkFFT.cache_size() == 4
        VkFFT.clear_cache!()
    end

    @testset "show" begin
        x = _upload(_noise(WIDE_COMPLEX, (16, 4)))
        r = _upload(_noise(WIDE_REAL, (16, 4)))
        @test sprint(show, VkFFT.plan_fft(x, (1, 2); zeropad=6:11)) == "2D out-of-place complex forward VkFFT plan for 16×4 $WIDE_COMPLEX on dims (1, 2) zeropad=6:11 [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_rfft(r, 1; zeropad=6:11)) == "1D out-of-place real-to-complex forward VkFFT plan for 16×4 $WIDE_REAL to 9×4 $WIDE_COMPLEX on dim (1,) zeropad=6:11 [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_dct(r, 1; zeropad=6:11)) == "1D out-of-place DCT-II VkFFT plan for 16×4 $WIDE_REAL on dim (1,) zeropad=6:11 [$(BACKEND.name)]"
    end

    @testset "no inverse and no adjoint" begin
        h = _noise(WIDE_COMPLEX, (16,))
        hr = _noise(WIDE_REAL, (16,))
        x = _upload(h)
        r = _upload(hr)

        cases = ((VkFFT.plan_fft(x, 1; zeropad=6:11), h),
                 (VkFFT.plan_ifft(x, 1; zeropad=6:11), h),
                 (VkFFT.plan_rfft(r, 1; zeropad=6:11), hr),
                 (VkFFT.plan_dct(r, 1; zeropad=6:11), hr),
                 (VkFFT.plan_idst(r, 1; zeropad=6:11), hr))
        for (plan, input) in cases
            @test_throws ArgumentError inv(plan)
            @test_throws ArgumentError AbstractFFTs.plan_inv(plan)
            cotangent = _upload(Array(plan * _upload(input)))
            @test_throws ArgumentError plan' * cotangent
        end

        @test occursin("has no inverse", _thrown(() -> inv(VkFFT.plan_fft(x, 1; zeropad=6:11))).msg)
        @test occursin("adjoint of a zero-padded", _thrown(() -> VkFFT.plan_fft(x, 1; zeropad=6:11)' * x).msg)

        # Positive control: the same shapes without padding invert and adjoin.
        @test inv(VkFFT.plan_fft(x, 1)) isa VkFFTPlan
        @test inv(VkFFT.plan_dct(r, 1)) isa VkFFTR2RPlan
        @test VkFFT.plan_dct(r, 1)' * r isa DeviceArray{WIDE_REAL, 1}
    end

    @testset "errors" begin
        x = _upload(_noise(WIDE_COMPLEX, (16, 4)))
        r = _upload(_noise(WIDE_REAL, (16, 4)))

        @testset "only the first transformed dimension can be padded" begin
            for entry in (VkFFT.plan_fft, VkFFT.plan_fft!, VkFFT.plan_bfft, VkFFT.plan_ifft)
                err = _thrown(() -> entry(x, (2,); zeropad=1:2))
                @test err isa ArgumentError
                @test occursin("Permute", err.msg)
                @test occursin("dimension 1", err.msg)
                @test !occursin("VKFFT_ERROR", err.msg)
            end
            @test_throws ArgumentError VkFFT.plan_dct(r, (2,); zeropad=1:2)
            @test_throws ArgumentError VkFFT.plan_idst(r, (2,); zeropad=1:2)

            # A leading length-1 dimension moves the paddable axis, exactly as
            # it moves the halved axis of a real transform.
            h116 = _noise(WIDE_COMPLEX, (1, 16))
            x116 = _upload(h116)
            @test_throws ArgumentError VkFFT.plan_fft(x116, (1, 2); zeropad=6:11)
            @test _relmax(VkFFT.plan_fft(x116, (2,); zeropad=6:11) * x116, fft(_zeroed(h116, 2, 6, 11), (2,))) < wide_rtol

            # Positive control on the same array.
            @test VkFFT.plan_fft(x, (1, 2); zeropad=1:2) isa VkFFTPlan
        end

        @testset "bounds" begin
            for bad in (0:4, -3:4, 5:20, 17:17, 5:4, 12:11)
                err = _thrown(() -> VkFFT.plan_fft(x, (1, 2); zeropad=bad))
                @test err isa ArgumentError
                @test occursin("non-empty 1-based range", err.msg)
            end

            # hi == size is legal and covers the tail, which is where VkFFT's
            # own documentation contradicts its own convention.
            @test VkFFT.plan_fft(x, (1, 2); zeropad=9:16) isa VkFFTPlan
            @test VkFFT.plan_fft(x, (1, 2); zeropad=1:1) isa VkFFTPlan
        end

        @testset "shapes zeropad does not take" begin
            for bad in ((6, 11), (6:11, 1:2), [6, 11], 6, "6:11")
                err = _thrown(() -> VkFFT.plan_fft(x, (1, 2); zeropad=bad))
                @test err isa ArgumentError
                @test occursin("one 1-based inclusive range", err.msg)
            end
        end

        @testset "nothing to pad" begin
            @test_throws ArgumentError VkFFT.plan_fft(x, (); zeropad=1:2)
            @test_throws ArgumentError VkFFT.plan_fft(_upload(_noise(WIDE_COMPLEX, (1, 1))), (1, 2); zeropad=1:1)
        end
    end
end
