# The rfft family: the numerical matrix against FFTW's rfft/brfft/irfft, both
# round trips, the plan algebra, the adjoint identities, and everything the
# planner has to refuse once a device array is involved. The refusals that need
# no device at all are in planner.jl.

# The sizes and shapes specific to this family: 16 an even halved axis (power
# of two), 13 an odd one (Bluestein prime), 120 mixed radix. The shared
# region-mapping code is exercised in full by the c2c matrix, so the shapes
# here are one full region, one trailing batch fold, one middle omit (the only
# omit pattern r2c accepts), one odd halved axis in 2d, and one leading
# length-1 axis, which moves the halved dimension off dimension 1.
const RSIZES_1D = (16, 13, 120)

const RSHAPE_CASES = (((16, 8, 4), ((1, 2, 3), (1,), (1, 3))),
                      ((13, 5), ((1, 2),)),
                      ((1, 16, 4), ((2,),)))

"""
    _test_rfft(::Type{Tr}, dims, region)

Runs the three real transforms plus plan reuse and both round trips on one configuration.
"""
function _test_rfft(::Type{Tr}, dims, region) where Tr
    Tc = Complex{Tr}
    n = length(dims)
    rtol = _rtol(Tr)
    h = _noise(Tr, dims)
    d = dims[first(region)]
    spectrum = rfft(h, region)

    x = _upload(h)
    forward = VkFFT.plan_rfft(x, region)
    @test size(forward) == dims
    @test AbstractFFTs.output_size(forward) == size(spectrum)
    @test _relmax(forward * x, spectrum) < rtol
    @test _relmax(mul!(DeviceArray{Tc, n}(undef, size(spectrum)), forward, x), spectrum) < rtol

    # Plan reuse: the same plan applied twice to the same input has to give
    # bit-identical results, which also catches a plan that quietly depends on
    # the buffer it was created with.
    first_run = Array(forward * x)
    @test Array(forward * x) == first_run
    @test Array(mul!(DeviceArray{Tc, n}(undef, size(spectrum)), forward, x)) == first_run

    # Each inverse gets its own upload of the spectrum, because applying one
    # clobbers it.
    backward = VkFFT.plan_brfft(_upload(spectrum), d, region)
    normalized = VkFFT.plan_irfft(_upload(spectrum), d, region)
    @test _relmax(backward * _upload(spectrum), brfft(spectrum, d, region)) < rtol
    @test _relmax(normalized * _upload(spectrum), irfft(spectrum, d, region)) < rtol
    @test _relmax(mul!(DeviceArray{Tr, n}(undef, dims), normalized, _upload(spectrum)), h) < rtol

    # Round trips. The unnormalized one lands on the product of the transformed
    # logical lengths, so the full d on the halved axis and never d ÷ 2 + 1.
    scale = prod(i -> dims[i], region)
    @test _relmax(backward * (forward * x), scale .* h) < rtol
    @test _relmax(normalized * (forward * x), h) < rtol
    @test _relmax(inv(forward) * (forward * x), h) < rtol

    return nothing
end

@testset verbose = true "rfft" begin
    rtol = _rtol(WIDE_REAL)
    h864 = _noise(WIDE_REAL, (8, 6, 4))
    x864 = _upload(h864)
    y564 = _upload(rfft(h864, (1, 3)))

    @testset "1d $Tr n=$n" for Tr in REAL_TYPES, n in RSIZES_1D
        _test_rfft(Tr, (n,), (1,))
    end

    @testset "$(join(dims, 'x')) region=$region $Tr" for (dims, regions) in RSHAPE_CASES,
                                                         region in regions,
                                                         Tr in REAL_TYPES
        _test_rfft(Tr, dims, region)
    end

    @testset "region as any iterable" begin
        reference = rfft(h864, (1, 3))
        for region in ((1, 3), (3, 1), [1, 3], (i for i in (1, 3)))
            @test _relmax(VkFFT.plan_rfft(x864, region) * x864, reference) < rtol
        end
        @test _relmax(VkFFT.plan_rfft(x864, 1) * x864, rfft(h864, 1)) < rtol
        @test _relmax(VkFFT.plan_rfft(x864, 1:2) * x864, rfft(h864, 1:2)) < rtol
    end

    @testset "plan traits" begin
        plan = VkFFT.plan_rfft(x864, (1, 3))
        @test plan isa AbstractFFTs.Plan{WIDE_REAL}
        @test eltype(plan) === WIDE_REAL
        @test size(plan) == (8, 6, 4)
        @test size(plan, 2) == 6
        @test ndims(plan) == 3
        @test AbstractFFTs.fftdims(plan) == (1, 3)
        @test AbstractFFTs.output_size(plan) == (5, 6, 4)
        @test plan.d == 8

        @test isconcretetype(typeof(plan))
        @test typeof(plan) === VkFFTRealPlan{WIDE_REAL, Complex{WIDE_REAL}, 3, BACKEND.name, 2}
        @test all(isconcretetype, fieldtypes(typeof(plan))[1:(end - 1)])
        @test fieldtype(typeof(plan), :pinv) === Union{Nothing, VkFFTRealPlan{Complex{WIDE_REAL}, WIDE_REAL, 3, BACKEND.name, 2}}

        inverse = VkFFT.plan_irfft(y564, 8, (1, 3))
        @test inverse isa AbstractFFTs.Plan{Complex{WIDE_REAL}}
        @test size(inverse) == (5, 6, 4)
        @test AbstractFFTs.output_size(inverse) == (8, 6, 4)
        @test typeof(inverse) === VkFFTRealPlan{Complex{WIDE_REAL}, WIDE_REAL, 3, BACKEND.name, 2}
        @test typeof(VkFFT.plan_brfft(y564, 9, (1, 3))) === VkFFTRealPlan{Complex{WIDE_REAL}, WIDE_REAL, 3, BACKEND.name, 2}
    end

    @testset "inference" begin
        # As on the complex-to-complex side, a tuple, integer or default region
        # carries its length in the type, so the whole plan type is inferable.
        @test (@inferred VkFFT.plan_rfft(x864, (1, 3))) isa VkFFTRealPlan{WIDE_REAL, Complex{WIDE_REAL}, 3, BACKEND.name, 2}
        @test (@inferred VkFFT.plan_rfft(x864, 1)) isa VkFFTRealPlan{WIDE_REAL, Complex{WIDE_REAL}, 3, BACKEND.name, 1}
        @test (@inferred VkFFT.plan_rfft(x864)) isa VkFFTRealPlan{WIDE_REAL, Complex{WIDE_REAL}, 3, BACKEND.name, 3}
        @test (@inferred VkFFT.plan_irfft(y564, 8, (1, 3))) isa VkFFTRealPlan{Complex{WIDE_REAL}, WIDE_REAL, 3, BACKEND.name, 2}
        @test (@inferred VkFFT.plan_brfft(y564, 8, (1, 3))) isa VkFFTRealPlan{Complex{WIDE_REAL}, WIDE_REAL, 3, BACKEND.name, 2}

        plan = VkFFT.plan_rfft(x864, (1, 3))
        @test (@inferred plan * x864) isa DeviceArray{Complex{WIDE_REAL}, 3}
        @test (@inferred mul!(DeviceArray{Complex{WIDE_REAL}, 3}(undef, (5, 6, 4)), plan, x864)) isa DeviceArray{Complex{WIDE_REAL}, 3}
    end

    @testset "show" begin
        @test sprint(show, VkFFT.plan_rfft(x864, (1, 3))) == "2D out-of-place real-to-complex forward VkFFT plan for 8×6×4 $WIDE_REAL to 5×6×4 $(Complex{WIDE_REAL}) on dims (1, 3) [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_rfft(x864, 1)) == "1D out-of-place real-to-complex forward VkFFT plan for 8×6×4 $WIDE_REAL to 5×6×4 $(Complex{WIDE_REAL}) on dim (1,) [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_brfft(y564, 8, (1, 3))) == "2D out-of-place complex-to-real unnormalized inverse VkFFT plan for 5×6×4 $(Complex{WIDE_REAL}) to 8×6×4 $WIDE_REAL on dims (1, 3) [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_irfft(y564, 8, (1, 3))) == "2D out-of-place complex-to-real normalized inverse VkFFT plan for 5×6×4 $(Complex{WIDE_REAL}) to 8×6×4 $WIDE_REAL on dims (1, 3) [$(BACKEND.name)]"
        @test sprint(show, VkFFT.plan_rfft(_upload(ones(Float32, 1, 1)))) == "trivial out-of-place real-to-complex forward VkFFT plan for 1×1 Float32 to 1×1 ComplexF32 on dims (1, 2) [$(BACKEND.name)]"
    end

    @testset "inv, plan_inv and \\" begin
        for region in ((1,), (1, 2), (1, 3), (1, 2, 3))
            @testset "region=$region" begin
                forward = VkFFT.plan_rfft(x864, region)
                spectrum = Array(forward * x864)
                volume = prod(i -> size(h864)[i], region)

                # A forward plan inverts to VkFFT's in-kernel normalized
                # inverse, so no ScaledPlan and no extra scaling kernel.
                inverse = inv(forward)
                @test inverse isa VkFFTRealPlan
                @test inverse.normalize
                @test inverse.direction == VkFFT.INVERSE
                @test inverse.d == 8
                @test inv(forward) === inverse                  # cached in the pinv field
                @test _relmax(inverse * _upload(spectrum), h864) < rtol
                @test _relmax(forward \ _upload(spectrum), h864) < rtol
                @test _relmax(AbstractFFTs.plan_inv(forward) * _upload(spectrum), h864) < rtol

                # ... and the normalized inverse inverts straight back to it.
                @test inv(inverse) === forward
                @test _relmax(inv(inverse) * x864, spectrum) < rtol

                # The unnormalized inverse is the one case that needs a
                # ScaledPlan, because VkFFT's normalize only exists on the
                # inverse direction.
                backward = VkFFT.plan_brfft(_upload(spectrum), 8, region)
                scaled = inv(backward)
                @test scaled isa AbstractFFTs.ScaledPlan
                @test scaled.p isa VkFFTRealPlan
                @test scaled.scale ≈ 1 / volume
                @test _relmax(scaled * (backward * _upload(spectrum)), spectrum) < rtol
                @test _relmax(backward \ (backward * _upload(spectrum)), spectrum) < rtol

                out = DeviceArray{WIDE_REAL, 3}(undef, (8, 6, 4))
                ldiv!(out, forward, _upload(spectrum))
                @test _relmax(out, h864) < rtol
            end
        end
    end

    @testset "adjoints" begin
        for Tr in REAL_TYPES,
            (dims, region) in (((16,), (1,)), ((13,), (1,)), ((16, 8), (1, 2)),
                               ((16, 8), (1,)), ((16, 8, 4), (1, 3)), ((1, 16), (2,)))
            @testset "$Tr $(join(dims, 'x')) region=$region" begin
                Tc = Complex{Tr}
                tol = _rtol(Tr)
                h = _noise(Tr, dims)
                d = dims[first(region)]
                spectrum = rfft(h, region)
                cotangent_c = _noise(Tc, size(spectrum))
                cotangent_r = _noise(Tr, dims)

                for plan in (VkFFT.plan_rfft(_upload(h), region),
                             VkFFT.plan_brfft(_upload(spectrum), d, region),
                             VkFFT.plan_irfft(_upload(spectrum), d, region))
                    @test (plan')' === plan
                    @test size(plan') == AbstractFFTs.output_size(plan)
                    @test AbstractFFTs.fftdims(plan') == AbstractFFTs.fftdims(plan)

                    # The identity AD needs, in the inner product a real
                    # transform is self-adjoint under: <v, P x> == <P' v, x>.
                    forward = plan.direction == VkFFT.FORWARD
                    input = forward ? h : spectrum
                    cotangent = forward ? cotangent_c : cotangent_r
                    left = _component_dot(cotangent, Array(plan * _upload(input)))
                    right = _component_dot(Array(plan' * _upload(cotangent)), input)
                    @test abs(left - right) <= tol * abs(left)
                end
            end
        end

        # Neither AbstractFFTs style fits a forward VkFFT real plan, because
        # recovering its adjoint that way runs a complex-to-real transform over
        # an array that is no longer the spectrum of a real signal. The
        # normalized inverse needs its own style for the usual reason: the
        # in-kernel 1/N leaves IRFFTAdjointStyle off by N^2.
        @test AbstractFFTs.AdjointStyle(VkFFT.plan_rfft(x864, 1)) === VkFFT.VkFFTRFFTAdjointStyle()
        @test AbstractFFTs.AdjointStyle(VkFFT.plan_brfft(y564, 8, 1)) === AbstractFFTs.IRFFTAdjointStyle(8)
        @test AbstractFFTs.AdjointStyle(VkFFT.plan_irfft(y564, 8, 1)) === VkFFT.VkFFTNormalizedIRFFTAdjointStyle()
    end

    @testset "ScaledPlan arithmetic" begin
        # A plan carries its scale in its own real precision, whatever the
        # literal was, so that the Float64 reciprocal inv would otherwise take
        # never reaches a device without double. The whole coercion is covered
        # in interface.jl. Here it is a real plan's turn to hold one.
        three = 3
        plan = VkFFT.plan_rfft(x864, (1, 2))
        @test (three * plan).scale === WIDE_REAL(3)
        reference = rfft(h864, (1, 2))
        @test _relmax((three * plan) * x864, 3 .* reference) < rtol
        @test _relmax((plan * three) * x864, 3 .* reference) < rtol
        @test _relmax(inv(three * plan) * ((three * plan) * x864), h864) < rtol
    end

    @testset "AbstractFFTs.TestUtils.test_plan" begin
        # test_complex_ffts and test_real_ffts both drive the global rfft and
        # plan_rfft, which would need AbstractFFTs.plan_rfft extended on the
        # device array type, the type piracy this package exists to avoid.
        # test_plan takes a plan directly. copy_input is on because an inverse
        # real plan consumes its spectrum.
        for (dims, region) in (((16,), (1,)), ((16, 4), (1, 2)), ((13, 3), (1,)))
            h = _noise(WIDE_REAL, dims)
            d = dims[first(region)]
            spectrum = rfft(h, region)
            scaled = prod(i -> dims[i], region) .* h

            AbstractFFTs.TestUtils.test_plan(VkFFT.plan_rfft(_upload(h), region), _upload(h), _upload(spectrum); copy_input=true)
            AbstractFFTs.TestUtils.test_plan(VkFFT.plan_brfft(_upload(spectrum), d, region), _upload(spectrum), _upload(scaled); copy_input=true)
            AbstractFFTs.TestUtils.test_plan(VkFFT.plan_irfft(_upload(spectrum), d, region), _upload(spectrum), _upload(h); copy_input=true)
        end
    end

    @testset "an inverse consumes its spectrum" begin
        # The documented contract, matching FFTW: the spectrum is scratch space
        # for the inverse. Pinned down here so nothing starts relying on the
        # opposite.
        spectrum = rfft(h864, (1, 3))
        y = _upload(spectrum)
        kept = copy(y)
        @test _relmax(VkFFT.plan_brfft(y, 8, (1, 3)) * y, brfft(spectrum, 8, (1, 3))) < rtol
        @test Array(y) != spectrum
        @test Array(kept) == spectrum
        @test _relmax(VkFFT.plan_irfft(kept, 8, (1, 3)) * kept, h864) < rtol
    end

    @testset "identity semantics instead of a silent no-op" begin
        # Every dimension has length 1, so there is no axis to halve and
        # VkFFT is never asked to transform anything. The results are exact.
        h = Float32[2.5;;]
        x = _upload(h)
        forward = VkFFT.plan_rfft(x)
        spectrum = forward * x
        @test Array(spectrum) == ComplexF32[2.5;;]
        @test Array(VkFFT.plan_brfft(spectrum, 1) * spectrum) == h
        @test Array(VkFFT.plan_irfft(spectrum, 1) * spectrum) == h
        @test Array(inv(forward) * spectrum) == h
    end

    @testset "errors" begin
        x16 = _upload(_noise(Float32, (16,)))
        y9 = _upload(_noise(ComplexF32, (9,)))

        @testset "region must start at the halved dimension" begin
            h85 = _noise(Float32, (8, 5))
            x85 = _upload(h85)
            spectrum85 = _upload(rfft(h85, (1, 2)))

            for region in ((2,), 2, [2])
                err = _thrown(() -> VkFFT.plan_rfft(x85, region))
                @test err isa ArgumentError
                @test occursin("dimension 1", err.msg)
                @test occursin("Permute", err.msg)
                @test !occursin("VKFFT_ERROR", err.msg) # a raw code would be a planner bug
            end

            @test_throws ArgumentError VkFFT.plan_brfft(spectrum85, 8, (2,))
            @test_throws ArgumentError VkFFT.plan_irfft(spectrum85, 8, (2,))

            # A leading length-1 dimension is dropped before VkFFT sees it, so
            # the halved dimension moves to the first one that is longer.
            h116 = _noise(Float32, (1, 16))
            x116 = _upload(h116)
            @test_throws ArgumentError VkFFT.plan_rfft(x116, (1,))
            @test_throws ArgumentError VkFFT.plan_rfft(x116, (1, 2))
            @test _relmax(VkFFT.plan_rfft(x116, (2,)) * x116, rfft(h116, (2,))) < BACKEND.rtol_f32

            # Positive controls, so the refusals above cannot be blanket ones.
            @test _relmax(VkFFT.plan_rfft(x85, (1, 2)) * x85, rfft(h85, (1, 2))) < BACKEND.rtol_f32
            @test _relmax(VkFFT.plan_rfft(x85, (1,)) * x85, rfft(h85, (1,))) < BACKEND.rtol_f32
        end

        @testset "empty regions" begin
            @test_throws ArgumentError VkFFT.plan_rfft(x16, ())
            @test_throws ArgumentError VkFFT.plan_brfft(y9, 16, ())
            @test_throws ArgumentError VkFFT.plan_irfft(y9, 16, ())
        end

        @testset "wrong real length" begin
            @test_throws ArgumentError VkFFT.plan_brfft(y9, 18, (1,)) # leaves 10 complex values
            @test_throws ArgumentError VkFFT.plan_brfft(y9, 15, (1,)) # leaves 8
            @test_throws ArgumentError VkFFT.plan_irfft(y9, 18, (1,))
            @test_throws ArgumentError VkFFT.plan_brfft(y9, 0, (1,))
            @test_throws ArgumentError VkFFT.plan_brfft(y9, -16, (1,))

            # d ÷ 2 + 1 does not recover d, so both lengths that leave 7
            # complex values plan, and each gives its own transform.
            h12 = _noise(WIDE_REAL, (12,))
            h13 = _noise(WIDE_REAL, (13,))
            s12 = rfft(h12)
            s13 = rfft(h13)
            @test length(s12) == length(s13) == 7

            p12 = VkFFT.plan_irfft(_upload(s12), 12, (1,))
            p13 = VkFFT.plan_irfft(_upload(s13), 13, (1,))
            @test p12 !== p13
            @test size(p12) == size(p13) == (7,)
            @test AbstractFFTs.output_size(p12) == (12,)
            @test AbstractFFTs.output_size(p13) == (13,)
            @test _relmax(p12 * _upload(s12), h12) < rtol
            @test _relmax(p13 * _upload(s13), h13) < rtol
        end

        @testset "unsupported array types" begin
            @test_throws ArgumentError VkFFT.plan_rfft(_noise(Float32, (16,)))
            @test_throws ArgumentError VkFFT.plan_brfft(_noise(ComplexF32, (9,)), 16)
            @test_throws ArgumentError VkFFT.plan_rfft(view(_upload(_noise(Float32, (3, 4, 5))), :, 2, :))
        end

        if BACKEND.offset_views
            _skip("the rfft offset-view refusals", "this backend's arrays carry their own offset, so the positive case is in its runner")
        else
            @testset "offset arrays at apply time" begin
                # A plan is reusable across buffers, so both arrays reaching
                # mul! can be offset views whose sizes clear every size check.
                # Their buffer handle is the base of the parent's allocation, so
                # without a layout check per application this transforms the
                # wrong elements and reports success.
                host = Float32[i for i in 1:16]
                big = _upload(host)
                tail = view(big, 9:16)
                @test tail isa DeviceArray
                @test _offset(tail) != 0

                spectra = _upload(_noise(ComplexF32, (10,)))
                spectrum_tail = view(spectra, 6:10)
                @test _offset(spectrum_tail) != 0

                forward = VkFFT.plan_rfft(_upload(_noise(Float32, (8,))))
                out = DeviceArray{ComplexF32, 1}(undef, (5,))

                @test_throws ArgumentError mul!(out, forward, tail)            # offset input
                @test_throws ArgumentError mul!(spectrum_tail, forward, _upload(_noise(Float32, (8,)))) # offset output
                @test_throws ArgumentError forward * tail

                backward = VkFFT.plan_brfft(_upload(_noise(ComplexF32, (5,))), 8)
                @test_throws ArgumentError mul!(DeviceArray{Float32, 1}(undef, (8,)), backward, spectrum_tail)
                @test_throws ArgumentError mul!(tail, backward, _upload(_noise(ComplexF32, (5,))))
                @test_throws ArgumentError backward * spectrum_tail

                # The refusal must not be a blanket one: the same plan still
                # applies to the head of the same buffer, and gives the head's
                # transform.
                head = view(big, 1:8)
                @test _offset(head) == 0
                @test _relmax(mul!(out, forward, head), rfft(host[1:8])) < BACKEND.rtol_f32

                # An array with no backend at all reaches mul! whenever its
                # element types and ndims match the plan, so it has to be
                # refused there too.
                @test_throws ArgumentError mul!(Array{ComplexF32}(undef, 5), forward, Array{Float32}(undef, 8))
            end
        end

        @testset "mismatched arrays" begin
            forward = VkFFT.plan_rfft(x16)
            backward = VkFFT.plan_brfft(y9, 16)

            @test_throws ArgumentError mul!(DeviceArray{ComplexF32, 1}(undef, (16,)), forward, x16)
            @test_throws ArgumentError mul!(DeviceArray{ComplexF32, 1}(undef, (8,)), forward, x16)
            @test_throws ArgumentError mul!(y9, forward, _upload(_noise(Float32, (8,))))
            @test_throws ArgumentError mul!(DeviceArray{Float32, 1}(undef, (17,)), backward, y9)
            @test_throws ArgumentError forward * _upload(_noise(Float32, (8,)))
            @test_throws ArgumentError forward * _upload(_noise(Float32, (4, 4)))
            @test_throws ArgumentError forward * y9                  # a spectrum into a forward plan
            @test_throws ArgumentError backward * x16                # reals into an inverse plan

            wrong = Any[y9, _upload(_noise(Float32, (4, 4)))]
            if BACKEND.fp64
                push!(wrong, _upload(_noise(Float64, (16,))))
                @test_throws ArgumentError forward * _upload(_noise(Float64, (16,)))
            end

            # The wrong-eltype and wrong-ndims fallbacks report the plan's own
            # element types, so assert the rendered text: a missing method would
            # mask these behind a MethodError.
            for bad in wrong
                err = _thrown(() -> forward * bad)
                @test err isa ArgumentError
                @test occursin("this VkFFT plan transforms 1-dimensional Float32 arrays into ComplexF32 ones", err.msg)
            end
        end

        @testset "freed plans" begin
            forward = VkFFT.plan_rfft(_upload(_noise(Float32, (32,))))
            VkFFT.unsafe_free!(forward)
            @test forward.destroyed
            @test forward.app == C_NULL
            @test_throws ArgumentError mul!(DeviceArray{ComplexF32, 1}(undef, (17,)), forward, _upload(_noise(Float32, (32,))))
            @test VkFFT.unsafe_free!(forward) === nothing # idempotent, as the finalizer needs it to be
        end
    end

    @testset "plan cache" begin
        VkFFT.clear_cache!()
        @test VkFFT.cache_size() == 0

        hr = _noise(WIDE_REAL, (16, 4))
        xr = _upload(hr)
        first_plan = VkFFT.plan_rfft(xr, (1, 2))
        @test VkFFT.cache_size() == 1
        @test VkFFT.plan_rfft(xr, (1, 2)) === first_plan
        @test VkFFT.plan_rfft(_upload(hr), (2, 1)) === first_plan     # keyed on shape, region canonicalized

        # A forward real plan and a forward complex plan of the same shape agree
        # on everything the complex-to-complex key holds, so the key needs the
        # real-to-complex flag to keep them apart.
        @test VkFFT.plan_fft(_upload(_noise(WIDE_COMPLEX, (16, 4))), (1, 2)) !== first_plan

        yr = _upload(_noise(WIDE_COMPLEX, (9, 4)))
        p16 = VkFFT.plan_irfft(yr, 16, (1, 2))
        p17 = VkFFT.plan_irfft(yr, 17, (1, 2))
        @test p16 !== p17                                            # d is in the key
        @test VkFFT.plan_irfft(yr, 16, (1, 2)) === p16
        @test VkFFT.plan_brfft(yr, 16, (1, 2)) !== p16               # normalize is too
        @test VkFFT.plan_ifft(yr, (1, 2)) !== p16
        @test VkFFT.plan_bfft(yr, (1, 2)) !== VkFFT.plan_brfft(yr, 16, (1, 2))
        @test VkFFT.cache_size() == 7

        VkFFT.clear_cache!()
        @test VkFFT.cache_size() == 0
        @test VkFFT.plan_rfft(xr, (1, 2)) !== first_plan
    end

    @testset "finalizers" begin
        VkFFT.clear_cache!()
        plan = VkFFT.plan_rfft(_upload(_noise(Float32, (256,))), 1)
        @test !plan.destroyed
        VkFFT.clear_cache!()
        plan = nothing
        GC.gc(true)
        GC.gc(true)
        @test true # a double free or a destroy without a live context would have crashed here
    end
end
