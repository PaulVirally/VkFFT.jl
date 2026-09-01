# VkFFT.unsafe_plan. What a raw plan computes has to match the typed planner
# that would have built the same config, the structural checks it keeps have to
# fire, and the precisions no Julia element type can express have to be
# reachable from here and from nowhere else.

# VkFFT's precision codes: 0 single, 1 double, 2 half, 3 double-double.
const WIDE_PRECISION = BACKEND.fp64 ? 1 : 0

# A double-double complex value is four native doubles: the real part as a
# high-order and a low-order double, then the imaginary part the same way. A
# ComplexF64 array of twice the length has exactly those bytes, which is what
# lets a prototype stand in for a precision Julia has no type for.
const QUAD_DOUBLES_PER_VALUE = 4

"""
    _quad_pack(h::Vector{Float64})

Returns a `ComplexF64` vector holding `h` in VkFFT's double-double complex layout.

Each value takes four doubles and only the high-order real one is written, which
is exact: a Float64 is its own double-double with a zero low-order part.
"""
function _quad_pack(h::Vector{Float64})
    flat = zeros(Float64, QUAD_DOUBLES_PER_VALUE * length(h))
    for i in eachindex(h)
        flat[QUAD_DOUBLES_PER_VALUE * (i - 1) + 1] = h[i]
    end
    return collect(reinterpret(ComplexF64, flat))
end

"""
    _quad_unpack(packed::Vector{ComplexF64}, n::Int)

Returns the `n` complex values a double-double buffer holds, summed back into `ComplexF64`.
"""
function _quad_unpack(packed::Vector{ComplexF64}, n::Int)
    flat = collect(reinterpret(Float64, packed))
    return [(flat[QUAD_DOUBLES_PER_VALUE * (i - 1) + 1] + flat[QUAD_DOUBLES_PER_VALUE * (i - 1) + 2]) +
            im * (flat[QUAD_DOUBLES_PER_VALUE * (i - 1) + 3] + flat[QUAD_DOUBLES_PER_VALUE * (i - 1) + 4]) for i in 1:n]
end

@testset verbose = true "unsafe_plan" begin
    rtol = _rtol(WIDE_COMPLEX)

    @testset "a DCT-II rebuilt from a raw config" begin
        h = _noise(WIDE_REAL, (16, 8))
        x = _upload(h)

        typed = VkFFT.plan_dct(x, (1, 2))
        raw = VkFFT.unsafe_plan(_raw_config((16, 8); precision=WIDE_PRECISION, dct=2, make_forward_only=1), x, similar(x))
        @test raw isa VkFFTUnsafePlan{WIDE_REAL, WIDE_REAL, 2, 2, BACKEND.name}

        expected = Array(typed * x)
        @test Array(raw * x) == expected
        @test Array(mul!(similar(x), raw, x)) == expected
        @test _relmax(raw * x, _r2r_ref(h, :dct, 2, (1, 2))) < _rtol(WIDE_REAL)
    end

    @testset "a c2c rebuilt from a raw config, both directions" begin
        h = _noise(WIDE_COMPLEX, (64,))
        x = _upload(h)

        forward = VkFFT.unsafe_plan(_raw_config((64,); precision=WIDE_PRECISION, make_forward_only=1), x, similar(x))
        @test Array(forward * x) == Array(VkFFT.plan_fft(x, 1) * x)
        @test _relmax(forward * x, fft(h)) < rtol

        spectrum = _upload(fft(h))
        inverse = VkFFT.unsafe_plan(_raw_config((64,); precision=WIDE_PRECISION, normalize=1, make_inverse_only=1), spectrum, similar(spectrum); direction=1)
        @test Array(inverse * spectrum) == Array(VkFFT.plan_ifft(spectrum, 1) * spectrum)
        @test _relmax(inverse * spectrum, h) < rtol
    end

    if BACKEND.fp16
        @testset "an fp16 c2c rebuilt from a raw config" begin
            h = _noise16(ComplexF16, (64,))
            x = _upload(h)

            typed = VkFFT.plan_fft(x, 1)
            raw = VkFFT.unsafe_plan(_raw_config((64,); precision=2, make_forward_only=1), x, similar(x))
            @test raw isa VkFFTUnsafePlan{ComplexF16, ComplexF16, 1, 1, BACKEND.name}

            @test Array(raw * x) == Array(typed * x)
            @test _relmax(raw * x, fft(ComplexF32.(h))) < BACKEND.rtol_f16

            # And the inverse direction, which is the same config with normalize
            # and the other half of the kernels.
            spectrum = _upload(ComplexF16.(fft(ComplexF32.(h))))
            inverse = VkFFT.unsafe_plan(_raw_config((64,); precision=2, normalize=1, make_inverse_only=1), spectrum, similar(spectrum); direction=1)
            @test Array(inverse * spectrum) == Array(VkFFT.plan_ifft(spectrum, 1) * spectrum)
            @test _relmax(inverse * spectrum, ComplexF32.(h)) < BACKEND.rtol_f16
        end
    else
        _skip("the fp16 raw config", "this backend has no half precision")
    end

    if BACKEND.quad
        @testset "double-double quad precision" begin
            # The one precision the typed API cannot reach, since no Julia
            # element type has four native doubles per complex value.
            n = 64
            h = _noise(Float64, (n,)) .- 0.5
            reference = fft(ComplexF64.(h))

            packed = _quad_pack(h)
            x = _upload(packed)
            y = _upload(zeros(ComplexF64, length(packed)))

            forward = VkFFT.unsafe_plan(_raw_config((n,); precision=3, make_forward_only=1), x, y)
            @test forward isa VkFFTUnsafePlan{ComplexF64, ComplexF64, 1, 1, BACKEND.name}
            @test size(forward) == (QUAD_DOUBLES_PER_VALUE * n ÷ 2,)
            @test _relmax(_quad_unpack(Array(mul!(y, forward, x)), n), reference) < BACKEND.rtol_f64

            # And a round trip through the normalized inverse, which is what
            # says the transform is really running in this precision and not
            # reinterpreting the buffer as something else.
            back = _upload(zeros(ComplexF64, length(packed)))
            inverse = VkFFT.unsafe_plan(_raw_config((n,); precision=3, normalize=1, make_inverse_only=1), y, back; direction=1)
            mul!(back, inverse, y)
            @test _relmax(_quad_unpack(Array(back), n), ComplexF64.(h)) < BACKEND.rtol_f64
        end
    else
        # On Metal the plan fails to compile and the Metal compiler writes
        # thousands of lines of diagnostics on its way out, because there is no
        # double to build a double-double from.
        _skip("double-double quad precision", "this backend has no double, so it has no double-double either")
    end

    @testset "an in-place config" begin
        h = _noise(WIDE_COMPLEX, (32,))
        x = _upload(h)
        raw = VkFFT.unsafe_plan(_raw_config((32,); precision=WIDE_PRECISION, inplace=1, make_forward_only=1), x, x)
        @test raw.inplace

        buffer = _upload(h)
        @test (raw * buffer) === buffer
        @test _relmax(buffer, fft(h)) < rtol

        # An in-place plan demands one array, and an out-of-place one demands
        # two.
        @test_throws ArgumentError mul!(similar(x), raw, x)
        outofplace = VkFFT.unsafe_plan(_raw_config((32,); precision=WIDE_PRECISION, make_forward_only=1), x, similar(x))
        @test_throws ArgumentError mul!(x, outofplace, x)

        # A config that says one buffer needs prototypes that agree, since
        # nothing else could apply it.
        @test_throws ArgumentError VkFFT.unsafe_plan(_raw_config((32,); precision=WIDE_PRECISION, inplace=1, make_forward_only=1), x, _upload(_noise(WIDE_COMPLEX, (16,))))
    end

    @testset "structural checks still fire" begin
        host = WIDE_COMPLEX[i for i in 1:16]
        x8 = _upload(_noise(WIDE_COMPLEX, (8,)))
        config = _raw_config((8,); precision=WIDE_PRECISION, make_forward_only=1)
        plan = VkFFT.unsafe_plan(config, x8, similar(x8))

        # A host array has no backend at all, which is the one wrong-device case
        # every machine can produce.
        @test_throws ArgumentError VkFFT.unsafe_plan(config, host, similar(x8))
        @test_throws ArgumentError mul!(Array{WIDE_COMPLEX}(undef, 8), plan, Array{WIDE_COMPLEX}(undef, 8))

        # Sizes come from the prototypes, so they are what an application is
        # checked against.
        @test_throws ArgumentError plan * _upload(_noise(WIDE_COMPLEX, (16,)))
        @test_throws ArgumentError mul!(_upload(_noise(WIDE_COMPLEX, (16,))), plan, x8)

        # A direction is VkFFTAppend's own encoding and nothing else.
        @test_throws ArgumentError VkFFT.unsafe_plan(config, x8, similar(x8); direction=0)
        @test_throws ArgumentError VkFFT.unsafe_plan(config, x8, similar(x8); direction=2)

        if BACKEND.offset_views
            _skip("the raw-plan offset-view refusals", "this backend's arrays carry their own offset, so the positive case is in its runner")
        else
            # An offset view has no layout the wrapper can use, at plan time and
            # at apply time both.
            big = _upload(host)
            tail = view(big, 9:16)
            @test _offset(tail) != 0
            @test_throws ArgumentError VkFFT.unsafe_plan(config, tail, similar(x8))
            @test_throws ArgumentError VkFFT.unsafe_plan(config, x8, tail)
            @test_throws ArgumentError mul!(similar(x8), plan, tail)
            @test_throws ArgumentError mul!(tail, plan, x8)
            @test_throws ArgumentError plan * tail

            # The head of the same buffer starts at offset 0, so the refusals
            # above are about the offset and not about the view.
            head = view(big, 1:8)
            @test _offset(head) == 0
            @test _relmax(mul!(similar(x8), plan, head), fft(host[1:8])) < rtol
        end
    end

    @testset "not cached, and freed like every other plan" begin
        VkFFT.clear_cache!()
        x = _upload(_noise(WIDE_COMPLEX, (64,)))
        config = _raw_config((64,); precision=WIDE_PRECISION, make_forward_only=1)

        first_plan = VkFFT.unsafe_plan(config, x, similar(x))
        @test VkFFT.unsafe_plan(config, x, similar(x)) !== first_plan
        @test VkFFT.cache_size() == 0

        VkFFT.unsafe_free!(first_plan)
        @test first_plan.destroyed
        @test first_plan.app == C_NULL
        @test_throws ArgumentError first_plan * x
        @test VkFFT.unsafe_free!(first_plan) === nothing # idempotent, as the finalizer needs it to be

        # Dropped on the floor and collected, which is where a missing finalizer
        # or a double free shows up.
        for _ in 1:16
            VkFFT.unsafe_plan(config, x, similar(x))
        end
        GC.gc(true)
        GC.gc(true)
        @test true # a double free would have taken the process down here
    end

    @testset "what a raw plan does not have" begin
        x = _upload(_noise(WIDE_COMPLEX, (16, 4)))
        plan = VkFFT.unsafe_plan(_raw_config((16, 4); precision=WIDE_PRECISION, make_forward_only=1), x, similar(x))

        @test plan isa AbstractFFTs.Plan{WIDE_COMPLEX}
        @test eltype(plan) === WIDE_COMPLEX
        @test size(plan) == (16, 4)
        @test AbstractFFTs.output_size(plan) == (16, 4)
        @test ndims(plan) == 2
        @test isconcretetype(typeof(plan))
        @test all(isconcretetype, fieldtypes(typeof(plan)))

        @test_throws ArgumentError AbstractFFTs.fftdims(plan)
        @test_throws ArgumentError inv(plan)
        @test_throws ArgumentError AbstractFFTs.plan_inv(plan)
        @test_throws ArgumentError plan'
        @test sprint(show, plan) == "unsafe out-of-place forward VkFFT plan for 16×4 $WIDE_COMPLEX to 16×4 $WIDE_COMPLEX [$(BACKEND.name)]"

        halved = VkFFT.unsafe_plan(_raw_config((16, 4); precision=WIDE_PRECISION, r2c=1, make_forward_only=1), _upload(_noise(WIDE_REAL, (16, 4))), _upload(_noise(WIDE_COMPLEX, (9, 4))))
        @test sprint(show, halved) == "unsafe out-of-place forward VkFFT plan for 16×4 $WIDE_REAL to 9×4 $WIDE_COMPLEX [$(BACKEND.name)]"

        inverse = VkFFT.unsafe_plan(_raw_config((16, 4); precision=WIDE_PRECISION, inplace=1, make_inverse_only=1), x, x; direction=1)
        @test sprint(show, inverse) == "unsafe in-place inverse VkFFT plan for 16×4 $WIDE_COMPLEX to 16×4 $WIDE_COMPLEX [$(BACKEND.name)]"
    end

    @testset "configs the typed planners refuse" begin
        # These are the point of the entry point: the typed planners refuse them
        # because VkFFT accepts them and computes something other than what was
        # asked, and here nothing stands in the way of building them.
        #
        # None of them is applied. Running one corrupts the process heap: the
        # transform VkFFT generates for a config like these is not the transform
        # the buffer was sized for, so it writes past the end of it, and the
        # crash surfaces later and somewhere else entirely. That is a property
        # of the entry point rather than something to assert on, and it is what
        # VkFFT.unsafe_plan's docstring warns about.
        x = _upload(_noise(WIDE_REAL, (16,)))
        spectrum = _upload(_noise(WIDE_COMPLEX, (9,)))

        # A DCT and a DST at once, which VkFFT resolves as the DST.
        @test VkFFT.unsafe_plan(_raw_config((16,); precision=WIDE_PRECISION, dct=2, dst=2, make_forward_only=1), x, similar(x)) isa VkFFTUnsafePlan

        # A real-to-complex plan carrying a DCT, which corrupts the spectrum.
        @test VkFFT.unsafe_plan(_raw_config((16,); precision=WIDE_PRECISION, r2c=1, dct=2, make_forward_only=1), x, spectrum) isa VkFFTUnsafePlan

        # An r2r type outside 1 to 4, which falls through every branch that
        # generates the pre- and post-processing an r2r transform needs.
        @test VkFFT.unsafe_plan(_raw_config((16,); precision=WIDE_PRECISION, dct=9, make_forward_only=1), x, similar(x)) isa VkFFTUnsafePlan
    end

    @testset "a raw config that VkFFT itself rejects still reports cleanly" begin
        x = _upload(_noise(WIDE_COMPLEX, (16,)))

        # precision is validated by the wrapper, which is the one thing it does
        # validate, and comes back as a wrapper error code rather than a crash.
        err = _thrown(() -> VkFFT.unsafe_plan(_raw_config((16,); precision=7), x, similar(x)))
        @test err isa VkFFTError
        @test err.code == -1

        # A convolution plan built with only one half of its kernels, which the
        # wrapper refuses for the same reason the planner does.
        conv = _thrown(() -> VkFFT.unsafe_plan(_raw_config((16, 1); precision=WIDE_PRECISION, perform_convolution=1, make_forward_only=1), x, similar(x)))
        @test conv isa VkFFTError
    end
end
