# Run with --startup-file=no. Paul's startup.jl activates whatever project the
# working directory holds, which silently overrides --project=benchmarks.
#
# Everything under test comes from runtests.jl, which includes ../run.jl.

_c2c(n, T) = Case((n,), (1,), :c2c, T, :forward, :oop, false, ispow2(n) ? :pow2 : :prime, string(n))

@testset "accuracy" begin
    @testset "draw counts n=$n" for (n, expected) in ((2^6, 20), (2^14, 20), (2^15, 5), (16381, 20), (65521, 5))
        inputs = accuracy_inputs(_c2c(n, ComplexF32))

        @test count(t -> t[1] == "uniform", inputs) == expected
        @test [t[1] for t in inputs[end - 1:end]] == ["chirp", "impulse"]
        @test [t[2] for t in inputs] == [collect(1:expected); 1; 1]
        @test all(t -> length(t[3]) == n && eltype(t[3]) === ComplexF32, inputs)
    end

    @testset "the reference is chosen by precision and not by size" begin
        for n in (64, 61, 4096), (T, name) in ((ComplexF64, "bigfloat256"), (ComplexF32, "fftw-f64"))
            x = uniform_input(StableRNG(19), n, T)
            @test last(reference_for("uniform", x)) == name
            @test last(reference_for("chirp", x)) == name
        end
        @test reference_for("impulse", ComplexF32[1, 0, 0]) == (ones(ComplexF64, 3), "exact")
    end

    # Getting this backwards would make every single precision number in the
    # file a measurement of how badly the input was rounded, and nothing else in
    # the suite would notice.
    @testset "the reference transforms the rounded input" begin
        original = uniform_input(StableRNG(11), 512)
        rounded = ComplexF32.(original)
        ref = first(reference_for("uniform", rounded))
        @test ref == fft(ComplexF64.(rounded))
        @test ref != fft(original)

        # An implementation exact on the array it was handed reads as a fifth of
        # an eps against its own input and a third of one against the original.
        ideal = ComplexF32.(fft(ComplexF64.(rounded)))
        honest = relative_error(ideal, ref)
        wrong = relative_error(ideal, fft(original))
        @test honest.eps_multiple < 0.25
        @test wrong.eps_multiple > 1.25 * honest.eps_multiple
    end

    # FFTW's own error is a fraction of an eps and grows with n. A reference or
    # an error measure that had gone wrong would land nowhere near this.
    @testset "fftw against the 256 bit reference" begin
        measured = [relative_error(fft(x), first(reference_for("uniform", x))).eps_multiple for x in (uniform_input(StableRNG(3), n) for n in (64, 1024, 16384))]

        @test all(e -> 0 < e < 5, measured)
        @test issorted(measured)
    end

    # A power of two carries the zeros through exactly, so the impulse there is
    # the cheap check that the chain is wired up at all. A prime goes through
    # FFTW's Bluestein path, whose own chirp arithmetic lands at a couple of eps,
    # and that row is a measurement rather than a check.
    @testset "the impulse is exact at a power of two" begin
        for T in (ComplexF32, ComplexF64)
            pow2 = first(impulse_input(64, T))
            prime = first(impulse_input(61, T))
            @test relative_error(fft(pow2), first(reference_for("impulse", pow2))).err == 0
            @test relative_error(fft(prime), first(reference_for("impulse", prime))).eps_multiple < 5
        end
    end

    @testset "rows land in accuracy.csv" begin
        out = mktempdir()
        tiny = filter(c -> prod(c.dims) <= 128, cases(:fftw))
        @test count(c -> c.family === :c2c && c.direction === :forward && c.placement === :oop, tiny) == 4 # 2^6 and 2^7 at both precisions

        redirect_stdout(devnull) do
            run_accuracy((FFTWImpl(1),), tiny, out)
        end
        lines = readlines(joinpath(out, "accuracy.csv"))
        rows = [split(line, ',') for line in lines[2:end]]

        @test lines[1] == join(ACCURACY_COLUMNS, ",")
        @test length(rows) == 4 * (20 + 3) # the uniform draws plus the chirp, the impulse and the round trip
        @test all(r -> length(r) == length(ACCURACY_COLUMNS), rows)
        @test Set(r[4] for r in rows) == Set(["uniform", "chirp", "impulse", "roundtrip"])
        @test Set(r[8] for r in rows if occursin("Float64", r[2]) && r[4] in ("uniform", "chirp")) == Set(["bigfloat256"])
        @test Set(r[8] for r in rows if occursin("Float32", r[2]) && r[4] in ("uniform", "chirp")) == Set(["fftw-f64"])
        @test all(r -> r[8] == "self", filter(r -> r[4] == "roundtrip", rows))
        @test all(r -> parse(Float64, r[6]) == 0, filter(r -> r[4] == "impulse", rows))

        trip(T) = maximum(parse(Float64, r[6]) for r in rows if r[4] == "roundtrip" && occursin(string(T), r[2]))
        @test 0 < trip(Float32) < 1e-5
        @test 0 < trip(Float64) < 1e-6 * trip(Float32)

        # A second pass over the same directory finds every pair recorded and
        # writes nothing.
        redirect_stdout(devnull) do
            run_accuracy((FFTWImpl(1),), tiny, out)
        end
        @test readlines(joinpath(out, "accuracy.csv")) == lines
    end
end
