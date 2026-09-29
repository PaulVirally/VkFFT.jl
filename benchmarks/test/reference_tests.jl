# Run with --startup-file=no. Paul's startup.jl activates whatever project the
# working directory holds, which silently overrides --project=benchmarks.

using FFTW
using Random
using StableRNGs
using Test

include(joinpath(@__DIR__, "..", "reference.jl"))

const rtol_ref = 1e-60 # two 256 bit paths to the same answer
const rtol_f64 = 1e-13 # the reference against FFTW, at FFTW's precision

@testset "reference" begin
    rng = StableRNG(2024)

    @testset "radix 2 against the direct sum n=$n" for n in (1, 2, 4, 8, 32, 128)
        x = uniform_input(rng, n)
        @test relative_error(reference_fft(x), direct_dft(x)).err < rtol_ref
    end

    # 33 and 65 sit just above a power of two in 2n-1, where the padded length jumps.
    @testset "bluestein against the direct sum n=$n" for n in (3, 5, 6, 7, 11, 12, 13, 17, 33, 60, 65, 97)
        x = uniform_input(rng, n)
        @test relative_error(reference_fft(x), direct_dft(x)).err < rtol_ref
    end

    @testset "round trip n=$n" for n in (8, 64, 7, 12, 97)
        x = uniform_input(rng, n)
        @test relative_error(reference_fft(reference_fft(x), false), x).err < rtol_ref
    end

    # Catches a sign or ordering convention error, which the quadratic sum shares.
    @testset "FFTW convention n=$n" for n in (16, 256, 13, 100)
        x = uniform_input(rng, n)
        @test relative_error(fft(x), reference_fft(x)).err < rtol_f64
        @test relative_error(ifft(x), reference_fft(x, false)).err < rtol_f64
    end

    @testset "relative error" begin
        ref = uniform_input(rng, 32)
        @test relative_error(ref, ref).err == 0
        @test relative_error(ref .* (1 + 1e-6), ref).err ≈ 1e-6

        # The eps is the result's and not the reference's, so rounding to ComplexF32 costs under one eps of it.
        rounded = relative_error(ComplexF32.(ref), ref)
        @test rounded.eps_multiple ≈ rounded.err / eps(Float32)
        @test 0 < rounded.eps_multiple < 1
    end

    # Half precision is where widening the unrounded original instead would do the most damage.
    @testset "exact widening $T" for T in (ComplexF16, ComplexF32)
        x = uniform_input(rng, 64, T)
        @test reference_fft(x) == reference_fft(ComplexF64.(x))
    end

    @testset "impulse n=$n" for n in (16, 13)
        x, y = impulse_input(n)
        @test relative_error(reference_fft(x), y).err < rtol_ref
    end

    @testset "chirp" begin
        x = chirp_input(64)
        @test all(≈(1), abs.(x))
        # The spectrum is flat to the precision of x, which is where rounding the chirp to ComplexF64 leaves it.
        @test maximum(abs, abs.(reference_fft(x)) .- sqrt(64)) < rtol_f64
    end
end
