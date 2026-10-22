# The accuracy half of the suite: the input classes, the choice of reference,
# and the pass that walks the case list once and writes accuracy.csv.
#
# It runs once per run rather than once per replication. The same input gives
# bit identical output, so there is nothing for a second process to measure. The
# spread here comes from the choice of input, which is why the random class is
# drawn several times and the two deterministic ones are drawn once.
#
# Only the ComplexF64 rows pay for the 256 bit reference. Everything narrower is
# measured against FFTW in Float64, which sits far enough below single precision
# error to read it honestly.

using FFTW
using Printf
using StableRNGs

const ACCURACY_SEED = 0x4143_4355_0000_0000 # the draws are reproducible from this and the length alone
const REF_MAX = 2^20 # a 256 bit reference costs 16 s per draw here and four times that at 2^22

"""
    accuracy_inputs(case::Case)

Returns the `(input_class, draw, x)` triples one case is measured on, in the case's own precision.

Uniform random is the average case and the only class with any spread, so it is
drawn 20 times up to 2^14 and 5 times from 2^15 to 2^20, from a `StableRNG`
seeded by the length. The chirp and the impulse are deterministic and appear
once each.

The draws are made in `Float64` and rounded to the case's precision once, so a
`ComplexF32` case sees exactly the `ComplexF64` case's input rounded. The two
precisions of one length are then measuring the same numbers.

# Returns
- A `Vector` of `(input_class::String, draw::Int, x::Vector{case.precision})`
"""
function accuracy_inputs(case::Case)
    n = prod(case.dims)
    rng = StableRNG(ACCURACY_SEED + n)

    inputs = [("uniform", draw, uniform_input(rng, n, case.precision)) for draw in 1:(n <= 2^14 ? 20 : 5)]
    push!(inputs, ("chirp", 1, chirp_input(n, case.precision)))
    push!(inputs, ("impulse", 1, first(impulse_input(n, case.precision))))
    return inputs
end

"""
    reference_for(input_class::String, x::AbstractVector)

Returns the reference spectrum of `x` and the name of what produced it.

The reference is the transform of the array the implementation was handed,
widened exactly, and never of whatever that array was rounded from. A
`ComplexF32` case is built in `ComplexF32` first and referenced as
`fft(ComplexF64.(x))`. Referencing the `Float64` original instead would fold the
input rounding into every single precision number in the file, and at
`eps(Float16) = 9.8e-4` it would drown a half precision row entirely.

`Float64` sits 2^-29 under `Float32`'s epsilon, which is far enough to measure a
single precision transform honestly and is vastly cheaper than 256 bits. A
`ComplexF64` result has no such headroom beneath it and pays for the BigFloat
path. The impulse transforms to all ones in closed form and pays for nothing.

# Arguments
- `input_class::String`: The class the input came from, since the impulse has a closed form and the others do not
- `x::AbstractVector`: The input exactly as the implementation received it

# Returns
- `(ref, name)`: the reference spectrum, and one of `bigfloat256`, `fftw-f64` or `exact`
"""
function reference_for(input_class::String, x::AbstractVector)
    input_class == "impulse" && return (ones(ComplexF64, length(x)), "exact")
    eltype(x) === ComplexF64 && return (reference_fft(x), "bigfloat256")
    return (fft(ComplexF64.(x)), "fftw-f64")
end

"""
    accuracy_rows(impls, case::Case)

Measures every implementation of one case and returns its accuracy.csv rows, less the run id.

The reference sweep runs on the powers of two up to 2^20 and the four primes,
and every other shape carries the round trip alone. A reference is computed once
per input and every implementation is measured against that same one, which
halves the BigFloat bill when a backend has two implementations and leaves two
rows differing by their own error rather than by their reference's.

The round trip is the forward transform followed by the normalized inverse,
measured against the input. It needs no reference at any size, so it is the only
accuracy number available for 2D, 3D, batched and 2^24, and it is what a user
actually experiences. A fresh `prepare` per draw costs nothing worth avoiding:
FFTW planning against warm wisdom is a lookup, and it keeps `refill!` and the
rest of the backend contract exactly as the timing half uses them.

# Arguments
- `impls`: The implementations to measure, already filtered by `supports`
- `case::Case`: A forward out of place complex to complex case, which is the one row per shape and precision the matrix always has

# Returns
- A `Vector` of `(implementation, input_class, draw, relerr, eps_multiple, reference)` tuples
"""
function accuracy_rows(impls, case::Case)
    rows = Tuple{String, String, Int, Float64, Float64, String}[]

    if case.class in (:pow2, :prime) && prod(case.dims) <= REF_MAX
        for (input_class, draw, x) in accuracy_inputs(case)
            ref, name = reference_for(input_class, x)
            for impl in impls
                prepared = prepare(impl, case, x)
                apply!(prepared)
                measured = relative_error(result(prepared), ref)
                push!(rows, (string(impl), input_class, draw, measured.err, measured.eps_multiple, name))
            end
        end
    end

    inverse = Case(case.dims, case.region, :c2c, case.precision, :inverse, :oop, case.tuned, case.class, case.label)
    x = input(case)
    for impl in impls
        forward = prepare(impl, case, x)
        apply!(forward)
        back = prepare(impl, inverse, result(forward))
        apply!(back)
        measured = relative_error(vec(result(back)), vec(x))
        push!(rows, (string(impl), "roundtrip", 1, measured.err, measured.eps_multiple, "self"))
    end
    return rows
end

"""
    run_accuracy(impls, all_cases, out)

Runs the accuracy pass over a case list once and appends its rows to accuracy.csv.

The subject is the forward out of place complex to complex case of every shape
and precision, which is the row the matrix always carries and so the natural
place to hang a case's accuracy. The inverse and in place rows of the same shape
would only measure it a second time.

A case, implementation pair already in the file is skipped, so a rerun into the
same directory resumes rather than repeating the expensive half. A case that
throws is warned about and left out, since a sweep on a machine nobody can
iterate on should not lose the other sixty five to one broken shape. The seed
the draws come from goes into the manifest.

The wisdom is the file the timing half keeps beside the code, which makes the
forward plans lookups. The inverse plans the round trip needs are not all in the
timing matrix, so the first pass on a machine pays `FFTW.MEASURE` for the ones
that are missing and then exports them.

# Returns
- Nothing
"""
function run_accuracy(impls, all_cases, out)
    machine = gethostname()
    run_id = string(machine, "-", basename(rstrip(out, '/')))
    path = joinpath(out, "accuracy.csv")
    wisdom = joinpath(@__DIR__, "fftw_wisdom_$machine")
    isfile(wisdom) && FFTW.import_wisdom(wisdom)
    record_accuracy_seed(joinpath(out, "manifest.toml"), ACCURACY_SEED)

    done = completed(path, 2)
    subject = filter(c -> c.family === :c2c && c.direction === :forward && c.placement === :oop, all_cases)
    for (position, case) in enumerate(subject)
        id = case_id(case)
        active = [impl for impl in impls if supports(impl, case) && !((id, string(impl)) in done)]
        isempty(active) && continue

        started = time_ns()
        try
            rows = accuracy_rows(active, case)
            append_csv(path, ACCURACY_COLUMNS, [(run_id, id, row...) for row in rows])
            @printf("[%3d/%3d] %s  %d rows  %.2f s\n", position, length(subject), id, length(rows), (time_ns() - started) * 1e-9)
        catch err
            # The case already has its row in cases.csv from the timing half, and
            # a contradicting one here would be worse than none. The missing
            # rows show up as a gap in the plot.
            @warn "accuracy case failed, carrying on" case=id exception=err
        end
    end

    FFTW.export_wisdom(wisdom)
    return nothing
end
