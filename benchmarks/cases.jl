# The case matrix. One Case is one shape, family, precision, direction and
# placement, and it is what every implementation of a backend runs head to head.
#
# The axes are deliberately not a full product. Complex to complex covers every
# shape class, the real families ride the powers of two and the primes, and the
# inverse and in place rows exist to settle one claim each and so ride the pow2
# ComplexF32 sweep alone.

using Random
using StableRNGs

include("reference.jl") # for uniform_input, so both halves of the suite draw the same distribution

# Shape, region and a short human name per class, from PLAN_BENCH section 4.
const SHAPES = vcat(
    [(:pow2, (2^k,), (1,), "2^$k") for k in 6:24],
    [(:prime, (n,), (1,), string(n)) for n in (4093, 8191, 16381, 65521)],
    [(:smooth, (n,), (1,), string(n)) for n in (1920, 6561, 15625, 46656, 1000000)],
    [(Symbol("2d"), (n, n), (1, 2), "$(n)^2") for n in (256, 512, 1024, 2048, 4096)],
    [(Symbol("3d"), (n, n, n), (1, 2, 3), "$(n)^3") for n in (64, 128, 256)],
    [(:batched, (256, 4096), (1,), "256x4096")],
)

# The cliff is a contiguous run of lengths straddling 4096, so it overlaps pow2
# at 4096 and the class has to stay part of a case's identity.
const CLIFF_SHAPES = [(:cliff, (n,), (1,), string(n)) for n in 4080:4110]

const INPUT_SEED = 0x5646_4654_0000_0000 # the timing input is fixed, so two runs weeks apart compare

"""
    CAPABILITIES

What each backend's device can do, so a combination it cannot run is never enumerated.

An entry gives the complex element types the device has, the largest transform
worth running on it, whether the cliff sweep belongs there, and whether the
device has a tuner to sweep at all. The CPU paths
cap at 2^22 elements: a 2^24 transform swallows the run for a point the sweep
already makes, and planning one with `FFTW.MEASURE` costs more than it returns.
The cliff is a CUDA against cuFFT measurement and is not enumerated elsewhere.

OpenCL's cap is the one written here for a CPU device, since that is the only
OpenCL device this table can assume. The same code on a GPU has no such limit,
and `capabilities` lifts the cap once the device is known. Leaving it at 2^22
everywhere would stop an OpenCL sweep on a card two points short of the CUDA
sweep beside it, which is exactly where the two are worth comparing.

Metal has no `Float64` at all, which is a hardware limitation rather than a gap
in the bindings, so it carries half precision in place of double. It is also the
one backend where VkFFT and its rival both take half, which is why `ComplexF16`
appears there and nowhere else. Its memory is the host's, so the whole shape list fits
and it has no cap.

CUDA takes half in VkFFT only through a wrapper built against a CUDA toolkit,
and the JLL's wrapper is not one, so a half row here would carry cuFFT alone. The card's 48 GB swallows the whole
shape list, whose largest case is a quarter of a gigabyte.

`tuning` is false for FFTW alone, since the tuner belongs to VkFFT and an FFTW
sweep holds no VkFFT implementation to turn it on for.
"""
const CAPABILITIES = Dict(
    :cuda => (types=(ComplexF32, ComplexF64), max_elements=typemax(Int), cliff=true, tuning=true),
    :fftw => (types=(ComplexF32, ComplexF64), max_elements=2^22, cliff=false, tuning=false),
    :metal => (types=(ComplexF32, ComplexF16), max_elements=typemax(Int), cliff=false, tuning=true),
    :opencl => (types=(ComplexF32, ComplexF64), max_elements=2^22, cliff=false, tuning=true),
)

# The shape classes the tuning rows ride. Tuning is off the main matrix on
# purpose: it answers one question about the plan rather than about the
# transform, and the two knobs it turns describe a memory system and a block
# scheduler, so a power of two run and the square 2D shapes are where an answer
# would show.
const TUNED_CLASSES = (:pow2, Symbol("2d"))

"""
    Case{N, M}

One transform to measure, and the join key for every row any wave writes about it.

`dims` is always the logical transform size, which for the `:c2r` family is the
size of the real signal rather than of the complex input `input` hands back.
`precision` is the complex element type the arithmetic runs at, so an `:r2c`
case at `ComplexF32` takes a `Float32` array in.

# Fields
- `dims::NTuple{N, Int}`: The size of the transform
- `region::NTuple{M, Int}`: The dimensions transformed over
- `family::Symbol`: `:c2c`, `:r2c` or `:c2r`
- `precision::DataType`: The complex element type, e.g. `ComplexF32`
- `direction::Symbol`: `:forward` or `:inverse`, the latter normalized
- `placement::Symbol`: `:oop` or `:inplace`
- `tuned::Bool`: Whether the plan is built with the autotuner on
- `class::Symbol`: The shape class, e.g. `:pow2`
- `label::String`: A short name for the shape within its class, e.g. `2^20`
"""
struct Case{N, M}
    dims::NTuple{N, Int}
    region::NTuple{M, Int}
    family::Symbol
    precision::DataType
    direction::Symbol
    placement::Symbol
    tuned::Bool
    class::Symbol
    label::String
end

"""
    case_id(case::Case)

Returns the string that identifies a case in every file the suite writes.

It is built out of the fields rather than hashed, so it says the same thing in
every process, on every machine and under every Julia version. The real type
rather than the complex alias goes in it because `string(Float32)` is a
guarantee and the printing of a type alias is not.
"""
case_id(case::Case) = string(case.class, "-", case.family, "-", real(case.precision), "-", case.direction, "-", case.placement, case.tuned ? "-tuned" : "", "-", join(case.dims, "x"), "-d", join(case.region, ""))

"""
    cases(backend::Symbol, caps=CAPABILITIES[backend])

Builds the case list for one backend, with everything its device cannot run left out.

There are no rows saying a thing was skipped: a combination outside `caps`
never becomes a case in the first place. A new backend is a new entry in
`CAPABILITIES` and nothing else.

# Arguments
- `backend::Symbol`: The backend to enumerate, e.g. `:fftw`
- `caps`: The capability entry to filter against, defaulting to the backend's own

# Returns
- A `Vector{Case}`
"""
function cases(backend::Symbol, caps=CAPABILITIES[backend])
    out = Case[]
    for (class, dims, region, label) in (caps.cliff ? vcat(SHAPES, CLIFF_SHAPES) : SHAPES)
        prod(dims) <= caps.max_elements || continue

        if class === :cliff
            ComplexF32 in caps.types && push!(out, Case(dims, region, :c2c, ComplexF32, :forward, :oop, false, class, label))
            continue
        end

        for T in caps.types
            push!(out, Case(dims, region, :c2c, T, :forward, :oop, false, class, label))
            if class === :pow2 || class === :prime
                push!(out, Case(dims, region, :r2c, T, :forward, :oop, false, class, label))
                push!(out, Case(dims, region, :c2r, T, :inverse, :oop, false, class, label))
            end
        end

        if class === :pow2 && ComplexF32 in caps.types
            push!(out, Case(dims, region, :c2c, ComplexF32, :inverse, :oop, false, class, label))
            push!(out, Case(dims, region, :c2c, ComplexF32, :forward, :inplace, false, class, label))
        end

        # The tuned row is a case of its own rather than another implementation
        # of the plain one, so the main matrix keeps the two implementations its
        # budget was written for and a tuned number never lands in a plot that
        # did not ask for one. Its duel is VkFFT against itself.
        if caps.tuning && class in TUNED_CLASSES && ComplexF32 in caps.types
            push!(out, Case(dims, region, :c2c, ComplexF32, :forward, :oop, true, class, label))
        end
    end
    return out
end

"""
    destroys_input(case::Case)

Returns whether the transform writes over the array it was handed.

An in place transform does by definition, and FFTW's larger complex to real
plans do as well. Two things read this. `prepare` keeps a pristine copy so that
`refill!` can put the input back before each sample, and the timer refuses to
batch such a case, since every application after the first in a batch would run
on the previous one's output and the values would run away.
"""
destroys_input(case::Case) = case.placement === :inplace || case.family === :c2r

"""
    input(case::Case)

Returns the fixed timing input for a case, uniform random in [-1, 1).

Every implementation of a case sees the same bytes, and the seed is the shape
alone, so the same numbers reach a case at every precision it is measured at.
The draws come from a `StableRNG` rather than the default one, which is what
makes the Mac and molering agree.

A `:c2r` case gets a Hermitian spectrum. The transform is oblivious and would
run just as fast on any complex array, but an inverse real transform is only
well defined once the imaginary parts of the DC and Nyquist bins are pinned, and
two libraries resolving that differently would read as an accuracy difference
when it is a difference of convention. The real families are enumerated on the
one dimensional classes only, so the first and last points of the flat draw are
those two bins.

# Returns
- An `Array` of the case's input element type: real for `:r2c`, complex otherwise, and of the halved first dimension for `:c2r`
"""
function input(case::Case)
    rng = StableRNG(foldl((s, d) -> 31 * s + UInt64(d), case.dims; init=INPUT_SEED))
    sz = case.family === :c2r ? (case.dims[1] ÷ 2 + 1, Base.tail(case.dims)...) : case.dims
    T = case.family === :r2c ? real(case.precision) : case.precision

    vals = uniform_input(rng, prod(sz), T)
    if case.family === :c2r
        vals[1] = real(vals[1])
        iseven(case.dims[1]) && (vals[end] = real(vals[end])) # the primes are odd and have no Nyquist bin
    end
    return reshape(vals, sz)
end
