# Pure mapping from (array size, region) to the four VkFFT layout fields, plus
# the two things that only work on VkFFT's own first axis: the halved axis of a
# real transform and a zero-padded range.

"""
    VkFFTLayout

The VkFFT layout a `(size(x), region)` pair maps onto.

`fft_dim == 0` marks a transform with nothing left to do (an empty region, or a
region whose every axis has length 1). VkFFT would happily build such a plan,
return `VKFFT_SUCCESS` and dispatch no kernel at all, leaving out-of-place
callers holding uninitialized memory, so the planner never hands that case to
the C library.

# Fields
- `fft_dim::Int`: Number of VkFFT axes, 0 for a transform that does nothing
- `size::NTuple{12, UInt64}`: Axis lengths after collapsing. Slots past `fft_dim` are 0
- `omit::NTuple{12, UInt64}`: 1 for a collapsed axis that is batched rather than transformed
- `number_batches::Int`: Outermost batch count. Never includes an omitted axis' length
"""
struct VkFFTLayout
    fft_dim::Int
    size::NTuple{VKFFT_MAX_FFT_DIMENSIONS, UInt64}
    omit::NTuple{VKFFT_MAX_FFT_DIMENSIONS, UInt64}
    number_batches::Int
end

const TRIVIAL_LAYOUT = VkFFTLayout(0, ntuple(_ -> UInt64(0), VKFFT_MAX_FFT_DIMENSIONS),
                                   ntuple(_ -> UInt64(0), VKFFT_MAX_FFT_DIMENSIONS), 1)

"""
    _is_trivial(layout::VkFFTLayout)

Returns if there are 0 transform dimensions (whether the layout transforms nothing), so no VkFFT app should be created.
"""
_is_trivial(layout::VkFFTLayout) = layout.fft_dim == 0

"""
    _canonical_region(region, n::Int)

Turns any iterable of FFT dimensions into a sorted tuple of unique in-bounds `Int`s.

Tuple and integer inputs keep their length in the type, so the plan's
region-length type parameter is inferable for them. Repeated or out-of-bounds
dimensions are refused, quoting `region` as the caller wrote it.
"""
function _canonical_region(region::NTuple{M, Int}, n::Int) where M
    dims = sort!(Int[d for d in region])
    allunique(dims) || throw(ArgumentError("region $region repeats a dimension. Each dimension can be transformed at most once"))
    for d in dims
        1 <= d <= n || throw(ArgumentError("region $region is out of bounds for a $n-dimensional array"))
    end

    return ntuple(i -> dims[i], Val(M))
end

_canonical_region(region::Integer, n::Int) = _canonical_region((Int(region),), n)
_canonical_region(region, n::Int) = _canonical_region(Tuple(Int(d) for d in region), n)

"""
    _map_region(dims::NTuple{N, Int}, region::NTuple{M, Int}, max_dims::Int=VKFFT_MAX_FFT_DIMENSIONS)

Maps an array size and a region onto VkFFT's `fft_dim` / `size` / `omit` / `number_batches`.

Length-1 axes are dropped first (VkFFT force-omits them anyway), then runs of
adjacent batch axes are merged into a single omitted axis, and the trailing
batch run folds into `number_batches` instead of occupying an axis.
"""
function _map_region(dims::NTuple{N, Int}, region::NTuple{M, Int}, max_dims::Int=VKFFT_MAX_FFT_DIMENSIONS) where {N, M}
    all(>(0), dims) || throw(ArgumentError("VkFFT cannot transform an array with a zero-length dimension. Got size $dims"))

    lengths = Int[]
    transformed = Bool[]
    for i in 1:N
        dims[i] == 1 && continue
        is_fft = i in region
        if !is_fft && !isempty(lengths) && !transformed[end]
            lengths[end] *= dims[i] # adjacent batch axes are interchangeable, so merge them
        else
            push!(lengths, dims[i])
            push!(transformed, is_fft)
        end
    end

    any(transformed) || return TRIVIAL_LAYOUT

    number_batches = 1
    if !transformed[end]
        number_batches = pop!(lengths)
        pop!(transformed)
    end

    fft_dim = length(lengths)
    fft_dim <= max_dims || throw(ArgumentError("region $region of a $(join(dims, '×')) array needs $fft_dim VkFFT axes but the library supports at most $max_dims. The transformed and batched dimensions interleave too much. Reshape or permute the array so the transformed dimensions sit next to each other first."))

    sizes = ntuple(i -> i <= fft_dim ? UInt64(lengths[i]) : UInt64(0), VKFFT_MAX_FFT_DIMENSIONS)
    omits = ntuple(i -> i <= fft_dim && !transformed[i] ? UInt64(1) : UInt64(0), VKFFT_MAX_FFT_DIMENSIONS)
    return VkFFTLayout(fft_dim, sizes, omits, number_batches)
end

"""
    _first_axis(dims::NTuple{N, Int})

Returns the first dimension longer than 1, or 0 when every dimension has length 1.

VkFFT force-omits length-1 axes, so this is the dimension that becomes VkFFT's
own first axis. Only two things work there: the halving of a real transform and
a zero-padded range.
"""
_first_axis(dims::NTuple{N, Int}) where N = something(findfirst(!=(1), dims), 0)

## Zero-padding

# The padded range as a 1-based inclusive (lo, hi) pair. (0, 0) is no padding.
const NO_ZEROPAD = (0, 0)

"""
    _canonical_zeropad(zeropad)

Turns the `zeropad` argument of a planning call into the `(lo, hi)` pair a plan carries.

`nothing` is no padding. Otherwise it is one 1-based inclusive range, optionally
wrapped in a one-element tuple, since only one axis can be padded.
"""
_canonical_zeropad(::Nothing) = NO_ZEROPAD
_canonical_zeropad(zeropad::AbstractUnitRange) = (Int(first(zeropad)), Int(last(zeropad)))
_canonical_zeropad(zeropad::Tuple{Any}) = _canonical_zeropad(zeropad[1])
_canonical_zeropad(zeropad) = throw(ArgumentError("zeropad takes one 1-based inclusive range naming the samples VkFFT may treat as zero, for example zeropad=5:11, or a one-element tuple holding one. Got $zeropad. Only the first transformed dimension can be padded, so there is never more than one range."))

"""
    _check_zeropad(dims::NTuple{N, Int}, region::NTuple{M, Int}, zeropad::NTuple{2, Int})

Throws unless a zero-padded range names real samples on the one axis VkFFT can pad.

VkFFT only skips the padded block correctly on its own first axis. On any other
axis the plan creates, runs, and reads uninitialized memory, an upstream VkFFT
bug, so the padded dimension has to be the first transformed one and that one
has to be the first dimension of the array longer than 1. VkFFT rejects no
range at all: it clamps or ignores whatever it cannot use, so the bounds are
checked here.
"""
function _check_zeropad(dims::NTuple{N, Int}, region::NTuple{M, Int}, zeropad::NTuple{2, Int}) where {N, M}
    zeropad == NO_ZEROPAD && return nothing

    lo, hi = zeropad
    M == 0 && throw(ArgumentError("zeropad names samples on the first transformed dimension, so it needs a non-empty region. Got region $region with zeropad=$lo:$hi."))

    axis = _first_axis(dims)
    axis == 0 && throw(ArgumentError("every dimension of this $(join(dims, '×')) array has length 1, so there is no axis to zero-pad. Drop the zeropad argument."))
    axis == region[1] || throw(ArgumentError("VkFFT can only zero-pad its own first axis, and that has to be dimension $axis, the first dimension of the $(join(dims, '×')) array longer than 1. Region $region starts at dimension $(region[1]) instead. On any other axis VkFFT accepts the padding and reads uninitialized memory, an upstream bug. Permute the array so the dimension to pad comes first."))

    n = dims[axis]
    1 <= lo <= hi <= n || throw(ArgumentError("a zero-padded range has to be a non-empty 1-based range inside dimension $axis, which has length $n. Got zeropad=$lo:$hi."))

    return nothing
end

"""
    _zeropad_fields(zeropad::NTuple{2, Int})

Returns the `zeropad_left`, `zeropad_right` and `perform_zeropad` config tuples for a padded range.

VkFFT reads a half-open 0-based `[left, right)` block, so a 1-based inclusive
`lo:hi` becomes `left = lo - 1` and `right = hi`, and `right = size` is what
covers the tail. The bounds are ignored unless the per-axis flag is set, so all
three are written together.
"""
function _zeropad_fields(zeropad::NTuple{2, Int})
    on = zeropad != NO_ZEROPAD
    lo, hi = zeropad
    left = ntuple(i -> (on && i == 1) ? UInt64(lo - 1) : UInt64(0), VKFFT_MAX_FFT_DIMENSIONS)
    right = ntuple(i -> (on && i == 1) ? UInt64(hi) : UInt64(0), VKFFT_MAX_FFT_DIMENSIONS)
    flags = ntuple(i -> (on && i == 1) ? Cint(1) : Cint(0), VKFFT_MAX_FFT_DIMENSIONS)
    return (left, right, flags)
end

"""
    _zeropad_string(zeropad::NTuple{2, Int})

Returns the ` zeropad=lo:hi` fragment `show` appends for a padded plan, or an empty string.
"""
_zeropad_string(zeropad::NTuple{2, Int}) = zeropad == NO_ZEROPAD ? "" : " zeropad=$(zeropad[1]):$(zeropad[2])"
