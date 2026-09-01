# The r2r family: VkFFT's DCT and DST of types I to IV, with FFTW's REDFT/RODFT
# semantics element for element. Both sides of an r2r transform have the same
# element type and the same size, so these plans need neither the real family's
# second element type nor its second size tuple. What they do carry is the kind
# and the type, which VkFFT applies to every transformed axis of a plan at once.

## Plan type

"""
    VkFFTR2RPlan{T, N, IP, B, M}

A first-class `AbstractFFTs.Plan` for VkFFT's real-to-real transforms.

The type parameters are (T) the element type, (N) `ndims` of the arrays, (IP)
whether the plan is in-place, (B) the backend tag (`:cuda`, `:opencl` or `:metal`) and (M) the
number of transformed dimensions. Input and output have the same size and the
same element type, so `p * x` gives an array shaped like `x`.

`kind` and `type` name the transform: `:dct` with `type` 2 is FFTW's `REDFT10`,
`:dst` with `type` 3 is `RODFT01`, and so on. VkFFT reads one DCT type and one
DST type per plan and applies it to every transformed axis, so a plan cannot mix
kinds or types the way FFTW's per-axis `kinds` tuple can.

An inverse plan carries the same `kind` and `type` as the forward plan it
undoes, and computes that transform's inverse: the type-3 transform of the same
kind for a type-2 plan, the type-2 for a type-3 plan, and the same type for
types 1 and 4. VkFFT applies the exact per-type scale factor in-kernel, so an
inverse plan is a true inverse and needs no scaling pass.

Plans are cached and reused, as `VkFFT.plan_fft`'s are.

# Fields
- `app::Ptr{Cvoid}`: The equivalent of an FFTW plan but for VkFFT. `C_NULL` when there is nothing to transform
- `sz::NTuple{N, Int}`: The size of both arrays
- `region::NTuple{M, Int}`: The dimensions on which to perform the transform, sorted
- `kind::Symbol`: `:dct` or `:dst`
- `type::Int`: The transform type, 1 to 4
- `direction::Int32`: -1 for forward, 1 for inverse
- `normalize::Bool`: Whether VkFFT applies the per-type 1/S in-kernel
- `zeropad::NTuple{2, Int}`: The zero-padded range on the first transformed dimension, `(0, 0)` for none
- `device_id::UInt64`: Identity of the device context the plan was built for
- `roots::Vector{Any}`: Backend objects (context, queue, ...) the app outlives nothing without
- `lock::ReentrantLock`: Held for the length of one application, because a VkFFT app is not reentrant
- `destroyed::Bool`: Set by the finalizer so a double destroy is a no-op
- `pinv::Union{Nothing, VkFFTR2RPlan{T, N, IP, B, M}}`: Cached raw inverse plan
"""
mutable struct VkFFTR2RPlan{T <: VkFFTReal, N, IP, B, M} <: AbstractVkFFTPlan{T}
    app::Ptr{Cvoid}
    sz::NTuple{N, Int}
    region::NTuple{M, Int}
    kind::Symbol
    type::Int
    direction::Int32
    normalize::Bool
    zeropad::NTuple{2, Int}
    device_id::UInt64
    roots::Vector{Any} # concrete field type (only ever read by the finalizer, never in mul!)
    lock::ReentrantLock
    destroyed::Bool
    pinv::Union{Nothing, VkFFTR2RPlan{T, N, IP, B, M}}

    function VkFFTR2RPlan{T, N, IP, B, M}(app::Ptr{Cvoid}, sz::NTuple{N, Int}, region::NTuple{M, Int}, kind::Symbol, type::Int, direction::Int32, normalize::Bool, zeropad::NTuple{2, Int}, device_id::UInt64, roots::Vector{Any}) where {T, N, IP, B, M}
        plan = new{T, N, IP, B, M}(app, sz, region, kind, type, direction, normalize, zeropad, device_id, roots, ReentrantLock(), false, nothing)
        app == C_NULL || finalizer(unsafe_free!, plan)
        return plan
    end
end

_plan_backend(::VkFFTR2RPlan{T, N, IP, B}) where {T, N, IP, B} = Val(B)

## Errors

"""
    _unsupported_r2r_eltype(::Type{T})

Throws the error for an element type the DCT/DST planners cannot plan for.
"""
_unsupported_r2r_eltype(::Type{T}) where T <: VkFFTComplex = throw(ArgumentError("VkFFT real-to-real plans transform a real array into a real array of the same size. Got $T. Use VkFFT.plan_fft for complex input."))
_unsupported_r2r_eltype(::Type{T}) where T = throw(ArgumentError("VkFFT real-to-real plans support Float16, Float32 and Float64. Got $T. $QUAD_HINT"))

## Planning

"""
    _check_r2r(dims::NTuple{N, Int}, region::NTuple{M, Int}, type::Int, entry::String)

Throws unless VkFFT can run this real-to-real transform with FFTW's semantics.

The length-1 check comes first, ahead of anything that could reach the C
library. A type-1 transform of an axis of length 1 sets that axis' internal
transform length to zero, and VkFFT's radix decomposition then spins forever
with nothing to cancel, an upstream VkFFT bug that hangs inside vkfft_create.
The other types create and run, but VkFFT force-omits a length-1 axis, which is
the identity along it, where FFTW returns a nonzero multiple of the input
instead. Both are refused.

The type check is second because VkFFT validates no range at all, an upstream
bug: a type of 5 or 99 falls through every branch that generates the pre- and
post-processing an r2r transform needs, leaving a plain real transform that is
no DCT and no DST.
"""
function _check_r2r(dims::NTuple{N, Int}, region::NTuple{M, Int}, type::Int, entry::String) where {N, M}
    for d in region
        dims[d] == 1 && throw(ArgumentError("a VkFFT real-to-real transform needs every transformed dimension to have length at least 2, and dimension $d of this $(join(dims, '×')) array has length 1. VkFFT hangs while planning a type-1 transform of a length-1 axis (an upstream VkFFT bug), and quietly computes the identity along it for the other types, where FFTW scales. Drop dimension $d from the region."))
    end
    1 <= type <= 4 || throw(ArgumentError("VkFFT.$entry takes a transform type of 1, 2, 3 or 4. Got type = $type. VkFFT accepts any value and quietly computes something that is no DCT and no DST at all, an upstream bug."))

    return nothing
end

"""
    _create_r2r_plan(::Type{T}, sz::NTuple{N, Int}, region::NTuple{M, Int}, kind::Symbol, type::Int, direction::Int32, normalize::Bool, zeropad::NTuple{2, Int}, ::Val{IP}, backend::Val{B}, device_id::UInt64, roots::Vector{Any}; coalesced_memory::Int=0, aim_threads::Int=0)

Returns the cached real-to-real plan for this configuration, creating the VkFFT application if needed.
"""
function _create_r2r_plan(::Type{T}, sz::NTuple{N, Int}, region::NTuple{M, Int}, kind::Symbol, type::Int, direction::Int32, normalize::Bool, zeropad::NTuple{2, Int}, ::Val{IP}, backend::Val{B}, device_id::UInt64, roots::Vector{Any}; coalesced_memory::Int=0, aim_threads::Int=0) where {T <: VkFFTReal, N, M, IP, B}
    layout = _map_region(sz, region, _max_dims())
    dct = kind === :dct ? Int32(type) : Int32(0)
    dst = kind === :dst ? Int32(type) : Int32(0)
    key = (B, device_id, T, sz, region, direction, normalize, IP, false, 0, dct, dst, zeropad, coalesced_memory, aim_threads)

    plan = _get_or_create_plan(key) do
        VkFFTR2RPlan{T, N, IP, B, M}(_create_app(T, layout, direction, normalize, IP, false, backend, roots; dct=dct, dst=dst, zeropad=zeropad, coalesced_memory=coalesced_memory, aim_threads=aim_threads), sz, region, kind, type, direction, normalize, zeropad, device_id, roots)
    end

    return plan::VkFFTR2RPlan{T, N, IP, B, M}
end

"""
    _make_r2r_plan(x, region::NTuple{M, Int}, kind::Symbol, type::Int, direction::Int32, normalize::Bool, zeropad::NTuple{2, Int}, ::Val{IP}, entry::String, tune)

Creates or looks up the real-to-real plan for an array, a canonical region and a kind.

Every guard runs before the backend is even looked up, so a configuration VkFFT
would hang or silently misread never reaches the C library.
"""
function _make_r2r_plan(x::AbstractArray{T, N}, region::NTuple{M, Int}, kind::Symbol, type::Int, direction::Int32, normalize::Bool, zeropad::NTuple{2, Int}, ::Val{IP}, entry::String, tune) where {T <: VkFFTReal, N, M, IP}
    sz = size(x)
    _check_r2r(sz, region, type, entry)
    _check_zeropad(sz, region, zeropad)
    sweep, force = _tune_mode(tune)

    backend = _backend(typeof(x))
    _check_layout(x)
    _check_precision(T, x)

    sweep && return _tuned_r2r_plan(T, sz, region, kind, type, direction, normalize, zeropad, Val(IP), backend, _device_id(x), _device_roots(x), x, force)

    return _create_r2r_plan(T, sz, region, kind, type, direction, normalize, zeropad, Val(IP), backend, _device_id(x), _device_roots(x))
end

function _make_r2r_plan(x::AbstractArray{T, N}, region::NTuple{M, Int}, kind::Symbol, type::Int, direction::Int32, normalize::Bool, zeropad::NTuple{2, Int}, ::Val{IP}, entry::String, tune) where {T, N, M, IP}
    _check_r2r(size(x), region, type, entry)
    return _unsupported_r2r_eltype(T)
end

"""
    _raw_inv(plan::VkFFTR2RPlan)

Creates the plan that undoes `plan`.

VkFFT's `normalize` divides the inverse by the exact per-type round-trip factor,
so a forward plan inverts to a genuine inverse with no scaling pass and no
`ScaledPlan`, and that inverse inverts straight back to the forward plan. This
family has no unnormalized inverse entry point, so there is no third case.
"""
function _raw_inv(plan::VkFFTR2RPlan{T, N, IP, B, M}) where {T, N, IP, B, M}
    _check_invertible(plan)
    forward = plan.direction == FORWARD
    direction = forward ? INVERSE : FORWARD
    return _create_r2r_plan(T, plan.sz, plan.region, plan.kind, plan.type, direction, forward, plan.zeropad, Val(IP), Val(B), plan.device_id, plan.roots)
end

# A real-to-real inverse is always normalized here, so nothing ever needs a
# ScaledPlan wrapped around it.
_wrap_inv(plan::VkFFTR2RPlan, raw::VkFFTR2RPlan) = raw

"""
    _r2r_entry(kind::Symbol, forward::Bool, inplace::Bool)

Returns the name of the entry point that builds a given real-to-real plan.
"""
_r2r_entry(kind::Symbol, forward::Bool, inplace::Bool) = "VkFFT.plan_" * (forward ? "" : "i") * string(kind) * (inplace ? "!" : "")

_check_io(plan::VkFFTR2RPlan{T, N, IP}, y::AbstractArray, x::AbstractArray) where {T, N, IP} = _check_uniform_io(plan, y, x, IP, _r2r_entry(plan.kind, plan.direction == FORWARD, false), _r2r_entry(plan.kind, plan.direction == FORWARD, true))

## Adjoints

"""
    VkFFTR2RAdjointStyle()

Adjoint style for VkFFT's real-to-real plans.

None of AbstractFFTs' styles fits. A DCT or DST matrix is real, so its adjoint is
its transpose, and in FFTW's normalization the eight transposes are: types 4 and
`RODFT00` are symmetric, types 2 and 3 of a kind are each other's transpose, and
`REDFT00` is its own transpose conjugated by a diagonal. All eight therefore come
out as one plan of the partner type between two diagonal weightings, which is
what `_r2r_transpose` tabulates and `_r2r_adjoint_recipe` adjusts per direction.
"""
struct VkFFTR2RAdjointStyle <: AbstractFFTs.AdjointStyle end

AbstractFFTs.AdjointStyle(::VkFFTR2RPlan) = VkFFTR2RAdjointStyle()
AbstractFFTs.output_size(plan::VkFFTR2RPlan, ::VkFFTR2RAdjointStyle) = size(plan)

# A weighting of the first and last element of every transformed axis. (1, 1) is
# no weighting at all, which is most of the table below.
const R2R_NO_WEIGHT = (1.0, 1.0)

"""
    _r2r_transpose(kind::Symbol, type::Int)

Returns the partner type and the two weightings that express one kind's transpose.

Read as `M' == left * M_partner * right`, with each weighting scaling the first
and the last element of every transformed axis. The three non-trivial entries are
the ones where FFTW's definition carries an endpoint term rather than a plain
cosine or sine: `REDFT00`'s two endpoints, `REDFT01`'s leading term, and
`RODFT01`'s trailing one.
"""
function _r2r_transpose(kind::Symbol, type::Int)
    if kind === :dct
        type == 1 && return (1, (0.5, 0.5), (2.0, 2.0))
        type == 2 && return (3, R2R_NO_WEIGHT, (2.0, 1.0))
        type == 3 && return (2, (0.5, 1.0), R2R_NO_WEIGHT)
        return (4, R2R_NO_WEIGHT, R2R_NO_WEIGHT)
    end
    type == 2 && return (3, R2R_NO_WEIGHT, (1.0, 2.0))
    type == 3 && return (2, (1.0, 0.5), R2R_NO_WEIGHT)
    return (type, R2R_NO_WEIGHT, R2R_NO_WEIGHT)
end

"""
    _r2r_adjoint_recipe(plan::VkFFTR2RPlan)

Returns the partner type and weightings whose composition is this plan's adjoint.

For a forward plan this is `_r2r_transpose` unchanged. An inverse plan computes
the inverse matrix, and transposing an inverse is inverting a transpose, so its
recipe is the forward one with the two weightings swapped and reciprocated.
"""
function _r2r_adjoint_recipe(plan::VkFFTR2RPlan)
    partner, left, right = _r2r_transpose(plan.kind, plan.type)
    plan.direction == FORWARD && return (partner, left, right)
    return (partner, (1 / right[1], 1 / right[2]), (1 / left[1], 1 / left[2]))
end

"""
    _r2r_weighted(x, sz::NTuple{N, Int}, region::NTuple{M, Int}, weight::NTuple{2, Float64})

Returns `x` scaled by a weighting of the first and last element of every transformed axis.

`x` itself comes back when the weighting is trivial, which is the common case, so
callers must not write into the result.
"""
function _r2r_weighted(x::AbstractArray{T, N}, sz::NTuple{N, Int}, region::NTuple{M, Int}, weight::NTuple{2, Float64}) where {T, N, M}
    weight == R2R_NO_WEIGHT && return x

    y = copy(x)
    for axis in region
        n = sz[axis]
        host = reshape(T[i == 1 ? weight[1] : (i == n ? weight[2] : 1.0) for i in 1:n], ntuple(i -> i == axis ? n : 1, Val(N)))
        y .*= convert(typeof(y), host)
    end

    return y
end

function AbstractFFTs.adjoint_mul(plan::VkFFTR2RPlan{T, N, IP, B, M}, x::AbstractArray, ::VkFFTR2RAdjointStyle) where {T, N, IP, B, M}
    _check_adjointable(plan)

    partner, left, right = _r2r_adjoint_recipe(plan)

    # The partner is out-of-place whatever the plan is, so weighting and
    # transforming never write into the caller's array.
    transform = _create_r2r_plan(T, plan.sz, plan.region, plan.kind, partner, plan.direction, plan.normalize, NO_ZEROPAD, Val(false), Val(B), plan.device_id, plan.roots)

    return _r2r_weighted(transform * _r2r_weighted(x, plan.sz, plan.region, right), plan.sz, plan.region, left)
end

## API entry points

"""
    plan_dct(x, region=1:ndims(x); type=2, zeropad=nothing, tune=false)

Creates an out-of-place forward discrete cosine transform VkFFT plan for `x`.

The plan is an `AbstractFFTs.Plan`, so `p * x`, `mul!(y, p, x)`, `inv(p)`,
`p \\ y`, `p'`, `size(p)` and `AbstractFFTs.fftdims(p)` all work. Output and
input have the same size and element type.

The transform is FFTW's, element for element and at every size: `type` 1 to 4
are `REDFT00`, `REDFT10`, `REDFT01` and `REDFT11`, unnormalized, so a round trip
through `VkFFT.plan_idct` of the same type comes back at a scale of exactly one.
VkFFT reads one type per plan and applies it to every transformed axis, so
FFTW's per-axis `kinds` tuple has no equivalent here.

Every transformed dimension has to have length at least 2. VkFFT drops a
length-1 axis, which computes the identity along it where FFTW scales, and it
hangs while planning a type-1 transform of one, an upstream VkFFT bug.

`x` must be a contiguous device array of `Float16`, `Float32` or `Float64`,
used as in `VkFFT.plan_fft`: only its type, size and device matter, and
`Float16` is a storage precision carrying the device and wrapper requirements
`VkFFT.plan_fft` lists.
Zero-padding and `tune` work as in `VkFFT.plan_fft`.

# Arguments
- `x`: The real input array
- `region=1:ndims(x)`: The dimensions on which to perform the transform
- `type=2`: The DCT type, 1 to 4
- `zeropad=nothing`: A 1-based inclusive range on the first transformed dimension VkFFT may treat as zero
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTR2RPlan`
"""
plan_dct(x::AbstractArray, region=_all_dims(x); type::Integer=2, zeropad=nothing, tune=false) = _make_r2r_plan(x, _canonical_region(region, ndims(x)), :dct, Int(type), FORWARD, false, _canonical_zeropad(zeropad), Val(false), "plan_dct", tune)

"""
    plan_dct!(x, region=1:ndims(x); type=2, zeropad=nothing, tune=false)

Creates an in-place forward discrete cosine transform VkFFT plan for `x`.

Like `VkFFT.plan_dct`, except that `p * x` overwrites `x` and `mul!(y, p, x)`
demands `y === x`.

# Arguments
- `x`: The real input array
- `region=1:ndims(x)`: The dimensions on which to perform the transform
- `type=2`: The DCT type, 1 to 4
- `zeropad=nothing`: A 1-based inclusive range on the first transformed dimension VkFFT may treat as zero
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTR2RPlan`
"""
plan_dct!(x::AbstractArray, region=_all_dims(x); type::Integer=2, zeropad=nothing, tune=false) = _make_r2r_plan(x, _canonical_region(region, ndims(x)), :dct, Int(type), FORWARD, false, _canonical_zeropad(zeropad), Val(true), "plan_dct!", tune)

"""
    plan_idct(x, region=1:ndims(x); type=2, zeropad=nothing, tune=false)

Creates an out-of-place inverse discrete cosine transform VkFFT plan for `x`.

`type` names the forward transform this undoes, not the transform this computes:
a `type=2` plan applies `REDFT01` divided by `prod(2 .* size(x)[region])`, which
is what inverts `REDFT10`. Types 1 and 4 invert to their own kind, over
`prod(2 .* size(x)[region] .- 2)` for type 1. VkFFT applies the factor inside the
transform kernels, so this costs one kernel launch rather than a transform plus a
scaling pass, and the round trip is exact rather than off by a scale.

Zero-padding and `tune` work as in `VkFFT.plan_bfft`, the padded range of an
inverse plan being skipped writes rather than skipped reads.

# Arguments
- `x`: The real input array
- `region=1:ndims(x)`: The dimensions on which to perform the transform
- `type=2`: The DCT type this inverts, 1 to 4
- `zeropad=nothing`: A 1-based inclusive range on the first transformed dimension VkFFT may treat as zero
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTR2RPlan`
"""
plan_idct(x::AbstractArray, region=_all_dims(x); type::Integer=2, zeropad=nothing, tune=false) = _make_r2r_plan(x, _canonical_region(region, ndims(x)), :dct, Int(type), INVERSE, true, _canonical_zeropad(zeropad), Val(false), "plan_idct", tune)

"""
    plan_idct!(x, region=1:ndims(x); type=2, zeropad=nothing, tune=false)

Creates an in-place inverse discrete cosine transform VkFFT plan for `x`.

# Arguments
- `x`: The real input array
- `region=1:ndims(x)`: The dimensions on which to perform the transform
- `type=2`: The DCT type this inverts, 1 to 4
- `zeropad=nothing`: A 1-based inclusive range on the first transformed dimension VkFFT may treat as zero
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTR2RPlan`
"""
plan_idct!(x::AbstractArray, region=_all_dims(x); type::Integer=2, zeropad=nothing, tune=false) = _make_r2r_plan(x, _canonical_region(region, ndims(x)), :dct, Int(type), INVERSE, true, _canonical_zeropad(zeropad), Val(true), "plan_idct!", tune)

"""
    plan_dst(x, region=1:ndims(x); type=2, zeropad=nothing, tune=false)

Creates an out-of-place forward discrete sine transform VkFFT plan for `x`.

FFTW's `RODFT00`, `RODFT10`, `RODFT01` and `RODFT11` for `type` 1 to 4,
unnormalized, element for element at every size. VkFFT reformulates the sine
transforms as cosine transforms internally, and the result is FFTW's with no sign
or index divergence anywhere. Everything else matches `VkFFT.plan_dct`.

# Arguments
- `x`: The real input array
- `region=1:ndims(x)`: The dimensions on which to perform the transform
- `type=2`: The DST type, 1 to 4
- `zeropad=nothing`: A 1-based inclusive range on the first transformed dimension VkFFT may treat as zero
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTR2RPlan`
"""
plan_dst(x::AbstractArray, region=_all_dims(x); type::Integer=2, zeropad=nothing, tune=false) = _make_r2r_plan(x, _canonical_region(region, ndims(x)), :dst, Int(type), FORWARD, false, _canonical_zeropad(zeropad), Val(false), "plan_dst", tune)

"""
    plan_dst!(x, region=1:ndims(x); type=2, zeropad=nothing, tune=false)

Creates an in-place forward discrete sine transform VkFFT plan for `x`.

# Arguments
- `x`: The real input array
- `region=1:ndims(x)`: The dimensions on which to perform the transform
- `type=2`: The DST type, 1 to 4
- `zeropad=nothing`: A 1-based inclusive range on the first transformed dimension VkFFT may treat as zero
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTR2RPlan`
"""
plan_dst!(x::AbstractArray, region=_all_dims(x); type::Integer=2, zeropad=nothing, tune=false) = _make_r2r_plan(x, _canonical_region(region, ndims(x)), :dst, Int(type), FORWARD, false, _canonical_zeropad(zeropad), Val(true), "plan_dst!", tune)

"""
    plan_idst(x, region=1:ndims(x); type=2, zeropad=nothing, tune=false)

Creates an out-of-place inverse discrete sine transform VkFFT plan for `x`.

`type` names the forward transform this undoes, exactly as in
`VkFFT.plan_idct`. The type-1 factor is `prod(2 .* size(x)[region] .+ 2)` here,
which is FFTW's for `RODFT00`.

# Arguments
- `x`: The real input array
- `region=1:ndims(x)`: The dimensions on which to perform the transform
- `type=2`: The DST type this inverts, 1 to 4
- `zeropad=nothing`: A 1-based inclusive range on the first transformed dimension VkFFT may treat as zero
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTR2RPlan`
"""
plan_idst(x::AbstractArray, region=_all_dims(x); type::Integer=2, zeropad=nothing, tune=false) = _make_r2r_plan(x, _canonical_region(region, ndims(x)), :dst, Int(type), INVERSE, true, _canonical_zeropad(zeropad), Val(false), "plan_idst", tune)

"""
    plan_idst!(x, region=1:ndims(x); type=2, zeropad=nothing, tune=false)

Creates an in-place inverse discrete sine transform VkFFT plan for `x`.

# Arguments
- `x`: The real input array
- `region=1:ndims(x)`: The dimensions on which to perform the transform
- `type=2`: The DST type this inverts, 1 to 4
- `zeropad=nothing`: A 1-based inclusive range on the first transformed dimension VkFFT may treat as zero
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTR2RPlan`
"""
plan_idst!(x::AbstractArray, region=_all_dims(x); type::Integer=2, zeropad=nothing, tune=false) = _make_r2r_plan(x, _canonical_region(region, ndims(x)), :dst, Int(type), INVERSE, true, _canonical_zeropad(zeropad), Val(true), "plan_idst!", tune)

## Base and AbstractFFTs methods

function LinearAlgebra.mul!(y::AbstractArray{T, N}, plan::VkFFTR2RPlan{T, N, IP, B, M}, x::AbstractArray{T, N}) where {T <: VkFFTReal, N, IP, B, M}
    _check_io(plan, y, x)

    if _is_trivial(plan)
        IP || copyto!(y, x)
        return y
    end

    _apply!(plan, y, x, plan.direction)
    return y
end

LinearAlgebra.mul!(y::AbstractArray, plan::VkFFTR2RPlan{T}, x::AbstractArray) where T = throw(ArgumentError("this VkFFT plan transforms $(ndims(plan))-dimensional $T arrays. Got an output of type $(typeof(y)) and an input of type $(typeof(x))"))

Base.:*(plan::VkFFTR2RPlan{T, N, true}, x::AbstractArray{T, N}) where {T <: VkFFTReal, N} = mul!(x, plan, x)
Base.:*(plan::VkFFTR2RPlan{T, N, false}, x::AbstractArray{T, N}) where {T <: VkFFTReal, N} = mul!(similar(x), plan, x)
Base.:*(plan::VkFFTR2RPlan{T}, x::AbstractArray) where T = throw(ArgumentError("this VkFFT plan transforms $(ndims(plan))-dimensional $T arrays. Got an input of type $(typeof(x))"))

function Base.show(io::IO, plan::VkFFTR2RPlan{T, N, IP, B, M}) where {T, N, IP, B, M}
    place = IP ? "in-place" : "out-of-place"
    name = uppercase(string(plan.kind)) * "-" * ("I", "II", "III", "IV")[plan.type] # the usual name, as in DCT-II
    label = plan.direction == FORWARD ? name : "normalized inverse $name"
    shape = join(plan.sz, '×')
    dims = "dim" * (M == 1 ? " " : "s ") * string(plan.region)
    padding = _zeropad_string(plan.zeropad)

    # ex: 2D out-of-place DCT-II VkFFT plan for 16×8 Float32 on dims (1, 2)
    # [opencl]
    if _is_trivial(plan) && !plan.destroyed
        print(io, "trivial $place $label VkFFT plan for $shape $T on $dims$padding [$B]")
    else
        print(io, "$(length(plan.region))D $place $label VkFFT plan for $shape $T on $dims$padding [$B]")
    end
end
