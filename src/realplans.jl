# The rfft family. Real-to-complex transforms get their own plan type because
# their two arrays differ in element type and in size, which VkFFTPlan's single
# T and single sz cannot express. The machinery underneath is shared with the
# complex-to-complex path: the backend interface, the layout mapper, the plan
# cache and app creation.

## Plan type

"""
    VkFFTRealPlan{T, S, N, B, M}

A first-class `AbstractFFTs.Plan` for VkFFT's real-to-complex and complex-to-real transforms.

The type parameters are (T) the input element type, (S) the output element type,
(N) `ndims` of both arrays, (B) the backend tag (`:cuda`, `:opencl` or `:metal`) and (M) the number
of transformed dimensions. A forward plan pairs a real T with a complex S and an
inverse plan the other way around, so `mul!` and `*` type check by dispatch
alone. There is no in-place parameter because AbstractFFTs has no in-place real
transform to implement.

`d` is the real length of the halved dimension. The halved complex length does
not recover it (d = 12 and d = 13 both give 7), which is why `d` is part of the
plan and of the cache key.

An inverse plan is only defined on a spectrum that really is the transform of a
real signal, which means one whose DC values, and its Nyquist values when `d` is
even, are real. VkFFT packs two such spectra into one complex pass, so for any
other input it returns something different from what FFTW returns.

Plans are cached and reused, as `VkFFT.plan_fft`'s are.

# Fields
- `app::Ptr{Cvoid}`: The equivalent of an FFTW plan but for VkFFT. `C_NULL` when there is nothing to transform
- `sz::NTuple{N, Int}`: The size of the input array
- `osz::NTuple{N, Int}`: The size of the output array
- `region::NTuple{M, Int}`: The dimensions on which to perform the FFT, sorted
- `d::Int`: The real length of the halved dimension
- `direction::Int32`: -1 for forward, 1 for inverse
- `normalize::Bool`: Whether VkFFT applies 1/N in-kernel, which is what makes this an `irfft` rather than a `brfft` plan
- `zeropad::NTuple{2, Int}`: The zero-padded range on the halved dimension, `(0, 0)` for none
- `device_id::UInt64`: Identity of the device context the plan was built for
- `roots::Vector{Any}`: Backend objects (context, queue, ...) the app outlives nothing without
- `lock::ReentrantLock`: Held for the length of one application, because a VkFFT app is not reentrant
- `destroyed::Bool`: Set by the finalizer so a double destroy is a no-op
- `pinv::Union{Nothing, VkFFTRealPlan{S, T, N, B, M}}`: Cached raw inverse plan
"""
mutable struct VkFFTRealPlan{T <: VkFFTNumber, S <: VkFFTNumber, N, B, M} <: AbstractVkFFTPlan{T}
    app::Ptr{Cvoid}
    sz::NTuple{N, Int}
    osz::NTuple{N, Int}
    region::NTuple{M, Int}
    d::Int
    direction::Int32
    normalize::Bool
    zeropad::NTuple{2, Int}
    device_id::UInt64
    roots::Vector{Any} # concrete field type (only ever read by the finalizer, never in mul!)
    lock::ReentrantLock
    destroyed::Bool
    pinv::Union{Nothing, VkFFTRealPlan{S, T, N, B, M}}

    function VkFFTRealPlan{T, S, N, B, M}(app::Ptr{Cvoid}, sz::NTuple{N, Int}, osz::NTuple{N, Int}, region::NTuple{M, Int}, d::Int, direction::Int32, normalize::Bool, zeropad::NTuple{2, Int}, device_id::UInt64, roots::Vector{Any}) where {T, S, N, B, M}
        plan = new{T, S, N, B, M}(app, sz, osz, region, d, direction, normalize, zeropad, device_id, roots, ReentrantLock(), false, nothing)
        app == C_NULL || finalizer(unsafe_free!, plan)
        return plan
    end
end

_plan_backend(::VkFFTRealPlan{T, S, N, B}) where {T, S, N, B} = Val(B)

## Errors

"""
    _unsupported_real_eltype(::Type{T})

Throws the error for an element type `VkFFT.plan_rfft` cannot plan for.
"""
_unsupported_real_eltype(::Type{T}) where T <: VkFFTComplex = throw(ArgumentError("VkFFT.plan_rfft transforms a real array into the complex spectrum of a real signal. Got $T. Use VkFFT.plan_fft for complex input, or VkFFT.plan_brfft and VkFFT.plan_irfft to go back from a spectrum to reals."))
_unsupported_real_eltype(::Type{T}) where T = throw(ArgumentError("VkFFT real-to-complex plans support Float16, Float32 and Float64. Got $T. $QUAD_HINT"))

"""
    _unsupported_spectrum_eltype(::Type{T})

Throws the error for an element type `VkFFT.plan_brfft` and `VkFFT.plan_irfft` cannot plan for.
"""
_unsupported_spectrum_eltype(::Type{T}) where T <: VkFFTReal = throw(ArgumentError("VkFFT.plan_brfft and VkFFT.plan_irfft transform the complex spectrum of a real signal back into reals. Got $T. Use VkFFT.plan_rfft for the forward direction."))
_unsupported_spectrum_eltype(::Type{T}) where T = throw(ArgumentError("VkFFT complex-to-real plans support ComplexF16, ComplexF32 and ComplexF64 spectra. Got $T. $QUAD_HINT"))

## Planning

"""
    _check_nonempty_region(region::NTuple{M, Int}, example::String)

Throws when a real transform is handed nothing to halve, naming `example` as the fix.
"""
_check_nonempty_region(region::NTuple{M, Int}, example::String) where M = M == 0 ? throw(ArgumentError("a real transform halves one dimension, so its region cannot be empty. Name the dimensions to transform, for example VkFFT.$example")) : nothing

"""
    _check_real_region(dims::NTuple{N, Int}, region::NTuple{M, Int})

Throws unless the region starts at the dimension a real transform is able to halve.

`dims` is the size of the real side of the transform, so the caller of an
inverse plan passes the output size rather than the spectrum's. Every other
region shape is fine: later batch dimensions, an omitted dimension between two
transformed ones, and a trailing batch run all work. The region has to be
non-empty by the time this runs, which `_check_nonempty_region` settles.
"""
function _check_real_region(dims::NTuple{N, Int}, region::NTuple{M, Int}) where {N, M}
    axis = _first_axis(dims)
    axis == 0 && return nothing # every dimension has length 1, so there is nothing to halve

    region[1] == axis || throw(ArgumentError("a VkFFT real transform halves its first transformed dimension, and that has to be dimension $axis, the first dimension of the $(join(dims, '×')) real array longer than 1. Region $region starts at dimension $(region[1]) instead. Permute the array so the dimension to halve comes first."))

    return nothing
end

"""
    _create_real_plan(::Type{T}, ::Type{S}, sz::NTuple{N, Int}, osz::NTuple{N, Int}, region::NTuple{M, Int}, d::Int, direction::Int32, normalize::Bool, backend::Val{B}, device_id::UInt64, roots::Vector{Any}; zeropad::NTuple{2, Int}=NO_ZEROPAD, coalesced_memory::Int=0, aim_threads::Int=0)

Returns the cached real plan for this configuration, creating the VkFFT application if needed.
"""
function _create_real_plan(::Type{T}, ::Type{S}, sz::NTuple{N, Int}, osz::NTuple{N, Int}, region::NTuple{M, Int}, d::Int, direction::Int32, normalize::Bool, backend::Val{B}, device_id::UInt64, roots::Vector{Any}; zeropad::NTuple{2, Int}=NO_ZEROPAD, coalesced_memory::Int=0, aim_threads::Int=0) where {T <: VkFFTNumber, S <: VkFFTNumber, N, M, B}
    layout = _map_region(direction == FORWARD ? sz : osz, region, _max_dims())
    key = (B, device_id, T, sz, region, direction, normalize, false, true, d, Int32(0), Int32(0), zeropad, coalesced_memory, aim_threads)

    plan = _get_or_create_plan(key) do
        VkFFTRealPlan{T, S, N, B, M}(_create_app(real(T), layout, direction, normalize, false, true, backend, roots; zeropad=zeropad, coalesced_memory=coalesced_memory, aim_threads=aim_threads), sz, osz, region, d, direction, normalize, zeropad, device_id, roots)
    end

    return plan::VkFFTRealPlan{T, S, N, B, M}
end

"""
    _make_rfft_plan(x, region::NTuple{M, Int}, zeropad::NTuple{2, Int}, tune)

Creates or looks up the forward real-to-complex plan for an array and a canonical region.
"""
function _make_rfft_plan(x::AbstractArray{T, N}, region::NTuple{M, Int}, zeropad::NTuple{2, Int}, tune) where {T <: VkFFTReal, N, M}
    sz = size(x)
    _check_nonempty_region(region, "plan_rfft(x, 1)")
    _check_real_region(sz, region)
    _check_zeropad(sz, region, zeropad)
    sweep, force = _tune_mode(tune)

    backend = _backend(typeof(x))
    _check_layout(x)
    _check_precision(T, x)

    osz = AbstractFFTs.rfft_output_size(sz, region)
    sweep && return _tuned_real_plan(T, Complex{T}, sz, osz, region, sz[region[1]], FORWARD, false, backend, _device_id(x), _device_roots(x), x, force; zeropad=zeropad)

    return _create_real_plan(T, Complex{T}, sz, osz, region, sz[region[1]], FORWARD, false, backend, _device_id(x), _device_roots(x); zeropad=zeropad)
end

_make_rfft_plan(x::AbstractArray{T, N}, region::NTuple{M, Int}, zeropad::NTuple{2, Int}, tune) where {T, N, M} = _unsupported_real_eltype(T)

"""
    _make_c2r_plan(x, d::Int, region::NTuple{M, Int}, normalize::Bool, entry::String, tune)

Creates or looks up the inverse complex-to-real plan for a spectrum, a real length and a region.
"""
function _make_c2r_plan(x::AbstractArray{T, N}, d::Int, region::NTuple{M, Int}, normalize::Bool, entry::String, tune) where {T <: VkFFTComplex, N, M}
    sz = size(x)
    _check_nonempty_region(region, "$entry(y, d, 1)")
    d >= 1 || throw(ArgumentError("the real length of an inverse real transform has to be at least 1. Got d = $d"))
    sz[region[1]] == d ÷ 2 + 1 || throw(ArgumentError("an inverse real transform of real length d = $d reads d ÷ 2 + 1 = $(d ÷ 2 + 1) complex values along dimension $(region[1]), but this array has $(sz[region[1]]). Pass the length of the real signal, not of its spectrum."))

    osz = AbstractFFTs.brfft_output_size(sz, d, region)
    _check_real_region(osz, region)
    sweep, force = _tune_mode(tune)

    backend = _backend(typeof(x))
    _check_layout(x)
    _check_precision(T, x)

    sweep && return _tuned_real_plan(T, real(T), sz, osz, region, d, INVERSE, normalize, backend, _device_id(x), _device_roots(x), x, force)

    return _create_real_plan(T, real(T), sz, osz, region, d, INVERSE, normalize, backend, _device_id(x), _device_roots(x))
end

_make_c2r_plan(x::AbstractArray{T, N}, d::Int, region::NTuple{M, Int}, normalize::Bool, entry::String, tune) where {T, N, M} = _unsupported_spectrum_eltype(T)

"""
    _raw_inv(plan::VkFFTRealPlan)

Creates the plan that undoes `plan`, up to the scale factor `_wrap_inv` puts back.

Same trade as on the complex-to-complex side: a forward plan inverts to VkFFT's
in-kernel normalized inverse, so an `irfft` costs one kernel launch rather than
a transform plus a scaling pass, and both inverse directions invert to the plain
forward plan.
"""
function _raw_inv(plan::VkFFTRealPlan{T, S, N, B, M}) where {T, S, N, B, M}
    _check_invertible(plan)
    forward = plan.direction == FORWARD
    direction = forward ? INVERSE : FORWARD
    return _create_real_plan(S, T, plan.osz, plan.sz, plan.region, plan.d, direction, forward, Val(B), plan.device_id, plan.roots)
end

"""
    _wrap_inv(plan::VkFFTRealPlan, raw::VkFFTRealPlan)

Adds the 1/N that `raw` does not apply, when `plan` is an unnormalized inverse.
"""
function _wrap_inv(plan::VkFFTRealPlan{T}, raw::VkFFTRealPlan) where T
    if plan.direction == INVERSE && !plan.normalize
        # The 1/N comes from the real side of the transform, which for this
        # inverse plan is its output. There the halved axis has its full length
        # d, which is the length VkFFT plans and scales from.
        return AbstractFFTs.ScaledPlan(raw, _normalization(T, plan.osz, plan.region))
    end
    return raw
end

"""
    _check_io(plan::VkFFTRealPlan, y, x)

Rejects output/input arrays that do not match what the real plan was built for.

Element types are already settled by dispatch, so what is left is the two
shapes, which differ from each other, and the memory layout of both arrays. The
layout check is the reason an offset view cannot reach `_buffer_handle`, where
it would be transformed from the base of its buffer instead of from its own
first element.
"""
function _check_io(plan::VkFFTRealPlan, y::AbstractArray, x::AbstractArray)
    plan.destroyed && throw(ArgumentError("this VkFFT plan has already been freed by VkFFT.unsafe_free!"))
    size(x) == plan.sz || throw(ArgumentError("this VkFFT plan was made for an input of size $(plan.sz). Got an input of size $(size(x))"))
    size(y) == plan.osz || throw(ArgumentError("this VkFFT plan writes an output of size $(plan.osz). Got an output of size $(size(y))"))

    _check_layout(x)
    _check_layout(y)

    return nothing
end

"""
    _trivial_apply!(y, x)

Applies a real plan with no axis left to transform, which is a copy through the element type.
"""
_trivial_apply!(y::AbstractArray{<:Complex}, x::AbstractArray{<:Real}) = (y .= x; return y)
_trivial_apply!(y::AbstractArray{<:Real}, x::AbstractArray{<:Complex}) = (y .= real.(x); return y)

"""
    _halfdim_vector(plan::VkFFTRealPlan, ::Val{N}, self_conjugate::Int, paired::Int)

Returns one value per bin of the halved axis, shaped to broadcast along that axis.

A real signal's spectrum stores only half of it, so every bin stands for a
conjugate pair except DC and, when the real length is even, Nyquist. Those two
are their own conjugates (`self_conjugate` is their value, `paired` the rest),
and every adjoint of a real transform treats them differently.
"""
function _halfdim_vector(plan::VkFFTRealPlan, ::Val{N}, self_conjugate::Int, paired::Int) where N
    halfdim = plan.region[1]
    n = plan.d ÷ 2 + 1
    values = [(i == 1 || 2 * (i - 1) == plan.d) ? self_conjugate : paired for i in 1:n]
    return reshape(values, ntuple(i -> i == halfdim ? n : 1, Val(N)))
end

## API entry points

"""
    plan_rfft(x, region=1:ndims(x); zeropad=nothing, tune=false)

Creates a forward real-to-complex VkFFT plan for `x`.

The plan is an `AbstractFFTs.Plan`, so `p * x`, `mul!(y, p, x)`, `inv(p)`,
`p \\ y`, `p'`, `size(p)` and `AbstractFFTs.fftdims(p)` all work. It is not
reachable through the global `rfft`/`plan_rfft`: those belong to whichever
package owns the array type.

The first transformed dimension is halved, so the output has
`size(x, d1) ÷ 2 + 1` along it and `size(x)` everywhere else. VkFFT can only
halve its own first axis, which means that dimension has to be the first one of
`x` longer than 1. Any other region is refused with advice to permute. There is
no in-place form, because AbstractFFTs has no in-place real transform.

`x` must be a contiguous device array of `Float16`, `Float32` or `Float64`,
used as in `VkFFT.plan_fft`: only its type, size and device matter, and
`Float16` is a storage precision carrying the device and wrapper requirements
`VkFFT.plan_fft` lists.

Zero-padding and `tune` work as in `VkFFT.plan_fft`. The padded range lies on
the halved dimension, which is the one axis VkFFT can pad and the axis it
already has to be.

# Arguments
- `x`: The real input array
- `region=1:ndims(x)`: The dimensions on which to perform the FFT
- `zeropad=nothing`: A 1-based inclusive range on the halved dimension VkFFT may treat as zero
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTRealPlan`
"""
plan_rfft(x::AbstractArray, region=_all_dims(x); zeropad=nothing, tune=false) = _make_rfft_plan(x, _canonical_region(region, ndims(x)), _canonical_zeropad(zeropad), tune)

"""
    plan_brfft(x, d, region=1:ndims(x); tune=false)

Creates an unnormalized inverse complex-to-real VkFFT plan for the spectrum `x`.

`d` is the length of the real signal along the halved dimension, which the
spectrum does not carry: `d = 12` and `d = 13` both leave 7 complex values.
Unnormalized: `p * (VkFFT.plan_rfft(r, region) * r)` is
`prod(size(r)[region])` times `r`. Use `VkFFT.plan_irfft` for the normalized
inverse, which is cheaper than scaling afterwards.

Applying the plan clobbers `x`, which VkFFT uses as scratch space whenever more
than one axis is transformed. FFTW's `brfft` destroys its input too. Copy the
spectrum first if you still need it.

`tune` works as in `VkFFT.plan_fft`.

# Arguments
- `x`: The complex input spectrum
- `d`: The length of the real signal along the halved dimension
- `region=1:ndims(x)`: The dimensions on which to perform the FFT
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTRealPlan`
"""
plan_brfft(x::AbstractArray, d::Integer, region=_all_dims(x); tune=false) = _make_c2r_plan(x, Int(d), _canonical_region(region, ndims(x)), false, "plan_brfft", tune)

"""
    plan_irfft(x, d, region=1:ndims(x); tune=false)

Creates a normalized inverse complex-to-real VkFFT plan for the spectrum `x`.

The 1/N is applied by VkFFT inside the transform kernels, so this costs one
kernel launch, not two. `N` is the product of the transformed axis lengths of
the real signal, `d` included, and never of the batched or omitted ones.

Applying the plan clobbers `x`, exactly as `VkFFT.plan_brfft` does, and `tune`
works as in `VkFFT.plan_fft`.

# Arguments
- `x`: The complex input spectrum
- `d`: The length of the real signal along the halved dimension
- `region=1:ndims(x)`: The dimensions on which to perform the FFT
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTRealPlan`
"""
plan_irfft(x::AbstractArray, d::Integer, region=_all_dims(x); tune=false) = _make_c2r_plan(x, Int(d), _canonical_region(region, ndims(x)), true, "plan_irfft", tune)

## Base and AbstractFFTs methods

function LinearAlgebra.mul!(y::AbstractArray{S, N}, plan::VkFFTRealPlan{T, S, N, B, M}, x::AbstractArray{T, N}) where {T, S, N, B, M}
    _check_io(plan, y, x)

    _is_trivial(plan) && return _trivial_apply!(y, x)

    _apply!(plan, y, x, plan.direction)
    return y
end

LinearAlgebra.mul!(y::AbstractArray, plan::VkFFTRealPlan{T, S}, x::AbstractArray) where {T, S} = throw(ArgumentError("this VkFFT plan transforms $(ndims(plan))-dimensional $T arrays into $S ones. Got an output of type $(typeof(y)) and an input of type $(typeof(x))"))

Base.:*(plan::VkFFTRealPlan{T, S, N}, x::AbstractArray{T, N}) where {T, S, N} = mul!(similar(x, S, plan.osz), plan, x)
Base.:*(plan::VkFFTRealPlan{T, S}, x::AbstractArray) where {T, S} = throw(ArgumentError("this VkFFT plan transforms $(ndims(plan))-dimensional $T arrays into $S ones. Got an input of type $(typeof(x))"))

"""
    VkFFTRFFTAdjointStyle()

Adjoint style for VkFFT's forward real-to-complex plans.

`AbstractFFTs.RFFTAdjointStyle` recovers the adjoint by reweighting the spectrum
and running the plan's inverse over the result. That reweighted array is no
longer the spectrum of a real signal, and VkFFT's complex-to-real transform only
matches FFTW's on ones that are, so this style takes a longer route where every
complex-to-real call sees a consistent spectrum.
"""
struct VkFFTRFFTAdjointStyle <: AbstractFFTs.AdjointStyle end

"""
    VkFFTNormalizedIRFFTAdjointStyle()

Adjoint style for VkFFT's in-kernel normalized complex-to-real inverse plans.

`AbstractFFTs.IRFFTAdjointStyle` assumes the plan is the unnormalized `brfft`,
so it multiplies the transformed volume back in. A plan that already carries a
1/N is off by `N^2` under that rule. This style divides by the volume instead.
"""
struct VkFFTNormalizedIRFFTAdjointStyle <: AbstractFFTs.AdjointStyle end

function AbstractFFTs.AdjointStyle(plan::VkFFTRealPlan)
    plan.direction == FORWARD && return VkFFTRFFTAdjointStyle()
    plan.normalize && return VkFFTNormalizedIRFFTAdjointStyle()
    return AbstractFFTs.IRFFTAdjointStyle(plan.d)
end

AbstractFFTs.output_size(plan::VkFFTRealPlan, ::VkFFTRFFTAdjointStyle) = plan.osz
AbstractFFTs.output_size(plan::VkFFTRealPlan, ::VkFFTNormalizedIRFFTAdjointStyle) = plan.osz

# An rfft is the one-axis real transform of the halved dimension followed by a
# plain complex transform of the others, and the two commute, so the adjoint is
# the unnormalized complex inverse of the others followed by the adjoint of the
# one-axis real transform. That second step is the halved axis' own
# complex-to-real transform over a spectrum whose paired bins are halved and
# whose self-conjugate bins are made real. Making them real is what leaves VkFFT
# a consistent spectrum to invert.
function AbstractFFTs.adjoint_mul(plan::VkFFTRealPlan{T, S, N, B, M}, x::AbstractArray, ::VkFFTRFFTAdjointStyle) where {T, S, N, B, M}
    _check_adjointable(plan)

    others_region = ntuple(i -> plan.region[i + 1], Val(M - 1)) # the region minus the halved dimension
    others = _create_plan(S, plan.osz, others_region, INVERSE, false, Val(false), Val(B), plan.device_id, plan.roots)
    axis = _create_real_plan(S, T, plan.osz, plan.sz, (plan.region[1],), plan.d, INVERSE, false, Val(B), plan.device_id, plan.roots)

    spectrum = others * x
    self_conjugate = convert(typeof(spectrum), _halfdim_vector(plan, Val(N), 1, 0))

    return axis * ((spectrum .+ self_conjugate .* conj.(spectrum)) ./ 2)
end

function AbstractFFTs.adjoint_mul(plan::VkFFTRealPlan{T, S, N}, x::AbstractArray, ::VkFFTNormalizedIRFFTAdjointStyle) where {T, S, N}
    _check_adjointable(plan)

    spectrum = inv(plan) * x
    weights = convert(typeof(spectrum), _halfdim_vector(plan, Val(N), 1, 2))
    return (_normalization(T, plan.osz, plan.region) .* weights) .* spectrum
end

function Base.show(io::IO, plan::VkFFTRealPlan{T, S, N, B, M}) where {T, S, N, B, M}
    forward = plan.direction == FORWARD
    kind = forward ? "real-to-complex" : "complex-to-real"
    direction = forward ? "forward" : (plan.normalize ? "normalized inverse" : "unnormalized inverse")
    shapes = "$(join(plan.sz, '×')) $T to $(join(plan.osz, '×')) $S"
    dims = "dim" * (M == 1 ? " " : "s ") * string(plan.region)
    padding = _zeropad_string(plan.zeropad)

    # ex: 2D out-of-place real-to-complex forward VkFFT plan for 16×8 Float32 to
    # 9×8 ComplexF32 on dims (1, 2) [opencl]
    if _is_trivial(plan) && !plan.destroyed
        print(io, "trivial out-of-place $kind $direction VkFFT plan for $shapes on $dims$padding [$B]")
    else
        print(io, "$(M)D out-of-place $kind $direction VkFFT plan for $shapes on $dims$padding [$B]")
    end
end
