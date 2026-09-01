# The plan type, the AbstractFFTs interface on it, and the plan cache.
# Everything backend-specific goes through the eleven underscore functions in
# the "backend interface" section, which the extensions implement. The eleventh,
# _device_key, lives in diskcache.jl with the rest of the persistence.

const VkFFTComplex = Union{ComplexF16, ComplexF32, ComplexF64}
const VkFFTReal = Union{Float16, Float32, Float64}
const VkFFTNumber = Union{VkFFTReal, VkFFTComplex}

# Half precision is storage only: VkFFT reads and writes 16-bit values and
# computes every butterfly in fp32, so the accuracy of a half-precision
# transform is set by its 11-bit mantissa and not by half-precision arithmetic.
const VkFFTHalf = Union{Float16, ComplexF16}

const FORWARD = Int32(-1) # -1 for forward, 1 for inverse, as VkFFTAppend takes it
const INVERSE = Int32(1)

## Backend interface

# An extension must implement the functions below. Only _backend and
# _buffer_handle really matter for safety: _backend decides whether we know the
# array type at all (so views, offset arrays and host arrays are rejected by
# dispatch), and _buffer_handle must be impossible to call with the wrong kind
# of memory, because handing VkFFT a USM pointer where it wants a cl_mem
# segfaults inside the driver with no error return.
#
# _stream_handle and _synchronize are only for backends that keep the default
# _with_execution. A backend whose submission handle exists for the length of
# one application replaces _with_execution instead and never hands out a
# standing handle.

"""
    _backend(::Type{<:AbstractArray})

Returns the backend tag of an array type, as a `Val` so the tag lands in the plan's type.
"""
_backend(::Type{A}) where A <: AbstractArray = throw(ArgumentError("VkFFT has no backend for arrays of type $A. Load the package for your device (VkFFTCUDA, VkFFTOpenCL or VkFFTMetal) and pass a dense device array of that backend. A non-contiguous view, or a wrapper such as an adjoint or a permuted view, has no dense layout to transform, so copy it first."))

"""
    _check_layout(x::AbstractArray)

Throws unless the array's memory is laid out the way the wrapper needs it.

Called once per plan and again on both arrays of every application, so it must
stay cheap.
"""
_check_layout(x::AbstractArray) = throw(ArgumentError("VkFFT cannot check the memory layout of an array of type $(typeof(x)). Pass a contiguous device array belonging to the backend this plan was built for."))

"""
    _buffer_handle(x::AbstractArray)

Returns the device buffer handle the wrapper wants for this array.
"""
_buffer_handle(x::AbstractArray) = throw(ArgumentError("VkFFT cannot take a device buffer handle from an array of type $(typeof(x))."))

"""
    _stream_handle(x::AbstractArray)

Returns the per-call submission handle (an OpenCL queue, a CUDA stream) to run on.
"""
_stream_handle(x::AbstractArray) = throw(ArgumentError("VkFFT cannot find a queue or stream for an array of type $(typeof(x))."))

"""
    _synchronize(x::AbstractArray)

Waits for work already submitted for this array to finish.
"""
_synchronize(x::AbstractArray) = throw(ArgumentError("VkFFT cannot synchronize an array of type $(typeof(x))."))

"""
    _with_execution(f, x::AbstractArray)

Calls `f` with the submission handle to run on and returns its result, once the work is done.

The default is what a backend with a standing queue or stream wants: take the
handle from `_stream_handle` and wait with `_synchronize`. A backend whose
handle is instead created per application (a Metal command buffer, which has to
be made, filled, submitted and waited on inside one scope) replaces this method
and owns that whole scope. `f` may call `_vkfft_execute` more than once on the
handle it is given, and those calls run in the order they were made.
"""
@inline function _with_execution(f, x::AbstractArray)
    res = f(_stream_handle(x))
    _synchronize(x) # TODO: relax once the async story across backends is nailed down
    return res
end

"""
    _device_id(x::AbstractArray)

Returns an integer identifying the device context the array lives in, for the cache key.
"""
_device_id(x::AbstractArray) = throw(ArgumentError("VkFFT cannot identify the device of an array of type $(typeof(x))."))

"""
    _device_roots(x::AbstractArray)

Returns the backend objects a plan must keep alive, in the order `_with_device_handles` wants.
"""
_device_roots(x::AbstractArray) = throw(ArgumentError("VkFFT cannot collect device objects from an array of type $(typeof(x))."))

"""
    _with_device_handles(f, roots::Vector{Any}, backend::Val)

Calls `f` with the `void**` array of device handles `vkfft_create` expects.

A backend whose applications are bound to a device context also makes that
context current here, because VkFFT builds an application in whatever context
the calling thread has. Whatever an implementation wraps around `f`, it must
leave the thread's Objective-C autorelease state alone: VkFFT's Metal plan
builder over-releases the strings it compiles kernels from, so a pool opened
here and drained afterwards would take the process down when the plan is built.
"""
_with_device_handles(f, roots::Vector{Any}, backend::Val) = throw(ArgumentError("VkFFT has no device-handle plumbing for backend $backend."))

"""
    _with_plan_context(f, backend::Val, roots::Vector{Any})

Calls `f` with the device context a plan was built in made current, from a finalizer.

VkFFT frees an application with the same API it built it with, so on a backend
where that API is context-bound the context has to be current again. `f` runs
from the garbage collector, so an implementation must not yield and must not
throw when the context is already gone. Backends whose applications are not
context-bound leave this at the default, which runs `f` where it stands.
"""
_with_plan_context(f, backend::Val, roots::Vector{Any}) = f()

"""
    _check_precision(::Type{T}, x::AbstractArray)

Throws unless the device behind `x` can run a transform whose element type is `T`.

Element types are settled by dispatch everywhere else, but half precision is a
property of the device rather than of the backend: the OpenCL extension that
carries it is present on one device and missing on the next, and VkFFT reports
nothing better than a failed compile for the difference. A backend whose devices
all handle every element type it accepts leaves this at the default, which allows
everything.
"""
_check_precision(::Type{T}, x::AbstractArray) where T = nothing

"""
    _precision(::Type{T})

Returns the `vkfft_config.precision` code for a Julia element type.
"""
_precision(::Type{ComplexF16}) = Cint(2)
_precision(::Type{ComplexF32}) = Cint(0)
_precision(::Type{ComplexF64}) = Cint(1)
_precision(::Type{Float16}) = Cint(2)
_precision(::Type{Float32}) = Cint(0)
_precision(::Type{Float64}) = Cint(1)

# What every unsupported-element-type message says about quad precision. VkFFT's
# double-double stores four native floats per complex value, and no Julia
# element type has that layout, so there is nothing for the typed API to
# dispatch on.
const QUAD_HINT = "Double-double quad precision has no matching Julia element type, so it is reachable only through VkFFT.unsafe_plan."

"""
    _all_dims(x)

Returns `(1, 2, ..., ndims(x))` with the length in the type, so regions stay inferable.
"""
_all_dims(x::AbstractArray{T, N}) where {T, N} = ntuple(identity, Val(N))

"""
    _normalization(::Type{T}, sz::Tuple, region::Tuple)

Returns the 1/N a normalized inverse applies, over the transformed axes only.

`AbstractFFTs.normalization` computes the same thing but trips over an empty
region, which is a legal region here (it plans an identity transform).
"""
_normalization(::Type{T}, sz::Tuple, region::Tuple) where T = one(real(T)) / prod(i -> sz[i], region; init=1)

## Plan type

"""
    AbstractVkFFTPlan{T}

Supertype of the VkFFT plan types, so that freeing, sizes and inversion are written once.

`T` is the element type of a plan's input, which is what
`AbstractFFTs.Plan{T}` means. It is not the output element type: a real
transform has different element types on its two sides.
"""
abstract type AbstractVkFFTPlan{T} <: AbstractFFTs.Plan{T} end

"""
    VkFFTPlan{T, N, IP, B, M}

A first-class `AbstractFFTs.Plan` backed by a VkFFT application.

The type parameters are (T) the element type, (N) `ndims` of the input array,
(IP) whether the plan is in-place, (B) the backend tag (`:cuda`, `:opencl` or `:metal`) and (M) the
number of transformed dimensions. `app` is `C_NULL` for a plan whose region
transforms nothing, which is applied as a copy in Julia rather than handed to
VkFFT (see `VkFFTLayout`).

Plans are cached and reused, so two calls to `VkFFT.plan_fft` with the same
array shape, region and device return the same object. `VkFFT.clear_cache!()`
drops the cache.

# Fields
- `app::Ptr{Cvoid}`: The equivalent of an FFTW plan but for VkFFT. `C_NULL` when there is nothing to transform
- `sz::NTuple{N, Int}`: The size of the input array
- `region::NTuple{M, Int}`: The dimensions on which to perform the FFT, sorted
- `direction::Int32`: -1 for forward, 1 for inverse
- `normalize::Bool`: Whether VkFFT applies 1/N in-kernel, which is what makes this an `ifft` rather than a `bfft` plan
- `zeropad::NTuple{2, Int}`: The zero-padded range on the first transformed dimension, `(0, 0)` for none
- `device_id::UInt64`: Identity of the device context the plan was built for
- `roots::Vector{Any}`: Backend objects (context, queue, ...) the app outlives nothing without
- `lock::ReentrantLock`: Held for the length of one application, because a VkFFT app is not reentrant
- `destroyed::Bool`: Set by the finalizer so a double destroy is a no-op
- `pinv::Union{Nothing, VkFFTPlan{T, N, IP, B, M}}`: Cached raw inverse plan
"""
mutable struct VkFFTPlan{T <: VkFFTComplex, N, IP, B, M} <: AbstractVkFFTPlan{T}
    app::Ptr{Cvoid}
    sz::NTuple{N, Int}
    region::NTuple{M, Int}
    direction::Int32
    normalize::Bool
    zeropad::NTuple{2, Int}
    device_id::UInt64
    roots::Vector{Any} # concrete field type (only ever read by the finalizer, never in mul!)
    lock::ReentrantLock
    destroyed::Bool
    pinv::Union{Nothing, VkFFTPlan{T, N, IP, B, M}}

    function VkFFTPlan{T, N, IP, B, M}(app::Ptr{Cvoid}, sz::NTuple{N, Int}, region::NTuple{M, Int}, direction::Int32, normalize::Bool, zeropad::NTuple{2, Int}, device_id::UInt64, roots::Vector{Any}) where {T, N, IP, B, M}
        plan = new{T, N, IP, B, M}(app, sz, region, direction, normalize, zeropad, device_id, roots, ReentrantLock(), false, nothing)
        app == C_NULL || finalizer(unsafe_free!, plan)
        return plan
    end
end

"""
    _plan_backend(plan::AbstractVkFFTPlan)

Returns the plan's backend tag as a `Val`, so a finalizer can reach the backend interface.
"""
_plan_backend(::VkFFTPlan{T, N, IP, B}) where {T, N, IP, B} = Val(B)

# app == C_NULL and not destroyed means "region transforms nothing". The
# destroyed check always comes first, because a destroyed plan also has a null
# app.
_is_trivial(plan::AbstractVkFFTPlan) = plan.app == C_NULL

"""
    unsafe_free!(plan::AbstractVkFFTPlan)

Frees the plan's VkFFT application now instead of waiting for the garbage collector, at most once.

Unsafe because nothing stops another reference to the same plan from being
applied afterwards, and because a cached plan stays in the cache. Call
`VkFFT.clear_cache!()` first if you mean to free everything.

This is also every plan's finalizer, so it runs from the garbage collector: a
flag write and one ccall, wrapped in whatever context guard the backend needs
(`_with_plan_context`). The backend objects behind `roots` are released by
their own finalizers afterwards, in whatever order the collector picks, which
is why the guard has to tolerate a context that is already gone.

# Returns
- `nothing`
"""
function unsafe_free!(plan::AbstractVkFFTPlan)
    plan.destroyed && return nothing
    plan.destroyed = true
    app = plan.app
    plan.app = C_NULL
    app == C_NULL && return nothing
    _with_plan_context(_plan_backend(plan), plan.roots) do
        _vkfft_destroy(app)
    end
    return nothing
end

## Plan cache

# (backend, device, input element type, input size, region, direction,
# normalize, in-place, real-to-complex, logical real length, DCT type, DST type,
# zero-padded range, coalesced memory, aim threads). The real length exists for
# the rfft family: an inverse real plan and an inverse complex plan agree on
# everything before it, and d ÷ 2 + 1 does not recover d, so d = 12 and d = 13
# would otherwise share an entry. The DCT and DST types separate the r2r kinds
# from each other and from c2c, the range keeps a padded plan away from an
# unpadded one of the same shape, and the last two are the tuning knobs, which
# generate different kernels and so let a tuned and an untuned plan of one shape
# coexist.
const PlanCacheKey = Tuple{Symbol, UInt64, DataType, Tuple{Vararg{Int}}, Tuple{Vararg{Int}}, Int32, Bool, Bool, Bool, Int, Int32, Int32, Tuple{Int, Int}, Int, Int}

# Unbounded, with an explicit clear_cache!. An LRU would have to evict plans
# that callers may still hold, which is not a correctness problem but was a
# performance one. At ~1.4 MB of host memory per live plan and a handful of
# distinct shapes per program, bounding it is not yet worth the complexity.
# TODO: bound it. An evicted plan costs a recompile to get back, so eviction
# wants a policy better than LRU-by-count before it pays for itself.
const PLAN_CACHE = Dict{PlanCacheKey, Any}()
const PLAN_CACHE_LOCK = ReentrantLock()

"""
    _get_or_create_plan(create, key::PlanCacheKey)

Returns the cached plan for a key, running `create` and inserting its result on a miss.

One lock covers the lookup and the creation both. Creating a plan compiles
kernels, so this serializes concurrent first-time planning. It never serializes
cache hits for long. The result is whatever the cache holds, so callers assert
their concrete plan type on it.
"""
function _get_or_create_plan(create, key::PlanCacheKey)
    @lock PLAN_CACHE_LOCK begin
        cached = get(PLAN_CACHE, key, nothing)
        cached === nothing || return cached
        fresh = create()
        PLAN_CACHE[key] = fresh
        return fresh
    end
end

"""
    clear_cache!()

Empties the plan cache, so the plans in it can be finalized.

# Returns
- `nothing`
"""
function clear_cache!()
    @lock PLAN_CACHE_LOCK empty!(PLAN_CACHE)
    return nothing
end

"""
    cache_size()

Returns the number of plans currently held by the plan cache.
"""
cache_size() = @lock PLAN_CACHE_LOCK length(PLAN_CACHE)

## Planning

"""
    _create_plan(::Type{T}, sz::NTuple{N, Int}, region::NTuple{M, Int}, direction::Int32, normalize::Bool, ::Val{IP}, backend::Val{B}, device_id::UInt64, roots::Vector{Any}; zeropad::NTuple{2, Int}=NO_ZEROPAD, coalesced_memory::Int=0, aim_threads::Int=0)

Returns the cached complex-to-complex plan for this configuration, creating the VkFFT application if needed.
"""
function _create_plan(::Type{T}, sz::NTuple{N, Int}, region::NTuple{M, Int}, direction::Int32, normalize::Bool, ::Val{IP}, backend::Val{B}, device_id::UInt64, roots::Vector{Any}; zeropad::NTuple{2, Int}=NO_ZEROPAD, coalesced_memory::Int=0, aim_threads::Int=0) where {T <: VkFFTComplex, N, M, IP, B}
    layout = _map_region(sz, region, _max_dims())
    key = (B, device_id, T, sz, region, direction, normalize, IP, false, 0, Int32(0), Int32(0), zeropad, coalesced_memory, aim_threads)

    plan = _get_or_create_plan(key) do
        VkFFTPlan{T, N, IP, B, M}(_create_app(T, layout, direction, normalize, IP, false, backend, roots; zeropad=zeropad, coalesced_memory=coalesced_memory, aim_threads=aim_threads), sz, region, direction, normalize, zeropad, device_id, roots)
    end

    return plan::VkFFTPlan{T, N, IP, B, M}
end

"""
    _app_config(::Type{T}, layout::VkFFTLayout, direction::Int32, normalize::Bool, inplace::Bool, r2c::Bool; dct::Int32=Int32(0), dst::Int32=Int32(0), zeropad::NTuple{2, Int}=NO_ZEROPAD, coordinate_features::Int=1, perform_convolution::Bool=false, kernel_convolution::Bool=false, conjugate_convolution::Bool=false, coalesced_memory::Int=0, aim_threads::Int=0)

Builds the `VkFFTConfig` for a layout and a set of plan properties.

This is the one place a `VkFFTConfig` is built, so it is also what the tuning
record key is computed over: two plans that hash to the same key here are the
same plan for VkFFT.

A convolution plan needs both halves of its pipeline generated even though it
only ever runs forward, so it is the one plan that sets neither
`make_forward_only` nor `make_inverse_only`. Building one half leaves VkFFT
dereferencing the missing half during the append, an upstream VkFFT bug that
crashes instead of reporting, which the wrapper refuses as an invalid argument.
"""
function _app_config(::Type{T}, layout::VkFFTLayout, direction::Int32, normalize::Bool, inplace::Bool, r2c::Bool; dct::Int32=Int32(0), dst::Int32=Int32(0), zeropad::NTuple{2, Int}=NO_ZEROPAD, coordinate_features::Int=1, perform_convolution::Bool=false, kernel_convolution::Bool=false, conjugate_convolution::Bool=false, coalesced_memory::Int=0, aim_threads::Int=0) where T <: VkFFTNumber
    forward = direction == FORWARD
    one_direction = !perform_convolution # a convolution plan needs both halves generated
    zeropad_left, zeropad_right, perform_zeropad = _zeropad_fields(zeropad)

    return VkFFTConfig(fft_dim=layout.fft_dim, size=layout.size, omit=layout.omit,
                       number_batches=layout.number_batches, precision=_precision(T),
                       r2c=r2c ? 1 : 0, dct=dct, dst=dst,
                       inplace=inplace ? 1 : 0, normalize=normalize ? 1 : 0,
                       make_forward_only=(one_direction && forward) ? 1 : 0,
                       make_inverse_only=(one_direction && !forward) ? 1 : 0,
                       zeropad_left=zeropad_left, zeropad_right=zeropad_right,
                       perform_zeropad=perform_zeropad,
                       coalesced_memory=coalesced_memory, aim_threads=aim_threads,
                       coordinate_features=coordinate_features,
                       perform_convolution=perform_convolution ? 1 : 0,
                       kernel_convolution=kernel_convolution ? 1 : 0,
                       conjugate_convolution=conjugate_convolution ? 1 : 0)
end

"""
    _app_from_config(config::VkFFTConfig, backend::Val, roots::Vector{Any})

Creates a VkFFT application for a configuration, compiling its kernels.

The one door to `_vkfft_create`. Compiling the kernels is what makes building a
plan slow, and the plan cache in front of this is what makes re-planning free.
"""
function _app_from_config(config::VkFFTConfig, backend::Val, roots::Vector{Any})
    app = Ref(Ptr{Cvoid}(C_NULL))
    res = _with_device_handles(roots, backend) do handles
        _vkfft_create(Ref(config), handles, app)
    end
    _check(res)

    return app[]
end

"""
    _create_app(::Type{T}, layout::VkFFTLayout, direction::Int32, normalize::Bool, inplace::Bool, r2c::Bool, backend::Val, roots::Vector{Any}; dct::Int32=Int32(0), dst::Int32=Int32(0), zeropad::NTuple{2, Int}=NO_ZEROPAD, coordinate_features::Int=1, perform_convolution::Bool=false, kernel_convolution::Bool=false, conjugate_convolution::Bool=false, coalesced_memory::Int=0, aim_threads::Int=0)

Creates the VkFFT application for a layout, or returns `C_NULL` when there is nothing to do.

Every plan family reaches the C library through here, so this is where the
configurations VkFFT accepts and then silently gets wrong (upstream VkFFT bugs)
are refused: a DCT and a DST on one plan (the DST wins), a real-to-complex
transform carrying either (which corrupts the spectrum), and a plan that both
performs a convolution and only prepares a kernel for one. No entry point can
express any of them, so these guard whatever calls in from below.
"""
function _create_app(::Type{T}, layout::VkFFTLayout, direction::Int32, normalize::Bool, inplace::Bool, r2c::Bool, backend::Val, roots::Vector{Any}; dct::Int32=Int32(0), dst::Int32=Int32(0), zeropad::NTuple{2, Int}=NO_ZEROPAD, coordinate_features::Int=1, perform_convolution::Bool=false, kernel_convolution::Bool=false, conjugate_convolution::Bool=false, coalesced_memory::Int=0, aim_threads::Int=0) where T <: VkFFTNumber
    dct == 0 || dst == 0 || throw(ArgumentError("a VkFFT plan runs one real-to-real kind, not both. Got dct = $dct and dst = $dst. VkFFT accepts the pair and computes the DST, or at other type combinations something that is no DCT and no DST while still round-tripping, an upstream VkFFT bug."))
    !r2c || (dct == 0 && dst == 0) || throw(ArgumentError("a VkFFT real-to-complex plan cannot also be a DCT or a DST. Got r2c with dct = $dct and dst = $dst. VkFFT accepts the combination and corrupts the spectrum, an upstream bug."))
    !perform_convolution || !kernel_convolution || throw(ArgumentError("a VkFFT plan either performs a convolution or prepares a kernel for one, not both. VkFFT reads the kernel flag first and quietly ignores the other."))
    !perform_convolution || direction == FORWARD || throw(ArgumentError("a VkFFT convolution plan runs its whole pipeline in the forward direction. Got direction $direction."))

    _is_trivial(layout) && return Ptr{Cvoid}(C_NULL)

    config = _app_config(T, layout, direction, normalize, inplace, r2c; dct=dct, dst=dst,
                         zeropad=zeropad, coordinate_features=coordinate_features,
                         perform_convolution=perform_convolution,
                         kernel_convolution=kernel_convolution,
                         conjugate_convolution=conjugate_convolution,
                         coalesced_memory=coalesced_memory, aim_threads=aim_threads)

    return _app_from_config(config, backend, roots)
end

"""
    _make_plan(x, region::NTuple{M, Int}, direction::Int32, normalize::Bool, ::Val{IP}, zeropad::NTuple{2, Int}, tune)

Creates or looks up the complex-to-complex plan for an array, a canonical region and a direction.
"""
function _make_plan(x::AbstractArray{T, N}, region::NTuple{M, Int}, direction::Int32, normalize::Bool, ::Val{IP}, zeropad::NTuple{2, Int}, tune) where {T <: VkFFTComplex, N, M, IP}
    _check_zeropad(size(x), region, zeropad)
    sweep, force = _tune_mode(tune)

    backend = _backend(typeof(x))
    _check_layout(x)
    _check_precision(T, x)

    sweep && return _tuned_plan(T, size(x), region, direction, normalize, Val(IP), backend, _device_id(x), _device_roots(x), x, force; zeropad=zeropad)

    return _create_plan(T, size(x), region, direction, normalize, Val(IP), backend, _device_id(x), _device_roots(x); zeropad=zeropad)
end

_make_plan(x::AbstractArray{T, N}, region::NTuple{M, Int}, direction::Int32, normalize::Bool, ::Val{IP}, zeropad::NTuple{2, Int}, tune) where {T, N, M, IP} = throw(ArgumentError("VkFFT complex-to-complex plans support ComplexF16, ComplexF32 and ComplexF64. Got $T. Real input goes through VkFFT.plan_rfft, VkFFT.plan_brfft and VkFFT.plan_irfft. $QUAD_HINT"))

"""
    _check_invertible(plan::AbstractVkFFTPlan)

Throws when a plan discards part of its input, which leaves it with no inverse.
"""
_check_invertible(plan::AbstractVkFFTPlan) = plan.zeropad == NO_ZEROPAD ? nothing : throw(ArgumentError("a zero-padded VkFFT plan has no inverse: it never reads the samples in $(plan.zeropad[1]):$(plan.zeropad[2]), and VkFFT's padded inverse leaves that range undefined rather than putting them back. Plan the inverse you want directly, without zeropad, or with zeropad if you accept an undefined output range."))

"""
    _check_adjointable(plan::AbstractVkFFTPlan)

Throws when a plan's adjoint is not expressible with the plans this package builds.
"""
_check_adjointable(plan::AbstractVkFFTPlan) = plan.zeropad == NO_ZEROPAD ? nothing : throw(ArgumentError("the adjoint of a zero-padded VkFFT plan is not available. It would have to zero the range $(plan.zeropad[1]):$(plan.zeropad[2]) of its output, and VkFFT leaves a padded range undefined instead. Apply the unpadded plan's adjoint and zero that range yourself."))

"""
    _raw_inv(plan::VkFFTPlan)

Creates the plan that undoes `plan`, up to the scale factor `_wrap_inv` puts back.

A forward plan inverts to VkFFT's in-kernel normalized inverse, so no separate
scaling kernel has to run on an `ifft`. Both inverse directions invert to the
plain forward plan.
"""
function _raw_inv(plan::VkFFTPlan{T, N, IP, B, M}) where {T, N, IP, B, M}
    _check_invertible(plan)
    forward = plan.direction == FORWARD
    direction = forward ? INVERSE : FORWARD
    return _create_plan(T, plan.sz, plan.region, direction, forward, Val(IP), Val(B), plan.device_id, plan.roots)
end

"""
    _wrap_inv(plan::VkFFTPlan, raw::VkFFTPlan)

Adds the 1/N that `raw` does not apply, when `plan` is an unnormalized inverse.

VkFFT's `normalize` only exists on the inverse direction, so the inverse of a
`bfft` plan (a normalized *forward* transform) needs an
`AbstractFFTs.ScaledPlan`.
"""
function _wrap_inv(plan::VkFFTPlan{T}, raw::VkFFTPlan{T}) where T
    if plan.direction == INVERSE && !plan.normalize
        return AbstractFFTs.ScaledPlan(raw, _normalization(T, plan.sz, plan.region))
    end
    return raw
end

"""
    _check_uniform_io(plan::AbstractVkFFTPlan, y, x, inplace::Bool, out_of_place_entry::String, in_place_entry::String)

Rejects output/input arrays that do not match a plan whose two sides have the same size.

A plan is reusable across buffers, so the array reaching `mul!` is not
necessarily the one passed to the planner. An offset array whose size happens
to match the plan would otherwise blow through size checks and be transformed
from the base of its buffer instead of from its own first element. This is the
one place every application must go through, keeping `_buffer_handle` a pure
extraction inside `GC.@preserve`.
"""
function _check_uniform_io(plan::AbstractVkFFTPlan, y::AbstractArray, x::AbstractArray, inplace::Bool, out_of_place_entry::String, in_place_entry::String)
    plan.destroyed && throw(ArgumentError("this VkFFT plan has already been freed by VkFFT.unsafe_free!"))
    size(x) == plan.sz || throw(ArgumentError("this VkFFT plan was made for an array of size $(plan.sz). Got an input of size $(size(x))"))
    size(y) == plan.sz || throw(ArgumentError("this VkFFT plan was made for an array of size $(plan.sz). Got an output of size $(size(y))"))
    if inplace
        y === x || throw(ArgumentError("an in-place VkFFT plan transforms one array. Pass it as both output and input, or plan with $out_of_place_entry for an out-of-place transform"))
    else
        y === x && throw(ArgumentError("an out-of-place VkFFT plan needs distinct output and input arrays. Plan with $in_place_entry for an in-place transform"))
    end

    _check_layout(x)
    y === x || _check_layout(y)

    return nothing
end

_check_io(plan::VkFFTPlan{T, N, IP}, y::AbstractArray, x::AbstractArray) where {T, N, IP} = _check_uniform_io(plan, y, x, IP, "VkFFT.plan_fft", "VkFFT.plan_fft!")

"""
    _apply!(plan::AbstractVkFFTPlan, y::AbstractArray, x::AbstractArray, direction::Int32)

Runs the plan's VkFFT application once, reading `x` and writing `y`.

Every `mul!` funnels through here after its own `_check_io` and trivial-plan
handling. One plan is one VkFFT application, and an application is not
reentrant: it holds the buffer and stream slots the kernels read their
arguments from, plus one device-side scratch buffer, all rewritten per
dispatch. Cached plans are shared across tasks by design, so the lock is what
makes that safe.
"""
function _apply!(plan::AbstractVkFFTPlan, y::AbstractArray, x::AbstractArray, direction::Int32)
    @lock plan.lock begin
        res = GC.@preserve x y begin
            _with_execution(y) do stream
                _vkfft_execute(plan.app, _buffer_handle(x), _buffer_handle(y), direction, stream)
            end
        end
        _check(res)
    end

    return nothing
end

## API entry points

"""
    plan_fft(x, region=1:ndims(x); zeropad=nothing, tune=false)

Creates an out-of-place forward complex-to-complex VkFFT plan for `x`.

The plan is an `AbstractFFTs.Plan`, so `p * x`, `mul!(y, p, x)`, `inv(p)`,
`p \\ y`, `p'`, `size(p)` and `AbstractFFTs.fftdims(p)` all work. It is not
reachable through the global `fft`/`plan_fft`: those belong to whichever package
owns the array type.

`x` must be a contiguous device array of `ComplexF16`, `ComplexF32` or
`ComplexF64` on a backend whose extension is loaded, and it is used only for its
type, size and device. Its contents are neither read nor written to.
`ComplexF16` is a storage precision: VkFFT reads and writes 16-bit values and
computes in fp32, so the accuracy comes from the 11-bit mantissa. It needs a
device that supports half precision, which on OpenCL means one reporting
`cl_khr_fp16`. On CUDA it needs more than the card: VkFFT compiles its
half-precision kernels by including `cuda_fp16.h` from the CUDA toolkit path
baked into `libvkfft`, so a wrapper built without one refuses half precision up
front. Building the wrapper against a local CUDA toolkit bakes a real path.

`zeropad=lo:hi` declares the samples `lo:hi` of the first transformed dimension
zero. A forward plan never reads them, so they need not be zeroed: whatever is
there does not reach the result. VkFFT only skips the block correctly
on its own first axis, so the padded dimension has to be the first one of `x`
longer than 1, and any other region is refused with advice to permute. A padded
plan has neither an inverse nor an adjoint, since nothing can put back samples
that were never read.

`tune=true` picks VkFFT's `coalesced_memory` and `aim_threads` for this shape on
this device, by building a small grid of candidate plans and timing warmed-up
applications of each. The winner is stored on disk, so the same call in a later
session and in a later process is a file read. The sweep allocates buffers of
the plan's own shape to time on, and it takes as long as building and timing
sixteen plans takes, which is the price of the first call only. `tune=:force`
sweeps again and overwrites what is stored, for a machine whose driver has
changed underneath its records, and `VkFFT.clear_tuning!()` drops them all.
Tuning applies to the plan being built and not to the inverse plans `inv`
derives from it.

# Arguments
- `x`: The input array
- `region=1:ndims(x)`: The dimensions on which to perform the FFT
- `zeropad=nothing`: A 1-based inclusive range on the first transformed dimension VkFFT may treat as zero
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTPlan`
"""
plan_fft(x::AbstractArray, region=_all_dims(x); zeropad=nothing, tune=false) = _make_plan(x, _canonical_region(region, ndims(x)), FORWARD, false, Val(false), _canonical_zeropad(zeropad), tune)

"""
    plan_fft!(x, region=1:ndims(x); zeropad=nothing, tune=false)

Creates an in-place forward complex-to-complex VkFFT plan for `x`.

Like `VkFFT.plan_fft`, except that `p * x` overwrites `x` and `mul!(y, p, x)`
demands `y === x`.

Zero-padding and `tune` work as in `VkFFT.plan_fft`.

# Arguments
- `x`: The input array
- `region=1:ndims(x)`: The dimensions on which to perform the FFT
- `zeropad=nothing`: A 1-based inclusive range on the first transformed dimension VkFFT may treat as zero
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTPlan`
"""
plan_fft!(x::AbstractArray, region=_all_dims(x); zeropad=nothing, tune=false) = _make_plan(x, _canonical_region(region, ndims(x)), FORWARD, false, Val(true), _canonical_zeropad(zeropad), tune)

"""
    plan_bfft(x, region=1:ndims(x); zeropad=nothing, tune=false)

Creates an out-of-place unnormalized inverse complex-to-complex VkFFT plan for `x`.

Unnormalized: `p * (VkFFT.plan_fft(x, region) * x)` is `prod(size(x)[region])`
times `x`. Use `VkFFT.plan_ifft` for the normalized inverse, which is cheaper
than scaling afterwards.

`zeropad=lo:hi` declares the samples `lo:hi` of the first transformed dimension
zero, as in `VkFFT.plan_fft`, except that an inverse plan skips writing that
range instead of skipping reads. It is left holding whatever the destination
already held, so it is undefined in the array `p * x` allocates, and the skip is
not reliable above one dimension. Treat that range as garbage, never as zeros.
`tune` works as in `VkFFT.plan_fft`.

# Arguments
- `x`: The input array
- `region=1:ndims(x)`: The dimensions on which to perform the FFT
- `zeropad=nothing`: A 1-based inclusive range on the first transformed dimension VkFFT may treat as zero
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTPlan`
"""
plan_bfft(x::AbstractArray, region=_all_dims(x); zeropad=nothing, tune=false) = _make_plan(x, _canonical_region(region, ndims(x)), INVERSE, false, Val(false), _canonical_zeropad(zeropad), tune)

"""
    plan_bfft!(x, region=1:ndims(x); zeropad=nothing, tune=false)

Creates an in-place unnormalized inverse complex-to-complex VkFFT plan for `x`.

Zero-padding and `tune` work as in `VkFFT.plan_bfft`.

# Arguments
- `x`: The input array
- `region=1:ndims(x)`: The dimensions on which to perform the FFT
- `zeropad=nothing`: A 1-based inclusive range on the first transformed dimension VkFFT may treat as zero
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTPlan`
"""
plan_bfft!(x::AbstractArray, region=_all_dims(x); zeropad=nothing, tune=false) = _make_plan(x, _canonical_region(region, ndims(x)), INVERSE, false, Val(true), _canonical_zeropad(zeropad), tune)

"""
    plan_ifft(x, region=1:ndims(x); zeropad=nothing, tune=false)

Creates an out-of-place normalized inverse complex-to-complex VkFFT plan for `x`.

The 1/N is applied by VkFFT inside the transform kernels, so this costs one
kernel launch, not two. `N` is the product of the transformed axis lengths only:
batched and omitted dimensions never enter it.

Zero-padding and `tune` work as in `VkFFT.plan_bfft`, the padded range of an
inverse plan being skipped writes rather than skipped reads.

# Arguments
- `x`: The input array
- `region=1:ndims(x)`: The dimensions on which to perform the FFT
- `zeropad=nothing`: A 1-based inclusive range on the first transformed dimension VkFFT may treat as zero
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTPlan`
"""
plan_ifft(x::AbstractArray, region=_all_dims(x); zeropad=nothing, tune=false) = _make_plan(x, _canonical_region(region, ndims(x)), INVERSE, true, Val(false), _canonical_zeropad(zeropad), tune)

"""
    plan_ifft!(x, region=1:ndims(x); zeropad=nothing, tune=false)

Creates an in-place normalized inverse complex-to-complex VkFFT plan for `x`.

Zero-padding and `tune` work as in `VkFFT.plan_ifft`.

# Arguments
- `x`: The input array
- `region=1:ndims(x)`: The dimensions on which to perform the FFT
- `zeropad=nothing`: A 1-based inclusive range on the first transformed dimension VkFFT may treat as zero
- `tune=false`: Whether to tune VkFFT's block and thread knobs for this shape, `:force` to re-sweep

# Returns
- A `VkFFTPlan`
"""
plan_ifft!(x::AbstractArray, region=_all_dims(x); zeropad=nothing, tune=false) = _make_plan(x, _canonical_region(region, ndims(x)), INVERSE, true, Val(true), _canonical_zeropad(zeropad), tune)

## Base and AbstractFFTs methods

Base.size(plan::AbstractVkFFTPlan) = plan.sz
AbstractFFTs.fftdims(plan::AbstractVkFFTPlan) = plan.region

function LinearAlgebra.mul!(y::AbstractArray{T, N}, plan::VkFFTPlan{T, N, IP, B, M}, x::AbstractArray{T, N}) where {T <: VkFFTComplex, N, IP, B, M}
    _check_io(plan, y, x)

    if _is_trivial(plan)
        IP || copyto!(y, x)
        return y
    end

    _apply!(plan, y, x, plan.direction)
    return y
end

LinearAlgebra.mul!(y::AbstractArray, plan::VkFFTPlan{T}, x::AbstractArray) where T = throw(ArgumentError("this VkFFT plan transforms $(ndims(plan))-dimensional $T arrays. Got an output of type $(typeof(y)) and an input of type $(typeof(x))"))

Base.:*(plan::VkFFTPlan{T, N, true}, x::AbstractArray{T, N}) where {T <: VkFFTComplex, N} = mul!(x, plan, x)
Base.:*(plan::VkFFTPlan{T, N, false}, x::AbstractArray{T, N}) where {T <: VkFFTComplex, N} = mul!(similar(x), plan, x)
Base.:*(plan::VkFFTPlan{T}, x::AbstractArray) where T = throw(ArgumentError("this VkFFT plan transforms $(ndims(plan))-dimensional $T arrays. Got an input of type $(typeof(x))"))

# A ScaledPlan's scale reaches the device as the scalar of an rmul!, so it has
# to be a type the device has arithmetic for. AbstractFFTs keeps whatever the
# caller wrote, and inv of a ScaledPlan takes the reciprocal of it, so an Int
# scale of 3 (which applies fine anywhere) inverts to a Float64 1/3 that a
# device without double cannot compile a kernel for. Narrowing the scale to the
# plan's own real precision here puts the ScaledPlan, its inverse and its
# adjoint all in a type every device that can run the plan can also scale in.
# The complex case is kept complex, since a complex scale on a complex plan is a
# phase and refusing it would be inventing a restriction AbstractFFTs does not
# have. On a plan with real output it still fails at application time, as it
# does today.
Base.:*(alpha::Real, plan::AbstractVkFFTPlan{T}) where T <: VkFFTNumber = AbstractFFTs.ScaledPlan(plan, convert(real(T), alpha))
Base.:*(alpha::Complex, plan::AbstractVkFFTPlan{T}) where T <: VkFFTNumber = AbstractFFTs.ScaledPlan(plan, convert(Complex{real(T)}, alpha))
Base.:*(plan::AbstractVkFFTPlan{T}, alpha::Number) where T <: VkFFTNumber = alpha * plan

AbstractFFTs.plan_inv(plan::AbstractVkFFTPlan) = _wrap_inv(plan, _raw_inv(plan))

function Base.inv(plan::AbstractVkFFTPlan)
    raw = plan.pinv
    if raw === nothing
        raw = _raw_inv(plan)
        plan.pinv = raw
    end
    return _wrap_inv(plan, raw)
end

"""
    VkFFTNormalizedAdjointStyle()

Adjoint style for VkFFT's in-kernel normalized inverse plans.

`AbstractFFTs.FFTAdjointStyle` assumes the plan is unnormalized, so it recovers
the adjoint as `N * inv(p)`. A plan that already carries a 1/N is off by `N^2`
under that rule. This style applies the 1/N once instead.
"""
struct VkFFTNormalizedAdjointStyle <: AbstractFFTs.AdjointStyle end

AbstractFFTs.AdjointStyle(plan::VkFFTPlan) = plan.normalize ? VkFFTNormalizedAdjointStyle() : AbstractFFTs.FFTAdjointStyle()
AbstractFFTs.output_size(plan::VkFFTPlan, ::VkFFTNormalizedAdjointStyle) = size(plan)
function AbstractFFTs.adjoint_mul(plan::VkFFTPlan{T}, x::AbstractArray, ::VkFFTNormalizedAdjointStyle) where T
    _check_adjointable(plan)
    return _normalization(T, plan.sz, plan.region) * (inv(plan) * x)
end

# Same as AbstractFFTs' generic FFTAdjointStyle method, except that
# _normalization survives an empty region, which is a legal region for a VkFFT
# plan.
function AbstractFFTs.adjoint_mul(plan::VkFFTPlan{T}, x::AbstractArray, ::AbstractFFTs.FFTAdjointStyle) where T
    _check_adjointable(plan)
    return (plan \ x) / _normalization(T, plan.sz, plan.region)
end

function Base.show(io::IO, plan::VkFFTPlan{T, N, IP, B, M}) where {T, N, IP, B, M}
    place = IP ? "in-place" : "out-of-place"
    direction = plan.direction == FORWARD ? "forward" : (plan.normalize ? "normalized inverse" : "unnormalized inverse")
    shape = join(plan.sz, '×')
    dims = "dim" * (M == 1 ? " " : "s ") * string(plan.region)
    padding = _zeropad_string(plan.zeropad)

    # ex: 2D out-of-place complex forward VkFFT plan for 64×64 ComplexF32 on
    # dims (1, 2) [opencl]
    if _is_trivial(plan) && !plan.destroyed
        print(io, "trivial $place complex $direction VkFFT plan for $shape $T on $dims$padding [$B]")
    else
        print(io, "$(length(plan.region))D $place complex $direction VkFFT plan for $shape $T on $dims$padding [$B]")
    end
end
