# The escape hatch. Every other file in this package exists to keep VkFFT's
# silent failure modes out of reach. This one hands a caller-supplied
# vkfft_config straight to the wrapper and keeps only the checks that are about
# Julia arrays rather than about the transform.

## Plan type

"""
    VkFFTUnsafePlan{T, S, N, M, B}

A first-class `AbstractFFTs.Plan` built from a raw `VkFFTConfig`.

The type parameters are (T) the input element type, (S) the output element type,
(N) `ndims` of the input, (M) `ndims` of the output and (B) the backend tag
(`:cuda`, `:opencl` or `:metal`). All five come from the prototype arrays
`VkFFT.unsafe_plan` was given, since a `vkfft_config` describes a byte layout
and knows nothing about Julia shapes.

None of the guards the other plan families run were applied to the
configuration behind this plan. `VkFFT.unsafe_plan` documents what that means.

Plans of this kind are never cached: two calls with equal configurations build
two VkFFT applications.

# Fields
- `app::Ptr{Cvoid}`: The equivalent of an FFTW plan but for VkFFT
- `sz::NTuple{N, Int}`: The size of the input array
- `osz::NTuple{M, Int}`: The size of the output array
- `direction::Int32`: -1 for forward, 1 for inverse
- `inplace::Bool`: Whether the configuration said one buffer, in which case an application needs `y === x`
- `device_id::UInt64`: Identity of the device context the plan was built for
- `roots::Vector{Any}`: Backend objects (context, queue, ...) the app outlives nothing without
- `lock::ReentrantLock`: Held for the length of one application, because a VkFFT app is not reentrant
- `destroyed::Bool`: Set by the finalizer so a double destroy is a no-op
"""
mutable struct VkFFTUnsafePlan{T, S, N, M, B} <: AbstractVkFFTPlan{T}
    app::Ptr{Cvoid}
    sz::NTuple{N, Int}
    osz::NTuple{M, Int}
    direction::Int32
    inplace::Bool
    device_id::UInt64
    roots::Vector{Any} # concrete field type (only ever read by the finalizer, never in mul!)
    lock::ReentrantLock
    destroyed::Bool

    function VkFFTUnsafePlan{T, S, N, M, B}(app::Ptr{Cvoid}, sz::NTuple{N, Int}, osz::NTuple{M, Int}, direction::Int32, inplace::Bool, device_id::UInt64, roots::Vector{Any}) where {T, S, N, M, B}
        plan = new{T, S, N, M, B}(app, sz, osz, direction, inplace, device_id, roots, ReentrantLock(), false)
        app == C_NULL || finalizer(unsafe_free!, plan)
        return plan
    end
end

_plan_backend(::VkFFTUnsafePlan{T, S, N, M, B}) where {T, S, N, M, B} = Val(B)

## Planning

"""
    _create_unsafe_plan(config::VkFFTConfig, in_prototype, out_prototype, direction::Int32, backend::Val{B})

Creates the VkFFT application for a raw configuration and wraps it in a plan.

Split out of `VkFFT.unsafe_plan` so that the element types, the two dimension
counts and the backend tag all reach the plan's type from the signature rather
than from run-time values.
"""
function _create_unsafe_plan(config::VkFFTConfig, in_prototype::AbstractArray{T, N}, out_prototype::AbstractArray{S, M}, direction::Int32, backend::Val{B}) where {T, N, S, M, B}
    roots = _device_roots(in_prototype)

    return VkFFTUnsafePlan{T, S, N, M, B}(_app_from_config(config, backend, roots), size(in_prototype), size(out_prototype), direction, config.inplace != 0, _device_id(in_prototype), roots)
end

"""
    unsafe_plan(config::VkFFTConfig, in_prototype, out_prototype; direction=-1)

Creates a plan from a `VkFFTConfig` verbatim, with none of the planner's guards in the way.

The two prototype arrays supply what a `vkfft_config` cannot: the backend, the
device context, the element types and the Julia array shapes an application is
checked against. Their contents are neither read nor written, and a
configuration with `inplace = 1` takes one array as both prototypes. Everything
about the transform itself comes from `config` as written, which is the point of
this entry point and the reason for its name.

What `unsafe_` means here is specific. The other entry points refuse a list of
configurations VkFFT accepts and then does something other than what was asked,
and not one of those refusals runs on this path:

- A configuration with `fft_dim = 0`, or one whose every axis is omitted, builds
  and reports success and dispatches no kernel, so an out-of-place caller is
  left holding whatever the output buffer already held.
- A type-1 DCT or DST of an axis of length 1 sets that axis' internal transform
  length to zero and VkFFT's radix decomposition never terminates. Nothing
  reports anything: the call into `vkfft_create` does not return.
- A `dct` and a `dst` set at once computes the DST, or at other type pairs
  something that is neither a DCT nor a DST while still round-tripping. An
  `r2c` plan carrying either corrupts the spectrum. A plan setting both
  `perform_convolution` and `kernel_convolution` quietly ignores the second. A
  `perform_convolution` plan built with `make_forward_only` or
  `make_inverse_only` leaves VkFFT dereferencing the half it was told to skip.
- A `dct` or `dst` outside 1 to 4 falls through every branch that generates the
  pre- and post-processing an r2r transform needs, leaving a plain real
  transform behind.
- `perform_zeropad` is only skipped correctly on VkFFT's own first axis. On any
  other axis the plan runs and reads uninitialized memory.
- Half precision on a device without it comes back as a failed compile rather
  than as anything a message can explain, and so does half precision on CUDA
  through a wrapper with no baked toolkit path. A half-precision convolution
  reports success and returns something that is no convolution.

The damage is not confined to the result. The transform VkFFT generates for a
configuration like these is not the transform the buffers were sized for, so
applying one writes past the end of them and corrupts the process heap. The
crash then lands wherever that heap is next touched, which has been inside
pocl's program teardown and inside Julia's own compiler. Work out the byte layout
your configuration implies before applying it.

Precision 3, VkFFT's double-double quad, is reachable only from here. No Julia
element type has its four-native-floats-per-value layout, so the prototypes
stand in for it by byte count and the packing and unpacking are the caller's.

The plan applies with `mul!(y, p, x)` and `p * x`, and it is freed by its
finalizer or by `VkFFT.unsafe_free!` like every other plan. It has no inverse,
no adjoint and no `AbstractFFTs.fftdims`: nothing here knows which Julia
dimensions `config.size` was built from. `size(p)` is the input prototype's size
and `AbstractFFTs.output_size(p)` the output prototype's.

# Arguments
- `config::VkFFTConfig`: The configuration, forwarded to the wrapper unchanged
- `in_prototype`: A device array carrying the input's element type, size and device
- `out_prototype`: A device array carrying the output's element type and size, on the same device
- `direction=-1`: Which direction to run, in VkFFTAppend's encoding of -1 for forward and 1 for inverse

# Returns
- A `VkFFTUnsafePlan`
"""
function unsafe_plan(config::VkFFTConfig, in_prototype::AbstractArray, out_prototype::AbstractArray; direction::Integer=-1) # -1 for forward, 1 for inverse
    dir = Int32(direction)
    dir == FORWARD || dir == INVERSE || throw(ArgumentError("direction is VkFFTAppend's own encoding: -1 for forward, 1 for inverse. Got $direction."))

    backend = _backend(typeof(in_prototype))
    backend === _backend(typeof(out_prototype)) || throw(ArgumentError("both prototypes of a VkFFT.unsafe_plan have to belong to one backend. Got an input of type $(typeof(in_prototype)) and an output of type $(typeof(out_prototype))."))

    _check_layout(in_prototype)
    in_prototype === out_prototype || _check_layout(out_prototype)
    _device_id(in_prototype) == _device_id(out_prototype) || throw(ArgumentError("both prototypes of a VkFFT.unsafe_plan have to live in one device context. Copy one of them across first."))

    if config.inplace != 0
        eltype(in_prototype) === eltype(out_prototype) && size(in_prototype) == size(out_prototype) || throw(ArgumentError("a config with inplace = 1 transforms one buffer, so both prototypes have to have the same element type and size. Got $(eltype(in_prototype)) of size $(size(in_prototype)) and $(eltype(out_prototype)) of size $(size(out_prototype)). Pass the same array twice."))
    end

    _ensure_library!()

    return _create_unsafe_plan(config, in_prototype, out_prototype, dir, backend)
end

"""
    _check_io(plan::VkFFTUnsafePlan, y, x)

Rejects output/input arrays that do not match the prototypes the plan was built from.

This is the whole of what an unsafe plan checks per application: the two shapes,
the in-place agreement the configuration asked for, the memory layout of both
arrays and the device context they live in. Nothing here says anything about the
transform, which is the caller's to get right.
"""
function _check_io(plan::VkFFTUnsafePlan, y::AbstractArray, x::AbstractArray)
    plan.destroyed && throw(ArgumentError("this VkFFT plan has already been freed by VkFFT.unsafe_free!"))
    size(x) == plan.sz || throw(ArgumentError("this VkFFT plan was made for an input of size $(plan.sz). Got an input of size $(size(x))"))
    size(y) == plan.osz || throw(ArgumentError("this VkFFT plan writes an output of size $(plan.osz). Got an output of size $(size(y))"))

    if plan.inplace
        y === x || throw(ArgumentError("this VkFFT plan was built from a config with inplace = 1, so it transforms one array. Pass it as both output and input, or build the plan from a config with inplace = 0"))
    else
        y === x && throw(ArgumentError("this VkFFT plan was built from a config with inplace = 0, so it needs distinct output and input arrays. Build the plan from a config with inplace = 1 for an in-place transform"))
    end

    _check_layout(x)
    y === x || _check_layout(y)

    _device_id(x) == plan.device_id || throw(ArgumentError("this VkFFT plan was built in a different device context than the input array's. A VkFFT application belongs to the context it was built in, so plan again in this one."))
    y === x || _device_id(y) == plan.device_id || throw(ArgumentError("this VkFFT plan was built in a different device context than the output array's. A VkFFT application belongs to the context it was built in, so plan again in this one."))

    return nothing
end

## Base and AbstractFFTs methods

function LinearAlgebra.mul!(y::AbstractArray{S, M}, plan::VkFFTUnsafePlan{T, S, N, M, B}, x::AbstractArray{T, N}) where {T, S, N, M, B}
    _check_io(plan, y, x)

    _apply!(plan, y, x, plan.direction)
    return y
end

LinearAlgebra.mul!(y::AbstractArray, plan::VkFFTUnsafePlan{T, S, N, M}, x::AbstractArray) where {T, S, N, M} = throw(ArgumentError("this VkFFT plan transforms $N-dimensional $T arrays into $M-dimensional $S ones. Got an output of type $(typeof(y)) and an input of type $(typeof(x))"))

function Base.:*(plan::VkFFTUnsafePlan{T, S, N, M}, x::AbstractArray{T, N}) where {T, S, N, M}
    plan.inplace && return mul!(x, plan, x)
    return mul!(similar(x, S, plan.osz), plan, x)
end

Base.:*(plan::VkFFTUnsafePlan{T, S, N, M}, x::AbstractArray) where {T, S, N, M} = throw(ArgumentError("this VkFFT plan transforms $N-dimensional $T arrays into $M-dimensional $S ones. Got an input of type $(typeof(x))"))

AbstractFFTs.output_size(plan::VkFFTUnsafePlan) = plan.osz

# A raw config carries VkFFT axes, not Julia dimensions, and the mapping from
# one to the other is exactly what this entry point skipped. There is nothing
# truthful to return.
AbstractFFTs.fftdims(plan::VkFFTUnsafePlan) = throw(ArgumentError("a VkFFT.unsafe_plan does not know which Julia dimensions its config's axes came from, so it has no fftdims. Read config.size and config.omit, which you wrote."))

# Inverting one of these would mean deriving a second config from the first, and
# every field that would have to change (direction, normalize, make_*_only, r2c,
# the dct/dst type) is one the caller set on purpose. Building the inverse
# config is theirs to do.
_raw_inv(plan::VkFFTUnsafePlan) = throw(ArgumentError("a VkFFT.unsafe_plan has no inverse. Build the config that inverts yours (the other direction, normalize, and for a real or real-to-real transform the partner type) and call VkFFT.unsafe_plan again."))
Base.inv(plan::VkFFTUnsafePlan) = _raw_inv(plan)
Base.adjoint(plan::VkFFTUnsafePlan) = throw(ArgumentError("a VkFFT.unsafe_plan has no adjoint. The adjoint of a transform depends on which transform it is, and this plan's config was not interpreted. Use the typed planners, whose adjoints are worked out, or build the adjoint out of plans of your own."))

function Base.show(io::IO, plan::VkFFTUnsafePlan{T, S, N, M, B}) where {T, S, N, M, B}
    place = plan.inplace ? "in-place" : "out-of-place"
    direction = plan.direction == FORWARD ? "forward" : "inverse"
    shapes = "$(join(plan.sz, '×')) $T to $(join(plan.osz, '×')) $S"

    # ex: unsafe out-of-place forward VkFFT plan for 16×8 ComplexF32 to 16×8
    # ComplexF32 [metal]
    print(io, "unsafe $place $direction VkFFT plan for $shapes [$B]")
end
