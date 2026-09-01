# Fused convolution. VkFFT runs the forward transform, the elementwise multiply
# against a pre-transformed kernel and the inverse transform in one application,
# which needs two apps: one that transforms the kernel and one that is pointed
# at its output. The kernel app is used once, at plan time. The convolution app
# lives as long as the plan, and so does the device buffer holding the
# transformed kernel, because VkFFT reads it at every dispatch and copies
# nothing.

## Plan type

# The convolution family stops at fp32. VkFFT builds a half-precision
# convolution plan, reports VKFFT_SUCCESS at every step and computes something
# that is no convolution, an upstream VkFFT bug: measured on Metal against a
# host circular convolution, a 16-point transform comes back O(1) away from the
# answer and a 64-point one comes back holding infinities, where the same
# shapes in fp32 are correct to 2e-7.
const VkFFTConvComplex = Union{ComplexF32, ComplexF64}

"""
    VkFFTConvPlan{T, N, B, M}

A first-class `AbstractFFTs.Plan` for VkFFT's fused circular convolution.

The type parameters are (T) the element type, (N) `ndims` of the arrays, (B) the
backend tag (`:cuda`, `:opencl` or `:metal`) and (M) the number of transformed
dimensions. Input and output have the same size and element type, so `p * x`
gives an array shaped like `x`. What `p * x` computes, what regions and kernels
a plan takes and why it has neither an inverse nor an adjoint is
`VkFFT.plan_conv`'s docstring.

Plans are not cached. A convolution app carries the kernel it was pointed at, so
two plans of the same shape and different kernels must never share one, and the
kernel lives on the device where a cache key cannot cheaply reach it.

# Fields
- `app::Ptr{Cvoid}`: The equivalent of an FFTW plan but for VkFFT, running the whole fft/multiply/inverse pipeline
- `sz::NTuple{N, Int}`: The size of both arrays
- `region::NTuple{M, Int}`: The dimensions on which to perform the convolution, sorted
- `features::Int`: The trailing component stack, VkFFT's `coordinateFeatures`
- `correlate::Bool`: Whether the transformed input is conjugated, giving a cross-correlation
- `kernel::Vector{Any}`: The one device array holding the transformed kernel, which VkFFT reads at every dispatch
- `device_id::UInt64`: Identity of the device context the plan was built for
- `roots::Vector{Any}`: Backend objects (context, queue, ...) the app outlives nothing without
- `lock::ReentrantLock`: Held for the length of one application, because a VkFFT app is not reentrant
- `destroyed::Bool`: Set by the finalizer so a double destroy is a no-op
"""
mutable struct VkFFTConvPlan{T <: VkFFTConvComplex, N, B, M} <: AbstractVkFFTPlan{T}
    app::Ptr{Cvoid}
    sz::NTuple{N, Int}
    region::NTuple{M, Int}
    features::Int
    correlate::Bool
    kernel::Vector{Any} # concrete field type, holding one device array nothing reads again
    device_id::UInt64
    roots::Vector{Any} # concrete field type (only ever read by the finalizer, never in mul!)
    lock::ReentrantLock
    destroyed::Bool

    function VkFFTConvPlan{T, N, B, M}(app::Ptr{Cvoid}, sz::NTuple{N, Int}, region::NTuple{M, Int}, features::Int, correlate::Bool, kernel::Vector{Any}, device_id::UInt64, roots::Vector{Any}) where {T, N, B, M}
        plan = new{T, N, B, M}(app, sz, region, features, correlate, kernel, device_id, roots, ReentrantLock(), false)
        app == C_NULL || finalizer(unsafe_free!, plan)
        return plan
    end
end

_plan_backend(::VkFFTConvPlan{T, N, B}) where {T, N, B} = Val(B)

## Errors

"""
    _unsupported_conv_eltype(::Type{T})

Throws the error for an element type `VkFFT.plan_conv` cannot plan for.
"""
_unsupported_conv_eltype(::Type{T}) where T <: VkFFTReal = throw(ArgumentError("VkFFT.plan_conv convolves complex arrays. Got $T. VkFFT's real-input convolution reads a kernel and a signal in the padded in-place real-to-complex layout, which this package does not build yet. Convert to ComplexF32 or ComplexF64 first."))
_unsupported_conv_eltype(::Type{ComplexF16}) = throw(ArgumentError("VkFFT cannot convolve in half precision. It builds the plan, reports success at every step and computes something that is no convolution, an upstream VkFFT bug: measured against a host circular convolution, 16 points come back O(1) away from the answer and 64 points come back holding infinities, where the same shapes in ComplexF32 are correct to 2e-7. Convert to ComplexF32 first. Half precision does work for VkFFT.plan_fft, VkFFT.plan_rfft and the real-to-real family."))
_unsupported_conv_eltype(::Type{T}) where T = throw(ArgumentError("VkFFT convolution plans support ComplexF32 and ComplexF64. Got $T. $QUAD_HINT"))

"""
    _check_conv_region(dims::NTuple{N, Int}, region::NTuple{M, Int}, layout::VkFFTLayout)

Throws unless VkFFT can convolve over this region of this array shape.

Three shapes are refused. A region that transforms nothing has no meaning here:
a convolution is against a kernel, so there is no identity to fall back on the
way `VkFFT.plan_fft` has one. A collapsed layout that omits an axis, which is
what an untransformed dimension between or before two transformed ones becomes,
does not work with a convolution: VkFFT rejects a leading or trailing omit as a
documented limit, and accepts a middle one while computing something that is no
convolution, an upstream bug, so both are refused here. And a layout with three
or more transformed axes computes something that is no convolution either, an
upstream bug measured on two backends against a kernel whose spectrum is all
ones, where the pipeline has to return its input and returns something O(1)
away from it instead.
"""
function _check_conv_region(dims::NTuple{N, Int}, region::NTuple{M, Int}, layout::VkFFTLayout) where {N, M}
    _is_trivial(layout) && throw(ArgumentError("a VkFFT convolution needs at least one dimension to transform, and region $region of this $(join(dims, '×')) array leaves none. A convolution against a kernel has no meaning along no axis at all, so this is refused rather than treated as a copy."))

    for i in 1:layout.fft_dim
        layout.omit[i] == 0 || throw(ArgumentError("VkFFT cannot convolve over region $region of a $(join(dims, '×')) array: the dimensions to transform have $(sum(layout.omit[j] != 0 for j in 1:layout.fft_dim)) untransformed one(s) before or between them, and a convolution needs the transformed dimensions to run from the first dimension longer than 1 up to a trailing batch of them. VkFFT rejects most such layouts and accepts a middle omit while computing something that is no convolution, an upstream bug. Reshape or permute the array so the dimensions to convolve come first."))
    end

    # TODO: lift once the three-axis pipeline is fixed upstream. The transformed
    # kernel is correct at any number of axes, so this is an upstream bug in the
    # convolution plan's own forward/multiply/inverse sequence and not a layout
    # misunderstanding.
    layout.fft_dim <= 2 || throw(ArgumentError("a VkFFT convolution runs over at most two transformed dimensions, and region $region of this $(join(dims, '×')) array asks for $(layout.fft_dim). VkFFT builds the plan and computes something that is no convolution, an upstream bug: with a kernel whose spectrum is all ones, where the pipeline has to return its input, it comes back O(1) away from it. Convolve over two dimensions and treat the rest as a trailing stack, or build the transform out of VkFFT.plan_fft and VkFFT.plan_ifft with the multiply of your own."))

    return nothing
end

## Planning

"""
    _conv_layout(layout::VkFFTLayout)

Returns the layout a convolution app is built from, and the component stack it carries.

Two things change, both working around upstream VkFFT bugs. A one-axis layout
is rewritten as two axes with a trailing length of 1, which is the same
transform: VkFFT's convolution codegen opens a bounds check per register and
never closes it for a single-axis transform that fits in one upload, and the
trailing axis moves the multiply onto the strided path, which compiles. Lengths
large enough to need several uploads take that path anyway, and the rewrite is
applied at every length rather than only where it is needed, since the two are
the same transform.

And the outermost batch count becomes `coordinate_features`. The two describe
the same bytes, one contiguous copy of the layout per component, but VkFFT's
convolution step offsets the kernel by the component index and not by the batch
index, so a batched convolution plan reads the first batch's kernel for every
batch and writes garbage for the rest.
"""
function _conv_layout(layout::VkFFTLayout)
    layout.fft_dim == 1 || return (VkFFTLayout(layout.fft_dim, layout.size, layout.omit, 1), layout.number_batches)

    sizes = ntuple(i -> i == 1 ? layout.size[1] : (i == 2 ? UInt64(1) : UInt64(0)), VKFFT_MAX_FFT_DIMENSIONS)
    return (VkFFTLayout(2, sizes, layout.omit, 1), layout.number_batches)
end

"""
    _make_conv_plan(x, kernel, region::NTuple{M, Int}, correlate::Bool, zeropad::NTuple{2, Int})

Creates the convolution plan for an array, a kernel and a canonical region.

Every guard runs before the C library is reached, and the two apps are built in
the order that leaves nothing leaked: the kernel app is created, used and freed
here, and the convolution app is handed to the plan, whose finalizer owns it
from then on.
"""
function _make_conv_plan(x::AbstractArray{T, N}, kernel::AbstractArray, region::NTuple{M, Int}, correlate::Bool, zeropad::NTuple{2, Int}) where {T <: VkFFTConvComplex, N, M}
    zeropad == NO_ZEROPAD || throw(ArgumentError("VkFFT.plan_conv does not take zeropad yet. Zero-padding a convolution has never been measured against a reference here, and VkFFT's padded blocks interact with the multiply in the middle of the pipeline in a way nothing in this package pins down. Zero the samples yourself and plan without it."))

    sz = size(x)
    layout = _map_region(sz, region, _max_dims())
    _check_conv_region(sz, region, layout)

    backend = _backend(typeof(x))
    _check_layout(x)

    # The kernel is read through the same layout as the signal, so it has to
    # have the same element type, the same size and the same device. Nothing
    # about it is checked by VkFFT, which reads whatever the buffer holds for
    # as many elements as the layout says.
    eltype(kernel) === T || throw(ArgumentError("a VkFFT convolution reads its kernel through the same transform as its input, so both need the same element type. Got an input of $T and a kernel of $(eltype(kernel))."))
    size(kernel) == sz || throw(ArgumentError("a VkFFT convolution kernel has to have the same size as the input, one kernel per component of a trailing stack. Got an input of size $sz and a kernel of size $(size(kernel)). Broadcast the kernel over the trailing dimensions yourself if one kernel is meant for all of them."))
    _backend(typeof(kernel)) === backend || throw(ArgumentError("a VkFFT convolution kernel has to live on the same backend as its input. Got an input of type $(typeof(x)) and a kernel of type $(typeof(kernel))."))
    _check_layout(kernel)
    _device_id(kernel) == _device_id(x) || throw(ArgumentError("a VkFFT convolution kernel has to live in the same device context as its input. Copy one of them across first."))

    conv, features = _conv_layout(layout)
    roots = _device_roots(x)

    # The kernel app is a plain forward transform with kernel_convolution set,
    # which is what makes VkFFT skip the internal reorderings whose output the
    # convolution step would not recognize. It transforms the kernel once and
    # is freed, so the plan carries only the buffer.
    kernel_app = _create_app(T, conv, FORWARD, false, false, false, backend, roots; coordinate_features=features, kernel_convolution=true)
    transformed = try
        buffer = similar(kernel)
        _check_layout(buffer)

        res = GC.@preserve kernel buffer begin
            _with_execution(buffer) do stream
                _vkfft_execute(kernel_app, _buffer_handle(kernel), _buffer_handle(buffer), FORWARD, stream)
            end
        end
        _check(res)

        buffer
    finally
        _with_plan_context(backend, roots) do
            _vkfft_destroy(kernel_app)
        end
    end

    return _create_conv_plan(T, sz, region, features, correlate, conv, transformed, backend, _device_id(x), roots)
end

_make_conv_plan(x::AbstractArray{T, N}, kernel::AbstractArray, region::NTuple{M, Int}, correlate::Bool, zeropad::NTuple{2, Int}) where {T, N, M} = _unsupported_conv_eltype(T)

"""
    _create_conv_plan(::Type{T}, sz::NTuple{N, Int}, region::NTuple{M, Int}, features::Int, correlate::Bool, layout::VkFFTLayout, transformed, backend::Val{B}, device_id::UInt64, roots::Vector{Any})

Creates the convolution app and the plan that owns it, and points the app at the transformed kernel.

There is no cache lookup here, unlike every other plan family. A convolution
app's identity includes the kernel buffer set on it, so two plans of the same
shape and different kernels cannot share one, and re-pointing a shared app per
application would race between the two calls with nothing holding a lock across
them. Convolution kernels are few in the programs this is for, so each plan owns
its app outright.
"""
function _create_conv_plan(::Type{T}, sz::NTuple{N, Int}, region::NTuple{M, Int}, features::Int, correlate::Bool, layout::VkFFTLayout, transformed::AbstractArray, backend::Val{B}, device_id::UInt64, roots::Vector{Any}) where {T <: VkFFTConvComplex, N, M, B}
    # normalize is set on every convolution plan. VkFFT's inverse half applies
    # 1/N over the last transformed axis whatever normalize says, and normalize
    # is what covers the others, so without it a multi-axis result comes back
    # scaled by the product of the remaining axis lengths.
    app = _create_app(T, layout, FORWARD, true, false, false, backend, roots; coordinate_features=features, perform_convolution=true, conjugate_convolution=correlate)

    plan = VkFFTConvPlan{T, N, B, M}(app, sz, region, features, correlate, Any[transformed], device_id, roots)

    res = GC.@preserve transformed _vkfft_set_kernel(app, _buffer_handle(transformed))
    _check(res)

    return plan
end

"""
    _check_io(plan::VkFFTConvPlan, y, x)

Rejects output/input arrays that do not match what the convolution plan was built for.
"""
function _check_io(plan::VkFFTConvPlan, y::AbstractArray, x::AbstractArray)
    if !plan.destroyed && y === x
        throw(ArgumentError("a VkFFT convolution plan writes its result to an array of its own. Got one array as both output and input, and there is no in-place convolution plan."))
    end

    return _check_uniform_io(plan, y, x, false, "VkFFT.plan_conv", "VkFFT.plan_conv")
end

## API entry points

"""
    plan_conv(x, kernel, region=1:ndims(x); correlate=false, zeropad=nothing)

Creates a fused circular convolution VkFFT plan for `x` against `kernel`.

The plan is an `AbstractFFTs.Plan`, so `p * x`, `mul!(y, p, x)`, `size(p)` and
`AbstractFFTs.fftdims(p)` work. Output and input have the same size and element
type. VkFFT runs the forward transform, the multiply against the transformed
kernel and the inverse transform as one application, and applies the whole 1/N
itself, so `p * x` is the plain circular convolution at a scale of exactly one.
Along one transformed axis of length `n`,
`(p * x)[m] == sum(x[j] * kernel[mod(m - j, n) + 1] for j in 1:n)`, and the
transform is separable over the transformed axes.

`correlate=true` conjugates the transformed input instead, which makes the
result the circular cross-correlation
`(p * x)[m] == sum(conj(x[j]) * kernel[mod(j + m - 2, n) + 1] for j in 1:n)`.
Conjugating the kernel rather than the input is a documented VkFFT option that no
version vendored here generates code for, so it is not offered.

`kernel` is read once, here. The plan transforms it into a device buffer of its
own and holds on to that buffer for as long as the plan lives, because VkFFT
reads it at every application and copies nothing. Writing to `kernel`
afterwards, or freeing it, does not affect the plan.

`x` and `kernel` must be contiguous device arrays of `ComplexF32` or
`ComplexF64` of the same size, on the same device, on a backend whose extension
is loaded. `x` is used only for its type, size and device. `ComplexF16` is
refused here even though the other families accept it: VkFFT builds a
half-precision convolution plan, reports success and returns something that is
no convolution.

The transformed dimensions have to run from the first dimension of `x` longer
than 1 up to a trailing run of untransformed ones, since VkFFT cannot convolve
over a layout that omits an axis, and there can be at most two of them, since
VkFFT computes something that is no convolution over three. Any other region is
refused rather than passed on. That trailing run is a stack of independent
components, each convolved against its own slice of the kernel, so one kernel
meant for all of them has to be broadcast into `kernel` by the caller.

One family of transformed lengths comes back wrong and cannot be refused from
here. VkFFT reaches for Bluestein's algorithm whenever an axis length has a
prime factor larger than 13, and a Bluestein axis long enough that VkFFT splits
it over several uploads computes something that is no convolution, an upstream
VkFFT bug: measured on pocl in Float64, lengths 4093, 8191 and 16381 return a
result with no
resemblance to the answer and no error code, while every length whose prime
factors are all at most 13 is correct up to 16384 and beyond, and shorter
Bluestein lengths such as 1021 and 727 are correct too. The length at which
VkFFT starts splitting depends on the device's shared memory, so the planner has
nothing to test against, which is why this is a warning and not a refusal. Check
one convolution against a host reference before trusting a transformed length
with a large prime factor.

There is no inverse and no adjoint. A convolution against a kernel with a zero
in its spectrum loses information, so it has no inverse in general, and the
adjoint would have to be written around the normalization VkFFT applies inside
the pipeline. Both refuse rather than return something plausible.

Convolution plans are not cached, so two calls with the same shapes give two
plans with two applications. A convolution app carries the kernel it was pointed
at, which is what makes it unshareable.

There is no `tune`. A convolution plan is a forward transform, a multiply and an
inverse in one app, and the two knobs the tuner sweeps set the block and thread
shape of the transforms, so tuning the plain `VkFFT.plan_fft` of the same axes
answers the same question at a third of the sweep. Tuning the pipeline directly
is worth adding once there is a measurement saying the multiply moves the
optimum.

# Arguments
- `x`: The prototype input array
- `kernel`: The convolution kernel, the same size as `x`, read at plan time
- `region=1:ndims(x)`: The dimensions on which to perform the convolution
- `correlate=false`: Whether to compute the circular cross-correlation instead
- `zeropad=nothing`: Not supported yet, and refused with an error rather than ignored

# Returns
- A `VkFFTConvPlan`
"""
plan_conv(x::AbstractArray, kernel::AbstractArray, region=_all_dims(x); correlate::Bool=false, zeropad=nothing) = _make_conv_plan(x, kernel, _canonical_region(region, ndims(x)), correlate, _canonical_zeropad(zeropad))

## Base and AbstractFFTs methods

function LinearAlgebra.mul!(y::AbstractArray{T, N}, plan::VkFFTConvPlan{T, N, B, M}, x::AbstractArray{T, N}) where {T <: VkFFTConvComplex, N, B, M}
    _check_io(plan, y, x)

    _apply!(plan, y, x, FORWARD) # a convolution pipeline only ever runs forward
    return y
end

LinearAlgebra.mul!(y::AbstractArray, plan::VkFFTConvPlan{T}, x::AbstractArray) where T = throw(ArgumentError("this VkFFT convolution plan transforms $(ndims(plan))-dimensional $T arrays. Got an output of type $(typeof(y)) and an input of type $(typeof(x))"))

Base.:*(plan::VkFFTConvPlan{T, N}, x::AbstractArray{T, N}) where {T <: VkFFTConvComplex, N} = mul!(similar(x), plan, x)
Base.:*(plan::VkFFTConvPlan{T}, x::AbstractArray) where T = throw(ArgumentError("this VkFFT convolution plan transforms $(ndims(plan))-dimensional $T arrays. Got an input of type $(typeof(x))"))

# AbstractFFTs derives output_size from the plan's adjoint style, and a
# convolution plan has none. Both sides have the same shape anyway.
AbstractFFTs.output_size(plan::VkFFTConvPlan) = size(plan)

# A fused convolution is not invertible in general, and the adjoint algebra
# around the 1/N VkFFT applies in the middle of the pipeline is easy to get
# silently wrong, so all three entry points refuse rather than guess.
_raw_inv(plan::VkFFTConvPlan) = throw(ArgumentError("a VkFFT convolution plan has no inverse. Deconvolution divides by the kernel's spectrum, which is not a transform this package plans and which does not exist at all where that spectrum has a zero. Build the plans you need out of VkFFT.plan_fft and VkFFT.plan_ifft."))
Base.inv(plan::VkFFTConvPlan) = _raw_inv(plan)
Base.adjoint(plan::VkFFTConvPlan) = throw(ArgumentError("the adjoint of a VkFFT convolution plan is not available. It is a correlation against the same kernel, but the normalization VkFFT applies inside the fused pipeline makes the scale factor easy to get wrong with nothing reporting it, so it is refused rather than guessed at. Build it out of VkFFT.plan_fft and VkFFT.plan_ifft."))

function Base.show(io::IO, plan::VkFFTConvPlan{T, N, B, M}) where {T, N, B, M}
    kind = plan.correlate ? "circular cross-correlation" : "circular convolution"
    shape = join(plan.sz, '×')
    dims = "dim" * (M == 1 ? " " : "s ") * string(plan.region)
    stack = plan.features == 1 ? "" : " ×$(plan.features) features"

    # ex: 2D out-of-place circular convolution VkFFT plan for 16×16 ComplexF32
    # on dims (1, 2) [opencl]
    print(io, "$(M)D out-of-place $kind VkFFT plan for $shape $T on $dims$stack [$B]")
end
