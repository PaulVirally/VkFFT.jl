# Everything the common testsets share. A runner includes this after it has
# defined `BACKEND`, `DeviceArray` and `_upload`, which are what differs between
# an OpenCL, a Metal and a CUDA run, plus `_offset` when its `BACKEND` says
# offset views are refused.

# The element types each family runs at. Half precision is not in here: it is
# its own set, because every reference for it is computed in single precision.
const COMPLEX_TYPES = BACKEND.fp64 ? (ComplexF32, ComplexF64) : (ComplexF32,)
const REAL_TYPES = BACKEND.fp64 ? (Float32, Float64) : (Float32,)

# The widest precision the backend runs, for the cases that only need one.
const WIDE_COMPLEX = BACKEND.fp64 ? ComplexF64 : ComplexF32
const WIDE_REAL = BACKEND.fp64 ? Float64 : Float32

# Shapes and regions that exercise every branch of the collapsing logic: full,
# leading, trailing, middle, interleaved, trailing-batch folding, and a 5d array
# whose batch runs merge.
const SHAPE_CASES = (((32, 17), ((1, 2), (1,), (2,))),
                     ((120, 8), ((1, 2), (2,))),
                     ((16, 8, 4), ((1, 2, 3), (1,), (2,), (3,), (1, 2), (2, 3), (1, 3))),
                     ((17, 5, 3), ((1, 2, 3), (2,), (1, 3))),
                     ((2, 3, 4, 5, 6), ((2, 4), (3,), (1, 2), (4, 5), (1, 5), (2, 3, 4))))

# Convolution shapes, regions, and what the region leaves as a trailing
# component stack, one case per codegen path: the 1D workaround at a power of
# two (64), a short Bluestein axis both small (17) and multi-radix (727), mixed
# radix (120), a multi-upload length (8192), plain 2D at a power of two (16x8)
# and through Bluestein (17x5), a stack over a 1D transform (features = 2) and a
# stack over a 2D one (features = 3). Three transformed axes are refused, so
# there is no 3D case here.
const CONV_CASES = (((64,), (1,), 1),
                    ((17,), (1,), 1),
                    ((120,), (1,), 1),
                    ((727,), (1,), 1),
                    ((8192,), (1,), 1),
                    ((16, 8), (1, 2), 1),
                    ((17, 5), (1, 2), 1),
                    ((16, 2), (1,), 2),
                    ((8, 5, 3), (1, 2), 3))

# The FFTW kind each (kind, type) pair claims to be, element for element.
const R2R_FFTW_KIND = Dict((:dct, 1) => FFTW.REDFT00, (:dct, 2) => FFTW.REDFT10,
                           (:dct, 3) => FFTW.REDFT01, (:dct, 4) => FFTW.REDFT11,
                           (:dst, 1) => FFTW.RODFT00, (:dst, 2) => FFTW.RODFT10,
                           (:dst, 3) => FFTW.RODFT01, (:dst, 4) => FFTW.RODFT11)

const R2R_KINDS = (:dct, :dst)

_rtol(::Type{Float16}) = BACKEND.rtol_f16
_rtol(::Type{Float32}) = BACKEND.rtol_f32
_rtol(::Type{Float64}) = BACKEND.rtol_f64
_rtol(::Type{ComplexF16}) = BACKEND.rtol_f16
_rtol(::Type{ComplexF32}) = BACKEND.rtol_f32
_rtol(::Type{ComplexF64}) = BACKEND.rtol_f64

"""
    _relmax(got, ref)

Returns the largest element-wise error relative to the peak magnitude of the reference.
"""
_relmax(got, ref) = maximum(abs.(Array(got) .- ref)) / maximum(abs.(ref))

"""
    _noise(::Type{T}, dims)

Returns a reproducible host array of the given element type and size.
"""
_noise(::Type{T}, dims) where T = rand(MersenneTwister(hash(dims) + hash(T)), T, dims)

"""
    _noise16(::Type{T}, dims)

Returns reproducible half-precision noise centred on zero.

Half precision saturates at 65504, and a forward transform puts the sum of its
input into the DC bin, so noise centred on 1/2 would overflow at the lengths
used here. Centring it keeps every bin of every case in range.
"""
_noise16(::Type{Float16}, dims) = Float16.(_noise(Float32, dims) .- 0.5f0)
_noise16(::Type{ComplexF16}, dims) = ComplexF16.(_noise(ComplexF32, dims) .- ComplexF32(0.5, 0.5))

"""
    _wide16(x)

Returns a host copy of a half-precision array widened to single precision.

FFTW has no half-precision transform, so both sides of every comparison against
it are fp32 and the half-precision values are widened into it.
"""
_wide16(x::AbstractArray{Float16}) = Float32.(Array(x))
_wide16(x::AbstractArray{ComplexF16}) = ComplexF32.(Array(x))

"""
    _kernel_noise(::Type{T}, dims)

Returns a reproducible host array of the given element type and size, from another stream than `_noise`.
"""
_kernel_noise(::Type{T}, dims) where T = rand(MersenneTwister(hash(dims) + hash(T) + 7919), T, dims)

"""
    _thrown(f)

Returns the exception `f` throws, or `nothing` when it does not throw.
"""
function _thrown(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

"""
    _skip(what::String, why::String)

Prints one line saying a capability is off, so a set that did not run is visible in the log.
"""
_skip(what::String, why::String) = println("SKIP $what on $(BACKEND.name): $why")

"""
    _zeroed(h, dim::Int, lo::Int, hi::Int)

Returns a copy of `h` with the slice `lo:hi` of dimension `dim` set to zero.
"""
function _zeroed(h::Array{T, N}, dim::Int, lo::Int, hi::Int) where {T, N}
    z = copy(h)
    selectdim(z, dim, lo:hi) .= zero(T)
    return z
end

"""
    _filled(h, dim::Int, lo::Int, hi::Int, value)

Returns a copy of `h` with the slice `lo:hi` of dimension `dim` set to `value`.

A padded forward plan has to give the same answer for this as for `_zeroed`,
which is the whole point of the feature: the caller never has to clear the block.
"""
function _filled(h::Array{T, N}, dim::Int, lo::Int, hi::Int, value) where {T, N}
    g = copy(h)
    selectdim(g, dim, lo:hi) .= T(value)
    return g
end

"""
    _component_dot(a, b)

Returns the inner product that reads a complex array as its real and imaginary parts.

The adjoint identity of a real transform only holds in this pairing, because a
real transform is real-linear and not complex-linear. `AbstractFFTs.TestUtils`
scores real plans the same way.
"""
_component_dot(a, b) = dot(real.(a), real.(b)) + dot(imag.(a), imag.(b))

"""
    _plan_for(op::Symbol, x, region, inplace::Bool)

Returns the VkFFT plan for one complex-to-complex transform, in-place or not.
"""
function _plan_for(op::Symbol, x, region, inplace::Bool)
    if op === :fft
        return inplace ? VkFFT.plan_fft!(x, region) : VkFFT.plan_fft(x, region)
    elseif op === :bfft
        return inplace ? VkFFT.plan_bfft!(x, region) : VkFFT.plan_bfft(x, region)
    elseif op === :ifft
        return inplace ? VkFFT.plan_ifft!(x, region) : VkFFT.plan_ifft(x, region)
    end
    error("unknown op $op")
end

"""
    _r2r_ref(h, kind::Symbol, type::Int, region)

Returns FFTW's own r2r transform of a host array, in FFTW's normalization.
"""
_r2r_ref(h, kind::Symbol, type::Int, region) = FFTW.plan_r2r(h, R2R_FFTW_KIND[(kind, type)], region) * h

"""
    _r2r_plan(x, kind::Symbol, region; type::Int, inverse::Bool=false, inplace::Bool=false, zeropad=nothing)

Returns whichever of the eight real-to-real entry points a case asks for.
"""
function _r2r_plan(x, kind::Symbol, region; type::Int, inverse::Bool=false, inplace::Bool=false, zeropad=nothing)
    if kind === :dct
        inverse && inplace && return VkFFT.plan_idct!(x, region; type=type, zeropad=zeropad)
        inverse && return VkFFT.plan_idct(x, region; type=type, zeropad=zeropad)
        inplace && return VkFFT.plan_dct!(x, region; type=type, zeropad=zeropad)
        return VkFFT.plan_dct(x, region; type=type, zeropad=zeropad)
    end
    inverse && inplace && return VkFFT.plan_idst!(x, region; type=type, zeropad=zeropad)
    inverse && return VkFFT.plan_idst(x, region; type=type, zeropad=zeropad)
    inplace && return VkFFT.plan_dst!(x, region; type=type, zeropad=zeropad)
    return VkFFT.plan_dst(x, region; type=type, zeropad=zeropad)
end

"""
    _r2r_inverse_type(type::Int)

Returns the type whose forward transform is what an inverse plan of this type computes.
"""
_r2r_inverse_type(type::Int) = type == 2 ? 3 : (type == 3 ? 2 : type)

"""
    _r2r_scale(kind::Symbol, type::Int, n::Int)

Returns the factor one transformed axis of length `n` contributes to an unnormalized round trip.

FFTW's logical transform sizes: s = 2n - 2 for DCT-I, 2n + 2 for DST-I and 2n
for every other kind and type. A normalized inverse plan divides by their
product over the transformed axes. Batched and omitted axes never enter it.
"""
function _r2r_scale(kind::Symbol, type::Int, n::Int)
    kind === :dct && type == 1 && return 2 * n - 2
    kind === :dst && type == 1 && return 2 * n + 2
    return 2 * n
end

"""
    _raw_config(sizes; kwargs...)

Returns a `VkFFTConfig` for a layout of the given axis lengths, fastest axis first.

Every other field comes from `kwargs` and defaults to VkFFT's own default of
zero, which is what the keyword constructor already does. This only exists so a
case reads as its axis lengths plus the two or three fields it is about.

# Arguments
- `sizes`: The axis lengths, as a tuple
- `kwargs...`: Any other `VkFFTConfig` field

# Returns
- A `VkFFTConfig`
"""
_raw_config(sizes; kwargs...) = VkFFTConfig(; fft_dim=length(sizes), size=ntuple(i -> i <= length(sizes) ? UInt64(sizes[i]) : UInt64(0), VkFFT.VKFFT_MAX_FFT_DIMENSIONS), kwargs...)
