# Every @ccall into libvkfft is in this file. They are `_`-prefixed names of the
# underlying C function.

# The array length baked into vkfft_config. The struct layout depends on it, so
# it cannot be queried at run time. _ensure_library! checks it against
# vkfft_max_dims() and errors if they differ.
const VKFFT_MAX_FFT_DIMENSIONS = 12

# Set by VkFFT.__init__ from the libvkfft_path preference.
libvkfft::String = ""

"""
    VkFFTConfig

Julia mirror of `vkfft_config`, the whole of what the wrapper exposes of VkFFT's configuration.

Field order and types are part of the wrapper's ABI. Zero means "VkFFT's
default" for every field, so a zeroed struct plus `fft_dim` and `size` is a
valid complex-to-complex plan. Build one with the keyword constructor, which
leaves every field VkFFT defaults to zero at zero, and hand it to
`VkFFT.unsafe_plan` to get a plan out of it. Nothing validates the combination:
read `VkFFT.unsafe_plan`'s docstring for what that costs.

Axes run fastest first, which is Julia's column-major order, and `size` slots
past `fft_dim` must be zero. `precision` 3 selects VkFFT's double-double quad
arithmetic, which stores four native floats per complex value and so has no
Julia element type of its own.

# Fields
- `fft_dim::UInt64`: Number of VkFFT axes, i.e. how many `size` slots are meaningful
- `size::NTuple{12, UInt64}`: Axis lengths, fastest axis first. Unused slots must be 0
- `omit::NTuple{12, UInt64}`: 1 = no FFT along this axis, but keep it in the layout
- `number_batches::UInt64`: Extra outermost batch count. 0 leaves VkFFT's default of 1
- `precision::Cint`: 0 = fp32 (Float32), 1 = fp64 (Float64), 2 = fp16 (Float16), 3 = quad (Float128)
- `r2c::Cint`: 1 = R2C forward / C2R inverse
- `dct::Cint`: DCT type 1..4. 0 = off
- `dst::Cint`: DST type 1..4. 0 = off
- `inplace::Cint`: 1 = one buffer. 0 = read `in`, write `out`
- `normalize::Cint`: 1 = apply 1/N in-kernel on the inverse
- `make_forward_only::Cint`: Skip inverse kernel generation
- `make_inverse_only::Cint`: Skip forward kernel generation
- `zeropad_left::NTuple{12, UInt64}`: VkFFT `fft_zeropad_left`
- `zeropad_right::NTuple{12, UInt64}`: VkFFT `fft_zeropad_right`
- `perform_zeropad::NTuple{12, Cint}`: VkFFT `performZeropadding`, per axis
- `coalesced_memory::UInt64`: Tuning, in bytes (0 = auto)
- `aim_threads::UInt64`: Tuning, threads per block (0 = auto)
- `num_shared_banks::UInt64`: Tuning (0 = auto)
- `coordinate_features::UInt64`: VkFFT `coordinateFeatures`. 0 leaves VkFFT's default of 1
- `number_kernels::UInt64`: VkFFT `numberKernels`. 0 leaves VkFFT's default of 1
- `perform_convolution::Cint`: 1 = fft, multiply, inverse in one plan
- `kernel_convolution::Cint`: 1 = this plan only transforms a convolution kernel
- `conjugate_convolution::Cint`: 1 = conjugate the transformed input, giving a cross-correlation
- `save_to_string::Cint`: 1 = keep the compiled binaries for `vkfft_save`
"""
struct VkFFTConfig
    fft_dim::UInt64
    size::NTuple{VKFFT_MAX_FFT_DIMENSIONS, UInt64}
    omit::NTuple{VKFFT_MAX_FFT_DIMENSIONS, UInt64}
    number_batches::UInt64
    precision::Cint
    r2c::Cint
    dct::Cint
    dst::Cint
    inplace::Cint
    normalize::Cint
    make_forward_only::Cint
    make_inverse_only::Cint
    zeropad_left::NTuple{VKFFT_MAX_FFT_DIMENSIONS, UInt64}
    zeropad_right::NTuple{VKFFT_MAX_FFT_DIMENSIONS, UInt64}
    perform_zeropad::NTuple{VKFFT_MAX_FFT_DIMENSIONS, Cint}
    coalesced_memory::UInt64
    aim_threads::UInt64
    num_shared_banks::UInt64
    coordinate_features::UInt64
    number_kernels::UInt64
    perform_convolution::Cint
    kernel_convolution::Cint
    conjugate_convolution::Cint
    save_to_string::Cint
end

"""
    VkFFTConfig(; fft_dim, size, omit, number_batches=0, precision=0, ...)

Builds a `VkFFTConfig` with every field that VkFFT defaults to zero left at zero.

`fft_dim` and `size` are the two required keywords, and a config with nothing
else set is a forward and inverse out-of-place complex-to-complex plan in fp32.
The keywords mirror the struct's fields one for one, so the `# Fields` list of
`VkFFTConfig` is what each of them means.

# Returns
- A `VkFFTConfig`
"""
function VkFFTConfig(; fft_dim::Integer, size::NTuple{VKFFT_MAX_FFT_DIMENSIONS, UInt64},
                     omit=ntuple(_ -> UInt64(0), VKFFT_MAX_FFT_DIMENSIONS),
                     number_batches::Integer=0, precision::Integer=0, r2c::Integer=0,
                     dct::Integer=0, dst::Integer=0, inplace::Integer=0, normalize::Integer=0,
                     make_forward_only::Integer=0, make_inverse_only::Integer=0,
                     zeropad_left=ntuple(_ -> UInt64(0), VKFFT_MAX_FFT_DIMENSIONS),
                     zeropad_right=ntuple(_ -> UInt64(0), VKFFT_MAX_FFT_DIMENSIONS),
                     perform_zeropad=ntuple(_ -> Cint(0), VKFFT_MAX_FFT_DIMENSIONS),
                     coalesced_memory::Integer=0, aim_threads::Integer=0, num_shared_banks::Integer=0,
                     coordinate_features::Integer=0, number_kernels::Integer=0,
                     perform_convolution::Integer=0, kernel_convolution::Integer=0,
                     conjugate_convolution::Integer=0, save_to_string::Integer=0)
    return VkFFTConfig(UInt64(fft_dim), size, omit, UInt64(number_batches), Cint(precision),
                       Cint(r2c), Cint(dct), Cint(dst), Cint(inplace), Cint(normalize),
                       Cint(make_forward_only), Cint(make_inverse_only), zeropad_left,
                       zeropad_right, perform_zeropad, UInt64(coalesced_memory),
                       UInt64(aim_threads), UInt64(num_shared_banks), UInt64(coordinate_features),
                       UInt64(number_kernels), Cint(perform_convolution), Cint(kernel_convolution),
                       Cint(conjugate_convolution), Cint(save_to_string))
end

"""
    _vkfft_create(config::Ref{VkFFTConfig}, handles::Ptr{Ptr{Cvoid}}, app::Ref{Ptr{Cvoid}})

Wrapper for the VkFFT C function vkfft_create.
"""
_vkfft_create(config::Ref{VkFFTConfig}, handles::Ptr{Ptr{Cvoid}}, app::Ref{Ptr{Cvoid}}) = @ccall libvkfft.vkfft_create(config::Ref{VkFFTConfig}, handles::Ptr{Ptr{Cvoid}}, app::Ref{Ptr{Cvoid}})::Cint

"""
    _vkfft_execute(app::Ptr{Cvoid}, in::Ptr{Cvoid}, out::Ptr{Cvoid}, direction::Integer, stream::Ptr{Cvoid})

Wrapper for the VkFFT C function vkfft_execute.
"""
_vkfft_execute(app::Ptr{Cvoid}, in::Ptr{Cvoid}, out::Ptr{Cvoid}, direction::Integer, stream::Ptr{Cvoid}) = @ccall libvkfft.vkfft_execute(app::Ptr{Cvoid}, in::Ptr{Cvoid}, out::Ptr{Cvoid}, direction::Cint, stream::Ptr{Cvoid})::Cint

"""
    _vkfft_set_kernel(app::Ptr{Cvoid}, kernel::Ptr{Cvoid})

Wrapper for the VkFFT C function vkfft_set_kernel.
"""
_vkfft_set_kernel(app::Ptr{Cvoid}, kernel::Ptr{Cvoid}) = @ccall libvkfft.vkfft_set_kernel(app::Ptr{Cvoid}, kernel::Ptr{Cvoid})::Cint

"""
    _vkfft_destroy(app::Ptr{Cvoid})

Wrapper for the VkFFT C function vkfft_destroy.
"""
_vkfft_destroy(app::Ptr{Cvoid}) = @ccall libvkfft.vkfft_destroy(app::Ptr{Cvoid})::Cvoid

"""
    _vkfft_error_name(code::Integer)

Wrapper for the VkFFT C function vkfft_error_name.
"""
_vkfft_error_name(code::Integer) = unsafe_string(@ccall libvkfft.vkfft_error_name(code::Cint)::Cstring)

"""
    _vkfft_max_dims()

Wrapper for the VkFFT C function vkfft_max_dims.
"""
_vkfft_max_dims() = @ccall libvkfft.vkfft_max_dims()::UInt64

"""
    _vkfft_backend()

Wrapper for the VkFFT C function vkfft_backend.
"""
_vkfft_backend() = @ccall libvkfft.vkfft_backend()::UInt64

"""
    _vkfft_config_size()

Wrapper for the VkFFT C function vkfft_config_size.
"""
_vkfft_config_size() = @ccall libvkfft.vkfft_config_size()::UInt64

"""
    _vkfft_cuda_toolkit_root()

Wrapper for the VkFFT C function vkfft_cuda_toolkit_root.
"""
_vkfft_cuda_toolkit_root() = unsafe_string(@ccall libvkfft.vkfft_cuda_toolkit_root()::Cstring)

"""
    _check(code::Integer)

Turns a nonzero VkFFT result code into a `VkFFTError`.
"""
_check(code::Integer) = iszero(code) ? nothing : throw(VkFFTError(Int(code), _vkfft_error_name(code)))

const LIBRARY_LOCK = ReentrantLock()
const LIBRARY_READY = Ref(false)
const LIBRARY_MAX_DIMS = Ref(0)
const LIBRARY_BACKEND = Ref(UInt64(0))

# `nothing` means the loaded wrapper does not export vkfft_cuda_toolkit_root,
# which is a different thing from exporting it and getting an empty path back.
const LIBRARY_CUDA_TOOLKIT_ROOT = Ref{Union{Nothing, String}}(nothing)

"""
    _ensure_library!()

Resolves and ABI-checks libvkfft, once, on the first call that needs the C library.

Doing this lazily rather than in `__init__` keeps `using VkFFT` (and
precompilation) working with no wrapper installed, which is what the CPU-only
region tests rely on.
"""
function _ensure_library!()
    LIBRARY_READY[] && return nothing

    @lock LIBRARY_LOCK begin
        LIBRARY_READY[] && return nothing

        isempty(libvkfft) && error("VkFFT does not know where libvkfft is. Set the libvkfft_path preference, e.g. `using Preferences; set_preferences!(VkFFT, \"libvkfft_path\" => \"/path/to/libvkfft_icd.dylib\")`, then restart Julia.")
        isfile(libvkfft) || error("The libvkfft_path preference points at \"$libvkfft\", which is not a file. Set it to the wrapper library built for your backend.")

        max_dims = Int(_vkfft_max_dims())
        max_dims == VKFFT_MAX_FFT_DIMENSIONS || error("libvkfft at \"$libvkfft\" was built with VKFFT_MAX_FFT_DIMENSIONS = $max_dims, but VkFFT.jl mirrors vkfft_config with $VKFFT_MAX_FFT_DIMENSIONS slots. Rebuild the wrapper with -DVKFFT_MAX_FFT_DIMENSIONS=$VKFFT_MAX_FFT_DIMENSIONS.")

        # vkfft_config only ever grows at its end, so a mirror that is short by
        # a field keeps every offset valid and the wrapper reads whatever
        # follows the struct instead. Nothing else notices.
        config_size = Int(_vkfft_config_size())
        config_size == sizeof(VkFFTConfig) || error("libvkfft at \"$libvkfft\" reads a $config_size byte vkfft_config, but VkFFT.jl mirrors it as $(sizeof(VkFFTConfig)) bytes. The wrapper and the package are from different versions. Rebuild libvkfft from the vkfft_wrapper.h this VkFFT.jl was written against, or update VkFFT.jl to match the wrapper.")

        LIBRARY_MAX_DIMS[] = max_dims
        LIBRARY_BACKEND[] = _vkfft_backend()

        # vkfft_cuda_toolkit_root came after the other ten symbols, so a
        # wrapper built before it exports nothing of the sort and the ccall
        # raises instead of returning. That is recorded as "unknown" rather
        # than allowed to fail the load, and every caller then behaves as this
        # package did before the accessor existed: nothing is refused up front
        # and VkFFT reports whatever it reports.
        LIBRARY_CUDA_TOOLKIT_ROOT[] = try
            _vkfft_cuda_toolkit_root()
        catch
            nothing
        end

        # vkfft_destroy is called from finalizers, i.e., from inside the garbage
        # collector, and the first ccall to a symbol has to resolve it through
        # the dynamic loader. Resolve it now instead since destroying a null app
        # does nothing.
        _vkfft_destroy(Ptr{Cvoid}(C_NULL))

        LIBRARY_READY[] = true
    end

    return nothing
end

"""
    _max_dims()

Returns the maximum number of VkFFT axes the loaded library supports.
"""
function _max_dims()
    _ensure_library!()
    return LIBRARY_MAX_DIMS[]
end

"""
    _backend_id()

Returns the VKFFT_BACKEND value baked into the loaded library (1 CUDA, 3 OpenCL, 5 Metal).
"""
function _backend_id()
    _ensure_library!()
    return LIBRARY_BACKEND[]
end

"""
    _cuda_toolkit_root()

Returns the CUDA toolkit path baked into the loaded library, or `nothing` when it cannot be asked.

An empty string is an answer and means half precision cannot compile on this
wrapper. `nothing` is the absence of one, from a wrapper predating
`vkfft_cuda_toolkit_root`, and callers treat it as permission rather than as a
refusal.
"""
function _cuda_toolkit_root()
    _ensure_library!()
    return LIBRARY_CUDA_TOOLKIT_ROOT[]
end
