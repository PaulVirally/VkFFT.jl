# Every @ccall into libvkfft is in this file. They are `_`-prefixed names of the
# underlying C function.

# The array length baked into vkfft_config. The struct layout depends on it, so
# it cannot be queried at run time. _library checks it against vkfft_max_dims()
# and errors if they differ.
const VKFFT_MAX_FFT_DIMENSIONS = 12

# The VKFFT_BACKEND value each backend's wrapper is built with.
const BACKEND_IDS = (cuda=1, opencl=3, metal=5)

# One wrapper per backend, opened by _library. The slots are fixed because
# _library reads them without the lock, which finalizers must not take.
const LIBRARIES = (cuda=Ref(C_NULL), opencl=Ref(C_NULL), metal=Ref(C_NULL))

# Where each backend extension puts the wrapper path from its JLL. A
# libvkfft_path preference built for the same backend wins over it.
const EXTENSION_LIBRARIES = (cuda=Ref(""), opencl=Ref(""), metal=Ref(""))

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
    _vkfft_create(lib::Ptr{Cvoid}, config::Ref{VkFFTConfig}, handles::Ptr{Ptr{Cvoid}}, app::Ref{Ptr{Cvoid}})

Wrapper for the VkFFT C function vkfft_create.
"""
_vkfft_create(lib::Ptr{Cvoid}, config::Ref{VkFFTConfig}, handles::Ptr{Ptr{Cvoid}}, app::Ref{Ptr{Cvoid}}) = @ccall $(dlsym(lib, :vkfft_create))(config::Ref{VkFFTConfig}, handles::Ptr{Ptr{Cvoid}}, app::Ref{Ptr{Cvoid}})::Cint

"""
    _vkfft_execute(lib::Ptr{Cvoid}, app::Ptr{Cvoid}, in::Ptr{Cvoid}, out::Ptr{Cvoid}, direction::Integer, stream::Ptr{Cvoid})

Wrapper for the VkFFT C function vkfft_execute.
"""
_vkfft_execute(lib::Ptr{Cvoid}, app::Ptr{Cvoid}, in::Ptr{Cvoid}, out::Ptr{Cvoid}, direction::Integer, stream::Ptr{Cvoid}) = @ccall $(dlsym(lib, :vkfft_execute))(app::Ptr{Cvoid}, in::Ptr{Cvoid}, out::Ptr{Cvoid}, direction::Cint, stream::Ptr{Cvoid})::Cint

"""
    _vkfft_set_kernel(lib::Ptr{Cvoid}, app::Ptr{Cvoid}, kernel::Ptr{Cvoid})

Wrapper for the VkFFT C function vkfft_set_kernel.
"""
_vkfft_set_kernel(lib::Ptr{Cvoid}, app::Ptr{Cvoid}, kernel::Ptr{Cvoid}) = @ccall $(dlsym(lib, :vkfft_set_kernel))(app::Ptr{Cvoid}, kernel::Ptr{Cvoid})::Cint

"""
    _vkfft_destroy(lib::Ptr{Cvoid}, app::Ptr{Cvoid})

Wrapper for the VkFFT C function vkfft_destroy.
"""
_vkfft_destroy(lib::Ptr{Cvoid}, app::Ptr{Cvoid}) = @ccall $(dlsym(lib, :vkfft_destroy))(app::Ptr{Cvoid})::Cvoid

"""
    _vkfft_error_name(lib::Ptr{Cvoid}, code::Integer)

Wrapper for the VkFFT C function vkfft_error_name.
"""
_vkfft_error_name(lib::Ptr{Cvoid}, code::Integer) = unsafe_string(@ccall $(dlsym(lib, :vkfft_error_name))(code::Cint)::Cstring)

"""
    _vkfft_max_dims(lib::Ptr{Cvoid})

Wrapper for the VkFFT C function vkfft_max_dims.
"""
_vkfft_max_dims(lib::Ptr{Cvoid}) = @ccall $(dlsym(lib, :vkfft_max_dims))()::UInt64

"""
    _vkfft_backend(lib::Ptr{Cvoid})

Wrapper for the VkFFT C function vkfft_backend.
"""
_vkfft_backend(lib::Ptr{Cvoid}) = @ccall $(dlsym(lib, :vkfft_backend))()::UInt64

"""
    _vkfft_config_size(lib::Ptr{Cvoid})

Wrapper for the VkFFT C function vkfft_config_size.
"""
_vkfft_config_size(lib::Ptr{Cvoid}) = @ccall $(dlsym(lib, :vkfft_config_size))()::UInt64

"""
    _vkfft_cuda_toolkit_root(lib::Ptr{Cvoid})

Wrapper for the VkFFT C function vkfft_cuda_toolkit_root.
"""
_vkfft_cuda_toolkit_root(lib::Ptr{Cvoid}) = unsafe_string(@ccall $(dlsym(lib, :vkfft_cuda_toolkit_root))()::Cstring)

"""
    _check(lib::Ptr{Cvoid}, code::Integer)

Turns a nonzero VkFFT result code into a `VkFFTError`.
"""
_check(lib::Ptr{Cvoid}, code::Integer) = iszero(code) ? nothing : throw(VkFFTError(Int(code), _vkfft_error_name(lib, code)))

const LIBRARY_LOCK = ReentrantLock()

"""
    _library(backend::Symbol)

Returns the ABI-checked libvkfft handle for `backend`, opening it on first use.
"""
function _library(backend::Symbol)
    slot = LIBRARIES[backend]
    slot[] == C_NULL || return slot[]

    return @lock LIBRARY_LOCK begin
        slot[] == C_NULL || return slot[]

        path = load_preference(@__MODULE__, "libvkfft_path", "")
        if isempty(path) || _vkfft_backend(dlopen(path)) != BACKEND_IDS[backend]
            path = EXTENSION_LIBRARIES[backend][]
        end
        isempty(path) && error("VkFFT does not know where the $backend libvkfft is. Load the $backend backend's JLL, or point the libvkfft_path preference at a wrapper you built for $backend: `using Preferences; set_preferences!(VkFFT, \"libvkfft_path\" => \"/path/to/libvkfft.so\")`.")
        lib = dlopen(path)

        max_dims = Int(_vkfft_max_dims(lib))
        max_dims == VKFFT_MAX_FFT_DIMENSIONS || error("libvkfft at \"$path\" was built with VKFFT_MAX_FFT_DIMENSIONS = $max_dims, but VkFFT.jl mirrors vkfft_config with $VKFFT_MAX_FFT_DIMENSIONS slots. Rebuild the wrapper with -DVKFFT_MAX_FFT_DIMENSIONS=$VKFFT_MAX_FFT_DIMENSIONS.")

        # vkfft_config only ever grows at its end, so a mirror that is short by
        # a field keeps every offset valid and the wrapper reads whatever
        # follows the struct instead. Nothing else notices.
        config_size = Int(_vkfft_config_size(lib))
        config_size == sizeof(VkFFTConfig) || error("libvkfft at \"$path\" reads a $config_size byte vkfft_config, but VkFFT.jl mirrors it as $(sizeof(VkFFTConfig)) bytes. The wrapper and the package are from different versions. Rebuild libvkfft from the vkfft_wrapper.h this VkFFT.jl was written against, or update VkFFT.jl to match the wrapper.")

        slot[] = lib
    end
end

_library(::Val{B}) where B = _library(B)

"""
    library_path(backend::Symbol)

Returns the path of the libvkfft that plans on `backend` (`:cuda`, `:opencl` or `:metal`) call into.
"""
library_path(backend::Symbol) = dlpath(_library(backend))

"""
    _cuda_toolkit_root()

Returns the CUDA toolkit path baked into the CUDA libvkfft, or `nothing` when it cannot be asked.

An empty string is an answer and means half precision cannot compile on this
wrapper. `nothing` is the absence of one, from a wrapper predating
`vkfft_cuda_toolkit_root`, and callers treat it as permission rather than as a
refusal.
"""
function _cuda_toolkit_root()
    lib = _library(:cuda)
    return dlsym(lib, :vkfft_cuda_toolkit_root; throw_error=false) === nothing ? nothing : _vkfft_cuda_toolkit_root(lib)
end
