module VkFFTCUDAExt

using CUDA
using VkFFT

import CUDA: CuArray, CuContext, CuDevice, DenseCuArray

# CUDA.jl represents every contiguous view, reshape and reinterpret of a CuArray
# as another CuArray carrying a byte offset, so DenseCuArray is an alias for
# CuArray and nothing else. Everything CUDA.jl cannot express (a strided view,
# an Adjoint, a PermutedDimsArray) stays a wrapper type and is refused by
# dispatch.
VkFFT._backend(::Type{<:DenseCuArray}) = Val(:cuda)

# VkFFT loads its nvrtc modules and allocates its scratch buffers in the context
# that is current when the app is built, and every launch and the free have to
# happen in that same context. The plan carries the context it was built in, and
# the planner only ever builds one in the current task's, so an array from

# another context has to be refused rather than planned for.
function VkFFT._check_layout(x::DenseCuArray)
    CUDA.context(x) == CUDA.context() || throw(ArgumentError("this array lives in a different CUDA context than the current task's. Switch to its device with CUDA.device! before planning, or copy the array to the current device."))

    return nothing
end

# convert(Ptr{Cvoid}, ::CuPtr) is refused on purpose by CUDA.jl (a device
# pointer is not a host pointer), so the reinterpret is the way across. Taking
# the pointer is also CUDA.jl's own accessor: it hands the allocation's stream
# ownership to the current task's stream and marks it dirty, which is what makes
# a later copyto! or Array() see the transform's output.
# Half precision on CUDA is a property of the wrapper rather than of the card.
# VkFFT builds its half-precision kernels around an include of cuda_fp16.h from
# the CUDA_TOOLKIT_ROOT_DIR baked into libvkfft at compile time, and it gives
# nvrtc no header list and no -I, so a wrapper built with an empty root cannot
# compile one. What comes back from VkFFT in that case is a failed compile plus
# the whole generated kernel on stdout, which is why the path is read here
# instead. A wrapper too old to have the accessor answers `nothing`, and that is
# not a refusal: it plans as it always did.
function VkFFT._check_precision(::Type{<:VkFFT.VkFFTHalf}, x::DenseCuArray)
    VkFFT._cuda_toolkit_root() == "" && throw(ArgumentError("this libvkfft was built without a CUDA toolkit path, so it cannot run a Float16 or ComplexF16 transform. VkFFT compiles its half-precision kernels by including cuda_fp16.h from the CUDA_TOOLKIT_ROOT_DIR baked into the wrapper, and nvrtc has no other way to find that header. Build libvkfft against your local CUDA toolkit, which bakes a real path, or plan in Float32 or ComplexF32."))

    return nothing
end

VkFFT._buffer_handle(x::DenseCuArray) = reinterpret(Ptr{Cvoid}, pointer(x))

VkFFT._stream_handle(::DenseCuArray) = convert(Ptr{Cvoid}, CUDA.stream().handle)

VkFFT._synchronize(::DenseCuArray) = CUDA.synchronize()

# The context handle alone is not enough. CUDA.jl keeps a unique id per context
# because a destroyed context's handle can come back for a new one, and a cache
# hit across that boundary would apply a dead VkFFT application.
function VkFFT._device_id(::DenseCuArray)
    context = CUDA.context()
    return xor(UInt64(UInt(context.handle)), context.id)
end

# _check_layout has already established that the array's context is the current
# task's, so the current device is the array's device too. Reading both from the
# task keeps this to two task-local lookups: CUDA.device(x) would push and pop a
# context on every planning call, cache hits included.
VkFFT._device_roots(::DenseCuArray) = Any[CUDA.device(), CUDA.context()]

# VkFFT compiles through nvrtc for a compute capability and loads the result as
# a module for a device, so the capability is what decides whether a cubin is
# even legal and the device name is what separates two cards of one capability.
# The CUDA driver version is deliberately not in here: it is a property of the
# process rather than of the plan, and nvrtc lives inside libvkfft, whose own
# contents the key already covers.
function VkFFT._device_key(roots::Vector{Any}, ::Val{:cuda})
    device = roots[1]::CuDevice
    capability = CUDA.capability(device)

    return string(CUDA.name(device), " | sm_", capability.major, capability.minor)
end

# device_handles for the CUDA backend is {CUdevice*} (i.e., a pointer to a
# CUdevice handle. The wrapper copies the value out, so the Ref doesn't have to
# outlive the call. The context has to outlive the app though, which is why the
# plan holds on to `roots`.
function VkFFT._with_device_handles(f, roots::Vector{Any}, ::Val{:cuda})
    device = roots[1]::CuDevice
    context = roots[2]::CuContext

    handle = Ref(device.handle)
    return CUDA.context!(context) do
        GC.@preserve handle begin
            handles = Ptr{Cvoid}[Base.unsafe_convert(Ptr{eltype(handle)}, handle)]
            GC.@preserve handles f(Base.unsafe_convert(Ptr{Ptr{Cvoid}}, handles))
        end
    end
end

# Runs from the garbage collector, where throwing loses the rest of the
# finalizer and yielding is illegal. CUDA.context! neither allocates a task nor
# takes a lock, and CUDA.jl frees its own device memory from finalizers the same
# way. It does throw when the context has been destroyed, and there is no
# version-stable way to check if a context is still alive (CUDA.jl 5 does so
# with a skip_destroyed keyword, CUDA.jl 6 with a function that is not in the
# CUDA namespace). A destroyed context has already freed everything the app
# owned, so there is nothing left to do.
function VkFFT._with_plan_context(f, ::Val{:cuda}, roots::Vector{Any})
    context = roots[2]::CuContext
    try
        CUDA.context!(f, context)
    catch
    end

    return nothing
end

end # module
