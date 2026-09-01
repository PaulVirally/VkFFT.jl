module VkFFTMetalExt

using Metal
using VkFFT

import Metal: MTL, MtlArray

# Metal has no double, and MtlArray refuses a Float64 element type at
# construction, so this only ever fires on an array smuggled past that check.
# Half precision needs no gate of its own: Metal has native half storage, and
# VkFFT's half-precision transforms are measured correct on it, so
# _check_precision stays at its permissive default here.
const NO_FLOAT64 = "Metal has no Float64. Plan in Float32 and ComplexF32, or run the double-precision transform on a CUDA or OpenCL device."

"""
    _objc_handle(obj)

Returns an Objective-C object as the opaque pointer the wrapper takes.

The Metal wrapper is handed the objects themselves rather than pointers to
them, on both `vkfft_create` and `vkfft_execute`, which is what makes this a
plain reinterpretation of the object pointer.
"""
_objc_handle(obj) = Ptr{Cvoid}(UInt(pointer(obj)))

"""
    _queue(x::MtlArray)

Returns the command queue of the task that owns this array's device.

Metal.jl batches its own kernels and blits into one command buffer per task on
this queue, and Metal runs the command buffers of a queue in the order they were
committed, so submitting the transform here is what keeps it ordered against
`copyto!` and broadcasts.
"""
_queue(x::MtlArray) = Metal.global_queue(Metal.device(x))

VkFFT._backend(::Type{<:MtlArray{<:Union{Float64, ComplexF64}}}) = throw(ArgumentError(NO_FLOAT64))

# Metal.jl represents every contiguous view, reshape and reinterpret of an
# MtlArray as another MtlArray carrying a byte offset, so MtlArray covers all of
# them. Everything Metal.jl cannot express that way (a strided view, an Adjoint,
# a PermutedDimsArray) stays a wrapper type and is refused by dispatch.
VkFFT._backend(::Type{<:MtlArray}) = Val(:metal)

# The Metal launch parameters carry no offset, so a buffer always enters the
# transform at its own first element, exactly as on OpenCL.
function VkFFT._check_layout(x::MtlArray)
    offset = pointer(x).offset
    offset == 0 || throw(ArgumentError("VkFFT needs a buffer whose data starts at offset 0. Got offset $offset bytes. Copy the view or reinterpreted array first."))

    Metal.device(x) == Metal.device() || throw(ArgumentError("this array lives on a different Metal device than the current task's. Switch to it with Metal.device! before planning, or copy the array to the current device."))

    return nothing
end

# pointer(x) is Metal.jl's own accessor, and the buffer it names is the whole
# allocation. _check_layout has already established that the array starts at the
# base of it.
VkFFT._buffer_handle(x::MtlArray) = _objc_handle(pointer(x).buffer)

VkFFT._synchronize(x::MtlArray) = Metal.synchronize(_queue(x))

VkFFT._device_id(x::MtlArray) = UInt64(UInt(pointer(Metal.device(x))))

# The wrapper copies the two objects into storage it owns but retains neither,
# so the plan is what keeps them alive. Holding the queue is the part that
# matters: a Metal device is a system singleton, while a command queue belongs
# to the task that made it, and VkFFT keeps a pointer to the one its application
# was built with.
VkFFT._device_roots(x::MtlArray) = Any[Metal.device(x), Metal.raw_queue(_queue(x))]

# Keys the tuning records. The registry ID separates two devices of one model,
# which a Mac with an external GPU can have, and is a cheap property read.
function VkFFT._device_key(roots::Vector{Any}, ::Val{:metal})
    device = roots[1]::MTL.MTLDevice

    return string(String(device.name), " | ", device.registryID)
end

# device_handles for the Metal backend is {MTLDevice, MTLCommandQueue}: the
# objects themselves, unlike CUDA and OpenCL where the entries are pointers to
# the handle variables.
#
# Nothing here may open an autorelease pool. vkfft_create builds VkFFT's lookup
# tables by submitting to the queue and waiting, and VkFFT's Metal plan builder
# over-releases the strings it compiles its kernels from, so a pool drained
# after this call segfaults inside objc_release. The call has to run on a thread
# holding no pool, which a bare Julia task is.
function VkFFT._with_device_handles(f, roots::Vector{Any}, ::Val{:metal})
    device = roots[1]::MTL.MTLDevice
    queue = roots[2]::MTL.MTLCommandQueue

    handles = Ptr{Cvoid}[_objc_handle(device), _objc_handle(queue)]
    return GC.@preserve handles roots f(Base.unsafe_convert(Ptr{Ptr{Cvoid}}, handles))
end

# Metal has no standing submission handle to hand out, which is why there is no
# _stream_handle method here: the wrapper takes a command buffer, one per
# application, and the caller commits it. _with_execution owns that.
#
# The command buffer comes back from the queue autoreleased, so the pool is what
# reclaims it. Releasing it by hand would be an over-release. Nothing commits
# until f has returned, so f is free to append several transforms to the one
# command buffer, and their encoders run in the order they were made.
function VkFFT._with_execution(f, x::MtlArray)
    queue = _queue(x)
    Metal.flush!(queue) # commit Metal.jl's open batch first, so its work runs before ours

    return Metal.@autoreleasepool begin
        command_buffer = MTL.MTLCommandBuffer(queue)
        res = f(_objc_handle(command_buffer))
        MTL.commit!(command_buffer)
        Metal.synchronize(queue)
        res
    end
end

end # module
