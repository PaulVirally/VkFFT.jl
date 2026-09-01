module VkFFTOpenCLExt

using OpenCL
using VkFFT

import OpenCL: cl, CLArray

# VkFFT drives clSetKernelArg with sizeof(cl_mem), so it needs an actual cl_mem
const CLBufferArray{T, N} = CLArray{T, N, cl.Buffer}

const BUFFER_BACKEND_HINT = "VkFFT needs OpenCL.jl's Buffer memory backend. Set it for the session with `task_local_storage(:CLMemoryBackend, OpenCL.cl.BufferBackend())`, or project-wide with `using Preferences; set_preferences!(OpenCL, \"default_memory_backend\" => \"buffer\")`."

VkFFT._backend(::Type{<:CLArray}) = Val(:opencl)

VkFFT._check_layout(x::CLArray) = throw(ArgumentError("$BUFFER_BACKEND_HINT Got a $(typeof(x))."))

function VkFFT._check_layout(x::CLBufferArray)
    x.offset == 0 || throw(ArgumentError("VkFFT needs a buffer whose data starts at offset 0. Got offset $(x.offset) bytes. Copy the view or reinterpreted array first."))

    array_context = Base.unsafe_convert(cl.cl_context, cl.context(x.data[].mem))
    task_context = Base.unsafe_convert(cl.cl_context, cl.context())
    array_context == task_context || throw(ArgumentError("this array lives in a different OpenCL context than the current task's. Switch to it with OpenCL.cl.context! before planning."))

    return nothing
end

# Half precision is a per-device OpenCL extension rather than a property of the
# backend, so it is checked on the device the plan is about to be built for.
# Without cl_khr_fp16 VkFFT emits half2 arithmetic the driver refuses to compile
# and the only thing coming back is VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM.
# OpenCL.jl already refuses to allocate a half-precision CLArray on such a
# device, so in practice this catches an array that arrived some other way, and
# it is what keeps a device that does report the extension planning normally.
function VkFFT._check_precision(::Type{<:VkFFT.VkFFTHalf}, x::CLArray)
    "cl_khr_fp16" in cl.device().extensions || throw(ArgumentError("this OpenCL device ($(cl.device().name)) does not report the cl_khr_fp16 extension, so it cannot run a half-precision transform. Plan in Float32 or ComplexF32, or use a device whose extension list includes cl_khr_fp16."))

    return nothing
end

VkFFT._buffer_handle(x::CLArray) = throw(ArgumentError("$BUFFER_BACKEND_HINT Got a $(typeof(x))."))

# convert(cl.Buffer, ::Managed) is OpenCL.jl's own accessor. Besides handing
# over the cl_mem, it takes queue ownership and marks the allocation dirty. This
# makes a future copyto! or Array() see the transform's output.
VkFFT._buffer_handle(x::CLBufferArray) = convert(Ptr{Cvoid}, Base.unsafe_convert(cl.cl_mem, convert(cl.Buffer, x.data[])))

VkFFT._stream_handle(x::CLArray) = convert(Ptr{Cvoid}, Base.unsafe_convert(cl.cl_command_queue, cl.queue()))

VkFFT._synchronize(x::CLArray) = cl.finish(cl.queue())

VkFFT._device_id(x::CLArray) = UInt64(reinterpret(UInt, Base.unsafe_convert(cl.cl_context, cl.context(x.data[].mem))))

VkFFT._device_roots(x::CLArray) = Any[cl.platform(), cl.device(), cl.context(), cl.queue()]

# An OpenCL tuning answer belongs to a driver and a device, and both change
# independently of each other: the same GPU under a new driver can want other
# knobs, and one platform can enumerate several models. Platform name, device
# name and CL_DRIVER_VERSION together pin all three, and pocl bumps its driver
# version with every release, which is what keeps a record tuned under one pocl
# from applying under a rebuilt one.
function VkFFT._device_key(roots::Vector{Any}, ::Val{:opencl})
    platform = roots[1]::cl.Platform
    device = roots[2]::cl.Device

    return string(platform.name, " | ", device.name, " | ", device.driver_version)
end

# device_handles for the OpenCL backend is {cl_platform_id*, cl_device_id*,
# cl_context*, cl_command_queue*}: an array of pointers to the handle variables,
# not the handles. The wrapper copies the values out, so nothing here has to 
# outlive the call (but the objects do, which is why the plan holds on to
# `roots`).
function VkFFT._with_device_handles(f, roots::Vector{Any}, ::Val{:opencl})
    platform = Ref(Base.unsafe_convert(cl.cl_platform_id, roots[1]::cl.Platform))
    device = Ref(Base.unsafe_convert(cl.cl_device_id, roots[2]::cl.Device))
    context = Ref(Base.unsafe_convert(cl.cl_context, roots[3]::cl.Context))
    queue = Ref(Base.unsafe_convert(cl.cl_command_queue, roots[4]::cl.CmdQueue))

    GC.@preserve platform device context queue begin
        handles = Ptr{Cvoid}[Base.unsafe_convert(Ptr{cl.cl_platform_id}, platform),
                             Base.unsafe_convert(Ptr{cl.cl_device_id}, device),
                             Base.unsafe_convert(Ptr{cl.cl_context}, context),
                             Base.unsafe_convert(Ptr{cl.cl_command_queue}, queue)]
        GC.@preserve handles f(Base.unsafe_convert(Ptr{Ptr{Cvoid}}, handles))
    end
end

end # module
