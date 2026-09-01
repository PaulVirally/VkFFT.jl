# The OpenCL runner. It supplies pocl's capability table, its uploader and its
# device array type, then includes the backend-parameterized sets in common/. It
# is also the only runner that includes the three pure-CPU sets, regions.jl,
# planner.jl and aqua.jl, since none of them touches a device and running them
# once per backend would give the same answer three times.
#
# The libvkfft_path preference is set here, before VkFFT is loaded, rather than
# checked into test/LocalPreferences.toml: the wrapper path is absolute and
# machine-specific, and VkFFT reads the preference in __init__ rather than at
# precompile time, so setting it in-process is enough. Override it with
# VKFFT_WRAPPER_PATH to test another build:
#
#     VKFFT_WRAPPER_PATH=/path/to/artifact/lib/libvkfft_icd.dylib \
#         julia --project=test -e 'using Pkg; Pkg.test()'
#
# VKFFT_CUDA_WRAPPER_PATH is the CUDA runner's variable and does nothing here,
# so setting it is an error rather than a silent fallback to the default
# wrapper. See test/wrapper_env.jl. VKFFT_METAL_WRAPPER_PATH is not an error,
# because the Metal child is spawned from this process and one run can point
# each half at a different build.
#
# The wrapper must be the ICD-linked build: Apple's OpenCL framework is not an
# ICD, so a wrapper linked straight against it is invisible to OpenCL.jl's
# loader and every create call comes back as
# VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE.
include(joinpath(@__DIR__, "..", "wrapper_env.jl"))
_reject_foreign_wrapper_var("VKFFT_WRAPPER_PATH", ["VKFFT_CUDA_WRAPPER_PATH"])

using Preferences
using UUIDs

const VKFFT_UUID = UUID("65dc4606-9ae3-4b78-8734-204937373618")
const OPENCL_UUID = UUID("08131aa3-fb12-5dee-8b74-c09406e224a2")
const WRAPPER_PATH = get(ENV, "VKFFT_WRAPPER_PATH", normpath(joinpath(@__DIR__, "..", "..", "..", "probes", "build-icd", "libvkfft_icd.dylib")))

set_preferences!(VKFFT_UUID, "libvkfft_path" => WRAPPER_PATH; force=true)
set_preferences!(OPENCL_UUID, "default_memory_backend" => "buffer"; force=true)

using AbstractFFTs
using Aqua
using FFTW
using LinearAlgebra
using OpenCL
using Random
using Test
using VkFFT
using pocl_jll

# VkFFT drives clSetKernelArg with a cl_mem, so OpenCL.jl has to hand out
# cl.Buffer-backed arrays. The preference above covers freshly started tasks.
# This covers this one, whose memory backend may already have been resolved.
task_local_storage(:CLMemoryBackend, cl.BufferBackend())

# The tuner writes its records to disk. Pointing them at a throwaway directory
# makes every run a cold run, and it keeps the suite from writing into, or
# clearing, the records of whoever is running it.
VkFFT.CACHE_DIR[] = mktempdir()

const DeviceArray{T, N} = CLArray{T, N, cl.Buffer}

# Tolerances measured on pocl against FFTW: worst element-wise relative error
# 2.5e-5 in fp32 and 4.9e-14 in fp64 at sizes up to 16k. The half-precision one
# is carried for a device that reports cl_khr_fp16, which pocl does not.
#
# 8191 and 4093 are fp64 only: pocl's fp32 kernel generation SIGBUSes at both
# (reproduced in plain C, so it is neither VkFFT's nor OpenCL.jl's fault). The
# other 1d sizes are one per VkFFT codegen path: 64 power of two, 17 small
# prime, 1021 single-upload Bluestein, 120 mixed radix, 16381 multi-upload
# Bluestein.
const BACKEND = (name = :opencl,
                 device_name = cl.device().name,
                 fp64 = true,
                 fp16 = "cl_khr_fp16" in cl.device().extensions,
                 quad = true,
                 offset_views = false,
                 pin_conv_bluestein = true,
                 rtol_f32 = 1e-4,
                 rtol_f64 = 1e-12,
                 rtol_f16 = 5e-3,
                 sizes_1d = (64, 17, 1021, 120, 16381),
                 sizes_1d_fp64_only = (8191, 4093))

"""
    _upload(h::Array)

Copies a host array into a fresh cl.Buffer-backed CLArray.
"""
function _upload(h::Array{T, N}) where {T, N}
    x = DeviceArray{T, N}(undef, size(h))
    copyto!(x, h)
    return x
end

"""
    _offset(x::CLArray)

Returns the element offset of an array into the buffer it shares.
"""
_offset(x::CLArray) = x.offset

const COMMON = normpath(joinpath(@__DIR__, "..", "common"))

@testset verbose = true "opencl" begin
    @testset "library" begin
        @test VkFFT._max_dims() == VkFFT.VKFFT_MAX_FFT_DIMENSIONS
        @test VkFFT._backend_id() == 3
        @test VkFFT._vkfft_config_size() == sizeof(VkFFT.VkFFTConfig)
        @test Base.get_extension(VkFFT, :VkFFTOpenCLExt) !== nothing

        # The eleventh symbol. Only a CUDA build ever bakes a path, so this
        # one reports the empty string, and the point of the assertion is that
        # the accessor answers at all rather than raising.
        @test VkFFT._vkfft_cuda_toolkit_root() isa String
        @test VkFFT._cuda_toolkit_root() == ""
        println("platform: ", cl.platform().name, ", device: ", cl.device().name)
        println("wrapper: ", WRAPPER_PATH)
    end

    include(joinpath(COMMON, "harness.jl"))
    include(joinpath(COMMON, "aqua.jl"))
    include(joinpath(COMMON, "regions.jl"))
    include(joinpath(COMMON, "planner.jl"))
    include(joinpath(COMMON, "errors.jl"))
    include(joinpath(COMMON, "fft.jl"))
    include(joinpath(COMMON, "rfft.jl"))
    include(joinpath(COMMON, "r2r.jl"))
    include(joinpath(COMMON, "zeropad.jl"))
    include(joinpath(COMMON, "conv.jl"))
    if BACKEND.fp16
        include(joinpath(COMMON, "fp16.jl"))
    else
        _skip("the fp16 set", "$(cl.device().name) does not report cl_khr_fp16")
    end
    include(joinpath(COMMON, "unsafe.jl"))
    include(joinpath(COMMON, "tuner.jl"))
    include(joinpath(COMMON, "interface.jl"))

    include("opencl.jl")
end
