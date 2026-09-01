# The Metal runner. It supplies Metal's capability table, its uploader and its
# device array type, then includes the backend-parameterized sets in common/.
# The two pure-CPU sets (regions.jl and planner.jl) belong to the OpenCL runner,
# since neither touches a device.
#
# gate.jl spawns this in a project of its own (see env.jl for why it cannot be
# the same process), and it is runnable by hand in any project that has VkFFT,
# Metal and FFTW:
#
#     julia --project=<that project> VkFFT.jl/test/metal/runtests.jl
#
# The wrapper it loads is the Metal build, from VKFFT_METAL_WRAPPER_PATH or the
# in-tree default. As in the OpenCL runner the preference is set here, before
# VkFFT is loaded, rather than checked in: the path is absolute and
# machine-specific, and VkFFT reads the preference in __init__ rather than at
# precompile time.
include("env.jl")

using Preferences
using UUIDs

set_preferences!(UUID("65dc4606-9ae3-4b78-8734-204937373618"), "libvkfft_path" => METAL_WRAPPER_PATH; force=true)

using AbstractFFTs
using FFTW
using LinearAlgebra
using Metal
using Random
using Test
using VkFFT

# The tuner writes its records to disk, so this runner gets a throwaway cache
# directory for the same reasons the OpenCL one does.
VkFFT.CACHE_DIR[] = mktempdir()

const DeviceArray{T, N} = MtlArray{T, N}

# Metal has no double, so no fp64 and no double-double quad either. Tolerances
# measured on an M3 Pro against FFTW: worst element-wise relative error 1.4e-5
# in fp32, at 16381 through Bluestein. Half precision is storage only, so its
# accuracy comes from the 11-bit mantissa: worst element-wise relative error
# 1.1e-3 over the c2c, rfft and r2r families at sizes up to 16381, on the double
# round trips that also carry a half-precision scale factor, and 6.5e-4 on any
# single transform.
#
# One 1d size per codegen path: 64 power of two, 8192 multi-upload power of two,
# 17 small prime, 1021 single-upload Bluestein, 120 mixed radix, 16381
# multi-upload Bluestein.
const BACKEND = (name = :metal,
                 device_name = String(Metal.device().name),
                 fp64 = false,
                 fp16 = true,
                 quad = false,
                 offset_views = false,
                 pin_conv_bluestein = false,
                 rtol_f32 = 1e-4,
                 rtol_f64 = 0.0,
                 rtol_f16 = 5e-3,
                 sizes_1d = (64, 8192, 17, 1021, 120, 16381),
                 sizes_1d_fp64_only = ())

"""
    _upload(h::Array)

Copies a host array into a fresh MtlArray.
"""
function _upload(h::Array{T, N}) where {T, N}
    x = DeviceArray{T, N}(undef, size(h))
    copyto!(x, h)
    return x
end

"""
    _offset(x::MtlArray)

Returns the byte offset of an array into the MTLBuffer it shares.
"""
_offset(x::MtlArray) = pointer(x).offset

const COMMON = normpath(joinpath(@__DIR__, "..", "common"))

@testset verbose = true "VkFFT.jl on Metal" begin
    @testset "library" begin
        @test VkFFT._max_dims() == VkFFT.VKFFT_MAX_FFT_DIMENSIONS
        @test VkFFT._backend_id() == 5
        @test VkFFT._vkfft_config_size() == sizeof(VkFFT.VkFFTConfig)
        @test Base.get_extension(VkFFT, :VkFFTMetalExt) !== nothing
        println("device: ", Metal.device().name)
        println("wrapper: ", METAL_WRAPPER_PATH)
    end

    include(joinpath(COMMON, "harness.jl"))
    include(joinpath(COMMON, "errors.jl"))
    include(joinpath(COMMON, "fft.jl"))
    include(joinpath(COMMON, "rfft.jl"))
    include(joinpath(COMMON, "r2r.jl"))
    include(joinpath(COMMON, "zeropad.jl"))
    include(joinpath(COMMON, "conv.jl"))
    include(joinpath(COMMON, "fp16.jl"))
    include(joinpath(COMMON, "unsafe.jl"))
    include(joinpath(COMMON, "tuner.jl"))
    include(joinpath(COMMON, "interface.jl"))

    include("metal.jl")
    include("trigger.jl")
end
