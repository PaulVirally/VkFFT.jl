# The CUDA runner. It supplies CUDA's capability table, its uploader and its
# device array type, then includes the backend-parameterized sets in common/.
# The two pure-CPU sets (regions.jl and planner.jl) belong to the OpenCL runner,
# since neither touches a device.
#
# This one is invoked by hand, on a machine with an NVIDIA card. Once per
# checkout, from VkFFT.jl:
#
#     julia --project=test/cuda -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#
# and then:
#
#     julia --project=test/cuda test/cuda/runtests.jl
#
# The wrapper comes from VkFFT_CUDA_jll. To test one you built, name it in
# VKFFT_WRAPPER_PATH:
#
#     VKFFT_WRAPPER_PATH=/path/to/libvkfft.so julia --project=test/cuda test/cuda/runtests.jl
using AbstractFFTs
using CUDA
using FFTW
using LinearAlgebra
using Preferences
using Random
using Test
using VkFFT
using VkFFT_CUDA_jll

# `nothing` deletes the preference, so a stale one cannot shadow the JLL.
set_preferences!(VkFFT, "libvkfft_path" => get(ENV, "VKFFT_WRAPPER_PATH", nothing); force=true)

# The tuner writes its records to disk, so this runner gets a throwaway cache
# directory for the same reasons the OpenCL one does.
VkFFT.CACHE_DIR[] = mktempdir()

const DeviceArray{T, N} = CuArray{T, N}

# An A6000 runs every precision VkFFT has, half and double-double included, and
# every transform family. Tolerances are the OpenCL ones, since the arithmetic
# is the same and the reference is FFTW either way. The half-precision one
# matches Metal's, since fp16 is a storage format in both.
#
# fp16 is the one capability read off the wrapper rather than off the card.
# VkFFT compiles half-precision kernels around an include of cuda_fp16.h from
# the path baked into libvkfft, so a wrapper built with an empty
# CUDA_TOOLKIT_ROOT_DIR (which is what the Yggdrasil recipe does) cannot run
# one, and the planner refuses it. A locally built wrapper bakes a real path
# and every half-precision case runs. A wrapper too old to export the accessor
# answers nothing, which is not the empty string and so counts as capable.
#
# One 1d size per codegen path: 64 power of two, 8192 multi-upload power of two,
# 17 small prime, 1021 single-upload Bluestein, 120 mixed radix, 16381
# multi-upload Bluestein. 8191 and 4093 run at every precision here: they are
# fp64 only on pocl, whose fp32 kernel generation SIGBUSes at both, and that is
# a pocl defect rather than a property of the sizes.
#
# offset_views is the one capability that flips a set off rather than on. A
# contiguous view of a CuArray is another CuArray whose pointer carries the
# offset, so the refusals the other backends assert do not apply, and the
# positive case is in cuda.jl instead.
const BACKEND = (name = :cuda,
                 device_name = CUDA.name(CUDA.device()),
                 fp64 = true,
                 fp16 = VkFFT._cuda_toolkit_root() != "",
                 quad = true,
                 offset_views = true,
                 pin_conv_bluestein = false,
                 rtol_f32 = 1e-4,
                 rtol_f64 = 1e-12,
                 rtol_f16 = 5e-3,
                 sizes_1d = (64, 8192, 17, 1021, 120, 16381, 8191, 4093),
                 sizes_1d_fp64_only = ())

"""
    _upload(h::Array)

Copies a host array into a fresh CuArray.
"""
function _upload(h::Array{T, N}) where {T, N}
    x = DeviceArray{T, N}(undef, size(h))
    copyto!(x, h)
    return x
end

# There is no `_offset` here on purpose. The other two runners define it so the
# common sets can say "this view really is offset" before asserting that the
# planner refuses it. A CuArray carries its offset in its own pointer and never
# names its parent, so the offset is only recoverable with the parent in hand,
# and offset_views = true means no common set ever asks.

const COMMON = normpath(joinpath(@__DIR__, "..", "common"))

@testset verbose = true "VkFFT.jl on CUDA" begin
    @testset "library" begin
        # The device, the wrapper and the toolkit root print above the
        # assertions, since every way this testset fails is a question about
        # which library got loaded.
        println("device: ", CUDA.name(CUDA.device()), ", capability: ", CUDA.capability(CUDA.device()))
        println("wrapper: ", VkFFT.library_path(:cuda))
        println("cuda toolkit root: ", repr(VkFFT._cuda_toolkit_root()))

        # The toolkit root accessor came after the other ten symbols, so a
        # wrapper built before it answers nothing, and `nothing isa String` on
        # its own reads as a type error rather than as a stale library.
        if VkFFT._cuda_toolkit_root() === nothing
            println("the wrapper at $(VkFFT.library_path(:cuda)) predates vkfft_cuda_toolkit_root, so its fp16 capability cannot be read off it and the assertion below fails. Rebuild it with remote/1_build_wrapper.sh. The assertion is strict here on purpose, since this runner should be pointed at a current wrapper.")
        end

        lib = VkFFT._library(:cuda)
        @test VkFFT._vkfft_max_dims(lib) == VkFFT.VKFFT_MAX_FFT_DIMENSIONS
        @test VkFFT._vkfft_backend(lib) == 1
        @test VkFFT._vkfft_config_size(lib) == sizeof(VkFFT.VkFFTConfig)
        @test Base.get_extension(VkFFT, :VkFFTCUDAExt) !== nothing
        @test VkFFT._cuda_toolkit_root() isa String
    end

    include(joinpath(COMMON, "harness.jl"))
    include(joinpath(COMMON, "errors.jl"))
    include(joinpath(COMMON, "fft.jl"))
    include(joinpath(COMMON, "rfft.jl"))
    include(joinpath(COMMON, "r2r.jl"))
    include(joinpath(COMMON, "zeropad.jl"))
    include(joinpath(COMMON, "conv.jl"))
    if BACKEND.fp16
        include(joinpath(COMMON, "fp16.jl"))
    else
        _skip("the fp16 set", "this libvkfft was built with an empty CUDA_TOOLKIT_ROOT_DIR")
    end
    include(joinpath(COMMON, "unsafe.jl"))
    include(joinpath(COMMON, "tuner.jl"))
    include(joinpath(COMMON, "interface.jl"))

    include("cuda.jl")
end
