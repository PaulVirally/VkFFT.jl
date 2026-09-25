# Entry point. One body of testsets lives in common/, parameterized on a
# capability table, and each of opencl/, metal/ and cuda/ is a thin runner that
# supplies that table plus an uploader and a device array type.
#
# The OpenCL and Metal runners share this process, each in a module of its own
# so that their tables and helpers do not collide. The Metal one needs a
# functional Metal and is skipped otherwise. The CUDA runner has its own project
# and is invoked by hand on a machine with an NVIDIA card:
#
#     julia --project=VkFFT.jl/test/cuda VkFFT.jl/test/cuda/runtests.jl
#
# Every wrapper comes from its JLL unless VKFFT_WRAPPER_PATH names one you
# built. VkFFT uses it for the backend it was built for:
#
#     VKFFT_WRAPPER_PATH=/path/to/libvkfft.dylib julia --project -e 'using Pkg; Pkg.test()'
using FFTW
using Metal
using Preferences
using Test
using VkFFT

# `nothing` deletes the preference, so a stale one in LocalPreferences.toml
# cannot shadow the JLLs.
set_preferences!(VkFFT, "libvkfft_path" => get(ENV, "VKFFT_WRAPPER_PATH", nothing); force=true)

@testset verbose = true "VkFFT.jl" begin
    @eval module OpenCLRunner include($(joinpath(@__DIR__, "opencl", "runtests.jl"))) end

    if Metal.functional()
        @eval module MetalRunner include($(joinpath(@__DIR__, "metal", "runtests.jl"))) end

        # Evaluated so that it runs in a world that can see the two modules.
        @eval @testset "OpenCL and Metal interleaved" begin
            for runners in ((OpenCLRunner, MetalRunner), (MetalRunner, OpenCLRunner)), R in runners
                h = rand(ComplexF32, 48, 4)
                x = R._upload(h)
                @test R._relmax(VkFFT.plan_fft(x, 1) * x, fft(h, 1)) < 1e-4
            end
        end
    else
        println("skipping the Metal suite: Metal is not functional here")
    end
end
