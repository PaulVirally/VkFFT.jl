# The Metal half runs in a child process with its own project, because a process
# resolves libvkfft once and the wrapper is one library per backend. See env.jl.
include("env.jl")

@testset "metal" begin
    # Metal.jl needs Apple silicon, and the wrapper has to have been built.
    # Neither is worth failing the OpenCL suite over, so the reason is printed
    # instead.
    reason = ""
    if !Sys.isapple()
        reason = "this is not macOS"
    elseif Sys.ARCH !== :aarch64
        reason = "Metal.jl needs Apple silicon, and this is $(Sys.ARCH)"
    elseif !isfile(METAL_WRAPPER_PATH)
        reason = "no Metal wrapper at \"$METAL_WRAPPER_PATH\". Build it with `cmake -S libvkfft -B libvkfft/build-metal -DVKFFT_BACKEND=5 && cmake --build libvkfft/build-metal`, or point VKFFT_METAL_WRAPPER_PATH somewhere else"
    end

    if isempty(reason)
        @test _run_in_metal_project(joinpath(@__DIR__, "runtests.jl"), [VKFFT_DIR],
                                    ["AbstractFFTs", "FFTW", "LinearAlgebra", "Metal", "Preferences", "Random", "Test", "UUIDs"])
    else
        println("skipping the Metal suite: $reason")
    end
end
