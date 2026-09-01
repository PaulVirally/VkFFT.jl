# The trigger package end to end. A fresh project whose only packages are VkFFT,
# VkFFTMetal and the Preferences the wrapper path is set through, so a passing
# run says that `using VkFFTMetal` on its own activates the extension and plans
# and applies on this machine's GPU. Metal itself is not a direct dependency
# there: the arrays come through VkFFTMetal.

@testset "VkFFTMetal.jl" begin
    if isdir(VKFFT_METAL_DIR)
        @test _run_in_metal_project(joinpath(@__DIR__, "trigger_child.jl"), [VKFFT_DIR, VKFFT_METAL_DIR], ["Preferences", "UUIDs"])
    else
        println("skipping the VkFFTMetal.jl gate: no trigger package at \"$VKFFT_METAL_DIR\"")
    end
end
