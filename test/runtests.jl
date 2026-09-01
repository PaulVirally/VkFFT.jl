# Entry point. One body of testsets lives in common/, parameterized on a
# capability table, and each of opencl/, metal/ and cuda/ is a thin runner that
# supplies that table plus an uploader and a device array type.
#
# This process runs the OpenCL half, because Pkg.test's project is the OpenCL
# one. The Metal half runs in a child, since a process resolves libvkfft once
# and the wrapper is one shared library per backend (see metal/env.jl). The CUDA
# runner has its own project and is invoked by hand on a machine with an NVIDIA
# card:
#
#     julia --project=VkFFT.jl/test/cuda VkFFT.jl/test/cuda/runtests.jl
using Test

@testset verbose = true "VkFFT.jl" begin
    include("opencl/runtests.jl")
    include("metal/gate.jl")
end
