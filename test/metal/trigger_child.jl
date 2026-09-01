# Run by trigger.jl in a project whose only direct dependencies are VkFFT and
# VkFFTMetal. That is the whole point: `using VkFFTMetal` has to activate the
# Metal extension and give a working plan with nothing else loaded, which is why
# there is no FFTW here and the ground truth is analytic.
include("env.jl")

using Preferences
using UUIDs

set_preferences!(UUID("65dc4606-9ae3-4b78-8734-204937373618"), "libvkfft_path" => METAL_WRAPPER_PATH; force=true)

using LinearAlgebra
using VkFFTMetal

const MtlArray = VkFFTMetal.Metal.MtlArray

failures = 0

"""
    check(name::String, ok::Bool)

Reports one result and counts the failures, since this runs without Test loaded.
"""
function check(name::String, ok::Bool)
    global failures += ok ? 0 : 1
    println(ok ? "[PASS] " : "[FAIL] ", name)
    return ok
end

n = 256
batch = 4

# The extension has to have activated, or everything below would be planning on
# a host array.
check("Metal extension active", Base.get_extension(VkFFT, :VkFFTMetalExt) !== nothing)
check("backend tag", VkFFT._backend(typeof(MtlArray{ComplexF32}(undef, n))) === Val(:metal))
check("Metal wrapper", VkFFT._backend_id() == 5)

# The transform of a delta is flat, and its inverse brings the delta back. No
# FFTW needed to say whether that happened.
host = zeros(ComplexF32, n, batch)
host[1, :] .= 1

x = MtlArray{ComplexF32, 2}(undef, (n, batch))
copyto!(x, host)

forward = VkFFT.plan_fft(x, 1)
check("plan type", forward isa VkFFTPlan{ComplexF32, 2, false, :metal, 1})

spectrum = Array(forward * x)
check("delta transforms flat", maximum(abs.(spectrum .- 1)) < 1e-5)

y = forward * x
back = Array(inv(forward) * y)
check("round trip", maximum(abs.(back .- host)) < 1e-5)

# One real transform too, so the r2c path is covered by the gate as well.
r = MtlArray{Float32, 1}(undef, n)
copyto!(r, ones(Float32, n))
rspectrum = Array(VkFFT.plan_rfft(r, 1) * r)
check("rfft of a constant", abs(rspectrum[1] - n) < 1e-3 && maximum(abs.(rspectrum[2:end])) < 1e-3)

println(failures == 0 ? "trigger package OK" : "$failures trigger checks failed")
exit(failures == 0 ? 0 : 1)
