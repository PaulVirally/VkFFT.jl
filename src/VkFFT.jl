module VkFFT

export VkFFTPlan, VkFFTRealPlan, VkFFTR2RPlan, VkFFTConvPlan, VkFFTUnsafePlan, VkFFTConfig, VkFFTError

using AbstractFFTs
using LinearAlgebra
using PrecompileTools
using Preferences
using Scratch

import SHA

include("errors.jl")
include("libinterface.jl")
include("regions.jl")
include("diskcache.jl")
include("plans.jl")
include("realplans.jl")
include("r2rplans.jl")
include("convplans.jl")
include("unsafeplans.jl")
include("tuner.jl")

"""
    __init__()

Reads the `libvkfft_path` preference into the session.

Reading it here rather than with `@load_preference` at precompile time keeps it
out of the precompile cache, so pointing the package at a different wrapper
build does not trigger a recompilation, making a missing wrapper a first-use
error rather than a load failure.
"""
function __init__()
    global libvkfft = load_preference(@__MODULE__, "libvkfft_path", "")
    return nothing
end

# Only the pure planning path. The C library may not be installed at precompile
# time, and a ccall into it here would make precompilation depend on it.
@setup_workload begin
    @compile_workload begin
        for dims in ((64,), (8, 8), (4, 5, 6), (2, 3, 4, 5, 6))
            n = length(dims)
            full = _canonical_region(ntuple(identity, n), n)
            _map_region(dims, full)
            _map_region(dims, _canonical_region(1, n))
            _map_region(dims, _canonical_region((n,), n))

            _check_nonempty_region(full, "plan_rfft(x, 1)")
            _check_real_region(dims, full)
            AbstractFFTs.rfft_output_size(dims, full)
            AbstractFFTs.brfft_output_size(AbstractFFTs.rfft_output_size(dims, full), dims[1], full)

            for type in 1:4
                _check_r2r(dims, full, type, "plan_dct")
                _r2r_transpose(:dct, type)
                _r2r_transpose(:dst, type)
            end
            _check_zeropad(dims, full, (1, dims[1]))
            _zeropad_fields((1, dims[1]))

            # A convolution takes one or two transformed axes and no more, so
            # the workload gives it its own regions rather than the full one.
            leading = _canonical_region(ntuple(identity, Val(min(n, 2))), n)
            _check_conv_region(dims, leading, _map_region(dims, leading))
            _conv_layout(_map_region(dims, leading))
            _conv_layout(_map_region(dims, _canonical_region(1, n)))
        end
        _canonical_region(1:2, 3)
        _canonical_region([2, 1], 3)
        _canonical_zeropad(2:5)
        _canonical_zeropad((2:5,))

        # The user-facing config constructor, which VkFFT.unsafe_plan callers
        # reach for directly.
        zeroed = ntuple(_ -> UInt64(0), VKFFT_MAX_FFT_DIMENSIONS)
        VkFFTConfig(fft_dim=1, size=ntuple(i -> i == 1 ? UInt64(64) : UInt64(0), VKFFT_MAX_FFT_DIMENSIONS))
        VkFFTConfig(fft_dim=2, size=ntuple(i -> i <= 2 ? UInt64(8) : UInt64(0), VKFFT_MAX_FFT_DIMENSIONS),
                    omit=zeroed, precision=2, dct=2, normalize=1, make_forward_only=1)
    end
end

end # module
