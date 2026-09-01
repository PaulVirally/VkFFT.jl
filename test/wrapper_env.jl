# Each runner reads a wrapper path variable of its own: VKFFT_WRAPPER_PATH for
# OpenCL, VKFFT_METAL_WRAPPER_PATH for Metal, VKFFT_CUDA_WRAPPER_PATH for CUDA.
# The three names differ because the Metal suite runs in a child of the OpenCL
# process and inherits its environment, so one shared name would hand that child
# an OpenCL wrapper and load the wrong backend.
#
# The cost of that separation is that a variable meant for one runner is
# invisible to another, which reads its own name, finds nothing and falls back
# to its default. This file is the guard against that. A runner includes it
# before it resolves its wrapper path and refuses to start when it can see a
# variable belonging to a runner other than itself.

# The OpenCL runner and the Metal child it spawns within the same process (see
# metal/env.jl) both include this file, so a second include must not
# redefine the function below. A top-level `return` does not skip the rest of
# an included file, so the guard has to wrap the definition itself. The
# `begin` is there only so the docstring still attaches to the function,
# which it does not do when placed directly inside an `if`.
if !@isdefined(_reject_foreign_wrapper_var)
    begin
        """
            _reject_foreign_wrapper_var(own::String, foreign::Vector{String})

        Errors when a wrapper path variable belonging to another runner is set and this runner's own is not.

        Only pairs that cannot both appear in a legitimate run belong in `foreign`.
        VKFFT_WRAPPER_PATH and VKFFT_METAL_WRAPPER_PATH do appear together, because the
        Metal child is spawned from the OpenCL process and one run can point each half
        at a different build. VKFFT_CUDA_WRAPPER_PATH appears with neither, because the
        CUDA runner is invoked by hand in a project nothing else uses.

        # Arguments
        - `own::String`: Name of the variable this runner reads
        - `foreign::Vector{String}`: Names of the variables no legitimate run of this runner carries

        # Returns
        - `nothing`
        """
        function _reject_foreign_wrapper_var(own::String, foreign::Vector{String})
            haskey(ENV, own) && return nothing
            for name in foreign
                haskey(ENV, name) || continue
                error("$name is set and $own is not. This runner reads $own, so $name would be ignored and the suite would run against the default wrapper rather than the one you named. The two names are separate on purpose: the Metal suite runs in a child of the OpenCL process and inherits its environment, so a single shared name would point that child at another backend's wrapper. Set $own to the wrapper you meant, or unset $name.")
            end
            return nothing
        end
    end
end
