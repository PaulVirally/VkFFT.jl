# The Metal half of the suite cannot share a process with the OpenCL half. VkFFT
# resolves libvkfft once, on the first call that needs it, and the wrapper is
# one shared library per backend, so a process is either an OpenCL process or a
# Metal one. Everything Metal therefore runs in a child with its own project,
# and this file builds those projects. It is included by gate.jl, which spawns
# the child, and by the Metal runner, which spawns a grandchild for the trigger
# package.

const VKFFT_DIR = normpath(joinpath(@__DIR__, "..", ".."))
const VKFFT_METAL_DIR = normpath(joinpath(VKFFT_DIR, "..", "VkFFTMetal.jl"))

# Mirrors the VKFFT_WRAPPER_PATH convention of the OpenCL runner, under a
# separate name so that this child does not inherit the parent's OpenCL path and
# load the wrong backend. The default is the in-tree Metal build, which is where
# `cmake -B build-metal -DVKFFT_BACKEND=5` leaves it.
#
# VKFFT_CUDA_WRAPPER_PATH belongs to the CUDA runner, which never shares a
# process or an environment with this one, so seeing it here means the caller
# meant a different runner. VKFFT_WRAPPER_PATH is not an error, since this file
# is loaded in the OpenCL process too and that variable is the OpenCL one.
include(joinpath(@__DIR__, "..", "wrapper_env.jl"))
_reject_foreign_wrapper_var("VKFFT_METAL_WRAPPER_PATH", ["VKFFT_CUDA_WRAPPER_PATH"])

const METAL_WRAPPER_PATH = get(ENV, "VKFFT_METAL_WRAPPER_PATH", normpath(joinpath(VKFFT_DIR, "..", "libvkfft", "build-metal", "libvkfft.dylib")))

"""
    _child_cmd(cmd::Cmd)

Returns a julia command that reads its project from `--project` and nothing else.

`Pkg.test` runs the suite with `JULIA_LOAD_PATH` pointing at its sandbox and no
`@stdlib` entry, and a child would inherit that and fail to load so much as
`Pkg`. Clearing both variables leaves the child with the default load path
around whatever `--project` it was given.
"""
_child_cmd(cmd::Cmd) = addenv(cmd, "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing)

"""
    _run_in_metal_project(script::String, devs::Vector{String}, packages::Vector{String})

Runs a script in a throwaway project and returns whether it exited cleanly.

The project holds the given local packages developed and the given registered
ones added. Resolution runs in a child process, because activating a project
mutates the process doing it and the caller still has its own suite to run.

# Arguments
- `script::String`: Path of the file to run
- `devs::Vector{String}`: Paths of local packages to develop
- `packages::Vector{String}`: Names of registered packages to add

# Returns
- `true` when the child exited with status 0
"""
function _run_in_metal_project(script::String, devs::Vector{String}, packages::Vector{String})
    dir = mktempdir()
    lines = ["using Pkg",
             "Pkg.develop([$(join(("PackageSpec(path=raw\"$d\")" for d in devs), ", "))]; io=devnull)"]
    isempty(packages) || push!(lines, "Pkg.add(String[$(join(("\"$p\"" for p in packages), ", "))]; io=devnull)")
    run(_child_cmd(`$(Base.julia_cmd()) --startup-file=no --project=$dir -e $(join(lines, '\n'))`))

    cmd = _child_cmd(`$(Base.julia_cmd()) --startup-file=no --project=$dir $script`)
    return success(pipeline(ignorestatus(cmd), stdout=stdout, stderr=stderr))
end
