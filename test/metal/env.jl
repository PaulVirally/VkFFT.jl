# Builds the throwaway project trigger.jl runs the trigger package in.

const VKFFT_DIR = normpath(joinpath(@__DIR__, "..", ".."))
const VKFFT_METAL_DIR = normpath(joinpath(VKFFT_DIR, "..", "VkFFTMetal.jl"))

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
    _run_in_metal_project(script::String, devs::Vector{String})

Runs a script in a throwaway project and returns whether it exited cleanly.

The project holds the given local packages developed. Resolution runs in a
child process, because activating a project mutates the process doing it and
the caller still has its own suite to run.

# Arguments
- `script::String`: Path of the file to run
- `devs::Vector{String}`: Paths of local packages to develop

# Returns
- `true` when the child exited with status 0
"""
function _run_in_metal_project(script::String, devs::Vector{String})
    dir = mktempdir()
    setup = "using Pkg; Pkg.develop([$(join(("PackageSpec(path=raw\"$d\")" for d in devs), ", "))]; io=devnull)"
    run(_child_cmd(`$(Base.julia_cmd()) --startup-file=no --project=$dir -e $setup`))

    cmd = _child_cmd(`$(Base.julia_cmd()) --startup-file=no --project=$dir $script`)
    return success(pipeline(ignorestatus(cmd), stdout=stdout, stderr=stderr))
end
