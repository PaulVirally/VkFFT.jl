using Documenter
using VkFFT

const ROOT = dirname(@__DIR__)

"""
    _remote_kwargs(root::String)

Returns the `makedocs` keywords describing the repository the source links point into.

Documenter hangs its source and edit links off a specific commit and treats a
repository it cannot resolve one for as fatal. A checkout with no commits yet,
and a build from a source tarball with no `.git` at all, both still have to
produce a site, so the remote is passed only when git can answer and the links
are dropped when it cannot.
"""
function _remote_kwargs(root::String)
    resolvable = try
        success(pipeline(`git -C $root rev-parse --verify HEAD`, stdout=devnull, stderr=devnull))
    catch
        false
    end

    resolvable && return (; repo=Documenter.Remotes.GitHub("PaulVirally", "VkFFT.jl"))

    @info "No resolvable git commit in $root, so the site is built without source links."
    return (; remotes=nothing)
end

# doctest=false is not a stopgap. Every example in this site needs a CUDA,
# OpenCL or Metal device plus a built libvkfft, and the documentation is built
# on a runner with neither. The examples are static blocks mirroring cases in
# test/common/, which is where they run against FFTW for real.
#
# checkdocs=:exports because most docstrings in src/ sit on `_`-prefixed
# internals that have no place in a manual. The exported names are the ones
# that have to appear, and they all do.
makedocs(;
    _remote_kwargs(ROOT)...,
    sitename = "VkFFT.jl",
    authors = "Paul Virally",
    modules = [VkFFT],
    doctest = false,
    checkdocs = :exports,
    # repolink and edit_link are set rather than inferred, because inferring
    # them needs a git remote and _remote_kwargs above tolerates not having one.
    format = Documenter.HTML(prettyurls=get(ENV, "CI", "false") == "true",
                             canonical="https://paulvirally.github.io/VkFFT.jl",
                             repolink="https://github.com/PaulVirally/VkFFT.jl",
                             edit_link="main"),
    pages = [
        "Home" => "index.md",
        "Backends and capabilities" => "backends.md",
        "Transforms" => "transforms.md",
        "The plan interface" => "plans.md",
        "Tuning" => "tuning.md",
        "Limitations" => "limitations.md",
        "Migrating from VkFFTCUDA 0.2" => "migration.md",
        "Benchmarks" => "benchmarks.md",
        "API reference" => "api.md",
    ],
)

# Guarded so a local build stays quiet: outside CI there is no key to deploy
# with and nothing to deploy to. DOCUMENTER_KEY comes from the CI workflow.
if get(ENV, "CI", "false") == "true"
    deploydocs(repo="github.com/PaulVirally/VkFFT.jl.git", push_preview=true)
end
