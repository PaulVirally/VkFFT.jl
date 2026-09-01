# Aqua's package-quality checks. Nothing here touches a device or the wrapper,
# so the OpenCL runner is the only one that includes it. Running the same
# checks once per backend would give the same answer three times.
#
# Each check gets its own testset, so a failure names the check instead of
# arriving as one big "aqua" failure.

@testset verbose = true "aqua" begin
    # The check that justifies the dependency. Entry points stay
    # module-qualified and we do not define AbstractFFTs.plan_fft on GPU array
    # types, because that is type piracy and it is why v0.2 of the old package
    # needed __precompile__(false). This check enforces that instead of
    # trusting a convention.
    #
    # Methods on our own plan types are not piracy, whoever owns the function.
    # AbstractFFTs.adjoint_mul, AbstractFFTs.plan_inv, AbstractFFTs.output_size
    # and LinearAlgebra.mul! all dispatch on a VkFFT plan, so they are ours to
    # define and Aqua agrees.
    @testset "piracies" begin
        Aqua.test_piracies(VkFFT)
    end

    # The extension and weak-dependency graph gets rewritten every time the JLL
    # wiring moves. These two catch a missing compat entry or an orphaned
    # dependency when it appears, rather than at registration time.
    @testset "deps compat" begin
        Aqua.test_deps_compat(VkFFT)
    end

    @testset "stale deps" begin
        Aqua.test_stale_deps(VkFFT)
    end

    # Cheap to run, and there are five parameterized plan families with
    # overlapping method tables, which is exactly where an ambiguity hides.
    @testset "ambiguities" begin
        Aqua.test_ambiguities(VkFFT)
    end

    @testset "unbound args" begin
        Aqua.test_unbound_args(VkFFT)
    end

    @testset "undefined exports" begin
        Aqua.test_undefined_exports(VkFFT)
    end

    # Enabled even though it asserts nothing today. Test dependencies live in
    # test/Project.toml, so Project.toml has no [extras] section to
    # cross-check. It costs one call, and it starts saying something the moment
    # somebody adds an [extras] block and forgets the matching [targets] entry.
    @testset "project extras" begin
        Aqua.test_project_extras(VkFFT)
    end

    # __init__ reads one preference and does nothing else, so a task or a timer
    # surviving `using VkFFT` would be a real regression, and the kind that
    # makes Pkg.precompile hang for whoever depends on us. The check loads the
    # package in a fresh process, which is where most of this set's running
    # time goes.
    @testset "persistent tasks" begin
        Aqua.test_persistent_tasks(VkFFT)
    end
end
