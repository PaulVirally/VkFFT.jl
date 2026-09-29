# The entry point.
#
#     julia --project=benchmarks benchmarks/run.jl --backend=fftw --reps=5 --out=results/
#     julia --project=benchmarks/envs/metal benchmarks/run.jl --backend=metal --reps=5 --out=results/
#     julia --project=benchmarks/envs/opencl benchmarks/run.jl --backend=opencl --device="Portable Computing Language" --reps=5 --out=results/
#     julia --project=benchmarks/envs/cuda benchmarks/run.jl --backend=cuda --reps=5 --out=results/
#
# Without --replication this is the driver: it builds the case list, prints what
# the run will cost, and spawns one fresh process per replication. Sample to
# sample jitter inside one process is the small variance. The one that decides
# whether a number reproduces is the spread between separate processes, with
# different allocation addresses, different plan choices and a different thermal
# starting point, so the replications have to be separate processes.
#
# With --replication=k it is that one replication, in process. With --accuracy it
# is the accuracy pass and with --tuning the tuner's grid sweep, and both of
# those run once for the whole run: the same input gives bit identical output on
# every repeat of it, and a grid measured twice is the same grid.

# This script runs in exactly one environment, so it pins it rather than
# trusting --project. A startup.jl that activates whatever project the working
# directory holds overrides the flag, and the documented invocation from the
# package root then dies on a missing dependency before the first case.
#
# Which environment that is comes out of --backend, which therefore has to be
# read before anything else. The GPU packages cannot share one project: Metal is
# macOS only, CUDA.jl is not installable on a Mac, and VkFFT compiles one
# backend into the wrapper anyway. Each device backend has a standalone
# environment under envs/ and fftw runs in benchmarks/ itself.
import Pkg
const BACKEND = let asked = filter(!isnothing, match.(r"^--backend=(.+)$", ARGS))
    isempty(asked) ? :fftw : Symbol(last(asked)[1])
end
const ENV_DIR = joinpath(@__DIR__, "envs", String(BACKEND))
BACKEND === :fftw || isdir(ENV_DIR) || error("the $BACKEND backend runs in benchmarks/envs/$BACKEND, and there is no such directory. It holds the measurement dependencies plus that backend's own packages, developed against the local clones of VkFFT.jl and its trigger package.")
Pkg.activate(BACKEND === :fftw ? (@__DIR__) : ENV_DIR)

include("cases.jl")
include("backends.jl")
include("timing.jl")
include("record.jl")
include("accuracy.jl")
BACKEND === :fftw || include("devices.jl")

using Random

const ORDER_SEED = 0x4f_52_44_52 # the case order is randomized per replication so drift decorrelates from size

"""
    main(argv)

Parses the flags and either drives the replications or runs one of them.

Machine identity, CPU, OS and package versions are detected rather than passed
in, so a mistyped argument cannot mislabel a run.

# Arguments
- `argv`: The command line, understanding `--backend`, `--device`, `--reps`, `--out`, `--max-elements`, `--dry-run`, `--replication`, `--accuracy` and `--tuning`

# Returns
- Nothing
"""
function main(argv)
    opts = Dict("backend" => "fftw", "device" => "", "reps" => "5", "out" => "results", "max-elements" => "", "dry-run" => "false", "replication" => "", "accuracy" => "false", "tuning" => "false")
    for arg in argv
        matched = match(r"^--([a-z-]+)(?:=(.*))?$", arg)
        matched === nothing && throw(ArgumentError("cannot read the argument $arg. Flags look like --backend=fftw"))
        haskey(opts, matched[1]) || throw(ArgumentError("there is no --$(matched[1]) flag. The flags are $(join(sort(collect(keys(opts))), ", "))"))
        opts[matched[1]] = matched[2] === nothing ? "true" : matched[2]
    end

    backend = Symbol(opts["backend"])
    out = abspath(opts["out"])
    # A device backend picks its device before anything else, since --device is
    # what chooses between an ICD's platforms, and reports back the name and
    # whatever version it exposes for the manifest. An fftw run labels itself
    # with the CPU it ran on.
    detected = backend === :fftw ? Dict{String, Any}() : detect_device(opts["device"])
    device = get(detected, "device", isempty(opts["device"]) ? Sys.cpu_info()[1].model : opts["device"])
    # --max-elements shortens the shape list without touching the capability
    # table, which is how a new machine gets a sweep it can watch finish before
    # it gets the hour long one.
    caps = capabilities(backend) # after detect_device, since an OpenCL cap depends on which device was picked
    all_cases = cases(backend, isempty(opts["max-elements"]) ? caps : (types=caps.types, max_elements=min(caps.max_elements, parse(Int, opts["max-elements"])), cliff=caps.cliff, tuning=caps.tuning))
    impls = implementations(backend)

    if !isempty(opts["replication"])
        mkpath(out)
        _run_replication(backend, impls, all_cases, out, parse(Int, opts["replication"]), device, detected)
        return nothing
    end

    if opts["accuracy"] == "true"
        mkpath(out)
        run_accuracy(impls, all_cases, out)
        return nothing
    end

    if opts["tuning"] == "true"
        caps.tuning || throw(ArgumentError("the $backend backend has no tuner, so --tuning has nothing to sweep. The tuner belongs to VkFFT and an fftw sweep holds no VkFFT plan to turn it on for."))
        mkpath(out)
        run_tuning(filter(c -> c.tuned, all_cases), out)
        return nothing
    end

    reps = parse(Int, opts["reps"])
    # The ceiling counts the implementations that actually run each case rather
    # than the whole tuple, since FFTW declines the half precision rows a Metal
    # sweep enumerates and the vendor declines every tuned one.
    duels = sum(count(impl -> supports(impl, case), impls) for case in all_cases; init=0)
    @printf("%s on %s: %d cases, %d implementations, %d duels, at most %.1f min of timing per replication\n", backend, device, length(all_cases), length(impls), duels, duels * BUDGET / 60)
    if opts["dry-run"] == "true"
        for case in all_cases
            println("  ", case_id(case))
        end
        return nothing
    end

    mkpath(out)
    # Before the replications, not after. The grid the tuner saw and the
    # parameters the timed plans run with then come from the same sweep, and no
    # replication pays for a sweep inside a duel.
    if caps.tuning
        println("the tuner's grid, in a fresh process")
        run(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $(@__FILE__) --backend=$(opts["backend"]) --device=$(opts["device"]) --out=$out --max-elements=$(opts["max-elements"]) --tuning`)
    end

    for k in 1:reps
        println("replication $k of $reps, in a fresh process")
        # --startup-file=no explicitly: a startup.jl that activates whatever
        # project the working directory holds would silently override --project.
        run(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $(@__FILE__) --backend=$(opts["backend"]) --device=$(opts["device"]) --out=$out --max-elements=$(opts["max-elements"]) --replication=$k`)
    end

    # Once for the run, not once per replication: accuracy is deterministic and
    # a second process would measure the same bits again.
    println("the accuracy pass, in a fresh process")
    run(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $(@__FILE__) --backend=$(opts["backend"]) --device=$(opts["device"]) --out=$out --max-elements=$(opts["max-elements"]) --accuracy`)
    return nothing
end

"""
    _run_replication(backend::Symbol, impls, all_cases, out, replication::Int, device, detected)

Runs one replication of the whole case list and appends its rows to the output directory.

The case order is shuffled from a seed derived from the replication index and
recorded in the manifest, so drift across a long run decorrelates from size
rather than looking like a trend in it. Triples already in summary.csv are
skipped, and a case that throws records its status and message in cases.csv
while the run carries on.

# Returns
- Nothing
"""
function _run_replication(backend::Symbol, impls, all_cases, out, replication::Int, device, detected=Dict{String, Any}())
    machine = gethostname()
    run_id = string(machine, "-", basename(rstrip(out, '/')))
    seed = ORDER_SEED + replication
    order = shuffle(StableRNG(seed), collect(eachindex(all_cases)))
    done = completed(joinpath(out, "summary.csv"), 3)
    # The wisdom lives beside the code and not in the output directory, so that
    # the half hour of FFTW.MEASURE a cold sweep costs is paid once per machine
    # rather than once per results directory.
    wisdom = joinpath(@__DIR__, "fftw_wisdom_$machine")

    isfile(wisdom) && FFTW.import_wisdom(wisdom)
    write_manifest(joinpath(out, "manifest.toml"), replication, seed, [case_id(all_cases[i]) for i in order], detected, device_telemetry())

    for (position, index) in enumerate(order)
        case = all_cases[index]
        id = case_id(case)
        active = [impl for impl in impls if supports(impl, case)]
        all((id, string(impl), string(replication)) in done for impl in active) && continue

        identity = (run_id, id, machine, device, backend, case.class, case.label, case.family, case.precision, join(case.dims, "x"), join(case.region, " "), case.direction, case.placement, case.tuned)
        try
            x = input(case) # one host array for every implementation, so they see the same bytes
            preps = Any[prepare(impl, case, x) for impl in active]
            timed = duel(preps; repeatable=!destroys_input(case))

            # cases.csv holds one row per case, so the happy path is written by
            # the first replication only. A failure is written whenever it happens.
            replication == 1 && append_csv(joinpath(out, "cases.csv"), CASE_COLUMNS, [(identity..., "ok", "")])
            for (impl, prep, taken, inner) in zip(active, preps, timed.samples, timed.inner)
                stat = summarize(taken)
                append_csv(joinpath(out, "summary.csv"), SUMMARY_COLUMNS, [(run_id, id, string(impl), replication, position, stat.repeats, inner, stat.min, stat.median, stat.q25, stat.q75, stat.ci_lo, stat.ci_hi, stat.ci_rel, prep.detail)])
                append_csv(joinpath(out, "samples.csv"), SAMPLE_COLUMNS, [(run_id, id, string(impl), replication, j, taken[j]) for j in eachindex(taken)])
            end
        catch err
            append_csv(joinpath(out, "cases.csv"), CASE_COLUMNS, [(identity..., "error", first(replace(sprint(showerror, err), r"\s+" => " "), 200))])
            @warn "case failed, carrying on" case=id
        end
        @printf("[%3d/%3d] rep %d  %s\n", position, length(order), replication, id)
    end

    FFTW.export_wisdom(wisdom) # so the next sweep on this machine plans nothing it has planned before
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
