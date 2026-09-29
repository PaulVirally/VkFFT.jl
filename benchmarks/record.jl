# Everything that reaches disk: the five CSVs, the manifest, and the resume scan.
#
# Long format throughout, and identity is split out of the timings so that
# twenty fields are not rewritten on every row. case_id is the join key across
# all five files and the thing plot.jl reads.

using Dates
using Pkg
using Printf
using TOML

const CASE_COLUMNS = (:run_id, :case_id, :machine, :device, :backend, :class, :label, :family, :precision, :dims, :region, :direction, :placement, :tuned, :status, :message)
const SUMMARY_COLUMNS = (:run_id, :case_id, :implementation, :replication, :order, :repeats, :inner, :min, :median, :q25, :q75, :ci_lo, :ci_hi, :ci_rel, :detail)
const SAMPLE_COLUMNS = (:run_id, :case_id, :implementation, :replication, :sample, :seconds)
const ACCURACY_COLUMNS = (:run_id, :case_id, :implementation, :input_class, :draw, :relerr, :eps_multiple, :reference)
# One row per candidate of one tuned case's grid, so sixteen rows a shape. There
# is no implementation column: the grid belongs to VkFFT's tuner and to nothing
# it is raced against.
const TUNING_COLUMNS = (:run_id, :case_id, :coalesced_memory, :aim_threads, :microseconds)

"""
    append_csv(path, columns, rows)

Appends rows to a CSV, writing the header only when the file is new.

Flushing on every call is what makes a run resumable: a crash at minute 40 costs
the case it was in and nothing before it.

# Arguments
- `path`: The file to append to, which need not exist yet
- `columns`: The column names, written as the header of a fresh file
- `rows`: An iterable of tuples, each as long as `columns`

# Returns
- Nothing
"""
function append_csv(path, columns, rows)
    fresh = !isfile(path)
    open(path, "a") do io
        fresh && println(io, join(columns, ","))
        for row in rows
            length(row) == length(columns) || throw(ArgumentError("a row of $(length(row)) fields does not fit the $(length(columns)) columns of $(basename(path))"))
            println(io, join(map(_csv_field, row), ","))
        end
        flush(io)
    end
    return nothing
end

_csv_field(v::AbstractFloat) = @sprintf("%.6g", v)

function _csv_field(v)
    s = string(v)
    return occursin(',', s) ? string('"', replace(s, '"' => "\"\""), '"') : s
end

"""
    completed(path, nkeys::Int)

Returns the key tuples a results file already holds, so that a rerun can skip them.

The keys are the `nkeys` columns starting at column two, which is where every
file that resumes keeps its identity: summary.csv is keyed by case, implementation
and replication, accuracy.csv by case and implementation, and tuning.csv by case
alone. None of those
fields can hold a comma, so the scan splits on commas rather than parsing CSV
properly, and everything comes back as a string so that the caller compares what
the file literally says.

# Arguments
- `path`: The file to scan, which need not exist
- `nkeys::Int`: How many columns after the run id make up the key

# Returns
- A `Set{NTuple{nkeys, String}}`, empty if the file does not exist
"""
function completed(path, nkeys::Int)
    done = Set{NTuple{nkeys, String}}()
    isfile(path) || return done
    for (i, line) in enumerate(eachline(path))
        i == 1 && continue
        fields = split(line, ',')
        push!(done, ntuple(j -> String(fields[1 + j]), nkeys))
    end
    return done
end

"""
    write_manifest(path, replication::Int, seed::Integer, order::Vector{String}, detected=Dict{String, Any}(), telemetry=Dict{String, Any}())

Writes the run's manifest, merging this replication's case order and device reading into what is already there.

Every replication is a fresh process and rewrites the same identity block with
the same values, while the `order`, `seed` and `telemetry` tables grow by one
entry each, keyed by the replication index. None of
it is passed in on the command line: identity is detected so that a mistyped
flag cannot mislabel a run. Fields that mean nothing on a CPU run, such as the
driver version and the card's temperature, are left out rather than written
empty, and a backend that reads no telemetry gets no table at all.

`detected` is what the backend found about the device it is about to measure,
and it is empty on a CPU run. Each backend fills in only what it exposes:
Metal offers a device name and no driver version under it, while OpenCL offers
the platform version and the device's own driver version as well.

# Arguments
- `path`: The manifest to write, which need not exist yet
- `replication::Int`: The replication index this process is running
- `seed::Integer`: The seed the case order was shuffled from
- `order::Vector{String}`: The case ids in the order this replication ran them
- `detected`: The device fields the backend detected, merged in as they come
- `telemetry`: What the device said about its own state at this replication's boundary

# Returns
- Nothing
"""
function write_manifest(path, replication::Int, seed::Integer, order::Vector{String}, detected=Dict{String, Any}(), telemetry=Dict{String, Any}())
    manifest = isfile(path) ? TOML.parsefile(path) : Dict{String, Any}()
    orders = get(manifest, "order", Dict{String, Any}())
    seeds = get(manifest, "seed", Dict{String, Any}())
    readings = get(manifest, "telemetry", Dict{String, Any}())
    orders[string(replication)] = order
    seeds[string(replication)] = Int(seed)
    isempty(telemetry) || (readings[string(replication)] = telemetry)

    merge!(manifest, Dict{String, Any}(
        "machine" => gethostname(),
        "date" => string(now()),
        "os" => string(Sys.KERNEL, " ", Sys.MACHINE),
        "cpu" => string(Sys.cpu_info()[1].model, " x", Sys.CPU_THREADS),
        "julia" => string(VERSION),
        "seed" => seeds,
        "vkfft_commit" => try readchomp(pipeline(`git -C $(normpath(joinpath(@__DIR__, ".."))) rev-parse HEAD`, stderr=devnull)) catch; "unknown" end,
        "packages" => Dict(p.name => string(p.version) for p in values(Pkg.dependencies()) if p.version !== nothing),
        "order" => orders,
    ), detected)
    isempty(readings) || (manifest["telemetry"] = readings)

    open(path, "w") do io
        TOML.print(io, manifest)
    end
    return nothing
end

"""
    record_accuracy_seed(path, seed::Integer)

Writes the seed the accuracy draws come from into the manifest, as a scalar.

The `seed` table beside it is keyed by replication index and the accuracy pass
has none, so a key in there would misstate what that table means.

# Returns
- Nothing
"""
function record_accuracy_seed(path, seed::Integer)
    manifest = isfile(path) ? TOML.parsefile(path) : Dict{String, Any}()
    manifest["accuracy_seed"] = Int(seed)
    open(io -> TOML.print(io, manifest), path, "w")
    return nothing
end
