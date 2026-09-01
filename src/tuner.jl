# The autotuner. VkFFT exposes two knobs that change how it decomposes a
# transform into blocks and threads, and the right values are a property of the
# device and the shape rather than of anything a caller knows. So the tuner
# sweeps a small grid, times the candidates honestly, keeps the fastest and
# writes the answer to disk, where the next process finds it.

## The grid
#
# coalesced_memory is a byte count: how wide a contiguous read VkFFT should
# assume the memory system rewards. VkFFT's own defaults are 32 bytes on CUDA
# and on NVIDIA and AMD OpenCL devices, and 64 on Metal, on Intel and on OpenCL
# vendors it does not recognize, doubled when the transform is half precision.
# The grid is those two values, one step above them, and 0 for whatever VkFFT
# would have picked.
#
# aim_threads is a thread count per block, and VkFFT defaults it to 128 on every
# backend. The grid is one halving, the default and one doubling, plus 0.
#
# 0 duplicates the default on every backend, on purpose: it makes one candidate
# a repeat of another and so puts the run-to-run noise of the timing right in
# the sweep, where it can be read.

const TUNE_COALESCED_MEMORY = (0, 32, 64, 128)
const TUNE_AIM_THREADS = (0, 64, 128, 256)

# Two applications to warm up, then as many as fit in roughly this much wall
# clock. mul! synchronizes, so this is device time plus one synchronization per
# application, and the bounds keep a huge transform from being timed once and a
# tiny one from being timed forever.
const TUNE_WARMUP = 2
const TUNE_TARGET_NS = 20_000_000
const TUNE_MIN_REPEATS = 3
const TUNE_MAX_REPEATS = 500

# The grid is timed through twice and each candidate keeps its faster pass.
# Measured on pocl: whichever candidate goes first reads about 10 percent slow
# on the first pass, however long it is warmed up, because the process itself is
# still settling. One extra pass costs a fraction of what building the
# candidates costs and removes a bias that would otherwise always land on the
# same grid point.
const TUNE_PASSES = 2

# Counted so that a test can prove the second tune of a shape is a lookup rather
# than a sweep.
const TUNE_SWEEPS = Ref(0)

# The grid of the most recent sweep, as (coalesced_memory, aim_threads,
# microseconds per application). Diagnostic, so a benchmark script or a test can
# print what the tuner saw rather than only what it concluded.
const LAST_SWEEP = Ref(Tuple{Int, Int, Float64}[])

"""
    sweep_count()

Returns how many autotuner sweeps this session has run.

A `tune=true` that finds a stored record does not sweep, so this counts the
expensive calls only. It exists to make "the second call was a lookup" a thing a
test can assert rather than infer from a stopwatch.
"""
sweep_count() = TUNE_SWEEPS[]

"""
    last_sweep()

Returns the grid the most recent autotuner sweep measured, fastest first.

Each entry is `(coalesced_memory, aim_threads, microseconds)`, the last being the
time one application took. Empty until a sweep has run in this session, and
overwritten by the next one, so two tasks tuning at once leave only one of the
two grids behind.
"""
last_sweep() = LAST_SWEEP[]

"""
    _tune_mode(tune)

Turns the `tune` keyword into `(sweep::Bool, force::Bool)`.

`false` plans as usual, `true` uses a stored record when there is one and sweeps
otherwise, and `:force` sweeps whatever is stored and overwrites it.
"""
_tune_mode(tune::Bool) = (tune, false)
_tune_mode(tune::Symbol) = tune === :force ? (true, true) : throw(ArgumentError("tune takes false, true or :force. Got :$tune. :force re-runs the sweep and overwrites the stored record, which is how a machine that has changed underneath its tuning records gets retuned."))
_tune_mode(tune) = throw(ArgumentError("tune takes false, true or :force. Got $tune."))

## Tuning records

"""
    _read_record(path::String)

Returns the `(coalesced_memory, aim_threads)` a record file holds, or `nothing`.

A file that is absent, unreadable or unparseable is a miss, not an error: a
record is an optimization, and a truncated or hand-edited one should cost a
sweep rather than a plan.
"""
function _read_record(path::String)
    isfile(path) || return nothing
    bytes = try
        read(path)
    catch
        return nothing
    end

    for line in eachline(IOBuffer(bytes))
        startswith(line, "#") && continue
        fields = split(line)
        length(fields) == 2 || continue
        coalesced_memory = tryparse(Int, fields[1])
        aim_threads = tryparse(Int, fields[2])
        (coalesced_memory === nothing || aim_threads === nothing) && continue
        (coalesced_memory >= 0 && aim_threads >= 0) || continue
        return (coalesced_memory, aim_threads)
    end

    return nothing
end

"""
    _write_record(path::String, coalesced_memory::Int, aim_threads::Int)

Writes a tuning record, with a comment line saying what the numbers are.

The write goes through a temporary name and a rename, which is what makes two
processes writing the same key benign: each writes its own temporary file, the
last rename wins, and no reader ever sees a half-written record. A write that
fails (a full disk, a read-only depot) is not a planning failure, so nothing
is thrown.
"""
function _write_record(path::String, coalesced_memory::Int, aim_threads::Int)
    temp = ""
    try
        temp, io = mktemp(dirname(path))
        write(io, "# VkFFT tuned coalesced_memory (bytes) and aim_threads (threads per block)\n$coalesced_memory $aim_threads\n")
        close(io)
        Base.Filesystem.rename(temp, path)
        return true
    catch
        isempty(temp) || rm(temp; force=true)
        return false
    end
end

"""
    clear_tuning!()

Deletes every stored tuning record.

The next `tune=true` on each shape sweeps again. `tune=:force` does the same for
one shape without touching the rest.

# Returns
- The number of files removed
"""
function clear_tuning!()
    dir = disk_cache_dir()
    removed = 0
    @lock DISK_CACHE_LOCK begin
        for name in readdir(dir)
            startswith(name, string(RECORD_FILE[1], "_")) || continue
            rm(joinpath(dir, name); force=true)
            removed += 1
        end
    end

    return removed
end

## Timing

"""
    _tuning_buffers(plan::AbstractVkFFTPlan, prototype::AbstractArray)

Returns the `(output, input)` pair one candidate plan is timed on.

Allocated from the caller's array rather than being the caller's array: the
planners document that the array they are handed is neither read nor written,
and the tuner has to write. The buffers are made once and reused across the
whole grid, since every candidate transforms the same shape.
"""
function _tuning_buffers(plan::VkFFTPlan{T, N, IP}, prototype::AbstractArray) where {T, N, IP}
    x = _tuning_input(prototype)
    return (IP ? x : similar(prototype), x) # an in-place plan takes one buffer twice
end

function _tuning_buffers(plan::VkFFTR2RPlan{T, N, IP}, prototype::AbstractArray) where {T, N, IP}
    x = _tuning_input(prototype)
    return (IP ? x : similar(prototype), x)
end

function _tuning_buffers(plan::VkFFTRealPlan{T, S}, prototype::AbstractArray) where {T, S}
    x = _tuning_input(prototype)
    return (similar(prototype, S, plan.osz), x)
end

"""
    _tuning_input(prototype::AbstractArray)

Returns a fresh device array of the prototype's type and shape, filled with ones.

Filled rather than left uninitialized because a buffer of whatever the allocator
handed back can hold denormals, and a device that penalizes those would make the
first candidate look slow for a reason that has nothing to do with the knobs.
"""
_tuning_input(prototype::AbstractArray) = fill!(similar(prototype), one(eltype(prototype)))

"""
    _time_applications(plan::AbstractVkFFTPlan, y, x, repeats::Int)

Returns the nanoseconds `repeats` warmed-up applications of `plan` take.

`mul!` synchronizes before it returns, so the host clock around a run of them
measures device time plus one synchronization per application, which is why there
is no event or profiler machinery here. The warmup and the repeat count are what
the measurement rests on: a clock around one unwarmed application would time an
asynchronous launch and a first-touch page fault instead of a transform.
"""
function _time_applications(plan::AbstractVkFFTPlan, y::AbstractArray, x::AbstractArray, repeats::Int)
    start = time_ns()
    for _ in 1:repeats
        mul!(y, plan, x)
    end

    return time_ns() - start
end

## The sweep

"""
    _uncache!(plan::AbstractVkFFTPlan)

Drops a plan from the plan cache and frees its VkFFT application.

The tuner reaches its candidates through the ordinary planning path, so every one
of them is in the plan cache by the time it has been timed. Freeing a loser
without removing it first would leave a destroyed plan behind for the next lookup
of that shape and those tuning values to return.
"""
function _uncache!(plan::AbstractVkFFTPlan)
    @lock PLAN_CACHE_LOCK begin
        for key in collect(keys(PLAN_CACHE))
            PLAN_CACHE[key] === plan && delete!(PLAN_CACHE, key)
        end
    end
    unsafe_free!(plan)

    return nothing
end

"""
    _tuning_params(build, config::VkFFTConfig, roots::Vector{Any}, backend::Val, prototype::AbstractArray, force::Bool)

Returns the `(coalesced_memory, aim_threads)` to plan this configuration with.

A stored record is returned as it stands, which is what makes `tune=true` cheap
on the second run and on the next process. Otherwise the grid is swept: every
candidate is built through `build`, then all of them are warmed up and timed,
twice over, and each keeps its faster pass. A candidate VkFFT refuses to build is
skipped rather than fatal, since a thread count or a coalescing width the device
cannot honour is a fact about the grid and not about the caller.

The winner stays in the plan cache and the losers are freed, so the plan the
caller gets afterwards is a cache hit rather than a seventeenth compilation, and
nothing is left holding a VkFFT application no one will apply.
"""
function _tuning_params(build, config::VkFFTConfig, roots::Vector{Any}, backend::Val{B}, prototype::AbstractArray, force::Bool) where B
    # The backend name and the first twelve hex digits of the key, so a
    # directory listing reads.
    key = _disk_key(config, roots, backend)
    path = joinpath(disk_cache_dir(), string(RECORD_FILE[1], "_", B, "_", first(key, 12), RECORD_FILE[2]))

    if !force
        stored = @lock DISK_CACHE_LOCK _read_record(path)
        stored === nothing || return stored
    end

    TUNE_SWEEPS[] += 1

    knobs = NTuple{2, Int}[]
    plans = AbstractVkFFTPlan[]
    for coalesced_memory in TUNE_COALESCED_MEMORY, aim_threads in TUNE_AIM_THREADS
        plan = try
            build(coalesced_memory, aim_threads)
        catch
            continue
        end
        push!(knobs, (coalesced_memory, aim_threads))
        push!(plans, plan)
    end
    isempty(plans) && return (0, 0) # nothing in the grid built, so leave it to VkFFT

    y, x = _tuning_buffers(plans[1], prototype)
    _time_applications(plans[1], y, x, TUNE_WARMUP)

    # One more warmed-up application sizes the repeat count so a candidate's run
    # takes roughly TUNE_TARGET_NS: long enough that a few percent of difference
    # between two candidates is above the noise, short enough that a
    # sixteen-point grid is not a coffee break. Computed once and used for every
    # candidate, so they are compared over the same amount of work.
    one_application = max(Int(_time_applications(plans[1], y, x, 1)), 1)
    repeats = clamp(TUNE_TARGET_NS ÷ one_application, TUNE_MIN_REPEATS, TUNE_MAX_REPEATS)

    times = fill(typemax(UInt64), length(plans))
    for _ in 1:TUNE_PASSES, i in eachindex(plans)
        # An inverse real transform uses its input as scratch space, so the
        # input is refilled before every candidate rather than once. Outside the
        # timed run, so it costs the measurement nothing.
        fill!(x, one(eltype(x)))
        _time_applications(plans[i], y, x, TUNE_WARMUP)
        times[i] = min(times[i], _time_applications(plans[i], y, x, repeats))
    end

    winner = argmin(times)
    order = sortperm(times)
    LAST_SWEEP[] = [(knobs[i][1], knobs[i][2], times[i] / repeats / 1e3) for i in order]

    for i in eachindex(plans)
        i == winner || _uncache!(plans[i])
    end

    @lock DISK_CACHE_LOCK _write_record(path, knobs[winner][1], knobs[winner][2])

    return knobs[winner]
end

## Per-family entry into the sweep
#
# One of these per plan family, sitting where the layout is already mapped and
# the _create_app argument list is already spelled out. Each builds the untuned
# configuration for the record key, hands the sweep a closure that plans with a
# candidate's values, and then plans once more with the winner, which the plan
# cache answers without compiling anything.

"""
    _tuned_plan(::Type{T}, sz::NTuple{N, Int}, region::NTuple{M, Int}, direction::Int32, normalize::Bool, ::Val{IP}, backend::Val{B}, device_id::UInt64, roots::Vector{Any}, prototype::AbstractArray, force::Bool; zeropad::NTuple{2, Int}=NO_ZEROPAD)

Returns the tuned complex-to-complex plan for this configuration.
"""
function _tuned_plan(::Type{T}, sz::NTuple{N, Int}, region::NTuple{M, Int}, direction::Int32, normalize::Bool, ::Val{IP}, backend::Val{B}, device_id::UInt64, roots::Vector{Any}, prototype::AbstractArray, force::Bool; zeropad::NTuple{2, Int}=NO_ZEROPAD) where {T <: VkFFTComplex, N, M, IP, B}
    plan(coalesced_memory, aim_threads) = _create_plan(T, sz, region, direction, normalize, Val(IP), backend, device_id, roots; zeropad=zeropad, coalesced_memory=coalesced_memory, aim_threads=aim_threads)

    layout = _map_region(sz, region, _max_dims())
    _is_trivial(layout) && return plan(0, 0)

    config = _app_config(T, layout, direction, normalize, IP, false; zeropad=zeropad)
    coalesced_memory, aim_threads = _tuning_params(plan, config, roots, backend, prototype, force)

    return plan(coalesced_memory, aim_threads)
end

"""
    _tuned_real_plan(::Type{T}, ::Type{S}, sz::NTuple{N, Int}, osz::NTuple{N, Int}, region::NTuple{M, Int}, d::Int, direction::Int32, normalize::Bool, backend::Val{B}, device_id::UInt64, roots::Vector{Any}, prototype::AbstractArray, force::Bool; zeropad::NTuple{2, Int}=NO_ZEROPAD)

Returns the tuned real-to-complex or complex-to-real plan for this configuration.
"""
function _tuned_real_plan(::Type{T}, ::Type{S}, sz::NTuple{N, Int}, osz::NTuple{N, Int}, region::NTuple{M, Int}, d::Int, direction::Int32, normalize::Bool, backend::Val{B}, device_id::UInt64, roots::Vector{Any}, prototype::AbstractArray, force::Bool; zeropad::NTuple{2, Int}=NO_ZEROPAD) where {T <: VkFFTNumber, S <: VkFFTNumber, N, M, B}
    plan(coalesced_memory, aim_threads) = _create_real_plan(T, S, sz, osz, region, d, direction, normalize, backend, device_id, roots; zeropad=zeropad, coalesced_memory=coalesced_memory, aim_threads=aim_threads)

    layout = _map_region(direction == FORWARD ? sz : osz, region, _max_dims())
    _is_trivial(layout) && return plan(0, 0)

    config = _app_config(real(T), layout, direction, normalize, false, true; zeropad=zeropad)
    coalesced_memory, aim_threads = _tuning_params(plan, config, roots, backend, prototype, force)

    return plan(coalesced_memory, aim_threads)
end

"""
    _tuned_r2r_plan(::Type{T}, sz::NTuple{N, Int}, region::NTuple{M, Int}, kind::Symbol, type::Int, direction::Int32, normalize::Bool, zeropad::NTuple{2, Int}, ::Val{IP}, backend::Val{B}, device_id::UInt64, roots::Vector{Any}, prototype::AbstractArray, force::Bool)

Returns the tuned real-to-real plan for this configuration.
"""
function _tuned_r2r_plan(::Type{T}, sz::NTuple{N, Int}, region::NTuple{M, Int}, kind::Symbol, type::Int, direction::Int32, normalize::Bool, zeropad::NTuple{2, Int}, ::Val{IP}, backend::Val{B}, device_id::UInt64, roots::Vector{Any}, prototype::AbstractArray, force::Bool) where {T <: VkFFTReal, N, M, IP, B}
    plan(coalesced_memory, aim_threads) = _create_r2r_plan(T, sz, region, kind, type, direction, normalize, zeropad, Val(IP), backend, device_id, roots; coalesced_memory=coalesced_memory, aim_threads=aim_threads)

    layout = _map_region(sz, region, _max_dims())
    _is_trivial(layout) && return plan(0, 0)

    dct = kind === :dct ? Int32(type) : Int32(0)
    dst = kind === :dst ? Int32(type) : Int32(0)
    config = _app_config(T, layout, direction, normalize, IP, false; dct=dct, dst=dst, zeropad=zeropad)
    coalesced_memory, aim_threads = _tuning_params(plan, config, roots, backend, prototype, force)

    return plan(coalesced_memory, aim_threads)
end
