# The duel timer and the statistics read off it.
#
# A case is a duel. Every implementation is prepared up front and then they are
# sampled in lockstep, so that a CPU changing boost state or a GPU ramping its
# clocks drifts under all of them at once. Sampling one to exhaustion before
# starting the next hands that drift to whichever went second and manufactures a
# few percent out of nothing.

using Random
using StableRNGs
using Statistics

const BUDGET = 0.3       # seconds of samples to aim for per implementation
const MIN_SAMPLES = 20
const MAX_SAMPLES = 2000
const CEILING = 10.0     # seconds after which a duel stops whatever it has
const TIMER_TARGET = 5e-3 # a batch this long keeps the one device wait per sample, about 100 us on Metal, near 2% of it
const MAX_INNER = 1000
const BOOTSTRAP_DRAWS = 1000
const BOOTSTRAP_SEED = 0x626f_6f74_0000_0000

"""
    duel(preps; budget=BUDGET, min_samples=MIN_SAMPLES, max_samples=MAX_SAMPLES, ceiling=CEILING, repeatable=true)

Times every prepared implementation of one case in lockstep and returns their sample times and batch sizes.

Each implementation is warmed up once and thrown away, since a first call
carries kernel compilation, first touch of its buffers and a cold clock. Four
further calls size that implementation's batch, where there is a batch to size.
Every round after that takes one sample from each of them in turn, each sample preceded by a `refill!`
outside the clock so that the transform always meets its pristine input. Rounds
stop once every implementation has its floor of samples and the slowest has
spent the budget, at the sample cap, or at the ceiling on the whole duel,
whichever lands first. Nothing is checked for convergence along the way: the
interval is computed once from the finished samples by `summarize`.

Stopping on the slowest keeps the sample counts equal, which is what
interleaving is for, and it bounds a duel at roughly the budget times the number
of implementations rather than at the budget each.

A sample times `inner` applications back to back, waits once for the device and
divides. The batch amortizes two fixed costs. One is the wait, about 100 us on
Metal against about 14 us of queued work per call at small sizes. The other is
the clock: `time_ns()` ticks at 41.7 ns on Apple Silicon, so a transform of a
few tens of nanoseconds timed one call at a time reads as zero, one tick or two.
The batch is sized so that it lasts `TIMER_TARGET` whatever the transform costs,
and it is held fixed for the duel so that both sides are measured the same way.

The two are not the same measurement and `inner` is recorded beside the times
because of it: a batch amortizes the per call overhead that a single timed call
pays in full, so a batched number is throughput under repetition and an
unbatched one is the latency of a single call.

A case that writes over its input cannot be batched at all, since every
application after the first would run on the previous one's output. Those pass
`repeatable=false` and keep `inner` at one, so each of their samples is one
application plus the wait.

# Arguments
- `preps`: The prepared objects, one per implementation, as `prepare` returned them
- `budget::Float64=BUDGET`: Seconds of samples to aim for per implementation
- `min_samples::Int=MIN_SAMPLES`: Samples taken before the budget is consulted at all
- `max_samples::Int=MAX_SAMPLES`: Hard cap on the samples per implementation
- `ceiling::Float64=CEILING`: Seconds after which the duel stops regardless
- `repeatable::Bool=true`: Whether the transform leaves its input alone, and so whether a sample may batch

# Returns
- `(samples, inner)`: a `Vector{Vector{Float64}}` of per application times in seconds, all of the same length, and the applications per sample behind them
"""
function duel(preps; budget=BUDGET, min_samples=MIN_SAMPLES, max_samples=MAX_SAMPLES, ceiling=CEILING, repeatable=true)
    for prep in preps
        _sample(prep, 1)
    end

    # The batch is sized from a timed call and then three more times from the
    # batch the last reading asked for. A batch of k calls reads the cost of one
    # call plus the device wait over k, so a short batch overestimates the cost
    # and asks for a batch that is still too short. On Metal, where the wait is
    # about 100 us, the second pass lands at a fifth to a half of the target and
    # the fourth lands on it. The nanosecond floor is there because a call at the
    # clock's resolution can time as zero, which would otherwise divide into an
    # infinity.
    inner = ones(Int, length(preps))
    if repeatable
        for (i, prep) in enumerate(preps), _ in 1:4
            inner[i] = clamp(ceil(Int, TIMER_TARGET / max(_sample(prep, inner[i]), 1e-9)), 1, MAX_INNER)
        end
    end

    samples = [Vector{Float64}(undef, max_samples) for _ in preps]
    spent = zeros(length(preps))
    started = time_ns()
    taken = 0
    while taken < max_samples
        for (i, prep) in enumerate(preps)
            seconds = _sample(prep, inner[i])
            samples[i][taken + 1] = seconds
            spent[i] += seconds * inner[i] # the batch's own wall clock, so the budget still means seconds
        end
        taken += 1

        (time_ns() - started) * 1e-9 >= ceiling && break
        taken >= min_samples && maximum(spent) >= budget && break
    end

    return (samples=[resize!(s, taken) for s in samples], inner=inner)
end

# The function barrier. `preps` holds implementations of different types, so
# reaching apply! through it would dispatch dynamically inside the timed region.
# Here `prep` is concrete and the dispatch happens on the way in, outside the
# clock. At the smallest sizes an FFTW transform is a few hundred nanoseconds,
# so that dispatch would be a visible part of the number.
function _sample(prep, inner)
    refill!(prep)
    started = time_ns()
    for _ in 1:inner
        apply!(prep)
    end
    synchronize!(prep)
    return (time_ns() - started) * 1e-9 / inner
end

"""
    summarize(samples::AbstractVector{Float64})

Summarizes one implementation's sample times, with a bootstrap interval on the median.

The interval is a percentile bootstrap: a thousand medians of a thousand
resamples drawn with replacement, read at the 2.5 and 97.5 percentiles. Its seed
is fixed, so the same samples give the same interval and a summary can be
recomputed from samples.csv without touching a device again.

# Returns
- A named tuple of `repeats`, `min`, `median`, `q25`, `q75`, `ci_lo`, `ci_hi` and `ci_rel`, the last being the interval's half width relative to the median
"""
function summarize(samples::AbstractVector{Float64})
    rng = StableRNG(BOOTSTRAP_SEED)
    draw = Vector{Float64}(undef, length(samples))
    resampled = sort!([median!(rand!(rng, draw, samples)) for _ in 1:BOOTSTRAP_DRAWS])

    sorted = sort(samples)
    middle = quantile(sorted, 0.5; sorted=true)
    ci_lo = quantile(resampled, 0.025; sorted=true)
    ci_hi = quantile(resampled, 0.975; sorted=true)
    return (repeats=length(samples), min=sorted[1], median=middle, q25=quantile(sorted, 0.25; sorted=true), q75=quantile(sorted, 0.75; sorted=true), ci_lo=ci_lo, ci_hi=ci_hi, ci_rel=(ci_hi - ci_lo) / 2 / middle)
end
