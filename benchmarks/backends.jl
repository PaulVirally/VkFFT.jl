# The interface every implementation satisfies, and FFTW as the first one.

using FFTW
using LinearAlgebra

"""
    prepare(impl, case::Case, x)

Builds everything an implementation needs to run one case and uploads the host array `x` into it.

This is the contract a new backend implements: `prepare`, `apply!`,
`synchronize!`, `refill!`, `result` and `supports`, and nothing else. `prepare`
allocates the buffers and builds the plan first and copies `x` in last, because
a planner is allowed to write over the arrays it plans on. The object it returns
must be concretely typed, since `apply!` runs once per timing sample and a
dynamic dispatch there is measurable against a transform of a few hundred
nanoseconds. It carries a `detail::String` field holding whatever about the plan
belongs in the output, which for FFTW is the planning flag and the thread count
and later is what the tuner chose.

`apply!(prepared)` queues the transform and returns without waiting for it.
`synchronize!(prepared)` waits for everything `apply!` queued. A sample is a
batch of `apply!` calls and then one `synchronize!` inside the clock, so it
measures queued throughput, which is what a loop of `mul!` calls sees.
`result(prepared)` brings the output back to the host as an `Array` for the
accuracy pass and is never timed.
`supports(impl, case)` answers whether the implementation can run a case at all,
which is how FFTW declines the half precision rows a Metal sweep enumerates.

`refill!(prepared)` puts the pristine input back in front of the transform and
is called before every sample, outside the clock. An in place transform and a
complex to real one both write over their input, so without it every sample
after the first runs on the previous sample's output, the values run away to
infinity, and the sweep no longer measures the well scaled random input the
method describes. An implementation whose input survives does nothing here and
pays nothing, since the branch folds away at compile time.

A device implementation has to synchronize inside `refill!`. The refill is an
asynchronous device to device copy issued on the stream the transform will run
on, so a `refill!` that returned as soon as the copy was enqueued would leave it
to finish inside the timed region. Copy, then wait, then return.
"""
function prepare end
function apply! end
function synchronize! end
function refill! end
function result end
function supports end

"""
    FFTWImpl(threads::Int)

FFTW at a fixed thread count, planned with `FFTW.MEASURE`.

One thread and all of them are two implementations and are never blended. FFTW
bakes a thread count into a plan when it builds it, so both plans keep their own
threading through a duel even though the setting is global.

# Fields
- `threads::Int`: The number of threads to plan with
"""
struct FFTWImpl
    threads::Int
end

struct FFTWPrepared{P, X, Y, S}
    plan::P
    x::X
    y::Y
    src::S # the pristine input, or nothing when the transform leaves its own alone
    detail::String
end

"""
    implementations(backend::Symbol)

Returns the implementations that race each other on a backend, in a fixed order.

A device backend's method lives in devices.jl, which run.jl includes only once
it has activated that backend's environment. A process is one backend: the C++
library compiles its own in, so a session holding the Metal wrapper cannot also
drive OpenCL, and a Mac cannot install CUDA.jl at all.
"""
implementations(backend::Symbol) = implementations(Val(backend))

implementations(::Val{B}) where B = throw(ArgumentError("no implementations are wired up for the $B backend. The backends are cuda, fftw, metal and opencl"))

implementations(::Val{:fftw}) = (FFTWImpl(1), FFTWImpl(Sys.CPU_THREADS))

"""
    capabilities(backend::Symbol)

Returns a backend's capability entry, with anything that depends on the device narrowed or widened to it.

`CAPABILITIES` is read before any device exists, so an entry whose truth varies
between two devices of one backend has to be settled here instead. OpenCL is the
one that does: its element types are the same everywhere but its size cap is a
property of the silicon.
"""
capabilities(backend::Symbol) = capabilities(Val(backend))

capabilities(::Val{B}) where B = CAPABILITIES[B]

"""
    device_telemetry()

Returns what the device says about its own state right now, for the manifest's per replication table.

The reading belongs to a replication rather than to a run, so that a
replication whose numbers look wrong can be checked against the state of the
device that produced them instead of guessed at. A backend that exposes nothing
of the sort answers with this empty default and gets no table.
"""
device_telemetry() = Dict{String, Any}()

# FFTW has no Float16, and the tuner it would be measured against belongs to
# VkFFT, so a tuned case is not FFTW's to run either.
supports(::FFTWImpl, case::Case) = case.precision in (ComplexF32, ComplexF64) && !case.tuned

function prepare(impl::FFTWImpl, case::Case, x)
    FFTW.set_num_threads(impl.threads)
    buf = similar(x)
    flags = FFTW.MEASURE

    if case.family === :r2c
        plan = FFTW.plan_rfft(buf, case.region; flags)
        out = similar(buf, complex(eltype(buf)), (size(buf, 1) ÷ 2 + 1, Base.tail(size(buf))...))
    elseif case.family === :c2r
        plan = FFTW.plan_irfft(buf, case.dims[1], case.region; flags)
        out = similar(buf, real(eltype(buf)), case.dims)
    elseif case.placement === :inplace
        plan = FFTW.plan_fft!(buf, case.region; flags)
        out = buf
    elseif case.direction === :forward
        plan = FFTW.plan_fft(buf, case.region; flags)
        out = similar(buf)
    else
        plan = FFTW.plan_ifft(buf, case.region; flags)
        out = similar(buf)
    end

    # A transform that writes over its input gets a pristine copy to restore
    # from. FFTW can be asked to preserve the input instead, but that flag costs
    # the implementation that needs it a scratch copy inside the timed region and
    # leaves the rival without one, so the refill happens outside the clock.
    copyto!(buf, x) # MEASURE writes over the arrays it plans on, so the upload comes last
    return FFTWPrepared(plan, buf, out, destroys_input(case) ? x : nothing, "MEASURE threads=$(impl.threads)")
end

apply!(prep::FFTWPrepared) = (mul!(prep.y, prep.plan, prep.x); nothing)

synchronize!(::FFTWPrepared) = nothing

function refill!(prep::FFTWPrepared)
    prep.src === nothing || copyto!(prep.x, prep.src)
    return nothing
end

result(prep::FFTWPrepared) = Array(prep.y)

Base.show(io::IO, impl::FFTWImpl) = print(io, "fftw-t", impl.threads)
