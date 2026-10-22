# The implementations that run on a device: VkFFT through whichever backend this
# process loaded, and the vendor's own FFT beside it where the device has one.
#
# One process is one backend. VkFFT compiles its backend in, so a session
# holding the Metal wrapper cannot also drive OpenCL, and only one branch of
# this file is ever live. run.jl sets BACKEND and activates the matching
# environment under envs/ before it includes this.

using AbstractFFTs
using LinearAlgebra

"""
    VkFFTImpl(tune::Bool=false)

VkFFT through the backend this process loaded, with the tuner off or on.

One type covers all three backends. The plan call and `mul!` are spelled the
same way everywhere, and the only thing that changes is the device array the
buffers are allocated as, which `DeviceArray` names.

The two tuner settings are two implementations of one case rather than two
cases, so they are sampled alternately inside one duel and the thing being
measured is the difference between them. A tuned plan is built with `tune=true`
and not `tune=:force`, so a run that has already swept the grid reads the record
instead of paying for it again.

# Fields
- `tune::Bool=false`: Whether the plan is built with the autotuner on
"""
struct VkFFTImpl
    tune::Bool
end

VkFFTImpl() = VkFFTImpl(false)

"""
    MPSGraphImpl()

Metal.jl's MPSGraph-backed FFT, which is the only other FFT a Julia program reaches on a Mac GPU.

It is reached through `AbstractFFTs.plan_fft` on an `MtlArray`, so it needs no
code of its own beyond naming the planner.
"""
struct MPSGraphImpl end

"""
    CUFFTImpl()

cuFFT through CUDA.jl, which is the library a CUDA user already has and the one VkFFT has to beat.

It is reached through `AbstractFFTs.plan_fft` on a `CuArray`, so like
`MPSGraphImpl` it needs no code of its own beyond naming the planner.
"""
struct CUFFTImpl end

struct DevicePrepared{P, X, Y, S}
    plan::P
    x::X
    y::Y
    src::S # a device resident pristine copy, or nothing when the transform leaves its own input alone
    detail::String
end

if BACKEND === :metal
    using Metal
    using VkFFTMetal

    const DeviceArray = MtlArray

    synchronize_device() = Metal.synchronize()

    """
        detect_device(request::String)

    Returns what the manifest records about the Metal device, which is its name and nothing else.

    A Mac has one GPU per task and Metal exposes no driver version under it, so
    `request` is ignored and there is no version to write. The macOS build the
    framework came with is already in the manifest's `os`.

    # Returns
    - A `Dict{String, Any}` carrying `device`
    """
    detect_device(::String) = Dict{String, Any}("device" => String(Metal.device().name))

    implementations(::Val{:metal}) = (VkFFTImpl(true), VkFFTImpl(), MPSGraphImpl(), FFTWImpl(Sys.CPU_THREADS))

    # A tuned implementation runs the tuned cases and nothing else, which is
    # what keeps the tuner off the main matrix while leaving both halves of one
    # tuned case in the same duel.
    supports(impl::VkFFTImpl, case::Case) = !impl.tune || case.tuned
elseif BACKEND === :cuda
    using CUDA
    using VkFFTCUDA

    const DeviceArray = CuArray

    synchronize_device() = CUDA.synchronize()

    """
        detect_device(request::String)

    Selects the CUDA device and returns what the manifest records about it.

    `request` is a device index, so `--device=1` reaches the second card on a
    box with two. Pinning the card with `CUDA_VISIBLE_DEVICES` is the better way
    and is what goes in the manifest, since it also keeps the driver from waking
    a card this run never touches.

    The card's name, its driver and the toolkit it is being driven by do not
    change over a run, so they live here. The clocks and the temperature do, and
    they go into the manifest's per replication table through
    `device_telemetry` instead.

    # Arguments
    - `request::String`: The index of the device to run on, or empty to keep the current one

    # Returns
    - A `Dict{String, Any}` carrying `device`, `cuda_driver` and `cuda_runtime`, plus `cuda_visible_devices` where it is set and `driver` and `gpu_clock_max_mhz` where NVML answers
    """
    function detect_device(request::String)
        if !isempty(request)
            index = tryparse(Int, request)
            index === nothing && throw(ArgumentError("--device on CUDA is a device index, such as --device=0, and $request is not one. Pinning the card with CUDA_VISIBLE_DEVICES is the better way and is what the manifest records."))
            CUDA.device!(index)
        end

        detected = Dict{String, Any}("device" => CUDA.name(CUDA.device()), "cuda_driver" => string(CUDA.driver_version()), "cuda_runtime" => string(CUDA.runtime_version()))
        haskey(ENV, "CUDA_VISIBLE_DEVICES") && (detected["cuda_visible_devices"] = ENV["CUDA_VISIBLE_DEVICES"])

        return merge(detected, _nvml() do device
            Dict{String, Any}("driver" => string(CUDA.NVML.driver_version()), "gpu_clock_max_mhz" => CUDA.NVML.max_clock_info(device).sm)
        end)
    end

    # A telemetry read must never kill a sweep, so a machine whose NVML does not
    # answer contributes none of these fields and says why once. A manifest
    # missing a field can be read. One holding an empty field cannot.
    function _nvml(fields)
        CUDA.has_nvml() || return Dict{String, Any}()
        try
            return fields(CUDA.NVML.Device(CUDA.uuid(CUDA.device())))
        catch err
            @warn "NVML did not answer, so the manifest goes without the fields it would have filled" exception=err maxlog=1
            return Dict{String, Any}()
        end
    end

    """
        device_telemetry()

    Returns the card's clocks, its temperature and NVML's clock event reason, for one replication's manifest entry.

    `clock_event_application_setting` is NVML's own answer for whether the
    applications clock setting is what is holding the clock down, which is the
    reason `nvidia-smi --lock-gpu-clocks` sets. It is recorded under that name
    rather than as a verdict on whether the clocks were locked, because an idle
    card can be holding a lock and naming its idle state instead. The two clocks
    beside it are what let a reader decide.

    # Returns
    - A `Dict{String, Any}` carrying `gpu_clock_sm_mhz`, `gpu_clock_memory_mhz`, `gpu_temperature_c` and `clock_event_application_setting`, or empty where NVML does not answer
    """
    device_telemetry() = _nvml() do device
        clocks = CUDA.NVML.clock_info(device)
        Dict{String, Any}("gpu_clock_sm_mhz" => clocks.sm, "gpu_clock_memory_mhz" => clocks.memory, "gpu_temperature_c" => CUDA.NVML.temperature(device), "clock_event_application_setting" => get(CUDA.NVML.clock_event_reasons(device), :application_setting, false))
    end

    # The rival is the vendor library and nothing else. FFTW gets a sweep of its
    # own, and asking FFTW.MEASURE to plan the 2^23 and 2^24 shapes an uncapped
    # card reaches costs minutes per plan, which is why the CPU paths cap at
    # 2^22 in the first place.
    implementations(::Val{:cuda}) = (VkFFTImpl(true), VkFFTImpl(), CUFFTImpl())

    supports(impl::VkFFTImpl, case::Case) = !impl.tune || case.tuned
else
    using pocl_jll # before OpenCL: it is what puts an OpenCL device on a Mac at all
    using OpenCL
    using VkFFTOpenCL

    const DeviceArray = CLArray

    synchronize_device() = cl.finish(cl.queue())

    """
        detect_device(request::String)

    Selects the OpenCL platform and returns what the manifest records about its device.

    `request` is matched against the platform names, so `--device="Portable
    Computing Language"` is how pocl is picked on a machine that also carries a
    vendor ICD. An empty request leaves OpenCL.jl's own choice alone.

    A tuning answer belongs to a driver as much as to a device, and pocl bumps
    its version with every release, so both versions are recorded beside the
    name.

    `device` is the label a figure's subtitle carries, so it has to name the
    processor that ran the test. pocl answers `CL_DEVICE_NAME` with the bare
    string `cpu`, which names neither a CPU nor a GPU, and an OpenCL CPU device
    is the host's processor in any case, so that is what goes in. The raw name
    stays beside it under `opencl_device`, because a tuning record on disk is
    keyed on the platform name, that device name and the driver version, and
    nothing else in the manifest would say what those records belong to.

    # Arguments
    - `request::String`: A substring of the platform name to select, or empty to keep the current one

    # Returns
    - A `Dict{String, Any}` carrying `device`, `opencl_device`, `opencl_platform`, `opencl_version` and `opencl_driver`
    """
    function detect_device(request::String)
        isempty(request) || cl.platform!(only(p for p in cl.platforms() if occursin(request, p.name)))

        # VkFFT needs cl.Buffer storage. A driver with coarse-grained SVM and no
        # buffer device address, NVIDIA's for one, leaves buffers off OpenCL.jl's
        # list of backends, and the default_memory_backend preference can only
        # pick from that list. Setting the backend on the task bypasses the list,
        # and it has to come after platform!, which clears the task's settings.
        task_local_storage(:CLMemoryBackend, cl.BufferBackend())

        label = cl.device().device_type === :cpu ? Sys.cpu_info()[1].model : cl.device().name
        return Dict{String, Any}("device" => label, "opencl_device" => cl.device().name, "opencl_platform" => cl.platform().name, "opencl_version" => cl.platform().version, "opencl_driver" => cl.device().driver_version)
    end

    # The rival belongs to the device rather than to the backend. pocl runs on a
    # CPU and races FFTW at one thread and at all of them. The same OpenCL code
    # on an A6000 races VkFFT on CUDA and cuFFT on that card, and neither is
    # reachable from this process: VkFFT compiles one backend into the wrapper
    # and cuFFT comes through CUDA.jl, which this environment does not hold. So
    # a card runs unopposed here, and the three way is a join on case_id between
    # this run's rows and the CUDA run's.
    implementations(::Val{:opencl}) = cl.device().device_type === :cpu ? (VkFFTImpl(true), VkFFTImpl(), FFTWImpl(1), FFTWImpl(Sys.CPU_THREADS)) : (VkFFTImpl(true), VkFFTImpl())

    # The 2^22 cap in the table is a fact about running OpenCL on a CPU, not
    # about OpenCL. A card takes the whole shape list, and capping it there would
    # end an OpenCL sweep two points short of the CUDA sweep it exists to be
    # compared against.
    capabilities(::Val{:opencl}) = cl.device().device_type === :cpu ? CAPABILITIES[:opencl] : merge(CAPABILITIES[:opencl], (max_elements=typemax(Int),))

    # On the CPU path a single precision transform of 4093 or 8191 points plans
    # cleanly and then kills the process with a bus error inside the generated
    # kernel, which no try block around the case can catch. Double precision at
    # those lengths runs and is correct, and a GPU ICD has neither problem. See
    # Known failures in docs/src/backends.md.
    supports(impl::VkFFTImpl, case::Case) = (!impl.tune || case.tuned) && (real(case.precision) === Float64 || cl.device().device_type !== :cpu || !(prod(case.dims) in (4093, 8191)))
end

# A tuned case's duel is VkFFT against itself, so the vendor sits it out rather
# than spending a third of the budget restating a number the plain case already
# carries for the same shape.
supports(::MPSGraphImpl, case::Case) = !case.tuned
supports(::CUFFTImpl, case::Case) = !case.tuned

"""
    prepare(impl::Union{VkFFTImpl, MPSGraphImpl, CUFFTImpl}, case::Case, x)

Allocates one case's device buffers, builds the plan and uploads the host input.

Everything is a whole array rather than a view of one. Neither Metal nor OpenCL
carries an offset in its launch parameters, so a buffer always enters a
transform at its own first element and an array at a nonzero offset is refused.

A transform that writes over its input keeps its pristine copy on the device
rather than on the host, so that `refill!` is a device to device copy and costs
no transfer per sample.
"""
function prepare(impl::Union{VkFFTImpl, MPSGraphImpl, CUFFTImpl}, case::Case, x)
    # VkFFT.plan_fft and AbstractFFTs.plan_fft take the same arguments, and
    # AbstractFFTs.plan_fft is Metal.jl's MPSGraph plan on an MtlArray and
    # CUDA.jl's cuFFT plan on a CuArray, so this name is the whole of the
    # difference between VkFFT and the vendor it is racing.
    planner = impl isa VkFFTImpl ? VkFFT : AbstractFFTs
    # VkFFT takes the tuner flag as a keyword and AbstractFFTs has no such
    # keyword, so it travels as a splat that is empty for everything but a tuned
    # VkFFT plan. Splatting it at every family rather than only at the one the
    # tuned cases use keeps a later tuned family from silently losing the flag.
    tuning = impl isa VkFFTImpl && impl.tune ? (tune=true,) : (;)
    buf = DeviceArray{eltype(x)}(undef, size(x))

    if case.family === :r2c
        plan = planner.plan_rfft(buf, case.region; tuning...)
        out = DeviceArray{complex(eltype(x))}(undef, (size(buf, 1) ÷ 2 + 1, Base.tail(size(buf))...))
    elseif case.family === :c2r
        plan = planner.plan_irfft(buf, case.dims[1], case.region; tuning...)
        out = DeviceArray{real(eltype(x))}(undef, case.dims)
    elseif case.placement === :inplace
        plan = planner.plan_fft!(buf, case.region; tuning...)
        out = buf
    elseif case.direction === :forward
        plan = planner.plan_fft(buf, case.region; tuning...)
        out = similar(buf)
    else
        plan = planner.plan_ifft(buf, case.region; tuning...)
        out = similar(buf)
    end

    pristine = destroys_input(case) ? copyto!(similar(buf), x) : nothing
    copyto!(buf, x)
    synchronize_device() # the upload is asynchronous, and the first sample must not wait on it
    return DevicePrepared(plan, buf, out, pristine, impl isa VkFFTImpl ? "tune=$(impl.tune)" : "")
end

apply!(prep::DevicePrepared) = (mul!(prep.y, prep.plan, prep.x); nothing)

synchronize!(::DevicePrepared) = synchronize_device()

# Copy, then wait, then return. The refill runs on the queue the transform will
# run on, so returning at enqueue time would leave the copy to finish inside the
# timed region, and at the small sizes that is most of what the sweep would then
# be measuring.
function refill!(prep::DevicePrepared)
    prep.src === nothing && return nothing
    copyto!(prep.x, prep.src)
    synchronize_device()
    return nothing
end

result(prep::DevicePrepared) = Array(prep.y)

"""
    run_tuning(subject, out)

Sweeps the tuner's grid once for every tuned case and appends what each candidate measured to tuning.csv.

The grid is sixteen candidates, `coalesced_memory` over four values crossed with
`aim_threads` over four, fastest first. What P7 reads off it is whether the
winner stands clear of the rest or the whole grid sits inside the noise, and
that second answer is the one that says whether tuning is worth its compile
time on a device.

The plan is built with `tune=:force` and not `tune=true`. A stored record turns
`tune=true` into a lookup, a lookup does not clear `last_sweep`, and what is
still sitting in it then belongs to the shape before this one and would be
written under this shape's name. `clear_tuning!` is the wrong tool for the
mirror image of that reason: it would throw away the records the timing half's
tuned plans read back, and the next sweep would pay for every one of them
again. `sweep_count` is read on both sides of the plan call as the check that a
grid really was measured, and a case whose count did not move records nothing
rather than recording the wrong thing.

The buffer is uninitialized because the sweep allocates and fills its own, so
the grid does not depend on the data and a host array built for it would be
work nothing reads.

# Arguments
- `subject`: The tuned cases to sweep, as `cases` enumerated them
- `out`: The results directory to append `tuning.csv` to

# Returns
- Nothing
"""
function run_tuning(subject, out)
    run_id = string(gethostname(), "-", basename(rstrip(out, '/')))
    path = joinpath(out, "tuning.csv")
    done = completed(path, 1)

    for (position, case) in enumerate(subject)
        id = case_id(case)
        (id,) in done && continue

        started = time_ns()
        try
            swept = VkFFT.sweep_count()
            VkFFT.plan_fft(DeviceArray{case.precision}(undef, case.dims), case.region; tune=:force)
            grid = VkFFT.last_sweep()
            VkFFT.sweep_count() == swept + 1 || error("the tuner reported no sweep for $id, so last_sweep still holds the grid of the shape before it")

            append_csv(path, TUNING_COLUMNS, [(run_id, id, coalesced, threads, microseconds) for (coalesced, threads, microseconds) in grid])
            @printf("[%3d/%3d] %s  %d candidates  best (%d, %d) at %.1f us, worst %.1f us  %.2f s\n",
                    position, length(subject), id, length(grid), grid[1][1], grid[1][2], grid[1][3], grid[end][3], (time_ns() - started) * 1e-9)
        catch err
            @warn "tuning case failed, carrying on" case=id exception=err
        end
        VkFFT.clear_cache!() # the sixteen candidates and their winner have been read
    end
    return nothing
end

Base.show(io::IO, impl::VkFFTImpl) = print(io, impl.tune ? "vkfft-tuned" : "vkfft")
Base.show(io::IO, ::MPSGraphImpl) = print(io, "mpsgraph")
Base.show(io::IO, ::CUFFTImpl) = print(io, "cufft")
