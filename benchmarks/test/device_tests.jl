# The device half of the harness, run from a backend environment and named on
# the command line:
#
#     julia --startup-file=no --project=benchmarks/envs/metal benchmarks/test/device_tests.jl --backend=metal
#     julia --startup-file=no --project=benchmarks/envs/opencl benchmarks/test/device_tests.jl --backend=opencl
#     julia --startup-file=no --project=benchmarks/envs/cuda benchmarks/test/device_tests.jl --backend=cuda
#
# It is not part of runtests.jl. That suite runs in the base environment, which
# holds no GPU package on purpose, and a process is one backend anyway.
#
# Everything under test comes from run.jl, which includes devices.jl once it has
# seen --backend.

using LinearAlgebra
using Test

include("../run.jl")

BACKEND === :fftw && error("device_tests.jl needs a device backend. Pass --backend=cuda, --backend=metal or --backend=opencl, and activate the matching project under benchmarks/envs.")

const rtol_device = Dict(Float64 => 1e-12, Float32 => 1e-4, Float16 => 5e-3)

@testset verbose=true "$BACKEND" begin
    detected = detect_device("")
    matrix = cases(BACKEND)
    devices = filter(impl -> !(impl isa FFTWImpl), implementations(BACKEND))

    # The capability entry is a claim about the device, and a claim about a
    # device is worth nothing until the device is asked.
    @testset "capabilities match the device" begin
        caps = capabilities(BACKEND)
        for T in caps.types
            @test DeviceArray{T}(undef, 4) isa DeviceArray{T}
        end

        if BACKEND === :metal
            # Not a gap in the bindings. Metal has no double precision in
            # hardware, and MtlArray refuses the element type outright.
            @test !(ComplexF64 in caps.types)
            @test_throws Exception DeviceArray{ComplexF64}(undef, 4)
        elseif BACKEND === :cuda
            # Half is left out because cuFFT's half path is not reachable from
            # CUDA.jl, so the row would carry VkFFT alone. The array type takes
            # the element and the planner is what refuses it, which is the thing
            # worth asking the device rather than asserting from a table.
            @test !(ComplexF16 in caps.types)
            @test DeviceArray{ComplexF16}(undef, 1024) isa DeviceArray{ComplexF16}
            @test_throws Exception AbstractFFTs.plan_fft(DeviceArray{ComplexF16}(undef, 1024), 1)

            # The cliff lives here and only here, and the card has to hold the
            # uncapped shape list the entry claims it holds. The largest case is
            # a 2^24 ComplexF64 and it needs an input, an output and a pristine
            # copy at once.
            @test caps.cliff && count(c -> c.class === :cliff, matrix) == 31
            @test caps.max_elements == typemax(Int)
            @test CUDA.free_memory() > 3 * 2^24 * sizeof(ComplexF64)

            # pocl's bus error at 4093 and 8191 is a fact about running OpenCL on
            # a CPU. Nothing declines a case here, and these two lengths in
            # particular have to be in the sweep, since the primes are the
            # Bluestein path and P3 is short two points without them.
            @test all(c -> supports(VkFFTImpl(), c), matrix)
            @test count(c -> prod(c.dims) in (4093, 8191), matrix) == 12

            # What the manifest needs to make a GPU run reproducible. A field the
            # machine does not answer is left out rather than written empty, and
            # what changes over a run is read per replication instead of once.
            reading = device_telemetry()
            @test issubset(("device", "cuda_driver", "cuda_runtime"), keys(detected))
            @test all(!isempty(string(v)) for v in values(detected))
            @test all(!isempty(string(v)) for v in values(reading))
            @test haskey(detected, "cuda_visible_devices") == haskey(ENV, "CUDA_VISIBLE_DEVICES")
            @test isempty(intersect(keys(detected), keys(reading)))
            if CUDA.has_nvml()
                @test issubset(("driver", "gpu_clock_max_mhz"), keys(detected))
                @test issubset(("gpu_clock_sm_mhz", "gpu_clock_memory_mhz", "gpu_temperature_c", "clock_event_application_setting"), keys(reading))
                @test reading["clock_event_application_setting"] isa Bool
                @test 0 < reading["gpu_clock_sm_mhz"] <= detected["gpu_clock_max_mhz"]
                @test 0 < reading["gpu_temperature_c"] < 120
            else
                @test isempty(reading)
            end

            # --device is an index here and the card is meant to be pinned with
            # CUDA_VISIBLE_DEVICES, so a platform name pasted over from the
            # OpenCL invocation says so instead of being ignored.
            @test_throws ArgumentError detect_device("A6000")
        else
            # Half precision on OpenCL is a per-device extension rather than a
            # property of the backend, so the extension list and what the
            # planner will actually do have to agree, whichever way the device
            # answers. A device without cl_khr_fp16 makes VkFFT emit half
            # arithmetic the driver refuses to compile.
            half = "cl_khr_fp16" in cl.device().extensions
            planned = try
                VkFFT.plan_fft(DeviceArray{ComplexF16}(undef, 1024), 1) isa VkFFT.VkFFTPlan
            catch
                false
            end
            @test planned == half
            @test ("cl_khr_fp64" in cl.device().extensions) == (ComplexF64 in caps.types)

            # The size cap belongs to the silicon rather than to the backend. A
            # CPU device keeps the table's 2^22 and a card takes the whole shape
            # list, which is what lets an OpenCL sweep reach the sizes the CUDA
            # sweep beside it reaches.
            @test caps.max_elements == (cl.device().device_type === :cpu ? 2^22 : typemax(Int))
            @test caps.types == CAPABILITIES[:opencl].types

            # A figure's subtitle names the processor that ran the test. pocl
            # answers CL_DEVICE_NAME with the bare string "cpu", which names
            # neither a CPU nor a GPU, so a CPU device is labelled with the
            # host's processor instead. The raw name has to survive somewhere,
            # since the tuning records on disk are keyed on it.
            @test detected["opencl_device"] == cl.device().name
            @test detected["device"] != "cpu"
            @test cl.device().device_type !== :cpu || detected["device"] == Sys.cpu_info()[1].model
            @test cl.device().device_type === :cpu || detected["device"] == cl.device().name

            # A single precision transform of either of these lengths plans
            # cleanly on pocl and then kills the process inside the generated
            # kernel, so declining it is the only thing that keeps a sweep alive.
            declined = filter(c -> !supports(VkFFTImpl(), c), matrix)
            @test cl.device().device_type !== :cpu || Set(prod(c.dims) for c in declined) == Set((4093, 8191))
            @test all(c -> c.precision === ComplexF32, declined)
        end
    end

    @testset "against fftw" begin
        small = filter(c -> prod(c.dims) <= 65536, matrix)
        # One case per family, precision, direction, placement and dimension
        # count, plus every prime, which is the Bluestein path and the one most
        # likely to differ between two libraries.
        # Plus a couple of tuned shapes. The tuner picks different block and
        # thread counts, which is the kind of change that shows up as a wrong
        # answer rather than as a slow one.
        subjects = unique(case_id, vcat(unique(c -> (c.family, c.direction, c.placement, c.precision, length(c.region)), small),
                                        filter(c -> c.class === :prime, small),
                                        filter(c -> c.tuned && prod(c.dims) in (4096, 65536), small)))

        for impl in devices, case in subjects
            supports(impl, case) || continue
            @testset "$impl $(case_id(case))" begin
                x = input(case)
                prepared = prepare(impl, case, x)
                apply!(prepared)

                expected = if case.family === :r2c
                    rfft(Float64.(x), case.region)
                elseif case.family === :c2r
                    irfft(ComplexF64.(x), case.dims[1], case.region)
                elseif case.direction === :inverse
                    ifft(ComplexF64.(x), case.region)
                else
                    fft(ComplexF64.(x), case.region)
                end
                measured = norm(vec(ComplexF64.(result(prepared))) - vec(expected)) / norm(vec(expected))

                # MPSGraph's normalized inverse returns all zeros in half
                # precision above 32768 points, silently, at every length tried.
                # Forward transforms are fine at any size. The shape stays in the
                # sweep because a wrong answer costs what a right one costs, and
                # accuracy.csv is where the error belongs, so this is where the
                # suite records that the answer is known bad.
                if impl isa MPSGraphImpl && case.direction === :inverse && case.precision === ComplexF16 && prod(case.dims) > 32768
                    @test_broken measured < rtol_device[real(case.precision)]
                else
                    @test measured < rtol_device[real(case.precision)]
                end
            end
        end
    end

    # Tuning is a plan flag, not a transform, so it is a case of its own whose
    # duel is VkFFT against itself. Every other implementation sits those cases
    # out rather than restating a number the plain case already carries.
    @testset "the tuned rows are VkFFT against itself" begin
        tuned = filter(c -> c.tuned, matrix)
        @test !isempty(tuned)
        @test all(c -> c.class in TUNED_CLASSES && (c.family, c.precision, c.direction, c.placement) == (:c2c, ComplexF32, :forward, :oop), tuned)
        @test Set(string(impl) for impl in implementations(BACKEND) if supports(impl, first(tuned))) == Set(("vkfft", "vkfft-tuned"))
        @test !any(c -> supports(VkFFTImpl(true), c), filter(c -> !c.tuned, matrix))
        @test prepare(VkFFTImpl(true), first(tuned), input(first(tuned))).detail == "tune=true"

        # The order the implementations come back in is load bearing. A sweep
        # frees every candidate it did not pick, the untuned plan's own
        # configuration is one of those sixteen, and a duel that built the
        # untuned plan first would be holding a freed plan by the time it came to
        # time it. Cold on purpose: with a record on disk nothing sweeps and
        # nothing is freed, so this only has teeth after clear_tuning!.
        VkFFT.clear_tuning!()
        cold = first(tuned)
        swept = VkFFT.sweep_count()
        preps = Any[prepare(impl, cold, input(cold)) for impl in implementations(BACKEND) if supports(impl, cold)]
        @test length(preps) == 2
        @test VkFFT.sweep_count() == swept + 1
        for prep in preps
            apply!(prep)
            @test maximum(abs, Array(prep.y)) > 0
        end

        # The grid the tuner saw, which is the other half of P7. tune=:force is
        # what guarantees last_sweep belongs to this shape, and sweep_count is
        # what proves it.
        smallest = argmin(c -> prod(c.dims), tuned)
        swept = VkFFT.sweep_count()
        VkFFT.plan_fft(DeviceArray{smallest.precision}(undef, smallest.dims), smallest.region; tune=:force)
        @test VkFFT.sweep_count() == swept + 1
        @test length(VkFFT.last_sweep()) == 16
        @test issorted(VkFFT.last_sweep(), by=last)
        @test all(((coalesced, threads, us),) -> coalesced in (0, 32, 64, 128) && threads in (0, 64, 128, 256) && us > 0, VkFFT.last_sweep())
    end

    # The one that would corrupt every small transform in the sweep without ever
    # failing visibly. A refill that returned at enqueue time would leave its
    # copy to finish inside the timed region, so what the contract demands is
    # that the queue is empty again by the time it returns. Metal.jl's device to
    # device copy happens to block already and OpenCL.jl's does not, so this has
    # teeth on pocl and guards Metal against a future where it stops blocking.
    # apply! returns as soon as it has queued, and synchronize! is what drains
    # the queue before the clock stops.
    @testset "refill! and synchronize! wait for the device" begin
        biggest = argmax(c -> prod(c.dims), filter(c -> c.placement === :inplace, matrix))
        prepared = prepare(VkFFTImpl(), biggest, input(biggest))
        apply!(prepared)
        refill!(prepared)

        for k in 1:4
            refilled = @elapsed refill!(prepared)
            drained = @elapsed synchronize_device()
            apply!(prepared)
            synchronized = @elapsed synchronize!(prepared)
            settled = @elapsed synchronize_device()

            k == 1 && continue # the first pass through a timer pays for compiling what it wraps
            @test drained < refilled / 10
            @test settled < synchronized / 10
        end
    end

    @testset "destructive cases start every sample pristine" begin
        destructive = filter(c -> prod(c.dims) <= 4096 && destroys_input(c), matrix)
        @test any(c -> c.placement === :inplace, destructive)
        @test any(c -> c.family === :c2r, destructive)

        for impl in devices, case in destructive
            supports(impl, case) || continue
            @testset "$impl $(case_id(case))" begin
                x = input(case)
                prepared = prepare(impl, case, x)
                for _ in 1:3
                    @test Array(prepared.x) == x
                    apply!(prepared)
                    refill!(prepared)
                end
                @test Array(prepared.x) == x
            end
        end

        # An in place transform is the one that certainly writes over its input,
        # so it is what shows the refill restoring something rather than checking
        # a buffer nothing touched.
        inplace = argmax(c -> prod(c.dims), filter(c -> c.placement === :inplace, destructive))
        pristine = input(inplace)
        prepared = prepare(VkFFTImpl(), inplace, pristine)
        apply!(prepared)
        @test Array(prepared.x) != pristine
        refill!(prepared)
        @test Array(prepared.x) == pristine

        # A transform that leaves its input alone carries no copy, so the refill
        # compiles away rather than costing it a device copy per sample.
        intact = first(filter(c -> !destroys_input(c), matrix))
        @test prepare(VkFFTImpl(), intact, input(intact)).src === nothing
    end
end
