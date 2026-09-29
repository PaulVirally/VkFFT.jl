using FFTW
using Statistics
using Test

include("../run.jl")

const rtol_fftw_f32 = 1e-4
const rtol_fftw_f64 = 1e-11

# A stand in implementation that burns a known number of nanoseconds, so the
# timer can be tested against a cost it does not have to measure to know.
struct Thunk
    nanos::Int
    calls::Base.RefValue{Int}
end

Thunk(nanos::Int) = Thunk(nanos, Ref(0))

function apply!(thunk::Thunk)
    thunk.calls[] += 1
    started = time_ns()
    while time_ns() - started < thunk.nanos
    end
    return nothing
end

synchronize!(::Thunk) = nothing
refill!(::Thunk) = nothing

# Records the order the timer calls the contract in, which is the thing the
# refill is easiest to break by moving.
struct Spy
    steps::Vector{Symbol}
end

refill!(spy::Spy) = (push!(spy.steps, :refill); nothing)
apply!(spy::Spy) = (push!(spy.steps, :apply); nothing)
synchronize!(spy::Spy) = (push!(spy.steps, :synchronize); nothing)

@testset verbose=true "benchmark harness" begin
    @testset "case matrix" begin
        matrix = cases(:fftw)
        ids = case_id.(matrix)

        @test length(unique(ids)) == length(ids)
        @test ids == case_id.(cases(:fftw))
        @test all(c -> prod(c.dims) <= 2^22, matrix) # the CPU paths cap at 2^22
        @test !any(c -> c.class === :cliff, matrix)  # the cliff is a CUDA against cuFFT measurement
        @test !any(c -> c.tuned, matrix)             # the tuner belongs to VkFFT, which an fftw sweep never loads

        # The axes are not a product: the real families ride pow2 and prime, and
        # the inverse and in place rows ride the pow2 ComplexF32 sweep alone.
        @test all(c -> c.class in (:pow2, :prime), filter(c -> c.family !== :c2c, matrix))
        @test all(c -> c.class === :pow2 && c.precision === ComplexF32, filter(c -> c.placement === :inplace, matrix))
        @test all(c -> c.class === :pow2 && c.precision === ComplexF32, filter(c -> c.family === :c2c && c.direction === :inverse, matrix))
        @test all(c -> c.direction === :inverse, filter(c -> c.family === :c2r, matrix))
        @test !any(c -> c.placement === :inplace && c.precision === ComplexF64, matrix)

        # 33 of the 37 shapes fit under the CPU cap: 66 complex to complex rows,
        # 84 real family rows over the 21 surviving pow2 and prime lengths, and
        # 34 inverse and in place rows.
        @test length(matrix) == 184

        # The whole matrix a backend with no limits enumerates, as PLAN_BENCH
        # section 4 counts it. CUDA is that backend: an A6000 holds 48 GB and
        # the largest shape in the list is a quarter of a gigabyte.
        @test CAPABILITIES[:cuda] == (types=(ComplexF32, ComplexF64), max_elements=typemax(Int), cliff=true, tuning=true)
        cuda = cases(:cuda)
        @test length(cuda) == 259
        @test count(c -> c.class === :cliff, cuda) == 31
        @test length(unique(case_id.(cuda))) == 259
        @test maximum(c -> prod(c.dims), cuda) == 2^24

        # The tuning rows ride the power of two sweep and the squares, at single
        # precision forward out of place and nowhere else, so P7 is 24 shapes
        # and the main matrix is the 235 it was without them.
        tuned = filter(c -> c.tuned, cuda)
        @test length(tuned) == 24
        @test Set(c.class for c in tuned) == Set(TUNED_CLASSES)
        @test all(c -> (c.family, c.precision, c.direction, c.placement) == (:c2c, ComplexF32, :forward, :oop), tuned)
        @test sort([backend for (backend, caps) in CAPABILITIES if !caps.tuning]) == [:fftw]

        # A tuned case and the plain case of the same shape differ in one field,
        # so the id has to carry it or the two overwrite each other.
        twins = filter(c -> c.dims == (1024,) && c.class === :pow2 && (c.family, c.precision, c.direction, c.placement) == (:c2c, ComplexF32, :forward, :oop), cuda)
        @test length(twins) == 2 && length(unique(case_id.(twins))) == 2
        @test Set(c.tuned for c in twins) == Set((false, true))

        # The cliff is enumerated on the one backend whose duel produces the
        # figure and nowhere else, which is also what keeps 4093 and 8191 out of
        # a pocl sweep's cliff.
        @test sort([backend for (backend, caps) in CAPABILITIES if caps.cliff]) == [:cuda]

        # The run from 4080 to 4110 straddles 4096, so that length is enumerated
        # twice and only the class tells the two rows apart. Drop the class from
        # case_id and the cliff silently overwrites a pow2 row.
        straddled = filter(c -> c.dims == (4096,) && !c.tuned && c.family === :c2c && c.precision === ComplexF32 && c.direction === :forward && c.placement === :oop, cuda)
        @test Set(c.class for c in straddled) == Set((:pow2, :cliff))
        @test length(unique(case_id.(straddled))) == 2

        # Every cliff row is the one combination PLAN_BENCH section 4 asks for,
        # so the 31 lengths stay one curve against one rival rather than four.
        @test all(c -> (c.family, c.precision, c.direction, c.placement) == (:c2c, ComplexF32, :forward, :oop), filter(c -> c.class === :cliff, cuda))
    end

    @testset "input" begin
        matrix = cases(:fftw)
        complex_case = first(filter(c -> c.family === :c2c && c.precision === ComplexF32, matrix))
        @test input(complex_case) == input(complex_case)
        @test eltype(input(complex_case)) === ComplexF32
        @test size(input(complex_case)) == complex_case.dims
        @test all(v -> -1 <= real(v) < 1 && -1 <= imag(v) < 1, input(complex_case))

        real_case = first(filter(c -> c.family === :r2c, matrix))
        @test eltype(input(real_case)) === real(real_case.precision)
        @test size(input(real_case)) == real_case.dims

        # A complex to real case takes the halved spectrum, not the full signal,
        # and that spectrum has to be Hermitian for the transform to be defined.
        for halved in filter(c -> c.family === :c2r && prod(c.dims) <= 8192, matrix)
            spectrum = input(halved)
            @test size(spectrum) == (halved.dims[1] ÷ 2 + 1,)
            @test imag(spectrum[1]) == 0
            @test iseven(halved.dims[1]) == (imag(spectrum[end]) == 0)
            @test any(v -> imag(v) != 0, spectrum[2:end - 1]) # only the two pinned bins are real
        end

        # The seed is the shape alone, so a shape reads the same numbers at
        # every precision it is measured at.
        pair = filter(c -> c.family === :c2c && c.direction === :forward && c.placement === :oop && c.dims == (1024,), matrix)
        @test length(pair) == 2
        @test ComplexF64.(input(pair[1])) ≈ input(pair[2]) rtol=rtol_fftw_f32
    end

    @testset "duel timer" begin
        slow, fast = Thunk(10_000_000), Thunk(6_000_000)
        timed = duel(Any[slow, fast]; budget=0.05)
        taken = timed.samples
        @test length(taken) == 2
        @test length(taken[1]) == length(taken[2])        # the round robin visits both equally
        @test timed.inner == [1, 1]                        # both are already longer than the timer target
        @test slow.calls[] == fast.calls[] == length(taken[1]) + 3 # plus a warmup and two sizing calls each
        @test sum(taken[1]) >= 0.05                        # the slowest spends the budget
        @test all(t -> t > 5e-3, taken[1])

        @test length(only(duel(Any[Thunk(1_000)]; budget=1e-9, min_samples=37).samples)) == 37
        @test length(only(duel(Any[Thunk(1_000)]; budget=1e9, max_samples=50).samples)) == 50

        # A thunk far too slow for the budget is stopped by the ceiling on the
        # whole duel, well before the floor of samples is reached.
        crawling = Thunk(20_000_000)
        @test length(only(duel(Any[crawling]; min_samples=1000, ceiling=0.1).samples)) < 20

        # A microsecond is 24 ticks of the 41.7 ns clock on Apple Silicon, so it
        # is batched. What comes back is still the cost of one application.
        quick = Thunk(1_000)
        batched = duel(Any[quick]; budget=1e-9, min_samples=5)
        @test 10 < only(batched.inner) <= TIMER_TARGET / 1e-6
        @test all(t -> 5e-7 < t < 5e-6, only(batched.samples))
        @test only(duel(Any[Thunk(1_000)]; budget=1e-9, min_samples=5, repeatable=false).inner) == 1

        # Every sample is a refill, then its batch of applies, then one wait,
        # warmup included, so a transform never meets the previous sample's
        # output and the clock never stops on work still queued.
        spy = Spy(Symbol[])
        rounds = length(only(duel(Any[spy]; budget=1e-9, min_samples=8, repeatable=false).samples))
        @test spy.steps == repeat([:refill, :apply, :synchronize], rounds + 1)

        batching = Spy(Symbol[])
        batches = duel(Any[batching]; budget=1e-9, min_samples=3)
        rounds, size = length(only(batches.samples)), only(batches.inner)
        applied = count(==(:apply), batching.steps)
        @test size > 1 # an apply of a few tens of nanoseconds is batched
        @test batching.steps[1:3] == [:refill, :apply, :synchronize]
        @test count(==(:refill), batching.steps) == rounds + 3 # one per sample, and one for the warmup and each sizing call
        @test count(==(:synchronize), batching.steps) == rounds + 3
        @test batching.steps[end] === :synchronize
        @test all(i -> batching.steps[i - 1] === :synchronize, findall(==(:refill), batching.steps)[2:end])
        @test 2 + size * rounds < applied <= 2 + MAX_INNER + size * rounds # the sizing batch is the only unknown
    end

    @testset "summary statistics" begin
        samples = collect(1.0:100.0)
        stat = summarize(samples)

        @test stat.repeats == 100
        @test stat.min == 1.0
        @test stat.median == median(samples)
        @test stat.q25 ≈ quantile(samples, 0.25)
        @test stat.q75 ≈ quantile(samples, 0.75)
        @test stat.ci_lo <= stat.median <= stat.ci_hi
        @test stat.ci_rel ≈ (stat.ci_hi - stat.ci_lo) / 2 / stat.median
        @test summarize(samples) == summarize(samples) # the bootstrap seed is fixed

        constant = summarize(fill(2.5, 40))
        @test constant.ci_lo == constant.ci_hi == 2.5
        @test constant.ci_rel == 0.0
    end

    @testset "csv writer" begin
        path = joinpath(mktempdir(), "rows.csv")
        append_csv(path, (:a, :b), [("x", 1.0)])
        append_csv(path, (:a, :b), [("y, z", 2.5), ("plain", 1 / 3)])
        lines = readlines(path)

        @test lines == ["a,b", "x,1", "\"y, z\",2.5", "plain,0.333333"]
        @test_throws ArgumentError append_csv(path, (:a, :b), [("one",)])
    end

    @testset "resume" begin
        path = joinpath(mktempdir(), "summary.csv")
        @test isempty(completed(path, 3))

        append_csv(path, SUMMARY_COLUMNS, [("run", "case-a", "fftw-t1", 2, 1, 20, 4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, "MEASURE threads=1")])
        @test completed(path, 3) == Set([("case-a", "fftw-t1", "2")])
        @test !(("case-a", "fftw-t6", "2") in completed(path, 3))
        @test completed(path, 2) == Set([("case-a", "fftw-t1")]) # the key accuracy.csv resumes on
    end

    @testset "fftw implementation" begin
        small = filter(c -> prod(c.dims) <= 65536, cases(:fftw))
        for case in unique(c -> (c.family, c.direction, c.placement, c.precision, length(c.region)), small)
            @testset "$(case_id(case))" begin
                x = input(case)
                prepared = prepare(FFTWImpl(1), case, x)
                apply!(prepared)

                expected = if case.family === :r2c
                    rfft(x, case.region)
                elseif case.family === :c2r
                    irfft(x, case.dims[1], case.region)
                elseif case.direction === :inverse
                    ifft(x, case.region)
                else
                    fft(x, case.region)
                end
                @test result(prepared) ≈ expected rtol=(real(case.precision) === Float32 ? rtol_fftw_f32 : rtol_fftw_f64)
            end
        end
    end

    @testset "destructive cases start every sample pristine" begin
        destructive = filter(c -> prod(c.dims) <= 256 && (c.placement === :inplace || c.family === :c2r), cases(:fftw))
        @test any(c -> c.placement === :inplace, destructive)
        @test any(c -> c.family === :c2r, destructive)

        for case in destructive
            @testset "$(case_id(case))" begin
                x = input(case)
                prepared = prepare(FFTWImpl(1), case, x)
                @test prepared.src !== nothing
                for _ in 1:3
                    @test prepared.x == x
                    apply!(prepared)
                    refill!(prepared)
                end
                @test prepared.x == x
            end
        end

        # An in place transform is the one that certainly writes over its input,
        # so it is what shows the refill is restoring something rather than
        # checking a buffer nothing touched. FFTW's small complex to real plans
        # happen to leave their input alone, and its large ones do not, which is
        # why every case that can destroy carries a source copy.
        inplace = first(filter(c -> c.placement === :inplace, destructive))
        pristine = input(inplace)
        prepared = prepare(FFTWImpl(1), inplace, pristine)
        apply!(prepared)
        @test prepared.x != pristine
        refill!(prepared)
        @test prepared.x == pristine

        # A transform that leaves its input alone carries no source copy, so the
        # refill compiles away rather than costing it a memcpy per sample.
        intact = first(filter(c -> c.family === :r2c, cases(:fftw)))
        @test prepare(FFTWImpl(1), intact, input(intact)).src === nothing
    end

    @testset "replication end to end" begin
        out = mktempdir()
        tiny = filter(c -> prod(c.dims) <= 128, cases(:fftw))
        impls = (FFTWImpl(1),)
        @test length(tiny) > 1

        redirect_stdout(devnull) do
            _run_replication(:fftw, impls, tiny, out, 1, "cpu")
        end
        rows = readlines(joinpath(out, "summary.csv"))
        @test length(rows) == length(tiny) + 1
        @test length(readlines(joinpath(out, "cases.csv"))) == length(tiny) + 1
        @test length(completed(joinpath(out, "summary.csv"), 3)) == length(tiny)

        manifest = TOML.parsefile(joinpath(out, "manifest.toml"))
        @test manifest["machine"] == gethostname()
        @test sort(manifest["order"]["1"]) == sort(case_id.(tiny))
        @test manifest["seed"]["1"] == ORDER_SEED + 1
        @test haskey(manifest["packages"], "FFTW")

        # A second pass over the same directory finds every triple recorded and
        # writes nothing.
        redirect_stdout(devnull) do
            _run_replication(:fftw, impls, tiny, out, 1, "cpu")
        end
        @test readlines(joinpath(out, "summary.csv")) == rows
    end

    @testset "reference" begin
        include("reference_tests.jl")
    end

    @testset "accuracy" begin
        include("accuracy_tests.jl")
    end

    # device_tests.jl is not here. It needs a device and one of the backend
    # environments, so it runs from those instead.
    @testset "figures" begin
        include("plot_tests.jl")
    end
end
