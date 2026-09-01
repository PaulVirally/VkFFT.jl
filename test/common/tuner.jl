# The autotuner and its on-disk records. The records live in the throwaway cache
# directory the runner set up, and sweeps are counted rather than timed:
# VkFFT.sweep_count() says whether a tune=true swept or looked a record up.

# Shapes used nowhere else in the suite, so that a record written here is one
# this file wrote. Small, because a sweep builds sixteen plans.
const TUNE_SHAPE = (40, 10)
const TUNE_REAL_SHAPE = (40, 6)

"""
    _record_files()

Returns the names of the tuning records currently on disk.
"""
_record_files() = filter(name -> startswith(name, "tune_"), readdir(VkFFT.disk_cache_dir()))

"""
    _config_for(dims::Tuple, region::Tuple; kwargs...)

Returns the `VkFFTConfig` a plain forward c2c plan of this shape would be built from.
"""
_config_for(dims::Tuple, region::Tuple; kwargs...) = VkFFT._app_config(ComplexF32, VkFFT._map_region(dims, region, VkFFT._max_dims()), VkFFT.FORWARD, false, false, false; kwargs...)

@testset verbose = true "autotuner" begin
    @testset "the disk key" begin
        roots = VkFFT._device_roots(_upload(_noise(ComplexF32, (16,))))
        key = VkFFT._device_key(roots, Val(BACKEND.name))
        @test !isempty(key)
        @test key == VkFFT._device_key(VkFFT._device_roots(_upload(_noise(ComplexF32, (16,)))), Val(BACKEND.name)) # stable within a session
        @test occursin(BACKEND.device_name, key)

        digest = VkFFT._library_digest()
        @test length(digest) == 64
        @test digest == VkFFT._library_digest() # computed once, kept

        base = _config_for((64, 4), (1, 2))
        record(config) = VkFFT._disk_key(config, roots, Val(BACKEND.name))

        @test length(record(base)) == 64
        @test record(base) != record(_config_for((64, 8), (1, 2)))
        @test record(base) == record(_config_for((64, 4), (1, 2)))

        # The tuning fields enter the key as zero: the record's content is the
        # values they should take, so they cannot also be part of its key.
        tuned = _config_for((64, 4), (1, 2); coalesced_memory=64, aim_threads=256)
        @test record(tuned) == record(base)

        # The per-axis tuples flatten into the key without losing their
        # boundaries, so a config differing only inside one of them gets its
        # own key.
        padded = _config_for((64, 4), (1, 2); zeropad=(2, 3))
        @test record(padded) != record(base)
    end

    @testset "a record round-trips on disk" begin
        path = joinpath(VkFFT.disk_cache_dir(), "tune_roundtrip.txt")
        @test VkFFT._write_record(path, 64, 256)
        @test VkFFT._read_record(path) == (64, 256)
        @test startswith(readline(path), "#") # the comment line survives the parse

        # Anything unparseable is a miss, not an error.
        write(path, "not a record\n")
        @test VkFFT._read_record(path) === nothing
        write(path, "-3 128\n")
        @test VkFFT._read_record(path) === nothing
        rm(path; force=true)
        @test VkFFT._read_record(path) === nothing
    end

    @testset "the tuner sweeps once and looks up after" begin
        VkFFT.clear_cache!()
        VkFFT.clear_tuning!()

        x = _upload(_noise(ComplexF32, TUNE_SHAPE))
        reference = fft(Array(x), (1, 2))
        sweeps = VkFFT.sweep_count()

        plan = VkFFT.plan_fft(x, (1, 2); tune=true)
        @test VkFFT.sweep_count() == sweeps + 1
        @test _relmax(plan * x, reference) < BACKEND.rtol_f32
        @test length(_record_files()) == 1

        grid = VkFFT.last_sweep()
        @test length(grid) == length(VkFFT.TUNE_COALESCED_MEMORY) * length(VkFFT.TUNE_AIM_THREADS)
        @test issorted(grid, by=last)
        @test all(entry -> entry[3] > 0, grid)
        @test Set(entry[1] for entry in grid) == Set(VkFFT.TUNE_COALESCED_MEMORY)
        @test Set(entry[2] for entry in grid) == Set(VkFFT.TUNE_AIM_THREADS)

        println("$(BACKEND.name) sweep on $(join(TUNE_SHAPE, 'x')) ComplexF32, fastest first:")
        for (coalesced_memory, aim_threads, microseconds) in grid
            println("  coalesced_memory=$coalesced_memory aim_threads=$aim_threads $(round(microseconds, digits=1)) us")
        end

        # The record holds the winner, and the winner is the plan that was kept.
        record = VkFFT._read_record(joinpath(VkFFT.disk_cache_dir(), only(_record_files())))
        @test record == (grid[1][1], grid[1][2])

        # Every loser was dropped from the plan cache and freed, so one plan of
        # this shape is left.
        @test VkFFT.cache_size() == 1

        # The second call is a lookup: no sweep, and the same plan back.
        second = VkFFT.plan_fft(x, (1, 2); tune=true)
        @test VkFFT.sweep_count() == sweeps + 1
        @test second === plan

        # And so it is in a fresh plan cache, where only the record survives.
        VkFFT.clear_cache!()
        third = VkFFT.plan_fft(x, (1, 2); tune=true)
        @test VkFFT.sweep_count() == sweeps + 1
        @test _relmax(third * x, reference) < BACKEND.rtol_f32

        # :force sweeps again over the same record.
        VkFFT.plan_fft(x, (1, 2); tune=:force)
        @test VkFFT.sweep_count() == sweeps + 2
        @test length(_record_files()) == 1

        # clear_tuning! removes exactly the records, and the next tune sweeps.
        @test VkFFT.clear_tuning!() == 1
        @test isempty(_record_files())
        VkFFT.clear_cache!()
        VkFFT.plan_fft(x, (1, 2); tune=true)
        @test VkFFT.sweep_count() == sweeps + 3
    end

    @testset "a tuned plan and an untuned one coexist" begin
        VkFFT.clear_cache!()
        VkFFT.clear_tuning!()

        x = _upload(_noise(ComplexF32, TUNE_SHAPE))
        reference = fft(Array(x), (1, 2))

        tuned = VkFFT.plan_fft(x, (1, 2); tune=true)
        plain = VkFFT.plan_fft(x, (1, 2))
        record = VkFFT._read_record(joinpath(VkFFT.disk_cache_dir(), only(_record_files())))

        # They are the same plan only when the sweep landed on VkFFT's defaults.
        @test (tuned === plain) == (record == (0, 0))
        @test _relmax(tuned * x, reference) < BACKEND.rtol_f32
        @test _relmax(plain * x, reference) < BACKEND.rtol_f32
    end

    @testset "tune reaches the real and real-to-real families" begin
        VkFFT.clear_cache!()
        VkFFT.clear_tuning!()

        h = _noise(Float32, TUNE_REAL_SHAPE)
        forward = VkFFT.plan_rfft(_upload(h), (1, 2); tune=true)
        @test _relmax(forward * _upload(h), rfft(h, (1, 2))) < BACKEND.rtol_f32

        spectrum = rfft(h, (1, 2))
        inverse = VkFFT.plan_irfft(_upload(spectrum), TUNE_REAL_SHAPE[1], (1, 2); tune=true)
        @test _relmax(inverse * _upload(spectrum), h) < BACKEND.rtol_f32

        cosine = VkFFT.plan_dct(_upload(h), (1, 2); tune=true)
        @test _relmax(cosine * _upload(h), FFTW.plan_r2r(h, FFTW.REDFT10, [1, 2]) * h) < BACKEND.rtol_f32

        @test length(_record_files()) == 3
    end

    @testset "tune refuses what it cannot read" begin
        x = _upload(_noise(ComplexF32, (16, 4)))
        @test_throws ArgumentError VkFFT.plan_fft(x, (1, 2); tune=:nope)
        @test_throws ArgumentError VkFFT.plan_fft(x, (1, 2); tune=1)
        @test_throws ArgumentError VkFFT.plan_rfft(_upload(_noise(Float32, (16, 4))), (1, 2); tune="yes")
    end

    @testset "a plan with nothing to transform tunes without sweeping" begin
        VkFFT.clear_tuning!()
        sweeps = VkFFT.sweep_count()

        x = _upload(_noise(ComplexF32, (8, 4)))
        plan = VkFFT.plan_fft(x, (); tune=true)
        @test VkFFT._is_trivial(plan)
        @test VkFFT.sweep_count() == sweeps
        @test isempty(_record_files())
        @test Array(plan * x) == Array(x)
    end
end
