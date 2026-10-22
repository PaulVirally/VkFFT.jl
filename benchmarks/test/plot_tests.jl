using CSV
using DataFrames
using Statistics
using Test

include("../plot.jl")

# Three cases, two implementations, three replications, with the timings chosen
# so that every aggregate the figures compute has an answer that can be read off
# by eye. fftw-t1 costs 4, 5 and 6 at n=64 against a flat 2, so the median ratio
# is 2.5 and the ratios span 2 to 3. pow2-128 carries accuracy rows and no
# timings, so the two loaders are not reading the same case list.
function _fixture(dir; accuracy=false)
    CSV.write(joinpath(dir, "cases.csv"), DataFrame(
        run_id="r", case_id=["pow2-64", "prime-4093", "broken", "pow2-128"], machine="m", device="cpu", backend="fftw",
        class=["pow2", "prime", "pow2", "pow2"], label=["2^6", "4093", "2^7", "2^7"], family="c2c", precision="ComplexF32",
        dims=["64", "4093", "128", "128"], region="1", direction="forward", placement="oop", tuned=false,
        status=["ok", "ok", "error", "ok"], message=""))

    rows = [(case_id=id, implementation=impl, replication=k, median=t)
            for (id, impl, ts) in (("pow2-64", "fftw-t1", (4.0, 5.0, 6.0)), ("pow2-64", "fftw-t6", (2.0, 2.0, 2.0)),
                                   ("prime-4093", "fftw-t1", (9.0, 9.0, 9.0)), ("prime-4093", "fftw-t6", (9.0, 9.0, 9.0)))
            for (k, t) in enumerate(ts)]
    CSV.write(joinpath(dir, "summary.csv"), insertcols!(DataFrame(rows), 1, :run_id => "r"))

    accuracy && CSV.write(joinpath(dir, "accuracy.csv"), DataFrame(
        run_id="r", case_id=["pow2-64", "pow2-64", "pow2-128", "pow2-128", "prime-4093", "prime-4093"],
        implementation=["fftw-t1", "fftw-t6", "fftw-t1", "fftw-t6", "fftw-t1", "fftw-t6"],
        input_class="uniform", draw=1, relerr=1.0e-7, eps_multiple=[0.6, 0.7, 0.8, 0.9, 1.3, 1.4], reference="fftw-f64"))
    return dir
end

@testset verbose=true "figures" begin
    @testset "load" begin
        df = load(_fixture(mktempdir()))

        # The two cases that ran, times two implementations, times three
        # replications. The case that errored has no timing and is not in here,
        # and neither is the one the timing sweep never reached.
        @test nrow(df) == 12
        @test !("broken" in df.case_id)
        @test df.n == [d == "pow2-64" ? 64 : 4093 for d in df.case_id]
        @test issubset([:class, :device, :median, :replication], propertynames(df))

        # A results directory that grows an accuracy.csv must not change what the
        # timing figures read.
        @test isequal(load(_fixture(mktempdir(), accuracy=true)), df)
    end

    @testset "between replication spread" begin
        dir = _fixture(mktempdir())
        fig = speedup(load(dir), "fftw-t1", "fftw-t6")
        @test fig isa Figure

        # The ratio is formed inside a replication, so n=64 reads 4/2, 5/2 and
        # 6/2: a median of 2.5 spanning 2 to 3. The prime is a flat 1.0.
        band = only(filter(p -> p isa Makie.Band, fig.content[1].scene.plots))
        @test band[1][][1][2] == 2.0 && band[2][][1][2] == 3.0
        line = only(filter(p -> p isa Makie.ScatterLines, fig.content[1].scene.plots))
        @test line[1][] == [Point2f(64, 2.5)]

        panels = absolute_time(load(dir))
        @test length(filter(c -> c isa Axis, panels.content)) == 1 # one device in the fixture
    end

    @testset "the accuracy pass is optional" begin
        # The accuracy pass runs once per run, so a directory that only timed
        # things has no accuracy.csv at all.
        @test isempty(load_accuracy(_fixture(mktempdir())))

        acc = load_accuracy(_fixture(mktempdir(), accuracy=true))
        @test nrow(acc) == 6
        @test issubset([:class, :precision, :eps_multiple, :n], propertynames(acc))
        @test accuracy(acc) isa Figure
    end

    @testset "the subtitle says what the slice holds" begin
        df = load(_fixture(mktempdir()))

        # The placement is spelled out and the family is not, because c2c is the
        # field's own shorthand and oop is only ours.
        @test subtitle(df) == "cpu, ComplexF32, c2c, forward, out of place"

        # A field the slice does not agree on belongs to the panel titles rather
        # than to the line under them.
        df.precision = repeat(["ComplexF32", "ComplexF64"], inner=6)
        @test subtitle(df) == "cpu, c2c, forward, out of place"

        df.placement .= "ip"
        @test subtitle(df) == "cpu, c2c, forward, in place"
    end

    @testset "ids are stylized for display only" begin
        @test label("fftw-t6") == "FFTW (6 threads)"
        @test label("fftw-t1") == "FFTW (1 thread)" # not "1 threads"
        @test label("vkfft-tuned") == "VkFFT (tuned)"
        @test label.(["cufft", "mpsgraph", "opencl", "metal"]) == ["cuFFT", "MPSGraph", "OpenCL", "Metal"]

        # An id nobody has given a spelling to prints as the CSV stores it,
        # rather than as a guess at how its library capitalizes itself.
        @test label("clfft") == "clfft"
    end

    @testset "render writes what it says" begin
        out = joinpath(mktempdir(), "figures")
        paths = render(_fixture(mktempdir(), accuracy=true), out)

        expected = ["$name-$mode.$extension" for name in ("p1-speedup", "p2-accuracy", "p3-time-vs-n") for mode in ("light", "dark") for extension in ("png", "svg")]
        @test sort(basename.(paths)) == sort(expected)
        @test sort(readdir(out)) == sort(expected)
        @test all(p -> occursin("<svg", read(p, String)), filter(endswith(".svg"), paths))

        # Each mode renders its own bytes, so the pair is two variants rather
        # than one figure saved twice under two names.
        @test read(joinpath(out, "p1-speedup-light.png")) != read(joinpath(out, "p1-speedup-dark.png"))

        # Without the accuracy pass the set is the same one less P2.
        @test length(render(_fixture(mktempdir()), joinpath(mktempdir(), "figures"))) == 8
    end
end
