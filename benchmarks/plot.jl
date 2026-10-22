# The figures, built from the CSVs and from nothing else.
#
#     julia --startup-file=no --project=benchmarks benchmarks/plot.jl <results-dir> --out=<dir>
#
# Nothing here measures anything or opens a device, so changing a colour costs a
# second rather than a benchmark run. One Theme carries every style decision, so
# a round of "lighter bands, bigger labels" is one edit in one place and every
# figure comes back restyled. The light and the dark variant are that same Theme
# under two modes: no figure function knows which one it is drawing in.

# A startup.jl that activates whatever project the working directory holds would
# silently override --project, so this script pins its environment rather than
# trusting the flag.
import Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using CSV
using DataFrames
using Statistics

# The figures ship in pairs so that a docs page or a README can hand the reader
# the one matching their own theme, through <picture> and prefers-color-scheme.
# Each mode draws from the palette column checked against its own surface. Three
# of the light steps sit above the dark lightness band, so one column cannot
# carry both pages honestly.
#
# `minor` differs per mode on purpose. One alpha does not fade the grid by the
# same amount against both surfaces.
const MODES = (
    light=(page="#f9f9f7", surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", minor=0.42,
           series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")),
    dark=(page="#0d0d0d", surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835", minor=0.58,
          series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181")),
)

const FIGSIZE = (880, 520)

# The fields a subtitle may carry. Whichever of them the slice holds one value
# of ends up on the panel, in this order.
const SUBTITLE_FIELDS = (:device, :precision, :family, :direction, :placement)

# The subtitle values that are our own abbreviation rather than the field's own
# shorthand. `c2c`, `r2c` and `c2r` stay as they are, since FFTW and cuFFT print
# them that way too and a reader arrives already knowing them. Nobody outside
# this suite has seen `oop`.
const SPELLED_OUT = Dict("oop" => "out of place", "ip" => "in place")

# The house spelling of every backend and implementation id the CSVs hold. The
# ids themselves are join keys that the resume scan depends on, so they are
# never rewritten on disk. An id with no entry here prints as it is stored
# rather than being guessed at.
const STYLIZED = Dict(
    "fftw" => "FFTW", "vkfft" => "VkFFT", "cufft" => "cuFFT", "mpsgraph" => "MPSGraph",
    "cuda" => "CUDA", "metal" => "Metal", "opencl" => "OpenCL",
)

"""
    bench_theme(mode::Symbol)

Builds the house theme in one of its two modes, `:light` or `:dark`.

Every colour a figure draws with comes from here, including the ones the figure
functions reach for by hand: `inkcolor` is the pen for annotations and rules,
and the palette is where a series colour comes from. The mode is therefore a
parameter of the theme and of nothing else.

# Arguments
- `mode::Symbol`: `:light` or `:dark`

# Returns
- A `Theme`
"""
function bench_theme(mode::Symbol)
    m = MODES[mode]
    return Theme(
        fonts=(regular="TeX Gyre Heros Makie", bold="TeX Gyre Heros Makie Bold"),
        fontsize=15,
        size=FIGSIZE,
        figure_padding=16,
        backgroundcolor=m.page,
        inkcolor=m.ink2,
        palette=(color=collect(m.series),),
        Band=(alpha=0.22,),
        ScatterLines=(linewidth=2, markersize=10),
        Legend=(framevisible=false, labelcolor=m.ink2, titlecolor=m.ink2, patchsize=(20, 14), rowgap=2),
        Axis=(
            backgroundcolor=m.surface,
            titlecolor=m.ink, titlesize=16, titlealign=:left, titlegap=8,
            subtitlecolor=m.ink2, subtitlesize=12.5, subtitlegap=6,
            xlabelcolor=m.ink2, ylabelcolor=m.ink2,
            xticklabelcolor=m.muted, yticklabelcolor=m.muted,
            xgridcolor=m.grid, ygridcolor=m.grid,
            # Nine intervals inside a decade, which is where a log axis wants
            # its minor grid. Makie steps linearly between two major ticks on a
            # log scale, so decade majors put the lines on 2, 3 and so on up to
            # 9 times the decade, where a reader expects them.
            xminorgridvisible=true, yminorgridvisible=true,
            xminorgridcolor=(m.grid, m.minor), yminorgridcolor=(m.grid, m.minor),
            xminorticks=IntervalsBetween(9), yminorticks=IntervalsBetween(9),
            xtickcolor=m.axis, ytickcolor=m.axis,
            leftspinecolor=m.axis, bottomspinecolor=m.axis,
            topspinevisible=false, rightspinevisible=false,
        ),
    )
end

_ink() = Makie.current_default_theme()[:inkcolor][]
_series(slot) = Makie.current_default_theme()[:palette][:color][][slot]

set_theme!(bench_theme(:light)) # so a figure function called on its own draws in the house style rather than in Makie's

"""
    subtitle(df)

Describes the slice a figure drew, as the line that sits under its title.

These figures end up in docs and in a README where the caption can be nowhere
near the image, so the panel itself says what it measured. Only the fields the
slice agrees on appear. A figure with a panel per precision therefore drops the
precision, which its titles already carry. `oop` and `ip` are spelled out,
because they are this suite's own abbreviations rather than the field's.

# Arguments
- `df`: The rows a figure drew, carrying the case identity columns

# Returns
- A `String`, possibly empty
"""
function subtitle(df)
    held = [string(first(df[!, f])) for f in SUBTITLE_FIELDS if allequal(df[!, f])]
    return join([get(SPELLED_OUT, v, v) for v in held], ", ")
end

"""
    label(id)

Turns a backend or implementation id from the CSVs into the name a figure prints.

`fftw-t6` is a join key and stays that way on disk. What a legend needs is FFTW
at six threads, spelled the way the library spells itself. A suffix that names a
thread count is expanded and any other suffix rides along in the parentheses, so
`vkfft-tuned` reads as VkFFT (tuned).

# Arguments
- `id`: A backend or implementation id, such as `"fftw-t6"` or `"opencl"`

# Returns
- A `String`
"""
function label(id)
    name, suffix = match(r"^([^-]+)-?(.*)$", string(id))
    spelled = get(STYLIZED, name, name)
    threads = match(r"^t(\d+)$", suffix)
    threads === nothing && return isempty(suffix) ? spelled : "$spelled ($suffix)"
    return "$spelled ($(threads[1]) thread$(threads[1] == "1" ? "" : "s"))"
end

"""
    load(dir)

Reads a results directory into one row per case, implementation and replication.

`cases.csv` holds what a case is and `summary.csv` holds what it cost, joined on
`case_id`. A case that failed carries no timing, so its row is dropped here
rather than turning into a gap in a figure later.

# Arguments
- `dir`: A results directory holding `cases.csv` and `summary.csv`

# Returns
- A `DataFrame` of the case fields, the timing fields, and `n`, the number of points in the transform
"""
function load(dir)
    identities = filter(:status => ==("ok"), CSV.read(joinpath(dir, "cases.csv"), DataFrame))
    timings = CSV.read(joinpath(dir, "summary.csv"), DataFrame)
    df = innerjoin(select(identities, Not(:run_id)), timings, on=:case_id)
    df.n = [prod(parse.(Int, split(string(d), 'x'))) for d in df.dims]
    return df
end

"""
    load_accuracy(dir)

Reads a results directory's accuracy rows into one row per case, implementation, input class and draw.

The accuracy pass runs once for a whole run and a directory that only timed
things never gets one, so a missing `accuracy.csv` is an ordinary outcome and
comes back as an empty frame rather than as an error.

# Arguments
- `dir`: A results directory holding `cases.csv`, and possibly `accuracy.csv`

# Returns
- A `DataFrame` of the case fields, the error fields, and `n`, or an empty one
"""
function load_accuracy(dir)
    path = joinpath(dir, "accuracy.csv")
    isfile(path) || return DataFrame()
    identities = filter(:status => ==("ok"), CSV.read(joinpath(dir, "cases.csv"), DataFrame))
    df = innerjoin(select(identities, Not(:run_id)), CSV.read(path, DataFrame), on=:case_id)
    df.n = [prod(parse.(Int, split(string(d), 'x'))) for d in df.dims]
    return df
end

"""
    speedup(df, baseline, subject)

Builds P1, the speedup of one implementation over another against transform length.

The ratio is formed inside a replication and only then aggregated across them,
so a replication that ran hot cancels out of the ratio instead of widening the
band. The band is the spread of those per replication ratios, because the
spread between processes is the error bar that belongs on a comparison. The
powers of two are the line and the primes sit on top as their own markers,
because the point of the figure is that the primes are off the trend.

# Arguments
- `df`: The frame `load` returns
- `baseline`: The implementation on top of the ratio, e.g. `"fftw-t1"`
- `subject`: The implementation underneath it, the one being sold, e.g. `"fftw-t6"`

# Returns
- A `Figure`
"""
function speedup(df, baseline, subject)
    sweep = filter(r -> r.family == "c2c" && r.precision == "ComplexF32" && r.direction == "forward" && r.placement == "oop" && r.class in ("pow2", "prime"), df)
    wide = dropmissing(unstack(sweep, [:class, :n, :replication], :implementation, :median), [baseline, subject])
    wide.ratio = wide[!, baseline] ./ wide[!, subject]
    agg = sort!(combine(groupby(wide, [:class, :n]), :ratio => median => :r, :ratio => minimum => :lo, :ratio => maximum => :hi), :n)

    fig = Figure()
    # A ratio is read as a multiple, so the ticks say so. Makie's own log ticks
    # fall back to half decades on a narrow range, and 10^-1.5 is not a number
    # anybody reads off a chart.
    ax = Axis(fig[1, 1], xscale=log10, yscale=log10, yticks=([0.001, 0.01, 0.1, 1, 10, 100, 1000], ["0.001x", "0.01x", "0.1x", "1x", "10x", "100x", "1000x"]), xlabel="Transform length n", ylabel="Speedup", title="Speedup of $(label(subject)) over $(label(baseline))", subtitle=subtitle(sweep))
    hlines!(ax, 1.0, color=(_ink(), 0.7), linestyle=:dash)

    pow2 = filter(:class => ==("pow2"), agg)
    band!(ax, pow2.n, pow2.lo, pow2.hi, color=_series(1))
    scatterlines!(ax, pow2.n, pow2.r, color=_series(1), label="powers of two")

    primes = filter(:class => ==("prime"), agg)
    rangebars!(ax, primes.n, primes.lo, primes.hi, color=_series(2), whiskerwidth=8)
    scatter!(ax, primes.n, primes.r, color=_series(2), marker=:diamond, markersize=14, label="primes")

    axislegend(ax, position=:lt)
    return fig
end

"""
    accuracy(acc)

Builds P2, the relative error of a transform in units of `eps` against its length, one panel per precision.

The error of an FFT grows with the length, so what a reader wants is the growth
rather than any one number, and `eps` is the unit that lets the two precisions
be read the same way. Uniform random is the only input class with any spread, so
it carries the line and the band around it is the range across the draws. Colour
belongs to the implementation and the marker to the shape class, which keeps a
prime on its own implementation's colour instead of turning it into a third
line.

# Arguments
- `acc`: The frame `load_accuracy` returns

# Returns
- A `Figure`
"""
function accuracy(acc)
    uniform = filter(r -> r.input_class == "uniform" && r.family == "c2c" && r.direction == "forward" && r.placement == "oop" && r.class in ("pow2", "prime"), acc)
    agg = sort!(combine(groupby(uniform, [:precision, :implementation, :class, :n]), :eps_multiple => median => :e, :eps_multiple => minimum => :lo, :eps_multiple => maximum => :hi), :n)
    precisions, impls = unique(agg.precision), unique(agg.implementation)

    # An error in units of eps lives inside two decades, where Makie's own log
    # ticks fall back to fractional powers. Nobody reads 10^0.3 off a chart, so
    # the ticks step by the square root of two and say the multiple itself.
    multiples = [0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0]

    # Colour carries the implementation and the marker carries the shape class,
    # so the key has a titled section for each. The shape swatches are drawn in
    # the annotation pen rather than in a series colour, which would read as a
    # third implementation. Both sections sit top left, which this figure leaves
    # empty because the error climbs from left to right.
    series_key = [LineElement(color=_series(slot), linewidth=2) for slot in eachindex(impls)]
    class_key = [MarkerElement(marker=:circle, color=_ink(), markersize=10), MarkerElement(marker=:diamond, color=_ink(), markersize=13)]

    fig = Figure(size=(FIGSIZE[1] * length(precisions), FIGSIZE[2]))
    for (panel, precision) in enumerate(precisions)
        # The y ticks here are not decades, so the minor grid that belongs
        # between two of them would land on nothing a reader is looking for.
        ax = Axis(fig[1, panel], xscale=log10, yscale=log10, yticks=(multiples, string.(multiples)), yminorgridvisible=false, xlabel="Transform length n", ylabel=panel == 1 ? "Relative L2 error, in multiples of eps" : "", title=precision, subtitle=subtitle(uniform))
        rows = filter(:precision => ==(precision), agg)
        for (slot, impl) in enumerate(impls) # the slot belongs to the implementation, not to its rank in this panel, so a colour means the same thing in every panel
            pow2 = filter(r -> r.implementation == impl && r.class == "pow2", rows)
            isempty(pow2) && continue
            band!(ax, pow2.n, pow2.lo, pow2.hi, color=_series(slot))
            scatterlines!(ax, pow2.n, pow2.e, color=_series(slot))

            primes = filter(r -> r.implementation == impl && r.class == "prime", rows)
            isempty(primes) || scatter!(ax, primes.n, primes.e, color=_series(slot), marker=:diamond, markersize=14)
        end

        Legend(fig[1, panel], [series_key, class_key], [label.(impls), ["powers of two", "primes"]], ["Implementation", "Shape"], tellheight=false, tellwidth=false, halign=:left, valign=:top, margin=(12, 12, 12, 12))
    end
    return fig
end

"""
    absolute_time(df)

Builds P3, the time one transform costs against its length, one panel per device.

A flat stretch on the left is the cost of making the call rather than of the
transform, and the climb on the right is the transform. The lead changes hands
where the two lines cross, which the crossing shows without help. The title
names the backend and the subtitle the silicon it ran on, so the panels stay
apart on a machine with more than one device.

# Arguments
- `df`: The frame `load` returns

# Returns
- A `Figure`
"""
function absolute_time(df)
    sweep = filter(r -> r.class == "pow2" && r.family == "c2c" && r.precision == "ComplexF32" && r.direction == "forward" && r.placement == "oop", df)
    agg = sort!(combine(groupby(sweep, [:device, :implementation, :n]), :median => median => :t, :median => minimum => :lo, :median => maximum => :hi), :n)
    devices, impls = unique(agg.device), unique(agg.implementation)

    # Every decade rather than every second one, in the unit a reader thinks in.
    # The panel spans seven decades, and a label on half of them turns reading a
    # value off it into guesswork.
    decades = 10.0 .^ (-9:0)
    labels = [t < 1e-6 ? "$(round(Int, t * 1e9)) ns" : t < 1e-3 ? "$(round(Int, t * 1e6)) μs" : t < 1 ? "$(round(Int, t * 1e3)) ms" : "1 s" for t in decades]

    fig = Figure(size=(FIGSIZE[1] * length(devices), FIGSIZE[2]))
    for (panel, device) in enumerate(devices)
        slice = filter(:device => ==(device), sweep)
        ax = Axis(fig[1, panel], xscale=log10, yscale=log10, yticks=(decades, labels), xlabel="Transform length n", ylabel=panel == 1 ? "Time per transform" : "", title="$(label(first(slice.backend))) Backend", subtitle=subtitle(slice))
        panel_rows = filter(:device => ==(device), agg)
        for (slot, impl) in enumerate(impls) # the slot belongs to the implementation, not to its rank in this panel, so a colour means the same thing in every panel
            line = filter(:implementation => ==(impl), panel_rows)
            isempty(line) && continue
            band!(ax, line.n, line.lo, line.hi, color=_series(slot))
            scatterlines!(ax, line.n, line.t, color=_series(slot), label=label(impl))
        end
        axislegend(ax, position=:lt)
    end
    return fig
end

"""
    render(dir, out)

Writes every figure the data in a results directory supports, in both modes and in both formats.

One invocation produces the whole set. The mode lives in the theme, so a figure
is built once per mode from the same call and saved under a name that says which
one it is. The PNG is rendered at twice the figure size so it holds up on a high
DPI display, and the SVG beside it is what a docs build should prefer.

# Arguments
- `dir`: A results directory holding `cases.csv` and `summary.csv`, and possibly `accuracy.csv`
- `out`: Where the figures go, created if it is not there

# Returns
- The paths written, in order
"""
function render(dir, out)
    df = load(dir)
    acc = load_accuracy(dir)
    mkpath(out)

    # The CPU run races FFTW at one thread against FFTW at all of them, which
    # reads as the speedup from threading. A backend with a vendor library to
    # race hands speedup its own pair instead.
    baseline = "fftw-t1"
    subject = only(filter(!=(baseline), unique(df.implementation)))

    paths = String[]
    for mode in (:light, :dark)
        set_theme!(bench_theme(mode))
        figures = ["p1-speedup" => speedup(df, baseline, subject), "p3-time-vs-n" => absolute_time(df)]
        isempty(acc) || insert!(figures, 2, "p2-accuracy" => accuracy(acc))

        for (name, fig) in figures, extension in ("png", "svg")
            path = joinpath(out, "$name-$mode.$extension")
            save(path, fig, px_per_unit=2)
            push!(paths, path)
        end
    end
    return paths
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || throw(ArgumentError("usage: plot.jl <results-dir> --out=<dir>"))
    flag = match(r"^--out=(.+)$", ARGS[2])
    flag === nothing && throw(ArgumentError("cannot read the argument $(ARGS[2]). The only flag is --out"))
    println("wrote ", length(render(ARGS[1], flag[1])), " files into ", flag[1])
end
