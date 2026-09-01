# Region mapping, exhaustively, with no GPU and no wrapper involved: _map_region
# and _canonical_region are pure functions of a size tuple and a region tuple.

"""
    _collapsed_pattern(dims::Tuple, region::Tuple)

Returns the `(length, transformed)` axis list that a mapped layout must reproduce.

Derived differently from `_map_region` on purpose, so it cross-checks rather
than restates it: the layout is read off as `[gap] T [gap] T ... T [gap]`, one
axis per transformed dimension with the product of the length-1-free batch
dimensions between them collapsed into one axis, and empty gaps dropped. The
trailing gap is what `number_batches` absorbs.
"""
function _collapsed_pattern(dims::Tuple, region::Tuple)
    kept = [i for i in 1:length(dims) if dims[i] != 1]
    fft_axes = [i for i in kept if i in region]
    boundaries = [0; fft_axes; length(dims) + 1]

    runs = Tuple{Int, Bool}[]
    for k in 1:(length(boundaries) - 1)
        low, high = boundaries[k], boundaries[k + 1]
        gap = prod(dims[i] for i in kept if low < i < high; init=1)
        gap == 1 || push!(runs, (gap, false))
        high <= length(dims) && push!(runs, (dims[high], true))
    end
    return runs
end

"""
    _expand(layout::VkFFT.VkFFTLayout)

Returns the layout's axes as `(length, transformed)` pairs, batch axes merged as they arrive.

Runs of transformed axes stay separate, because a transform along two adjacent
axes is not the same transform as one along their product.
"""
function _expand(layout::VkFFT.VkFFTLayout)
    runs = Tuple{Int, Bool}[]
    for i in 1:layout.fft_dim
        push!(runs, (Int(layout.size[i]), layout.omit[i] == 0))
    end
    return runs
end

"""
    _check_invariants(dims::Tuple, region::Tuple)

Asserts every structural property a mapped layout has to satisfy.
"""
function _check_invariants(dims::Tuple, region::Tuple)
    layout = VkFFT._map_region(dims, region)
    pattern = _collapsed_pattern(dims, region)
    transform_volume = prod(i -> dims[i], region; init=1)

    if !any(last, pattern)
        # Nothing survives collapsing, so no VkFFT app may be built at all.
        @test VkFFT._is_trivial(layout)
        @test transform_volume == 1
        return nothing
    end

    @test !VkFFT._is_trivial(layout)
    @test 1 <= layout.fft_dim <= VkFFT.VKFFT_MAX_FFT_DIMENSIONS
    @test layout.number_batches >= 1

    # Slots past fft_dim must be zero. The wrapper reads all of them.
    @test all(i -> layout.size[i] == 0 && layout.omit[i] == 0, (layout.fft_dim + 1):VkFFT.VKFFT_MAX_FFT_DIMENSIONS)

    expanded = _expand(layout)
    @test all(len -> len >= 2, first.(expanded))               # length-1 axes were dropped
    @test last(expanded[end])                                  # trailing batch run was folded away
    @test !any(k -> !expanded[k][2] && !expanded[k + 1][2], 1:(length(expanded) - 1)) # batch runs merged

    # The interleaving pattern, plus the folded trailing batch, must reconstruct
    # the region.
    expected = copy(pattern)
    if !last(expected[end])
        @test layout.number_batches == expected[end][1]
        pop!(expected)
    else
        @test layout.number_batches == 1
    end
    @test expanded == expected

    # Volume is conserved, and the transformed volume is exactly the
    # normalization factor.
    @test prod(first.(expanded)) * layout.number_batches == prod(dims)
    @test prod(len for (len, is_fft) in expanded if is_fft; init=1) == transform_volume

    return nothing
end

@testset verbose = true "regions" begin
    @testset "canonicalization" begin
        @test VkFFT._canonical_region((2, 1), 3) == (1, 2)
        @test VkFFT._canonical_region(3:-1:1, 3) == (1, 2, 3)
        @test VkFFT._canonical_region([3, 1], 3) == (1, 3)
        @test VkFFT._canonical_region(2, 3) == (2,)
        @test VkFFT._canonical_region((), 3) == ()
        @test VkFFT._canonical_region(1:3, 3) == (1, 2, 3)

        # Tuple and integer regions keep their length in the type, so the plan's
        # region-length type parameter is inferable for them.
        @test (@inferred VkFFT._canonical_region((3, 1), 3)) === (1, 3)
        @test (@inferred VkFFT._canonical_region(2, 3)) === (2,)

        @test_throws ArgumentError VkFFT._canonical_region((1, 1), 3)
        @test_throws ArgumentError VkFFT._canonical_region((0,), 3)
        @test_throws ArgumentError VkFFT._canonical_region((4,), 3)
        @test_throws ArgumentError VkFFT._canonical_region((-1,), 3)
    end

    @testset "golden cases" begin
        # dims, region => (fft_dim, size[1:fft_dim], omit[1:fft_dim],
        # number_batches)
        cases = ((((8,), (1,)),                 (1, (8,), (0,), 1)),
                 (((8, 5), (1, 2)),             (2, (8, 5), (0, 0), 1)),
                 (((8, 5), (1,)),               (1, (8,), (0,), 5)),           # trailing batch folds
                 (((8, 5), (2,)),               (2, (8, 5), (1, 0), 1)),       # leading batch stays as an omit
                 (((8, 5, 4), (1, 3)),          (3, (8, 5, 4), (0, 1, 0), 1)), # interleaved
                 (((8, 5, 4), (2,)),            (2, (8, 5), (1, 0), 4)),       # both kinds at once
                 (((8, 5, 4), (3,)),            (2, (40, 4), (1, 0), 1)),      # adjacent batch axes merge
                 (((8, 5, 4), (1, 2)),          (2, (8, 5), (0, 0), 4)),
                 (((2, 3, 4, 5, 6), (2, 4)),    (4, (2, 3, 4, 5), (1, 0, 1, 0), 6)),
                 (((2, 3, 4, 5, 6), (1, 2)),    (2, (2, 3), (0, 0), 120)),
                 (((4, 1, 5), (1, 2, 3)),       (2, (4, 5), (0, 0), 1)),       # length-1 axis dropped
                 (((4, 1, 5), (1, 3)),          (2, (4, 5), (0, 0), 1)),       # ... same layout
                 (((1, 8), (1, 2)),             (1, (8,), (0,), 1)),
                 (((3, 1, 4), (2, 3)),          (2, (3, 4), (1, 0), 1)))

        for ((dims, region), (fft_dim, sizes, omits, number_batches)) in cases
            layout = VkFFT._map_region(dims, VkFFT._canonical_region(region, length(dims)))
            @testset "$(join(dims, 'x')) region=$region" begin
                @test layout.fft_dim == fft_dim
                @test layout.size[1:fft_dim] == UInt64.(sizes)
                @test layout.omit[1:fft_dim] == UInt64.(omits)
                @test layout.number_batches == number_batches
            end
        end
    end

    @testset "nothing to transform" begin
        # The SILENT NO-OP trap: VkFFT builds these plans, returns VKFFT_SUCCESS
        # and runs no kernel, so the planner must not hand them over.
        for (dims, region) in (((8,), ()), ((8, 5), ()), ((1,), (1,)), ((1, 1), (1, 2)),
                               ((4, 1), (2,)), ((1, 5, 1), (1, 3)), ((2, 1, 3), (2,)))
            layout = VkFFT._map_region(dims, VkFFT._canonical_region(region, length(dims)))
            @test VkFFT._is_trivial(layout)
            @test layout.fft_dim == 0
        end
    end

    @testset "exhaustive invariants" begin
        # Exhaustive over every size-and-mask combination up to 3d, which is
        # where every interleaving class (leading, middle and trailing runs,
        # merged batch axes, dropped length-1 axes) first occurs. 4d adds only
        # longer run patterns, so it sweeps every mask over the (1, 2) alphabet
        # plus two shapes with unequal lengths for the batch-merge arithmetic.
        # The 12-axis limit has its own testset below.
        for n in 1:3, dims in Iterators.product(ntuple(_ -> (1, 2, 3), n)...)
            for mask in 0:(2^n - 1)
                region = Tuple(i for i in 1:n if (mask >> (i - 1)) & 1 == 1)
                _check_invariants(dims, VkFFT._canonical_region(region, n))
            end
        end
        for dims in (Iterators.product(ntuple(_ -> (1, 2), 4)...)..., (2, 3, 2, 3), (3, 1, 2, 3))
            for mask in 0:15
                region = Tuple(i for i in 1:4 if (mask >> (i - 1)) & 1 == 1)
                _check_invariants(dims, VkFFT._canonical_region(region, 4))
            end
        end
    end

    @testset "deep interleaving" begin
        # The measured limit: omitDimension = [1,0,1,0,...] at FFTdim = 12
        # works. One more run has to be refused with the reshape advice rather
        # than a raw VkFFT code.
        layout = VkFFT._map_region(ntuple(_ -> 2, 12), VkFFT._canonical_region(ntuple(i -> 2i, 6), 12))
        @test layout.fft_dim == 12
        @test layout.omit[1:12] == UInt64.((1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0))
        @test layout.number_batches == 1

        err = try
            VkFFT._map_region(ntuple(_ -> 2, 13), VkFFT._canonical_region(ntuple(i -> 2i - 1, 7), 13))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Reshape or permute", err.msg)
        @test occursin("at most 12", err.msg)
    end

    @testset "max_dims is not hardcoded" begin
        # The planner takes the axis limit as an argument. The ABI guard in
        # _ensure_library! is what ties VKFFT_MAX_FFT_DIMENSIONS to the loaded
        # library's vkfft_max_dims().
        dims = (2, 3, 4, 5)
        region = (2, 4) # batch, fft, batch, fft: four axes, nothing to fold
        @test VkFFT._map_region(dims, region, 4).fft_dim == 4
        @test_throws ArgumentError VkFFT._map_region(dims, region, 3)
    end

    @testset "zero-length dimensions" begin
        @test_throws ArgumentError VkFFT._map_region((0,), (1,))
        @test_throws ArgumentError VkFFT._map_region((4, 0), (1, 2))
    end

    @testset "layout is concretely typed" begin
        @test isconcretetype(VkFFT.VkFFTLayout)
        @test (@inferred VkFFT._map_region((8, 5, 4), (1, 3))) isa VkFFT.VkFFTLayout
    end

    @testset "first axis" begin
        # The axis a real transform halves and the only one that can be padded
        # are the same axis: the first dimension VkFFT does not force-omit.
        @test VkFFT._first_axis((8,)) == 1
        @test VkFFT._first_axis((8, 5)) == 1
        @test VkFFT._first_axis((1, 8)) == 2
        @test VkFFT._first_axis((1, 1, 5, 1)) == 3
        @test VkFFT._first_axis((1,)) == 0
        @test VkFFT._first_axis((1, 1)) == 0
    end

    @testset "zeropad canonicalization" begin
        @test VkFFT._canonical_zeropad(nothing) == (0, 0)
        @test VkFFT._canonical_zeropad(5:11) == (5, 11)
        @test VkFFT._canonical_zeropad((5:11,)) == (5, 11)
        @test VkFFT._canonical_zeropad(1:1) == (1, 1)
        @test VkFFT._canonical_zeropad(Base.OneTo(8)) == (1, 8)

        for bad in ((5, 11), (5:11, 1:2), [5, 11], 5, "5:11", 5.0:11.0)
            err = try
                VkFFT._canonical_zeropad(bad)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("one 1-based inclusive range", err.msg)
        end
    end

    @testset "zeropad mapping" begin
        # A 1-based inclusive lo:hi is VkFFT's half-open 0-based [lo - 1, hi),
        # and hi == size is what covers the tail. Only slot 1 is ever written,
        # and the bounds and the flag always travel together.
        for ((lo, hi), (left, right)) in (((5, 11), (4, 11)), ((1, 8), (0, 8)),
                                          ((9, 16), (8, 16)), ((1, 1), (0, 1)),
                                          ((16, 16), (15, 16)))
            fields = VkFFT._zeropad_fields((lo, hi))
            @test fields[1][1] == UInt64(left)
            @test fields[2][1] == UInt64(right)
            @test fields[3][1] == Cint(1)
            for i in 2:VkFFT.VKFFT_MAX_FFT_DIMENSIONS
                @test fields[1][i] == UInt64(0)
                @test fields[2][i] == UInt64(0)
                @test fields[3][i] == Cint(0)
            end
        end

        unpadded = VkFFT._zeropad_fields(VkFFT.NO_ZEROPAD)
        @test all(iszero, unpadded[1])
        @test all(iszero, unpadded[2])
        @test all(iszero, unpadded[3])

        @test VkFFT._zeropad_string((5, 11)) == " zeropad=5:11"
        @test VkFFT._zeropad_string(VkFFT.NO_ZEROPAD) == ""
    end

    @testset "zeropad validation" begin
        # Accepted: the padded dimension is the first one longer than 1, is
        # transformed, and the range is a non-empty 1-based range inside it.
        for (dims, region, range) in (((16,), (1,), (5, 11)), ((16,), (1,), (1, 16)),
                                      ((16,), (1,), (16, 16)), ((16, 4), (1,), (2, 3)),
                                      ((16, 4), (1, 2), (2, 3)), ((1, 16), (2,), (5, 11)),
                                      ((1, 16, 4), (2, 3), (5, 11)), ((16, 1, 4), (1, 3), (5, 11)))
            @test VkFFT._check_zeropad(dims, region, range) === nothing
        end

        # No padding is always fine, whatever the region.
        for (dims, region) in (((16,), ()), ((1, 1), (1, 2)), ((16, 4), (2,)))
            @test VkFFT._check_zeropad(dims, region, VkFFT.NO_ZEROPAD) === nothing
        end

        # Refused: a strided axis (VkFFT reads uninitialized memory there), an
        # empty region, an array with nothing to pad, and every out-of-range or
        # empty span.
        for (dims, region, range, fragment) in (((16, 4), (2,), (1, 2), "Permute"),
                                                ((16, 4, 3), (2, 3), (1, 2), "Permute"),
                                                ((1, 16), (1, 2), (1, 2), "Permute"),
                                                ((16,), (), (1, 2), "non-empty region"),
                                                ((1, 1), (1, 2), (1, 1), "no axis to zero-pad"),
                                                ((16,), (1,), (0, 4), "non-empty 1-based range"),
                                                ((16,), (1,), (5, 17), "non-empty 1-based range"),
                                                ((16,), (1,), (5, 4), "non-empty 1-based range"),
                                                ((16,), (1,), (17, 20), "non-empty 1-based range"),
                                                ((4, 16), (1, 2), (1, 5), "non-empty 1-based range")) # the bound is the padded axis', not the other one's
            canonical = VkFFT._canonical_region(region, length(dims))
            err = try
                VkFFT._check_zeropad(dims, canonical, range)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin(fragment, err.msg)
        end
    end
end
