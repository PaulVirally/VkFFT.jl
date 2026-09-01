# Everything the planner refuses, or computes, without a device: every case here
# takes a host `Array`, which has no backend at all, or calls a pure function
# directly. None of it can differ between an OpenCL, a Metal and a CUDA process,
# so the OpenCL runner is the only one that includes it. The refusals that do
# need a device array live with their family.

@testset verbose = true "planner" begin
    @testset "error type" begin
        @test sprint(showerror, VkFFTError(3005, "VKFFT_ERROR_UNSUPPORTED_FFT_OMIT")) == "Fatal VkFFT error: VKFFT_ERROR_UNSUPPORTED_FFT_OMIT (code: 3005)"
        @test VkFFTError <: Exception

        # The name comes from VkFFT's own getVkFFTErrorString, never from a
        # table in Julia.
        @test VkFFT._vkfft_error_name(0) == "VKFFT_SUCCESS"
        @test VkFFT._vkfft_error_name(3005) == "VKFFT_ERROR_UNSUPPORTED_FFT_OMIT"
        @test VkFFT._check(0) === nothing
        @test_throws VkFFTError VkFFT._check(3005)
    end

    @testset "unsupported element types" begin
        # The messages have to name what is supported rather than a VkFFT code.
        for (bad, expected) in ((Array{Int}(undef, 8), "ComplexF16, ComplexF32 and ComplexF64"),
                                (Array{BigFloat}(undef, 8), "ComplexF16, ComplexF32 and ComplexF64"),
                                (Array{Float32}(undef, 8), "VkFFT.plan_rfft"))
            err = _thrown(() -> VkFFT.plan_fft(bad))
            @test err isa ArgumentError
            @test occursin(expected, err.msg)
            @test !occursin("VKFFT_ERROR", err.msg)
        end

        for (entry, bad, expected) in ((VkFFT.plan_rfft, Array{Int}(undef, 8), "Float16, Float32 and Float64"),
                                       (x -> VkFFT.plan_brfft(x, 8), Array{Int}(undef, 5), "ComplexF16, ComplexF32 and ComplexF64 spectra"),
                                       (VkFFT.plan_dct, Array{Int}(undef, 8), "Float16, Float32 and Float64"),
                                       (x -> VkFFT.plan_conv(x, x), Array{Int}(undef, 8), "ComplexF32 and ComplexF64"))
            err = _thrown(() -> entry(bad))
            @test err isa ArgumentError
            @test occursin(expected, err.msg)
        end

        # Quad precision has no Julia element type, so every one of those
        # messages points at the escape hatch instead of promising it later.
        for err in (_thrown(() -> VkFFT.plan_fft(Array{Int}(undef, 8))),
                    _thrown(() -> VkFFT.plan_rfft(Array{Int}(undef, 8))),
                    _thrown(() -> VkFFT.plan_dct(Array{Int}(undef, 8))))
            @test occursin("VkFFT.unsafe_plan", err.msg)
            @test !occursin("not implemented", err.msg)
        end

        # The real families name their own entry points rather than half
        # precision when a complex array arrives.
        @test_throws ArgumentError VkFFT.plan_rfft(Array{ComplexF32}(undef, 16))
        @test_throws ArgumentError VkFFT.plan_rfft(Array{Float16}(undef, 16))
        @test_throws ArgumentError VkFFT.plan_brfft(Array{Float32}(undef, 9), 16)
        @test_throws ArgumentError VkFFT.plan_irfft(Array{ComplexF16}(undef, 9), 16)
        @test occursin("plan_fft", _thrown(() -> VkFFT.plan_rfft(Array{ComplexF32}(undef, 16))).msg)

        @test_throws ArgumentError VkFFT.plan_dct(Array{ComplexF32}(undef, 16))
        @test_throws ArgumentError VkFFT.plan_dst(Array{Float16}(undef, 16))
        @test_throws ArgumentError VkFFT.plan_idct(Array{Int}(undef, 16))
        @test occursin("plan_fft", _thrown(() -> VkFFT.plan_dct(Array{ComplexF32}(undef, 16))).msg)

        # A bad r2r type is refused before the element type is looked at, so the
        # caller hears about the mistake VkFFT would not catch.
        @test occursin("1, 2, 3 or 4", _thrown(() -> VkFFT.plan_dct(Array{ComplexF32}(undef, 16); type=9)).msg)

        # Half precision works for every other family, and a convolution in it
        # builds, reports success and computes something that is no convolution,
        # so it gets its own refusal rather than the generic one.
        host16 = _noise(ComplexF16, (16, 8))
        err = _thrown(() -> VkFFT.plan_conv(host16, host16, (1, 2)))
        @test err isa ArgumentError
        @test occursin("cannot convolve in half precision", err.msg)
        @test !occursin("VKFFT_ERROR", err.msg)

        host = Array{Int}(undef, 16, 8)
        err = _thrown(() -> VkFFT.plan_conv(host, host, (1, 2)))
        @test err isa ArgumentError
        @test occursin("ComplexF32 and ComplexF64", err.msg)
        @test occursin("VkFFT.unsafe_plan", err.msg)
    end

    @testset "r2r guards, no C library" begin
        # Every case here is a host Array, which has no backend at all, so the
        # only way any of them can throw an ArgumentError naming an r2r
        # constraint is from the pure planner path. The positive control at the
        # end shows what a host array gets once the guards are happy.
        @testset "a length-1 transformed dimension" begin
            # Type 1 is the dangerous one: VkFFT sets that axis' internal length
            # to 2*1 - 2 = 0 and its radix decomposition spins forever inside
            # vkfft_create, with nothing to cancel. This test must never reach
            # the library, and cannot: a plain Array has no backend.
            for kind in R2R_KINDS, type in 1:4
                for (dims, region) in (((1,), (1,)), ((1, 1), (1, 2)), ((16, 1), (1, 2)),
                                       ((1, 16), (1,)), ((4, 1, 5), (1, 2, 3)))
                    err = _thrown(() -> _r2r_plan(Array{Float32}(undef, dims), kind, region; type=type))
                    @test err isa ArgumentError
                    @test occursin("length at least 2", err.msg)
                    @test !occursin("VKFFT_ERROR", err.msg)
                    @test !occursin("no backend", err.msg) # the guard, not the array type
                end
            end

            # Inverse plans are planned through the same guard.
            @test_throws ArgumentError VkFFT.plan_idct(Array{Float32}(undef, 1), 1; type=1)
            @test_throws ArgumentError VkFFT.plan_idst!(Array{Float32}(undef, 1, 1); type=1)
        end

        @testset "type range" begin
            for kind in R2R_KINDS, type in (-1, 0, 5, 6, 99)
                err = _thrown(() -> _r2r_plan(Array{Float32}(undef, 16), kind, 1; type=type))
                @test err isa ArgumentError
                @test occursin("1, 2, 3 or 4", err.msg)
            end

            # The message has to name the entry point the caller used.
            @test occursin("plan_idst!", _thrown(() -> VkFFT.plan_idst!(Array{Float32}(undef, 16); type=7)).msg)
        end

        @testset "positive control" begin
            # Legal size, legal type: the guards pass and the array type is what
            # fails, which is how we know the refusals above are the guards.
            for kind in R2R_KINDS, type in 1:4
                err = _thrown(() -> _r2r_plan(Array{Float32}(undef, 16), kind, 1; type=type))
                @test err isa ArgumentError
                @test occursin("no backend", err.msg)
            end
        end

        @testset "the combinations VkFFT accepts and gets wrong" begin
            # No entry point can express either of these, so the guard sits
            # where the config is built and protects anything calling in from
            # below. A trivial layout means nothing is created even when a call
            # gets through.
            roots = Any[]
            @test VkFFT._create_app(Float32, VkFFT.TRIVIAL_LAYOUT, VkFFT.FORWARD, false, false, false, Val(BACKEND.name), roots; dct=Int32(2)) == C_NULL
            @test VkFFT._create_app(Float32, VkFFT.TRIVIAL_LAYOUT, VkFFT.FORWARD, false, false, true, Val(BACKEND.name), roots) == C_NULL

            err = _thrown(() -> VkFFT._create_app(Float32, VkFFT.TRIVIAL_LAYOUT, VkFFT.FORWARD, false, false, false, Val(BACKEND.name), roots; dct=Int32(2), dst=Int32(2)))
            @test err isa ArgumentError
            @test occursin("one real-to-real kind", err.msg)

            err = _thrown(() -> VkFFT._create_app(Float32, VkFFT.TRIVIAL_LAYOUT, VkFFT.FORWARD, false, false, true, Val(BACKEND.name), roots; dst=Int32(3)))
            @test err isa ArgumentError
            @test occursin("corrupts the spectrum", err.msg)
        end
    end

    @testset "convolution layout" begin
        # A one-axis layout is rewritten as two axes with a trailing length of
        # 1, which is the same transform and the only shape VkFFT's convolution
        # codegen compiles at a single-upload length.
        layout, features = VkFFT._conv_layout(VkFFT._map_region((64,), (1,)))
        @test layout.fft_dim == 2
        @test layout.size[1] == 64
        @test layout.size[2] == 1
        @test layout.number_batches == 1
        @test features == 1

        # The outermost batch count becomes the component stack, because VkFFT
        # offsets the kernel by the component index and never by the batch one.
        layout, features = VkFFT._conv_layout(VkFFT._map_region((8, 5, 3), (1, 2)))
        @test layout.fft_dim == 2
        @test (layout.size[1], layout.size[2]) == (8, 5)
        @test layout.number_batches == 1
        @test features == 3

        layout, features = VkFFT._conv_layout(VkFFT._map_region((16, 2), (1,)))
        @test layout.fft_dim == 2
        @test (layout.size[1], layout.size[2]) == (16, 1)
        @test features == 2

        layout, features = VkFFT._conv_layout(VkFFT._map_region((8, 5, 4), (1, 2)))
        @test layout.fft_dim == 2
        @test layout.number_batches == 1
        @test features == 4

        # The layout the app is built from always describes exactly the bytes
        # the array holds.
        for (dims, region, stack) in CONV_CASES
            layout, features = VkFFT._conv_layout(VkFFT._map_region(dims, region))
            @test features == stack
            @test features * prod(layout.size[i] for i in 1:layout.fft_dim) == prod(dims)
        end
    end

    @testset "convolution region validation" begin
        for (dims, region) in (((16,), (1,)), ((16, 8), (1, 2)), ((16, 8), (1,)),
                               ((16, 8, 4), (1, 2)), ((16, 8, 4), (1,)),
                               ((1, 16, 4), (2,)), ((1, 16, 4), (2, 3)),
                               ((16, 1, 4), (1, 3)))
            canonical = VkFFT._canonical_region(region, length(dims))
            @test VkFFT._check_conv_region(dims, canonical, VkFFT._map_region(dims, canonical)) === nothing
        end

        # A leading or a middle untransformed axis becomes an omit, which no
        # convolution layout may carry. A region that transforms nothing has no
        # convolution to compute at all.
        for (dims, region, fragment) in (((16, 8), (2,), "permute"),
                                         ((16, 8, 4), (2, 3), "permute"),
                                         ((16, 8, 4), (1, 3), "permute"),
                                         ((16, 8, 4), (3,), "permute"),
                                         ((16, 8, 4, 2), (1, 4), "permute"),
                                         ((16, 8, 4), (1, 2, 3), "at most two transformed dimensions"),
                                         ((8, 5, 4, 2), (1, 2, 3), "at most two transformed dimensions"),
                                         ((8, 5, 4, 2), (1, 2, 3, 4), "at most two transformed dimensions"),
                                         ((16, 8), (), "at least one dimension"),
                                         ((1, 1), (1, 2), "at least one dimension"),
                                         ((16, 1), (2,), "at least one dimension"))
            canonical = VkFFT._canonical_region(region, length(dims))
            err = _thrown(() -> VkFFT._check_conv_region(dims, canonical, VkFFT._map_region(dims, canonical)))
            @test err isa ArgumentError
            @test occursin(fragment, err.msg)
            @test !occursin("VKFFT_ERROR", err.msg)
        end
    end
end
