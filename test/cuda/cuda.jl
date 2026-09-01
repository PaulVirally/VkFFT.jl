# What only a CUDA process can assert: the layout cases CUDA accepts where the
# other backends refuse, the context guard, the per-task streams the plan lock
# exists for, and the half-precision gate, which reads the wrapper rather than
# the card.

@testset verbose = true "cuda only" begin
    @testset "half precision needs a wrapper with a toolkit path" begin
        # VkFFT writes an #include of cuda_fp16.h into every half-precision
        # kernel, taking the directory from the CUDA_TOOLKIT_ROOT_DIR baked
        # into libvkfft, and hands nvrtc no headers and no -I. A wrapper built
        # with an empty root therefore cannot compile one, and what VkFFT
        # reports is a failed compile behind several hundred lines of generated
        # CUDA on stdout, so the planner refuses first. A locally built wrapper
        # bakes a real path and the whole fp16 set runs.
        root = VkFFT._cuda_toolkit_root()
        x8 = _upload(_noise(ComplexF32, (8,)))

        if BACKEND.fp16
            for T in (Float16, ComplexF16)
                @test VkFFT._check_precision(T, x8) === nothing
            end
            @test VkFFT.plan_fft(_upload(_noise16(ComplexF16, (16,))), 1) isa VkFFTPlan
        else
            @test root == ""
            for T in (Float16, ComplexF16)
                err = _thrown(() -> VkFFT._check_precision(T, x8))
                @test err isa ArgumentError
                @test occursin("cuda_fp16.h", err.msg)
                @test !occursin("VKFFT_ERROR", err.msg)
            end
            @test_throws ArgumentError VkFFT.plan_fft(_upload(_noise16(ComplexF16, (16,))), 1)
        end

        # Positive control: nothing but half precision is gated here.
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            @test VkFFT._check_precision(T, x8) === nothing
        end
    end

    @testset "offset dense views are the positive case here" begin
        # A cl_mem and an MTLBuffer always name the base of their allocation, so
        # OpenCL and Metal refuse an offset view. A CUDA device pointer carries
        # the offset, so the same views plan and apply, and their transform is
        # the transform of the view rather than of the parent.
        host = _noise(ComplexF64, (16,))
        x = _upload(host)

        slice = view(x, 3:10)
        @test slice isa CuArray
        @test UInt(pointer(slice)) - UInt(pointer(x)) == 2 * sizeof(ComplexF64)
        @test _relmax(VkFFT.plan_fft(slice) * slice, fft(host[3:10])) < BACKEND.rtol_f64

        # A plan is reusable across buffers, so an offset slice also has to work
        # at apply time.
        plan = VkFFT.plan_fft(_upload(_noise(ComplexF64, (8,))))
        out = _upload(_noise(ComplexF64, (8,)))
        tail = view(x, 9:16)
        @test _relmax(mul!(out, plan, tail), fft(host[9:16])) < BACKEND.rtol_f64

        reshaped = reshape(x, 4, 4)
        @test reshaped isa CuArray
        @test _relmax(VkFFT.plan_fft(reshaped) * reshaped, fft(reshape(host, 4, 4))) < BACKEND.rtol_f64

        real_view = reinterpret(Float64, x)
        @test real_view isa CuArray
        @test _relmax(VkFFT.plan_rfft(real_view, 1) * real_view, rfft(collect(reinterpret(Float64, host)))) < BACKEND.rtol_f64
    end

    @testset "strided views are still refused" begin
        # The refusal the common set asserts for a SubArray applies here too:
        # what CUDA accepts is a dense view, not a strided one.
        x = _upload(_noise(ComplexF64, (16,)))
        strided = view(x, 1:2:15)
        @test_throws ArgumentError VkFFT.plan_fft(strided)

        plan = VkFFT.plan_fft(_upload(_noise(ComplexF64, (8,))))
        out = _upload(_noise(ComplexF64, (8,)))
        @test_throws ArgumentError mul!(out, plan, strided)
    end

    @testset "streams" begin
        # A NULL handle would be the legacy default stream, which this package
        # never executes on.
        h = _noise(ComplexF32, (1024, 4))
        reference = fft(h, 1)
        x = _upload(h)
        @test VkFFT._stream_handle(x) != C_NULL

        # One cached plan, several tasks. CUDA.jl gives every task its own
        # stream, and the plan cache hands them all the same VkFFT application.
        # An application is not reentrant, so this is what the plan's lock is
        # for.
        VkFFT.clear_cache!()
        shared = VkFFT.plan_fft(x, 1)
        tasks = map(1:8) do _
            Threads.@spawn begin
                xi = _upload(h)
                plan = VkFFT.plan_fft(xi, 1)
                (plan === shared, _relmax(plan * xi, reference))
            end
        end
        outcomes = fetch.(tasks)
        @test all(o -> o[1], outcomes)
        @test maximum(o -> o[2], outcomes) < BACKEND.rtol_f32

        # An explicit non-default stream. mul! already synchronizes the
        # task-local stream, which under stream! is this one, so the explicit
        # synchronize only confirms that the work really went to the stream the
        # caller named.
        stream = CuStream()
        y = CUDA.stream!(stream) do
            @test VkFFT._stream_handle(x) == convert(Ptr{Cvoid}, stream.handle)
            shared * x
        end
        CUDA.synchronize(stream)
        @test _relmax(y, reference) < BACKEND.rtol_f32
    end

    @testset "a plan belongs to the context it was built in" begin
        # VkFFT loads its nvrtc modules and allocates its scratch buffers in the
        # context that is current when the app is built, so an array from
        # another context has to be refused rather than planned for. With one
        # visible device there is no second context to reach, and the check is
        # then a tautology worth stating: the current context is the array's.
        x = _upload(_noise(ComplexF32, (16,)))
        @test CUDA.context(x) == CUDA.context()
        @test VkFFT._check_layout(x) === nothing
        @test VkFFT.plan_fft(x, 1) isa VkFFTPlan
    end
end
