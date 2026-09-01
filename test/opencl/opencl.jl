# What only an OpenCL process can assert: the half-precision extension gate and
# the memory backend the wrapper needs. Both are properties of OpenCL.jl and of
# the device, so neither has a counterpart on Metal or CUDA.

@testset verbose = true "opencl only" begin
    @testset "half precision needs cl_khr_fp16" begin
        # Without the extension VkFFT emits half2 arithmetic and the only thing
        # coming back is VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM, so the planner
        # gates on the device's extension list. OpenCL.jl also refuses to
        # allocate a half-precision CLArray on such a device, which is why the
        # gate is reached by calling it directly.
        x8 = _upload(_noise(ComplexF32, (8,)))

        if BACKEND.fp16
            for T in (Float16, ComplexF16)
                @test VkFFT._check_precision(T, x8) === nothing
            end
        else
            @test_throws ErrorException DeviceArray{ComplexF16, 1}(undef, (8,))
            for T in (Float16, ComplexF16)
                err = _thrown(() -> VkFFT._check_precision(T, x8))
                @test err isa ArgumentError
                @test occursin("cl_khr_fp16", err.msg)
                @test !occursin("VKFFT_ERROR", err.msg)
            end
        end

        # Positive control: the precisions this device does report pass the
        # gate, and so does an element type no half check applies to.
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            @test VkFFT._check_precision(T, x8) === nothing
        end
    end

    @testset "wrong memory backend" begin
        # A USM-backed CLArray is a different type, so it never reaches
        # _buffer_handle: passing a USM pointer where VkFFT wants a cl_mem
        # segfaults inside the driver with no error return, which is why this is
        # dispatch and not a runtime check.
        task_local_storage(:CLMemoryBackend, cl.USMBackend())
        try
            usm = CLArray{ComplexF32}(undef, 8)
            @test !(usm isa DeviceArray)
            @test_throws ArgumentError VkFFT.plan_fft(usm)
            @test_throws ArgumentError VkFFT._buffer_handle(usm)
        finally
            task_local_storage(:CLMemoryBackend, cl.BufferBackend())
        end
    end
end
