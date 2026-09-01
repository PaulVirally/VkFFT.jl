# What only a Metal process can assert: the refusal of Float64, the reshape
# positive control on an offset view, and the two command-buffer lifetimes the
# wrapper's autorelease and flush behaviour rest on.

@testset verbose = true "metal only" begin
    @testset "Metal has no Float64" begin
        # Metal.jl already refuses the allocation, so this only fires on a type
        # that got past it. It still has to be the sentence naming the reason
        # rather than a VkFFT result code.
        for A in (MtlArray{ComplexF64, 1, Metal.PrivateStorage}, MtlArray{Float64, 2, Metal.PrivateStorage})
            err = _thrown(() -> VkFFT._backend(A))
            @test err isa ArgumentError
            @test occursin("Metal has no Float64", err.msg)
            @test !occursin("VKFFT_ERROR", err.msg)
        end

        @test_throws ErrorException MtlArray{ComplexF64}(undef, 4)
        @test VkFFT._backend(MtlArray{ComplexF32, 1, Metal.PrivateStorage}) === Val(:metal)
    end

    @testset "a reshape keeps offset 0" begin
        # A buffer crosses into Metal as the whole MTLBuffer and the launch
        # parameters carry no offset, which is why an offset view is refused in
        # the common set. A reshape of the same allocation keeps offset 0 and
        # stays transformable, so the refusal is about the offset and not about
        # the wrapper type. Metal.jl is the one backend here that represents a
        # reshape as another MtlArray, so this cannot live in common/.
        host = ComplexF32[i for i in 1:16]
        big = _upload(host)
        square = reshape(big, 4, 4)
        @test pointer(square).offset == 0
        @test _relmax(VkFFT.plan_fft(square) * square, fft(reshape(host, 4, 4))) < BACKEND.rtol_f32
    end

    @testset "plan reuse under load" begin
        # One plan, many applications, which is the case a leaked or
        # over-released command buffer shows up in. Interleaved GC and fresh
        # plans put a finalizer inside the window where a pool is open.
        h64 = _noise(ComplexF32, (64, 8))
        reference = fft(h64, 1)
        x64 = _upload(h64)
        y64 = similar(x64)
        plan = VkFFT.plan_fft(x64, 1)

        worst = 0.0
        for i in 1:2000
            mul!(y64, plan, x64)
            if i % 250 == 0
                worst = max(worst, _relmax(y64, reference))
                VkFFT.plan_fft(_upload(_noise(ComplexF32, (32 + i,))), 1)
                GC.gc(false)
            end
        end
        @test worst < BACKEND.rtol_f32

        # Interleaving with Metal.jl's own work is what the flush before every
        # submission is for: the transform has to see the broadcast that
        # preceded it, and the copy back has to see the transform.
        for _ in 1:64
            x64 .= 2 .* x64
            mul!(y64, plan, x64)
            x64 .= inv(plan) * y64 ./ 2
        end
        @test _relmax(x64, h64) < BACKEND.rtol_f32
    end

    @testset "several transforms per command buffer" begin
        # The wrapper appends one compute encoder per call and never commits, so
        # a caller may fill one command buffer with several transforms. mul!
        # does not, but the hook it goes through has to keep the door open.
        h32 = _noise(ComplexF32, (32,))
        x32 = _upload(h32)
        y32 = similar(x32)
        z32 = similar(x32)
        forward = VkFFT.plan_fft(x32, 1)
        inverse = inv(forward)

        res = GC.@preserve x32 y32 z32 begin
            VkFFT._with_execution(z32) do stream
                first = VkFFT._vkfft_execute(forward.app, VkFFT._buffer_handle(x32), VkFFT._buffer_handle(y32), forward.direction, stream)
                first == 0 || return first
                VkFFT._vkfft_execute(inverse.app, VkFFT._buffer_handle(y32), VkFFT._buffer_handle(z32), inverse.direction, stream)
            end
        end
        @test res == 0
        @test _relmax(y32, fft(h32)) < BACKEND.rtol_f32
        @test _relmax(z32, h32) < BACKEND.rtol_f32
    end
end
