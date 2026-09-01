# VkFFT.jl

Julia bindings for [VkFFT](https://github.com/DTolm/VkFFT), Dmitrii
Tolmachev's runtime-compiled GPU FFT library. This package has the plan types,
the AbstractFFTs plan interface, region mapping and the plan cache. The code
for each device lives in an extension, one per backend.

## Which package do you want?

`VkFFTCUDA`, `VkFFTOpenCL` or `VkFFTMetal`, not this one. Each is a ten-line
package that depends on this one and on the GPU package for your device, which
is what activates the matching extension. They re-export everything here.

- [VkFFTCUDA.jl](https://github.com/PaulVirally/VkFFTCUDA.jl) for `CuArray`
- [VkFFTOpenCL.jl](https://github.com/PaulVirally/VkFFTOpenCL.jl) for `CLArray`
- [VkFFTMetal.jl](https://github.com/PaulVirally/VkFFTMetal.jl) for `MtlArray`

## Setup

<!-- TODO(jll): install instructions pending VkFFT_{CUDA,OpenCL,Metal}_jll
     registration. Until then, the libvkfft_path preference is the only way
     in. This block is rewritten by the JLL wiring wave. -->

There is no JLL yet, so point the package at a locally built
[`libvkfft`](https://github.com/PaulVirally/libvkfft) wrapper once:

```julia
using Preferences, VkFFT
set_preferences!(VkFFT, "libvkfft_path" => "/path/to/libvkfft.so")
```

The wrapper is built per backend (`-DVKFFT_BACKEND=1` for CUDA, `3` for OpenCL,
`5` for Metal), because VkFFT picks its backend at compile time. That is also
why one Julia process drives one backend. The build flags and the rest of the
setup are in the README of the package for your device.

## Use

The only thing that changes between backends is the array type and the `using`
line:

```julia
using VkFFTOpenCL, OpenCL, LinearAlgebra   # or VkFFTCUDA and CUDA, or VkFFTMetal and Metal

x = CLArray{ComplexF32}(undef, 256, 64)
copyto!(x, rand(ComplexF32, 256, 64))

p = VkFFT.plan_fft(x, 1) # transform along dimension 1, batch over dimension 2
y = p * x                # or mul!(y, p, x)
x2 = inv(p) * y          # normalized inverse, 1/N applied inside the kernel

q = VkFFT.plan_fft!(x)   # in-place, both dimensions
q * x                    # overwrites x
```

Entry points stay module-qualified, `VkFFT.plan_fft` rather than a bare
`plan_fft`, because `AbstractFFTs.plan_fft` on a GPU array type belongs to
whoever owns that type. What you get back is an `AbstractFFTs.Plan`, so `*`,
`mul!`, `inv`, `\`, `ldiv!`, `size`, `adjoint` and `AbstractFFTs.fftdims` all
work.

## Documentation

The complex, real and real-to-real families, zero-padding, fused convolution,
half and quad precision, the autotuner, the per-backend capability matrix and
the sharp edges are in the
[documentation](https://paulvirally.github.io/VkFFT.jl/stable/).

## Tests

```julia
using Pkg; Pkg.test("VkFFT")
```

The suite runs on [pocl](https://portablecl.org), so it needs no GPU, and it
checks every transform against FFTW. On Apple silicon it runs the Metal suite
too. The header comment of each runner under `test/` says which wrapper it
expects and which environment variable repoints it.
