# VkFFT.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://paulvirally.github.io/VkFFT.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://paulvirally.github.io/VkFFT.jl/dev/)

Julia bindings for [VkFFT](https://github.com/DTolm/VkFFT), Dmitrii
Tolmachev's runtime-compiled GPU FFT library.

This package defines all the exported functions, plan types, plan cache, etc.
This package is not the one you load if you want to use VkFFT in Julia. Rather,
you must load one of the backends.

## Which package do you want?

Load `VkFFTCUDA`, `VkFFTOpenCL` or `VkFFTMetal`, not this package. Each one of
these re-exports all the symbols from this package. You want to use

- [VkFFTCUDA.jl](https://github.com/PaulVirally/VkFFTCUDA.jl) for `CuArray`
- [VkFFTOpenCL.jl](https://github.com/PaulVirally/VkFFTOpenCL.jl) for `CLArray`
- [VkFFTMetal.jl](https://github.com/PaulVirally/VkFFTMetal.jl) for `MtlArray`

## Setup

```julia
using Pkg
Pkg.add("VkFFTOpenCL") # or VkFFTCUDA, or VkFFTMetal
```

For devs working on this package itself, see [Building the
wrapper](https://paulvirally.github.io/VkFFT.jl/stable/building/).

## Use

The only thing that changes between backends is the array type and the `using`
line:

```julia
using VkFFTOpenCL, OpenCL, LinearAlgebra # or VkFFTCUDA and CUDA, or VkFFTMetal and Metal

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
whoever owns that type (e.g., `CUDA.jl` owns `fft`, and we don't compete with
that). What you get back from `VkFFT.plan_fft` is an `AbstractFFTs.Plan`, so
`*`, `mul!`, `inv`, `\`, `ldiv!`, `size`, `adjoint` and `AbstractFFTs.fftdims`
all work.

## Documentation

The complex, real and real-to-real families, zero-padding, fused convolution,
half and quad precision, the autotuner, the per-backend capability matrix and
the sharp edges are in the
[documentation](https://paulvirally.github.io/VkFFT.jl/stable/).

## Tests

```julia
using Pkg; Pkg.test("VkFFT")
```

The suite runs on [pocl](https://portablecl.org), so it doesn't need a GPU, and
it checks every transform against FFTW. On Apple Silicon it runs the Metal suite
too.
