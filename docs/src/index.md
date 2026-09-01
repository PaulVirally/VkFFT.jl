# VkFFT.jl

Julia bindings for [VkFFT](https://github.com/DTolm/VkFFT), DTolm's
runtime-compiled GPU FFT library. Bindings are available for three backends:
CUDA, OpenCL, and Metal.

VkFFT implements complex-to-complex transforms, real-to-complex,
complex-to-real, cosine and sine transforms of types I to IV, fused circular
convolution, fused zero-padding, half precision, and quad precision. An
autotuner is provided which picks VkFFT's block and thread hints per device and
per shape for the best performance.

VkFFT uses Bluestein's algorithm for axes with lengths that have a prime factor
above 13. Thus, any size FFT can be done.

## VkFFT Backends

You don't want to use VkFFT.jl directly. Instead, use one of `VkFFTCUDA`,
`VkFFTOpenCL` or `VkFFTMetal`, depending on your device. See [Backends and
capabilities](backends.md) for what each backend is capable of.

## Install

```@raw html
<!-- TODO(jll): install instructions pending VkFFT_{CUDA,OpenCL,Metal}_jll
     registration. Until then, the libvkfft_path preference is the only way
     in. This block is rewritten by the JLL wiring wave. -->
```

```julia
using Pkg
Pkg.add("VkFFTOpenCL") # or VkFFTCUDA, or VkFFTMetal
```

VkFFT.jl calls [`libvkfft`](https://github.com/PaulVirally/libvkfft), a small C
wrapper around VkFFT. There is no JLL for that wrapper yet, so it has to be
built once and named in a preference:

```julia
using Preferences, VkFFT
set_preferences!(VkFFT, "libvkfft_path" => "/path/to/libvkfft.so")
```

VkFFT chooses its backend when it is compiled, so the wrapper is built once
per backend: `-DVKFFT_BACKEND=1` for CUDA, `3` for OpenCL and `5` for Metal.

OpenCL needs one more preference, and each backend has a short list of its
limitations. See [Backends and capabilities](backends.md) for more information.

## Quickstart

```julia
using VkFFTOpenCL, OpenCL
using LinearAlgebra

h = rand(ComplexF32, 1024, 64)
x = CLArray{ComplexF32}(undef, size(h))
copyto!(x, h)

p = VkFFT.plan_fft(x, 1) # transform dimension 1, batch over dimension 2
y = p * x                # or mul!(y, p, x)
back = inv(p) * y        # normalized inverse, 1/N applied in-kernel
norm(back - h) / norm(h) # Relative error, should be < 1e-6 for single precision
```

`y` is `fft(h, 1)` and `back` is `h` again, both to single-precision.

The same program on the other two backends changes two lines, the `using` and
the array type:

```julia
using VkFFTCUDA, CUDA
# ...
x = CuArray{ComplexF32}(undef, size(h))
```

```julia
using VkFFTMetal, Metal
# ...
x = MtlArray{ComplexF32}(undef, size(h))
```

`VkFFT.plan_fft` is module-qualified. This means that `fft(x)` and `plan_fft(x)`
will call the function from whichever package owns the array type (i.e., CUFFT
on a `CuArray`, MPSGraph on an `MtlArray`, and a `MethodError` on a `CLArray`,
since OpenCL.jl has no FFT of its own).

## Read more

- [Backends and capabilities](backends.md) for details on the backends, what
  each device runs, and the setup each backend needs.
- [Transforms](transforms.md) for the entry points, their arguments and the
  size and layout rules.
- [The plan interface](plans.md) for how to use plans, how they cache data and
  the module-qualified entry points.
- [Tuning](tuning.md) for the autotuner and its records.
- [Limitations](limitations.md) for the layout rules, the lifetime rules or
  the raw configuration.
- [Migrating from VkFFTCUDA 0.2](migration.md) if you used the old package.
- [API reference](api.md) for the docstrings.
