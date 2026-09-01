# API reference

Every entry point is called as `VkFFT.plan_fft`, never as a bare `plan_fft`.
See [Specify-VkFFT](@ref).

`Base.:*`, `LinearAlgebra.mul!`, `Base.inv`, `\`, `ldiv!`, `Base.size`,
`adjoint`, `AbstractFFTs.fftdims`, `AbstractFFTs.plan_inv`,
`AbstractFFTs.adjoint_mul` and `Base.show` have no docstrings of their own.
They are documented on [The plan interface](plans.md).

## Plan types

```@docs
VkFFT.VkFFTPlan
VkFFT.VkFFTRealPlan
VkFFT.VkFFTR2RPlan
VkFFT.VkFFTConvPlan
VkFFT.VkFFTUnsafePlan
```

## Complex-to-complex transforms

```@docs
VkFFT.plan_fft
VkFFT.plan_fft!
VkFFT.plan_bfft
VkFFT.plan_bfft!
VkFFT.plan_ifft
VkFFT.plan_ifft!
```

## Real-to-complex transforms

```@docs
VkFFT.plan_rfft
VkFFT.plan_brfft
VkFFT.plan_irfft
```

## Real-to-real transforms

```@docs
VkFFT.plan_dct
VkFFT.plan_dct!
VkFFT.plan_idct
VkFFT.plan_idct!
VkFFT.plan_dst
VkFFT.plan_dst!
VkFFT.plan_idst
VkFFT.plan_idst!
```

## Convolution

```@docs
VkFFT.plan_conv
```

## The raw configuration

```@docs
VkFFT.VkFFTConfig
VkFFT.VkFFTConfig()
VkFFT.unsafe_plan
```

## Freeing, caches and tuning records

```@docs
VkFFT.unsafe_free!
VkFFT.clear_cache!
VkFFT.cache_size
VkFFT.clear_tuning!
VkFFT.last_sweep
VkFFT.sweep_count
VkFFT.disk_cache_dir
```

## Errors

```@docs
VkFFT.VkFFTError
```
