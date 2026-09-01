# Migrating from VkFFTCUDA v0.2

v0.3 is a rewrite. The bindings moved into VkFFT.jl and `VkFFTCUDA` became a
loader: it depends on VkFFT.jl and CUDA.jl, and it re-exports VkFFT.jl. The
package name and UUID did not change, so upgrading in place works.

## What changed

| VkFFTCUDA 0.2 | Now |
| --- | --- |
| `plan_fft(x)`, through a method on `AbstractFFTs.plan_fft(::CuArray, region)` | `VkFFT.plan_fft(x)` |
| `fft(x)` routed to VkFFT | `fft(x)` routes to CUFFT. Write `VkFFT.plan_fft(x) * x` |
| `Pkg.build("VkFFTCUDA")`, which cloned a repository and ran `sudo make install` | Nothing to build at install time. Point the `libvkfft_path` preference at a wrapper |
| `__precompile__(false)` | Precompiles normally |
| CUDA only | CUDA, OpenCL and Metal |
| Complex-to-complex only | Complex-to-complex, real-to-complex, DCT and DST, fused convolution, fused zero-padding |
| `ComplexF32` and `ComplexF64` | Those, plus `ComplexF16` and double-double quad |

## The entry points are module-qualified now

Every planner gained a `VkFFT.` prefix:

```julia
using VkFFTCUDA, CUDA
using LinearAlgebra

x = CuArray{ComplexF32}(undef, 1024, 64)
copyto!(x, rand(ComplexF32, 1024, 64))

p = VkFFT.plan_fft(x, 1) # was plan_fft(x, 1)
y = p * x
back = inv(p) * y
```

A bare `fft(x)` or `plan_fft(x)` on a `CuArray` now goes to CUFFT. See
[Specify VkFFT](@ref).

Nothing else about a plan changed. You still get a first-class
`AbstractFFTs.Plan`, so `*`, `mul!`, `inv`, `\`, `ldiv!`, `size`, `adjoint` and
`AbstractFFTs.fftdims` all work on it, on the new families too. See [The plan
interface](plans.md).

## There is nothing to build

`Pkg.build("VkFFTCUDA")` and the `deps/build.jl` behind it are gone. VkFFT.jl
finds its wrapper through a preference instead, and will find it through a JLL
once one is registered. See [Install](@ref).

## What is new

New families, on [Transforms](transforms.md): real-to-complex and back, cosine
and sine transforms of types I to IV, fused zero-padding and fused circular
convolution. `ComplexF16` and `Float16` plan like any other element type on a
device that supports them, and double-double quad precision is reachable
through [the raw configuration](limitations.md).

[Tuning](tuning.md) covers the autotuner, which persists what it measures
across processes.

Two new backends. The same code runs on OpenCL and on Metal through
`VkFFTOpenCL` and `VkFFTMetal`, with the array type as the only difference. See
[Backends and capabilities](backends.md).
