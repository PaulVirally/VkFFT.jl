# Backends and capabilities

VkFFT.jl uses CUDA through CUDA.jl, OpenCL through OpenCL.jl and Metal
through Metal.jl. Planning, region mapping, and plan cache are the same on all
three backends. The only difference is what element types a device can use, what
a contiguous slice of a larger buffer is allowed to be, and a short list of
transforms that have upstream bugs on Metal.

## Element types

|  | CUDA | OpenCL | Metal |
| --- | --- | --- | --- |
| `Float32`, `ComplexF32` | yes | yes | yes |
| `Float64`, `ComplexF64` | yes | yes on every device tested | **no** |
| `Float16`, `ComplexF16` | only through a wrapper built against a CUDA toolkit | only where the device reports `cl_khr_fp16` | yes |
| Double-double quad, through [`VkFFT.unsafe_plan`](@ref) | yes | yes | **no** |
| Contiguous views at a nonzero offset | yes | **no** | **no** |

Note that half precision in VkFFT is only a storage format. Arithmetic is still
performed in single precision, though data is read and written in half
precision.

Every transform listed in [Transforms](transforms.md) runs on all three
backends. There are two exceptions: fused convolution for `ComplexF16` is not
implemented, and there is no real-input convolution (i.e., no r2r nor r2c
convolution).

## CUDA

For half-precision, you need to build
[`libvkfft`](https://www.github.com/PaulVirally/libvkfft) yourself against a
CUDA toolkit.

A plan belongs to the CUDA context it was built in. VkFFT loads its nvrtc
modules and allocates its scratch buffers in whatever context is current when
the application is built. Every launch and the freeing of the plan have to
happen in that same original context too. An array from another context is
refused. Execution takes the stream from `CUDA.stream()` on every call (not the
legacy default stream).

## OpenCL

VkFFT drives `clSetKernelArg` with a `cl_mem`, so OpenCL.jl has to return
buffer-backed arrays rather than its default unified-memory ones. You can do
this by setting the `default_memory_backend` preference to `"buffer"`:

```julia
using Preferences, OpenCL
set_preferences!(OpenCL, "default_memory_backend" => "buffer")
```

Passing a unified-memory `CLArray` to the planner throws an error.

Half precision is a per-device extension. Without `cl_khr_fp16` VkFFT emits
half arithmetic the driver will not compile. The planner checks the device's
extension list first and informs you if it is not present.

Note that the wrapper has to be linked against an ICD loader.

A `CLArray` at a nonzero offset into its buffer is refused. You should copy such
an array first.

OpenCL also runs on CPU. Add `pocl_jll` to enable this:

```julia
using pocl_jll # must come before OpenCL
using OpenCL
using VkFFTOpenCL
using LinearAlgebra

# Only needed when another ICD is installed. pocl calls itself this.
cl.platform!(first(p for p in cl.platforms() if p.name == "Portable Computing Language"))

h = rand(ComplexF32, 1024)
x = CLArray{ComplexF32}(undef, size(h))
copyto!(x, h)

p = VkFFT.plan_fft(x, 1)
y = p * x
norm(Array(inv(p) * y) - h) / norm(h) # Relative error should be < 1e-6
```

## Metal

Metal does not support double precision, so you cannot use `Float64` and
`ComplexF64`. This is a hardware limitation.

Passing an `MtlArray` at a nonzero offset to a plan throws an error. Similarly,
using an array on a different Metal device than the current task's throws.

There is one Metal-only crash to know about before you write anything around a
planning call. See [Do not plan inside an autorelease pool](@ref).

## Known failures

On pocl, single-precision kernel generation fails at lengths 8191 and 4093. You
should run these lengths in double precision on pocl.

The fused convolution is incorrect for a Bluestein axis long enough that VkFFT
splits it across several uploads. The length at which the split starts depends
on the device's shared memory, so there is nothing for the planner to compare
against. Check one convolution against a host reference before trusting a
transformed length with a large prime factor. Note that this is problematic only
for large prime lengths and for fused convolutions only (i.e., Fourier
transforms are fine).

## Tolerances

The relative tolerances the test suite scores against FFTW, element by
element against the peak magnitude of the reference:

|  | CUDA | OpenCL | Metal |
| --- | --- | --- | --- |
| single precision | 1e-4 | 1e-4 | 1e-4 |
| double precision | 1e-12 | 1e-12 | not applicable |
| half precision | 5e-3 | 5e-3 | 5e-3 |

The measured errors sit well inside those. On pocl the worst element-wise
error over sizes up to 16k is 2.5e-5 in single precision and 4.9e-14 in
double. On an M3 Pro it is 1.4e-5 in single precision, at 16381 through
Bluestein, and 1.1e-3 in half precision over the complex, real and
real-to-real families.

## Twelve axes

VkFFT takes at most twelve axes per plan. Note that this is a choice made by the
[`libvkfft`](https://www.github.com/PaulVirally/libvkfft) wrapper, not a
limitation of the original C++ library VkFFT.

The cap applies after the region is collapsed, not to `ndims`. Length-1 axes are
dropped, adjacent untransformed axes merge into one, and a trailing run of
untransformed axes folds into VkFFT's batch count instead of occupying an axis
at all. So the only way to reach twelve is to alternate transformed and
untransformed dimensions many times over, as in transforming every other axis of
a 25-dimensional array, and the error tells you to permute the array so the
transformed dimensions sit next to each other. You are extremely unlikely to hit
this.

## Multiple backends need multiple processes

The C++ VkFFT library chooses its backend when it is compiled. A Julia session
can only drive one backend, whichever wrapper the `libvkfft_path` preference
names. Loading `VkFFTCUDA` and `VkFFTMetal` into the same session does not give
you a process that can do both. This is why this package's own Metal test suite
runs in a child process. A program that needs two backends needs two processes.

`VKFFT_BACKEND` is 1 for CUDA, 3 for OpenCL and 5 for Metal.

The first call that reaches the C library checks the wrapper's
`VKFFT_MAX_FFT_DIMENSIONS` and its configuration size against the mirror
compiled into VkFFT.jl. A wrapper and a package from different versions throws
an error.
