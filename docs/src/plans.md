# The plan interface

Every planner in VkFFT.jl returns an `AbstractFFTs.Plan`, so `mul!`, `*`, `inv`,
`\`, `ldiv!`, `size`, `adjoint`, `AbstractFFTs.fftdims` and
`AbstractFFTs.plan_inv` all work. These methods belong to AbstractFFTs, `Base`
and `LinearAlgebra`, so they have no docstrings of their own and are documented
down here instead.

## Specify VkFFT

Use `VkFFT.plan_fft`, not `plan_fft`. A bare `fft(x)` or `plan_fft(x)` on a
`CuArray` goes to CUFFT, and on a `CLArray` or `MtlArray` to whatever its owner
provides.

`AbstractFFTs.plan_fft(x, region)` is one generic function that the whole
ecosystem dispatches on, and CUDA.jl already owns that dispatch for `CuArray`.
Overwriting this would be type piracy which would make the order in which you
load the module what decides which transform runs. This also breaks
precompilation.

## Traits

```julia
x = rand(ComplexF32, 8, 6, 4) # 3D array of size 8×6×4
x = CLArray(x) # MtlArray(x) for Metal, CuArray(x) for CUDA
p = VkFFT.plan_fft(x, (1, 3))

size(p)                       # (8, 6, 4), the input size
size(p, 2)                    # 6
ndims(p)                      # 3
length(p)                     # 192
eltype(p)                     # ComplexF32
AbstractFFTs.fftdims(p)       # (1, 3), sorted
AbstractFFTs.output_size(p)   # (8, 6, 4)
```

`size` is the input size and `AbstractFFTs.output_size` the output size. The
two differ only for the real family, where the halved dimension shortens.
`eltype` is the input element type, which for an inverse real plan is the
complex one.

Plan types are concrete. A region given as a tuple, an integer or left at its
default carries its length in the type, so the whole plan type is inferable and
applying it needs no dynamic dispatch. A region given as a range only gets its
length at run time, which costs that one type parameter and nothing else.

## Applying a plan

```julia
using LinearAlgebra

y = p * x     # out-of-place, allocates the output
mul!(y, p, x) # the same transform into a buffer you own
```

For an in-place plan, `p * x` returns `x` and `mul!(y, p, x)` demands
`y === x`. For an out-of-place plan the two arrays have to be distinct.

The array reaching `mul!` need not be the one the plan was built from, only the
same size, element type, layout and device.

## Inverses

```julia
inv(p) * y # the inverse plan, applied
p \ y      # the same thing

out = similar(x)
ldiv!(out, p, y) # ... into a buffer you own

AbstractFFTs.plan_inv(p) # the inverse plan itself, not memoized
```

`inv` memoizes its answer on the plan, so `inv(p) === inv(p)`. Both routes come
out of the plan cache, so neither compiles a second set of kernels.

A forward plan inverts to VkFFT's in-kernel normalized inverse, one kernel
launch instead of a transform plus a scaling pass, and it inverts straight back
to the forward plan. The real-to-real family is the same, since VkFFT applies
the exact per-type round-trip factor in-kernel.

One case needs an `AbstractFFTs.ScaledPlan`: the inverse of a `plan_bfft` is a
normalized forward transform, and VkFFT's normalization only exists on the
inverse direction.

Zero-padded plans, convolution plans and plans from
[`VkFFT.unsafe_plan`](@ref) have no inverse, and each says why.

## Adjoints

```julia
p'          # the adjoint plan
(p')' === p # true
size(p')    # AbstractFFTs.output_size(p)
```

`p'` satisfies `dot(v, p * x) == dot(p' * v, x)`, and for the real families it
satisfies it in the inner product a real-linear transform is self-adjoint
under, which pairs real and imaginary parts separately. Zero-padded plans and
convolution plans have no adjoint, and say so.

## Scaling

`alpha * plan` and `plan * alpha` give an `AbstractFFTs.ScaledPlan`:

```julia
scaled = 3 * p
scaled.scale      # Float32(3), not Int(3)
inv(scaled).scale # Float32(1/3)

rotated = im * p
rotated.scale    # ComplexF32(im)
```

The scale is narrowed to the plan's own real precision.

## The plan cache

Plans are cached and reused, so planning the same transform twice is free:

```julia
VkFFT.plan_fft(x, (1, 2)) === VkFFT.plan_fft(x, (2, 1)) # true

VkFFT.cache_size()   # how many plans the cache holds
VkFFT.clear_cache!() # drop them all
```

The key is the backend, the device context, the input element type, the input
size, the canonicalized region, the direction, whether the plan normalizes,
whether it is in-place, whether it is real-to-complex, the real length of the
halved dimension, the DCT and DST types, the zero-padded range, and the two
tuning knobs. The array's identity is not in it, so a plan built from one
buffer is returned for another of the same shape on the same device.

The cache is unbounded and cleared only by [`VkFFT.clear_cache!`](@ref).
Concurrent planning of one shape is serialized, and every task gets the same
plan back. Convolution plans and plans from [`VkFFT.unsafe_plan`](@ref) are
never cached.

Clearing the cache lets the plans in it be finalized. See [Freeing a
plan](@ref).

## Printing

```julia
VkFFT.plan_fft(x, (1, 3))
# 2D out-of-place complex forward VkFFT plan for 8×6×4 ComplexF32 on dims (1, 3) [opencl]

VkFFT.plan_ifft(x, 2)
# 1D out-of-place complex normalized inverse VkFFT plan for 8×6×4 ComplexF32 on dim (2,) [opencl]

VkFFT.plan_fft(x, ())
# trivial out-of-place complex forward VkFFT plan for 8×6×4 ComplexF32 on dims () [opencl]
```

The leading count is the number of transformed dimensions, not `ndims`. A
`trivial` plan transforms nothing and applies as a copy. The tag in brackets is
the backend.
