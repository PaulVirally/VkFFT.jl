# Transforms

The examples below use `CLArray`, but `CuArray` and `MtlArray` work the same
way.

The first argument to every entry point is a dummy array. Only its element type,
its size and its device are used. Its contents are never read nor written, so
you can plan from one buffer and apply the plan to another of the same shape.

## Regions

`region` is any iterable of the dimensions to transform, as in
[AbstractFFTs](https://github.com/JuliaMath/AbstractFFTs.jl), and it defaults
to every dimension. Order does not matter. Repeats and out-of-bounds dimensions
are refused and throw an error.

```julia
v = rand(ComplexF32, 8, 128, 6) # 3 dimensions to work with
w = CLArray{ComplexF32}(undef, size(v))
copyto!(w, v)

VkFFT.plan_fft(w, 2)      # transform over dimension 2, batch over 1 and 3
VkFFT.plan_fft(w, (1, 3)) # transform over dimension 1 and 3, do nothing to 2
VkFFT.plan_fft(w, 1:2)    # any iterable works
VkFFT.plan_fft(w, ())     # transforms nothing, and applies as a copy
```

A region that transforms nothing, or one where every dimension has length 1, is
applied as a copy in Julia.

Length-1 dimensions are ignored. See [Twelve axes](@ref) for the cap on how many
axes a region can need.

## Complex to complex

`VkFFT.plan_fft`, `VkFFT.plan_bfft` and `VkFFT.plan_ifft`, each with an
in-place `!` form. Input and output have the same size and the same element
type, one of `ComplexF16`, `ComplexF32` and `ComplexF64`.

```julia
h = rand(ComplexF32, 1024, 64)
x = CLArray{ComplexF32}(undef, size(h))
copyto!(x, h)

forward = VkFFT.plan_fft(x, 1) # out-of-place, dimension 1, batched over 2
y = forward * x                # fft(h, 1)
mul!(y, forward, x)            # same thing as above, but in-place into y

unnormalized = VkFFT.plan_bfft(x, 1) # bfft(h, 1), no 1/N
normalized = VkFFT.plan_ifft(x, 1)   # ifft(h, 1), 1/N applied in-kernel

back = inv(forward) * y # h again

inplace = VkFFT.plan_fft!(x, 1) # mutating
inplace * x # x now holds fft(h, 1)
```

`VkFFT.plan_ifft` is a normalized inverse of its own and not a scaled `bfft`.
VkFFT applies the 1/N inside the transform kernels, so an `ifft` is one kernel
launch instead of two. The `N` is the product of the transformed axis lengths
only, and batched and omitted dimensions never enter it.

The in-place forms overwrite their argument: `p * x` returns `x` itself, and
`mul!(y, p, x)` requires that `y === x`.

## Real to complex, and back

`VkFFT.plan_rfft` forward, `VkFFT.plan_brfft` and `VkFFT.plan_irfft` back. The
real side is one of `Float16`, `Float32` and `Float64` and the output has the
matching complex type. There is no in-place form, because AbstractFFTs has no
in-place real transform.

The first transformed dimension is halved, so the output has
`size(x, d1) ÷ 2 + 1` along it and `size(x)` everywhere else. That dimension
has to be the first one of the array longer than 1, and using any other region
throws an error.

```julia
h = rand(Float32, 1024, 64)
r = CLArray{Float32}(undef, size(h))
copyto!(r, h)

forward = VkFFT.plan_rfft(r, 1) # 1024 reals in, 513 complex values out
spectrum = forward * r

normalized = VkFFT.plan_irfft(spectrum, 1024, 1)   # applies to h
unnormalized = VkFFT.plan_brfft(spectrum, 1024, 1) # applies to 1024 .* h

signal = normalized * spectrum # h again, and spectrum is now garbage
```

Both inverses need the real length, because the spectrum does not know which
inverse to take. 513 complex values could have come from 1024 reals or from
1025. You need to tell VkFFT which one to use.

Applying an inverse consumes its input, which VkFFT uses as scratch whenever
more than one axis is transformed. Copy the spectrum first if you still need it.
FFTW's `brfft` destroys its input too.

An inverse is only defined on a spectrum that really is the transform of a
real signal, meaning one whose DC values, and its Nyquist values when the real
length is even, are real. VkFFT packs two such spectra into one complex pass,
so for any other input it returns something different from what FFTW returns.

## Real to real

`VkFFT.plan_dct` and `VkFFT.plan_dst` forward, `VkFFT.plan_idct` and
`VkFFT.plan_idst` back, each with an in-place `!` form. Input and output have
the same size and the same real element type.

`type` is 1 to 4 and defaults to 2. It selects FFTW's `REDFT00`, `REDFT10`,
`REDFT01` and `REDFT11` for the cosine transforms and the matching `RODFT`
kinds for the sine ones. The transform is FFTW's element for element at every
size, prime lengths included.

```julia
h = rand(Float32, 1024, 64)
r = CLArray{Float32}(undef, size(h))
copyto!(r, h)

cosine = VkFFT.plan_dct(r, 1) # REDFT10, unnormalized, batched over 2
y = cosine * r
back = inv(cosine) * y        # exactly h, the factor is applied in-kernel

VkFFT.plan_dst(r, 1; type=4)  # RODFT11
VkFFT.plan_idct(r, 1; type=3) # inverts a type-3 DCT
VkFFT.plan_dct!(r, 1)         # in-place
```

`VkFFT.plan_idct` and `VkFFT.plan_idst` take the type of the forward transform
they undo, not the type they compute. An inverse of a type-2 DCT applies
`REDFT01` over `prod(2 .* size(r)[region])`. Types 1 and 4 invert to their own
kind, over `prod(2 .* size(r)[region] .- 2)` for a type-1 DCT and
`prod(2 .* size(r)[region] .+ 2)` for a type-1 DST. VkFFT applies the factor
in-kernel, so a round trip lands on a scale of one and no `ScaledPlan` ever
appears in this family.

Two restrictions have no equivalent in FFTW. One kind and one type apply to
every transformed axis, so FFTW's per-axis `kinds` tuple cannot be expressed
here. And every transformed dimension has to have length at least 2, since
VkFFT computes the identity along a length-1 axis where FFTW scales, and it
hangs while planning a type-1 transform of one.

## Zero-padding

`zeropad=lo:hi` declares the samples `lo:hi` of the first transformed dimension
zero. It is available on the six complex-to-complex entry points, on
`VkFFT.plan_rfft` and on all eight real-to-real ones.

```julia
p = VkFFT.plan_fft(x, 1; zeropad=513:1024) # only the first 512 rows matter
p * x                                      # the transform of the zeroed signal
```

A forward plan never reads that range, so you do not have to clear it. VkFFT
does not touch that range.

An inverse plan skips *writing* the range instead, and leaves it holding
whatever the destination already held. You should treat that range as undefined,
not as zeros, and note that the skip is not reliable above one dimension.

The padded dimension has to be the first transformed one, and that one has to be
the first dimension of the array longer than 1. Any other region throws an
error. The range itself is checked here too, because VkFFT clamps or ignores
whatever it cannot use instead of rejecting it. `hi` may equal the axis length,
which is how you cover the tail.

A padded plan has neither an inverse nor an adjoint.

## Fused convolution

`VkFFT.plan_conv` runs the forward transform, the multiply against a
pre-transformed kernel and the inverse transform as one VkFFT application.
Element types are `ComplexF32` and `ComplexF64`.

```julia
h = rand(ComplexF32, 256, 256)
k = rand(ComplexF32, 256, 256)
x = CLArray{ComplexF32}(undef, size(h))
kernel = CLArray{ComplexF32}(undef, size(k))
copyto!(x, h)
copyto!(kernel, k)

p = VkFFT.plan_conv(x, kernel)                 # circular convolution
y = p * x                                      # at a scale of exactly one

c = VkFFT.plan_conv(x, kernel; correlate=true) # cross-correlation instead
```

VkFFT applies the whole 1/N itself, so `p * x` is the plain circular
convolution. Along one transformed axis of length `n`,
`(p * x)[m] == sum(x[j] * kernel[mod(m - j, n) + 1] for j in 1:n)`, and the
transform is separable over the transformed axes. `correlate=true` conjugates
the transformed input, which gives
`(p * x)[m] == sum(conj(x[j]) * kernel[mod(j + m - 2, n) + 1] for j in 1:n)`.

`kernel` has to be the same element type, the same size and on the same device
as `x`, and it is read once, here. The plan transforms it into a buffer of its
own, so writing to your kernel array afterwards, or freeing it, does not affect
the plan.

At most two dimensions can be transformed, and they have to run from the first
dimension longer than 1 up to a trailing run of untransformed ones. Any other
region is refused. That trailing run is a stack of independent components, each
convolved against its own slice of the kernel, so broadcast the kernel yourself
if one kernel is meant for all of them.

What this family does not have:

| | |
| --- | --- |
| in-place form | none |
| `inv` and `'` | none |
| `zeropad` | refused with an error, not ignored |
| `tune` | none, tune the plain `VkFFT.plan_fft` of the same axes instead |
| plan cache | convolution plans are never cached |
| `ComplexF16` | refused, VkFFT returns something that is no convolution |

One family of transformed lengths comes back wrong with nothing reported. See
[Known failures](@ref).
