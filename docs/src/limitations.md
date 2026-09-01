# Limitations

## Arrays you can pass to a plan

You may only pass a dense device array of the backend you loaded. Passing a
non-contiguous view, an adjoint, a permuted view and a host `Array` all throw an
error.

A *contiguous* slice of a larger allocation depends on the backend. There are no
issues here with CUDA.jl. OpenCL buffers cross as `cl_mem` handles and Metal
buffers as `MTLBuffer` objects, and neither carries an offset the launch
parameters can use, so a `CLArray` or `MtlArray` at a nonzero offset is refused.
You must copy the array first.

```julia
host = ComplexF32[i for i in 1:8]
big = CLArray{ComplexF32}(undef, 8)
copyto!(big, host)

head = view(big, 1:4) # offset 0
tail = view(big, 5:8) # nonzero offset

p = VkFFT.plan_fft(head)
p * head # fft(host[1:4])
p * tail # ArgumentError
```

The layout is checked on every transform, not only at plan time, because a plan
is reusable across buffers and an offset view of the right size clears every
size check.

A plan is also bound to the device context it was built in. Passing an array
from another CUDA or OpenCL context, or from another Metal device, throws an
error.

## Precision

Using the wrong element type will throw an error according to the table below:

| You passed | You should do instead |
| --- | --- |
| A real array into `VkFFT.plan_fft` | `VkFFT.plan_rfft`, `VkFFT.plan_brfft`, `VkFFT.plan_irfft` |
| A complex array into `VkFFT.plan_rfft` or a real-to-real planner | `VkFFT.plan_fft` |
| A real array into `VkFFT.plan_conv` | Convert first, since VkFFT's real-input convolution needs a padded in-place layout this package does not build |
| An `Int` or `BigFloat` array | Use the element types VkFFT supports, see [Element types](@ref) (or [`VkFFT.unsafe_plan`](@ref) for quad precision) |

`VkFFT.plan_conv` throws an error for `ComplexF16` on every backend. Upstream,
VkFFT has issues with convolutions on half precision (it gives the wrong
values), so we throw an error in this package.

Two more errors can happen relating to the device or the wrapper rather than the
element type, and both are on [Backends and capabilities](backends.md). Half
precision on OpenCL needs `cl_khr_fp16`, and half precision on CUDA needs a
wrapper built against a CUDA toolkit.

## Bugs in VkFFT

Each configuration below is one VkFFT technically accepts, reports
`VKFFT_SUCCESS` for, but gives the wrong numerical output values. The planner
throws an error for every one of them, so this list is here to explain the
guards, not to warn you off anything you can still do:

| Configuration | What VkFFT does |
| --- | --- |
| A region that transforms nothing, or whose every axis has length 1 | Reports success and dispatches no kernel, leaving an out-of-place caller holding uninitialized memory. The planner applies it as a copy in Julia instead |
| A type-1 DCT or DST of an axis of length 1 | Never returns from `vkfft_create`. The internal transform length goes to zero and the radix decomposition has nothing to cancel. The other types create, run, and compute the identity where FFTW scales |
| A DCT and a DST on one plan | Computes the DST, or at other type pairs something that is neither while still round-tripping |
| A real-to-complex plan carrying a DCT or a DST | Corrupts the spectrum |
| A DCT or DST type outside 1 to 4 | Falls through every branch that generates the pre- and post-processing, leaving a plain real transform |
| A plan that both convolves and prepares a kernel | Reads the kernel flag first and ignores the other |
| A convolution plan with only one direction's kernels generated | Dereferences the half it was told to skip |
| Zero-padding on any axis but VkFFT's own first one | Creates, runs and reads uninitialized memory |
| A convolution over three or more axes, or over a layout that omits an axis | Rejects a leading or trailing omit and accepts a middle one while computing something that is no convolution |

See also [Known failures](@ref).

## Freeing a plan

Every plan is freed by its finalizer, so nothing has to be freed by hand.
[`VkFFT.unsafe_free!`](@ref) frees one now instead, at most once:

```julia
p = VkFFT.plan_fft(x, 1)
VkFFT.unsafe_free!(p)
p * x                 # ArgumentError, this plan has already been freed
VkFFT.unsafe_free!(p) # a second call is a no-op
```

It is unsafe for two reasons. Another reference to the same plan can still be
applied afterwards, which throws rather than crashing but is still not what its
holder expected. And a cached plan stays in the cache, so the next call that
would have hit it gets a freed plan back. Call [`VkFFT.clear_cache!`](@ref)
first if you mean to free everything.

A plan holds references to its backend objects, because the context and the
queue a VkFFT application was built against have to outlive it.

## Do not plan inside an autorelease pool

This is for Metal only. VkFFT's Metal plan builder over-releases the strings it
compiles its kernels from, so an Objective-C autorelease pool drained after a
planning call dies inside `objc_release` and takes the process with it:

```julia
p = VkFFT.plan_fft(x)                    # fine
Metal.@autoreleasepool VkFFT.plan_fft(x) # crashes when the pool drains
```

VkFFT.jl never opens a pool around planning, so this only bites if you wrap the
call yourself. Applying a plan is unaffected. Where Julia is embedded in a
Cocoa application that drains a pool on the main thread, plan from a worker
task. The over-release is an upstream VkFFT bug with a fix pending in
[DTolm/VkFFT#227](https://github.com/DTolm/VkFFT/pull/227).

## The raw configuration

[`VkFFT.unsafe_plan`](@ref) builds a plan from a [`VkFFTConfig`](@ref) you
filled in yourself, forwarded to the wrapper unchanged. Two prototype arrays
supply what a configuration cannot: the backend, the device context, the element
types and the Julia array shapes an application is checked against. Their
contents are neither read nor written.

```julia
h = rand(Float32, 16, 8)
x = CLArray{Float32}(undef, size(h))
copyto!(x, h)

config = VkFFTConfig(fft_dim=2,
                     size=ntuple(i -> i <= 2 ? UInt64((16, 8)[i]) : UInt64(0),
                                 VkFFT.VKFFT_MAX_FFT_DIMENSIONS),
                     dct=2, make_forward_only=1)
raw = VkFFT.unsafe_plan(config, x, similar(x))

Array(raw * x) == Array(VkFFT.plan_dct(x, (1, 2)) * x) # identical
```

Zero means VkFFT's default for every field, so a zeroed configuration plus
`fft_dim` and `size` is a valid out-of-place complex-to-complex plan in single
precision. Axes run fastest first, which is Julia's column-major order, and
`size` slots past `fft_dim` have to be zero.

`unsafe_` means something specific: not one of the errors in [Bugs in
VkFFT](@ref) runs on this path. The damage is not just a wrong answer. The
transform VkFFT generates for a configuration like those is not the transform
the buffers were sized for, so applying one writes past the end of them and
corrupts the process heap, and the crash then lands wherever that heap is next
touched. Work out the byte layout your configuration implies before applying it.

A raw plan keeps the structural checks that are about Julia arrays rather than
about the transform: the two shapes, the in-place agreement the configuration
asked for, the memory layout of both arrays and the device context they live
in. It has no inverse, no adjoint and no `AbstractFFTs.fftdims`, since nothing
here knows which Julia dimensions `config.size` was built from. `size` is the
input prototype's size and `AbstractFFTs.output_size` the output prototype's.
Raw plans are never cached, and they are freed by their finalizer like every
other plan.

## Quad precision

VkFFT's `precision = 3` is double-double: four native doubles per complex value,
the real part as a high-order and a low-order double and then the imaginary part
the same way. No native Julia element type has that layout, which is why quad is
reachable only through the raw configuration. A `ComplexF64` array of twice the
length holds exactly those bytes, so it can stand in as a prototype, and the
packing and unpacking are yours.

```julia
# Only the high-order real double is written, which is exact: a Float64 is
# its own double-double with a zero low-order part.
function quad_pack(h::Vector{Float64})
    flat = zeros(Float64, 4 * length(h))
    for i in eachindex(h)
        flat[4 * (i - 1) + 1] = h[i]
    end
    return collect(reinterpret(ComplexF64, flat))
end

n = 64
packed = quad_pack(rand(n) .- 0.5)
x = CLArray{ComplexF64}(undef, length(packed))
copyto!(x, packed)
y = CLArray{ComplexF64}(undef, length(packed))

config = VkFFTConfig(fft_dim=1,
                     size=ntuple(i -> i == 1 ? UInt64(n) : UInt64(0),
                                 VkFFT.VKFFT_MAX_FFT_DIMENSIONS),
                     precision=3, make_forward_only=1)
forward = VkFFT.unsafe_plan(config, x, y)
mul!(y, forward, x)
```

The inverse is the same configuration with `normalize=1` and
`make_inverse_only=1`, passed to `unsafe_plan` with `direction=1` (a keyword of
[`VkFFT.unsafe_plan`](@ref), not a configuration field). Metal has no double
precision, so it has no double-double either.
