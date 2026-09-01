# Benchmarks

The numbers are not here yet.

The plan is to measure VkFFT against CUFFT on an NVIDIA card, and against FFTW
on the host for the same shapes, over the size classes the test suite already
covers: powers of two, small primes, mixed radix, a Bluestein length that fits
in one upload and one that does not. Complex-to-complex, real-to-complex and
real-to-real, at each precision the backend runs, in-place and out of place,
with the plan built once and applied many times so that planning cost and
transform cost are reported separately.

Two comparisons matter more than the raw throughput. The first is prime and
near-prime lengths, since avoiding a performance cliff there is one of the
reasons to use VkFFT at all. The second is the normalized inverse, which VkFFT
applies inside the transform kernels instead of in a second pass, so an `ifft`
should cost one kernel launch where a scaled `bfft` costs two.

The autotuner gets its own numbers: what [Tuning](tuning.md) buys over VkFFT's
defaults, per shape and per device. On pocl's CPU device the answer so far is
nothing measurable, which says nothing about a GPU.
