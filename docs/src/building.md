# Building the wrapper

For working on VkFFT.jl or testing against VkFFT's development tip. Everyone
else installs a loader package instead. See [Install](@ref).

VkFFT.jl calls [`libvkfft`](https://github.com/PaulVirally/libvkfft), a small C
wrapper around VkFFT. The JLLs ship a prebuilt one per backend. To build your
own:

```sh
cmake -S . -B build -DVKFFT_BACKEND=3 -DVKFFT_MAX_FFT_DIMENSIONS=12 -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

VkFFT picks its backend at compile time, so one build drives one backend:
`-DVKFFT_BACKEND=1` for CUDA, `3` for OpenCL, `5` for Metal.

Keep `VKFFT_MAX_FFT_DIMENSIONS` at 12. It sets the array lengths in
`vkfft_config`, which VkFFT.jl mirrors. The first call into the library checks
that value and `sizeof(vkfft_config)` against the package and throws when either
differs.

Point VkFFT.jl at what you built:

```julia
using Preferences, VkFFT
set_preferences!(VkFFT, "libvkfft_path" => "/path/to/libvkfft.so")
```

The preference wins over the JLL, so the loader package can stay installed. The
path is resolved on the first call into the library, not at `using` time.
