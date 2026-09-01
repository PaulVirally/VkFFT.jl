# Tuning

VkFFT takes two hints about how to decompose a transform into blocks and
threads: `coalesced_memory` and `aim_threads`. `coalesced_memory` is a byte
width the memory system is assumed to reward and `aim_threads` is a thread count
per block. The optimal values depend on the device and the shape of the
transform. Specifying `tune=true` finds the best values by measuring the time
taken by many transforms.

```julia
p = VkFFT.plan_fft(x, (1, 2); tune=true)   # sweep, or read a stored answer
p = VkFFT.plan_fft(x, (1, 2); tune=:force) # sweep again and overwrite it

VkFFT.last_sweep()     # what the last sweep measured, fastest first
VkFFT.sweep_count()    # how many sweeps this session actually ran
VkFFT.disk_cache_dir() # where the records live
VkFFT.clear_tuning!()  # delete every record, and return how many
```

[`VkFFT.last_sweep`](@ref) returns `(coalesced_memory, aim_threads,
microseconds)` entries, fastest first. It is empty until a sweep has run in this
session, and the next sweep overwrites it.

```julia
for (coalesced_memory, aim_threads, microseconds) in VkFFT.last_sweep()
    println("$coalesced_memory bytes, $aim_threads threads: $microseconds us")
end
```

[`VkFFT.sweep_count`](@ref) counts the expensive calls only. A `tune=true` that
finds a stored record and skips the grid does not move it.

`tune` works on the complex-to-complex, real-to-complex and real-to-real
families. It is not available on `VkFFT.plan_conv`, and it applies to the plan
being built rather than to the inverse plans `inv` derives from it. On a region
that transforms nothing it plans the copy, sweeps nothing and writes no record.

## What the sweep measures

The grid is 16 candidates, `coalescedMemory` over 0, 32, 64 and 128 bytes
crossed with `aimThreads` over 0, 64, 128 and 256. A 0 leaves the choice to
VkFFT, whose own defaults are 128 threads everywhere and:

| device | `coalescedMemory` |
| --- | --- |
| CUDA, NVIDIA and AMD OpenCL | 32 bytes |
| Metal, Intel, unrecognized OpenCL vendors | 64 bytes |
| any of the above, half precision | doubled |


Each candidate is a real plan of the shape being tuned. Timing runs two
warmed-up applications, then sizes a repeat count so that one candidate's run
takes roughly 20 milliseconds, clamped to between 3 and 500 repeats and computed
once so every candidate is compared over the same amount of work. `mul!`
synchronizes before it returns, so the host clock measures device time plus one
synchronization per application. The whole grid is timed twice and each
candidate keeps its faster pass.

The sweep allocates buffers of the plan's own shape and fills them with ones.
It does not touch the array you passed.

When the sweep finishes, the winner stays in the plan cache and every loser is
removed from it and freed, so the plan you get back is a cache hit and not a
seventeenth compilation.

The cost of tuning is one sweep, sixteen plans built and timed, on the first
call on a shape in the first process.

## Records on disk

The outcomes of the tuning go into a
[Scratch.jl](https://github.com/JuliaPackaging/Scratch.jl) space, which
survives across sessions and is garbage collected with the package rather than
with the depot. Use [`VkFFT.disk_cache_dir`](@ref) to see where this scratch
space is. A record is one file holding a comment line and two integers, named
for the backend and the first twelve hex digits of its key.

The key covers the backend tag, the `VKFFT_BACKEND` the wrapper was built for,
the device's cross-process identity, the SHA-256 of the wrapper library itself,
the configuration size the wrapper reads, and every field of the plan's
configuration except the two tuning fields. The device identity is the
human-readable one:

| backend | identity |
| --- | --- |
| OpenCL | platform name, device name, `CL_DRIVER_VERSION` |
| CUDA | device name, compute capability |
| Metal | device name, registry ID |

A record that is absent, unreadable or unparseable is a miss, not an error.
Writes go through a temporary file and a rename, so two processes writing the
same key is benign and no reader sees a half-written record. A write that
fails, on a full disk or a read-only depot, throws nothing.

`tune=:force` re-sweeps one shape and overwrites its record, which is how a
machine whose driver has changed underneath its records gets retuned.
[`VkFFT.clear_tuning!`](@ref) deletes all of them.

Nothing compiled is ever cached. Two integers per tuned configuration is all
that persists. VkFFT does have an interface for saving compiled applications,
and this package does not call it.
