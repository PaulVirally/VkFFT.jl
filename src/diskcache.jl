# The on-disk tuning records and the key they are filed under. A record stores
# the swept (coalesced_memory, aim_threads) winner for one plan configuration
# on one device, so a later session or process can skip the sweep. The key has
# to go stale whenever anything that could move the optimum changes, which is
# why it covers the device, the wrapper library and the whole configuration.

## The disk key

"""
    _device_key(roots::Vector{Any}, backend::Val)

Returns a cross-process identity for the device a plan is about to be built for.

`_device_id` cannot do this job. It is a context or object handle, which is
exactly right for the in-memory plan cache and meaningless in the next
process. This is the human-readable device identity instead: enough of the
platform, model and driver that two devices which would tune differently never
produce the same string. It is read from `roots` rather than from an array,
because the key is needed where a plan's configuration is built and the array
is long out of scope by then.
"""
_device_key(roots::Vector{Any}, backend::Val) = throw(ArgumentError("VkFFT has no cross-process device identity for backend $backend, so it cannot key a tuning record. Plan without tune=true, or add a _device_key method for this backend."))

# The wrapper library's own contents, which pin the VkFFT version and the
# wrapper build together. Hashing a multi-megabyte file is not something to do
# per plan, and the path cannot change without a restart (VkFFT.__init__ reads
# it), so it is done once.
const LIBRARY_DIGEST = Ref("")

"""
    _library_digest()

Returns the SHA-256 of the loaded libvkfft, computed on first use and kept for the session.

This is what makes a record tuned under one VkFFT version miss under another.
A new VkFFT can lay the same transform out differently, and nothing else in
the key would notice the upgrade.
"""
function _library_digest()
    isempty(LIBRARY_DIGEST[]) || return LIBRARY_DIGEST[]

    _ensure_library!()
    @lock LIBRARY_LOCK begin
        isempty(LIBRARY_DIGEST[]) || return LIBRARY_DIGEST[]
        LIBRARY_DIGEST[] = bytes2hex(open(SHA.sha256, libvkfft))
    end

    return LIBRARY_DIGEST[]
end

# Where the tuning fields sit in VkFFTConfig, so that the record key can leave
# them out without anything hardcoding a field number.
const CONFIG_COALESCED_MEMORY = findfirst(==(:coalesced_memory), fieldnames(VkFFTConfig))
const CONFIG_AIM_THREADS = findfirst(==(:aim_threads), fieldnames(VkFFTConfig))

"""
    _disk_key(config::VkFFTConfig, roots::Vector{Any}, backend::Val{B})

Returns the hex SHA-256 the tuning record for this configuration is filed under.

The digest covers everything that can move the tuned optimum or make a stored
answer stale: the backend tag and the wrapper's own `VKFFT_BACKEND`, the
device identity, the wrapper library's contents (which pin the VkFFT version),
`sizeof(vkfft_config)` as the wrapper reads it, and every field of the
configuration except the two tuning fields, which enter as zero. The record's
whole content is the values those fields should take, so they cannot also be
part of its own key. A record left behind by a VkFFT upgrade misses rather
than applies, which is why the library contents are in here.
"""
function _disk_key(config::VkFFTConfig, roots::Vector{Any}, backend::Val{B}) where B
    parts = Any[string(B), _device_key(roots, backend), _library_digest(), _backend_id(), _vkfft_config_size()]

    for i in 1:fieldcount(VkFFTConfig)
        if i == CONFIG_COALESCED_MEMORY || i == CONFIG_AIM_THREADS
            push!(parts, 0)
        else
            value = getfield(config, i)
            value isa Tuple ? append!(parts, value) : push!(parts, value)
        end
    end

    # One canonical NUL-separated string, hashed once. The config always
    # flattens to the same number of components (its field count and tuple
    # lengths are compile-time constants), so with a separator no component can
    # contain, equal strings mean equal components. NUL cannot appear: the
    # device key is the only free-form component and it is built from C
    # strings, and everything else is an integer's decimal digits or an ASCII
    # backend tag.
    return bytes2hex(SHA.sha256(join(parts, '\0')))
end

## The scratch space

# One space, with the key in every filename, rather than a space per package
# version. What makes a record stale is the VkFFT version, the device or the
# configuration, and all three are in the key already, so versioning the
# directory as well would only orphan records that are still good.
const CACHE_SPACE = "tuning"
const CACHE_DIR = Ref("")
const DISK_CACHE_LOCK = ReentrantLock()

# Filename prefix and extension of a tuning record, which is two integers a
# human may want to read.
const RECORD_FILE = ("tune", ".txt")

"""
    disk_cache_dir()

Returns the directory the tuning records live in.

A `Scratch.jl` space, so it survives across sessions and is garbage collected
with the package rather than with the depot. The filenames carry a backend
name and the first twelve hex digits of the key, which is enough to tell at a
glance what a directory listing is holding.

# Returns
- The path of the scratch space, created if it did not exist
"""
function disk_cache_dir()
    isempty(CACHE_DIR[]) || return CACHE_DIR[]

    @lock DISK_CACHE_LOCK begin
        isempty(CACHE_DIR[]) || return CACHE_DIR[]
        CACHE_DIR[] = @get_scratch!(CACHE_SPACE)
    end

    return CACHE_DIR[]
end
