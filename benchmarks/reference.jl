# Ground truth for the accuracy half of the suite. Pure numerics over
# AbstractVector, knowing nothing about VkFFT, devices, cases or CSV.
#
# The reference runs at 256 bits, roughly 2^-256 under Float64's 2^-53, so what
# it measures is the other transform's error and not its own. Twiddles come from
# BigFloat trig rather than from Float64 values widened after the fact, which
# would put a Float64 rounding back inside the thing we call exact.
#
# Powers of two go through a recursive radix 2 transform, every other length
# through Bluestein layered on that same path, so both are n log n. The direct
# quadratic sum is here only as the thing those two are validated against at
# small n, being far too slow for the sweep.

using Random

const REF_BITS = 256

"""
    reference_fft(x::AbstractVector, forward::Bool=true)

Computes the DFT of `x` at 256 bits, as a `Vector{Complex{BigFloat}}`.

The forward direction is unnormalized, with the sign convention
`y[m] = sum_j x[j] exp(-2pi i (j-1)(m-1)/n)`. That is FFTW's and VkFFT's
forward direction. The inverse takes the opposite sign and the `1/n`, so the
two compose back to `x`.

`x` is widened exactly, and the answer is the transform of the array you hand
over rather than of whatever that array was rounded from. Pass the same
`ComplexF16` or `ComplexF32` array the implementation under test was given.
Widening a `Float64` original instead measures how badly the input was rounded,
which at `eps(Float16) = 9.8e-4` drowns out everything you were trying to see.

The `rfft` reference is the first `n / 2 + 1` points of this applied to the
complex widening of the real input, and the `irfft` reference is the inverse of
the full hermitian spectrum. Neither needs a path of its own.

# Arguments
- `x::AbstractVector`: The input, of any element type convertible to `Complex{BigFloat}`
- `forward::Bool=true`: Whether to compute the forward transform rather than the normalized inverse

# Returns
- The transform of `x`, a `Vector{Complex{BigFloat}}`
"""
function reference_fft(x::AbstractVector, forward::Bool=true)
    n = length(x)
    return setprecision(BigFloat, REF_BITS) do
        z = Complex{BigFloat}.(x)
        forward || (z = conj.(z)) # the inverse is the forward transform between conjugates
        y = ispow2(n) ? _radix2_fft(z, [cis(-2 * BigFloat(pi) * k / n) for k in 0:(n >> 1) - 1]) : _bluestein_fft(z)
        forward ? y : conj.(y) ./ n
    end
end

"""
    _radix2_fft(x::AbstractVector{Complex{BigFloat}}, w::Vector{Complex{BigFloat}})

Computes the forward DFT of a power of two length `x` by decimation in time.

`w` holds `exp(-2pi i k/n)` for `k` in `0:n/2-1` at the top level `n`, and a
sub-transform of length `m` reads it with stride `n/m`. Sharing one table costs
`n/2` trig evaluations instead of one per butterfly, which at 2^20 is the
difference between seconds and hours.
"""
function _radix2_fft(x::AbstractVector{Complex{BigFloat}}, w::Vector{Complex{BigFloat}})
    m = length(x)
    m == 1 && return Complex{BigFloat}[x[1]]

    h = m >> 1
    even = _radix2_fft(x[1:2:end], w)
    odd = _radix2_fft(x[2:2:end], w)

    step = length(w) ÷ h
    y = Vector{Complex{BigFloat}}(undef, m)
    for k in 1:h
        t = w[1 + (k - 1) * step] * odd[k]
        y[k] = even[k] + t
        y[k + h] = even[k] - t
    end
    return y
end

"""
    _bluestein_fft(x::AbstractVector{Complex{BigFloat}})

Computes the forward DFT of an arbitrary length `x` as a chirp filtered convolution.

Writing `(j-1)(m-1)` as `((j-1)^2 + (m-1)^2 - (m-j)^2)/2` turns the transform
into a convolution against a chirp, which a zero padded power of two cyclic
convolution evaluates. Its inverse transform is the forward one applied to
conjugates, so there is only ever the one kernel.
"""
function _bluestein_fft(x::AbstractVector{Complex{BigFloat}})
    n = length(x)
    len = nextpow(2, 2n - 1)
    pi_n = BigFloat(pi) / n

    # exp(i pi d^2/n) has period 2n in d^2, so reduce before forming the angle.
    v = [cis(-pi_n * mod((j - 1)^2, 2n)) for j in 1:n]
    h = zeros(Complex{BigFloat}, len)
    for d in 0:n - 1
        h[d + 1] = cis(pi_n * mod(d^2, 2n))
        d > 0 && (h[len + 1 - d] = h[d + 1]) # the chirp is even in d
    end

    u = zeros(Complex{BigFloat}, len)
    u[1:n] .= x .* v

    w = [cis(-2 * BigFloat(pi) * k / len) for k in 0:(len >> 1) - 1]
    c = _radix2_fft(u, w) .* _radix2_fft(h, w)
    return v .* (conj.(_radix2_fft(conj.(c), w))[1:n] ./ len)
end

"""
    direct_dft(x::AbstractVector)

Computes the unnormalized forward DFT of `x` by the defining quadratic sum at 256 bits.

The independent check on `reference_fft`, not a path the sweep ever takes. It
spends O(n^2) BigFloat trig evaluations, so keep `n` in the hundreds.
"""
function direct_dft(x::AbstractVector)
    return setprecision(BigFloat, REF_BITS) do
        z = Complex{BigFloat}.(x)
        n = length(z)
        two_pi_n = 2 * BigFloat(pi) / n
        [sum(z[j] * cis(-two_pi_n * mod((j - 1) * (m - 1), n)) for j in 1:n) for m in 1:n]
    end
end

"""
    relative_error(y::AbstractVector, ref::AbstractVector)

Computes the relative L2 error of `y` against `ref`, bare and as a multiple of `eps`.

The difference accumulates at 256 bits whenever either side is a `BigFloat`, so
a `Float16` result and a `Float64` one are read on the same scale. When neither
side is, the accumulation runs in `Float64`. It computes the same number there:
the two values agree to within an ulp or so, which makes their difference exact,
and the sums that follow are over nonnegative terms and carry no cancellation.
Accumulating a `Float32` result against a `Float64` reference at 256 bits would
spend thirty times the arithmetic to print the same digits.

The `eps` is `y`'s, which is what makes "3 eps" comparable across precisions and
is the number to plot.

# Returns
- `(err, eps_multiple)`: the relative L2 error, and that error over `eps(real(eltype(y)))`
"""
function relative_error(y::AbstractVector, ref::AbstractVector)
    length(y) == length(ref) || throw(DimensionMismatch("y has $(length(y)) points, reference has $(length(ref))"))

    err = if BigFloat in (real(float(eltype(y))), real(float(eltype(ref))))
        setprecision(BigFloat, REF_BITS) do
            zy, zref = Complex{BigFloat}.(y), Complex{BigFloat}.(ref)
            Float64(sqrt(sum(abs2, zy .- zref) / sum(abs2, zref)))
        end
    else
        zy, zref = ComplexF64.(y), ComplexF64.(ref)
        sqrt(sum(abs2, zy .- zref) / sum(abs2, zref))
    end

    return (err=err, eps_multiple=err / eps(real(float(eltype(y)))))
end

"""
    uniform_input(rng::AbstractRNG, n::Integer, T::Type{<:Number}=ComplexF64)

Returns `n` points drawn uniformly from [-1, 1), in both parts when `T` is complex.

The average case, and the one the headline accuracy plot uses. The timing half
of the suite draws its fixed input here too, so that both halves mean the same
thing by uniform random. Feed it a `StableRNG` so a draw reproduces across Julia
versions. The values are well scaled, so nothing drifts into the subnormal range
where x86 arithmetic slows by one to two orders of magnitude.
"""
uniform_input(rng::AbstractRNG, n::Integer, T::Type{<:Complex}=ComplexF64) = T.(complex.(2 .* rand(rng, n) .- 1, 2 .* rand(rng, n) .- 1))

uniform_input(rng::AbstractRNG, n::Integer, T::Type{<:Real}) = T.(2 .* rand(rng, n) .- 1)

"""
    chirp_input(n::Integer, T::Type{<:Complex}=ComplexF64)

Returns `n` unit modulus points of the linear chirp `exp(i pi (j-1)^2/n)`.

Its spectrum is flat, so every output bin is a sum of `n` unit modulus terms
cancelling down to about `sqrt(n)`. That is the worst realistic case, and it
stresses cancellation in a way uniform random input does not. The points are
formed at 256 bits and rounded to `T` once, so the input carries no error the
transform did not put there.
"""
function chirp_input(n::Integer, T::Type{<:Complex}=ComplexF64)
    return setprecision(BigFloat, REF_BITS) do
        pi_n = BigFloat(pi) / n
        T[cis(pi_n * mod((j - 1)^2, 2n)) for j in 1:n]
    end
end

"""
    impulse_input(n::Integer, T::Type{<:Complex}=ComplexF64)

Returns the unit impulse of length `n` and its exact forward DFT, all ones.

Exact by construction and in closed form, so this is the cheap check to run
before paying for a reference transform.

# Returns
- `(x, y)`: the impulse and its exact unnormalized forward DFT, both `Vector{T}`
"""
impulse_input(n::Integer, T::Type{<:Complex}=ComplexF64) = ([j == 1 ? one(T) : zero(T) for j in 1:n], ones(T, n))
