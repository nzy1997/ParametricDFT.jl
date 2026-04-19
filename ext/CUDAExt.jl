module CUDAExt

using CUDA
using ParametricDFT
using LinearAlgebra
using ChainRulesCore

# Device transfer

"""Move array to GPU via CuArray{T}(x) to preserve Float64 precision."""
function ParametricDFT.to_device(x::AbstractArray{T}, ::Val{:gpu}) where T
    return CuArray{T}(x)
end
ParametricDFT.to_device(x, ::Val{:gpu}) = x  # scalars pass through

# GPU-native top-k truncation with frequency weighting

# Cached frequency weights to avoid per-call GPU allocations
const _freq_weight_cache = Dict{Tuple{Int,Int}, CuArray{Float64,2}}()

function _get_freq_weights_gpu(m::Int, n::Int)
    key = (m, n)
    if !haskey(_freq_weight_cache, key)
        center_i = (m + 1) / 2
        center_j = (n + 1) / 2
        max_dist = sqrt((m / 2)^2 + (n / 2)^2)
        is = Float64.(1:m)
        js = Float64.(1:n)'
        freq_dists = sqrt.((is .- center_i) .^ 2 .+ (js .- center_j) .^ 2)
        freq_weights = 1.0 .- (freq_dists ./ max_dist) .* 0.5
        _freq_weight_cache[key] = CuArray(freq_weights)
    end
    return _freq_weight_cache[key]
end

function _topk_mask_gpu(x::CuArray{T}, k::Integer) where T
    m, n = size(x)
    k2 = min(Int(k), m * n)
    RT = real(T)

    freq_weights = _get_freq_weights_gpu(m, n)
    scores = abs.(x) .* (1.0 .+ freq_weights)

    # Sort on GPU, extract threshold via 1-element slice (no @allowscalar)
    sorted = sort(vec(scores); rev=true)
    threshold_arr = sorted[k2:k2]
    mask = RT.(reshape(vec(scores) .>= threshold_arr, m, n))
    return mask
end

function ParametricDFT.topk_truncate(x::CuArray{T}, k::Integer) where {T}
    mask_gpu = _topk_mask_gpu(x, k)
    return x .* mask_gpu
end

# rrule for topk_truncate on GPU (gradient flows through kept elements)

function ChainRulesCore.rrule(::typeof(ParametricDFT.topk_truncate), x::CuArray{T}, k::Integer) where {T}
    mask_gpu = _topk_mask_gpu(x, k)
    y = x .* mask_gpu
    function topk_truncate_pullback(ȳ)
        return (ChainRulesCore.NoTangent(), ȳ .* mask_gpu, ChainRulesCore.NoTangent())
    end
    return y, topk_truncate_pullback
end

# GPU batched top-k truncation: all B per-image masks via one sort(dims=1) call.
# Replaces the per-image `map(1:B)` loop in batched_loss_mse on GPU.

function _topk_mask_gpu_batched(x_batched::CuArray{T, N}, m::Int, n::Int,
                                k::Integer) where {T, N}
    @assert N == m + n + 1
    B = size(x_batched, N)
    Ntot = 2^(m + n)
    RT = real(T)
    k2 = min(Int(k), Ntot)

    freq_weights = _get_freq_weights_gpu(2^m, 2^n)
    freq_flat = reshape(freq_weights, Ntot)                 # (Ntot,)

    # Scores: |x| * (1 + w), broadcast w across the B columns
    x_flat = reshape(x_batched, Ntot, B)
    scores = abs.(x_flat) .* (one(RT) .+ freq_flat)

    # Sort each column descending, take the k-th largest per column as threshold
    sorted = sort(scores; dims = 1, rev = true)             # (Ntot, B)
    thresholds = sorted[k2:k2, :]                            # (1, B) — broadcasts

    mask_flat = RT.(scores .>= thresholds)                   # (Ntot, B)
    return reshape(mask_flat, size(x_batched))
end

function ParametricDFT.batched_topk_truncate(x_batched::CuArray{T}, m::Int, n::Int,
                                             k::Integer) where T
    return x_batched .* _topk_mask_gpu_batched(x_batched, m, n, k)
end

function ChainRulesCore.rrule(::typeof(ParametricDFT.batched_topk_truncate),
                              x_batched::CuArray{T}, m::Int, n::Int,
                              k::Integer) where T
    mask = _topk_mask_gpu_batched(x_batched, m, n, k)
    y = x_batched .* mask
    function pullback(ȳ)
        return (ChainRulesCore.NoTangent(), ȳ .* mask,
                ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),
                ChainRulesCore.NoTangent())
    end
    return y, pullback
end

function ParametricDFT.batched_inv(A::CuArray{T,3}) where T
    d, _, n = size(A)
    if d == 2
        # Explicit 2×2 inverse: (1/(ad-bc)) * [d -b; -c a]
        # Pure broadcasting — single GPU kernel
        a = A[1:1, 1:1, :]
        b = A[1:1, 2:2, :]
        c = A[2:2, 1:1, :]
        dd = A[2:2, 2:2, :]
        det = a .* dd .- b .* c
        inv_det = one(T) ./ det
        return cat(
            cat(dd .* inv_det, -(b .* inv_det); dims=2),
            cat(-(c .* inv_det), a .* inv_det; dims=2);
            dims=1
        )
    else
        # General case: per-slice inverse
        C = similar(A)
        for k in 1:n
            C[:, :, k] .= inv(A[:, :, k])
        end
        return C
    end
end

function ParametricDFT.batched_matmul(A::CuArray{T,3}, B::CuArray{T,3}) where T
    d1, d2A, n = size(A)
    d2B, d3, n2 = size(B)
    @assert d2A == d2B "Inner dimensions must match: got $d2A and $d2B"
    @assert n == n2 "Batch sizes must match: got $n and $n2"
    C = similar(A, T, d1, d3, n)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), A, B, zero(T), C)
    return C
end

end # module
