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

# GPU-compatible top-k truncation with frequency weighting

function _topk_mask(x_cpu::Array{T}, k::Integer) where T
    m, n = size(x_cpu)
    k2 = min(Int(k), length(x_cpu))
    center_i, center_j = (m + 1) ÷ 2, (n + 1) ÷ 2
    max_dist = sqrt((m/2)^2 + (n/2)^2)
    scores = similar(x_cpu, Float64)
    mags = abs.(x_cpu)
    for j in 1:n, i in 1:m
        freq_dist = sqrt((i - center_i)^2 + (j - center_j)^2)
        freq_weight = 1.0 - (freq_dist / max_dist) * 0.5
        scores[i, j] = mags[i, j] * (1.0 + freq_weight)
    end
    idx = partialsortperm(vec(scores), 1:k2, rev=true)
    RT = real(T)
    mask_cpu = zeros(RT, m, n)
    for flat_idx in idx
        mask_cpu[flat_idx] = one(RT)
    end
    return CuArray{RT}(mask_cpu)
end

function ParametricDFT.topk_truncate(x::CuArray{T}, k::Integer) where {T}
    mask_gpu = _topk_mask(Array(x), k)
    return x .* mask_gpu
end

# rrule for topk_truncate on GPU (gradient flows through kept elements)

function ChainRulesCore.rrule(::typeof(ParametricDFT.topk_truncate), x::CuArray{T}, k::Integer) where {T}
    mask_gpu = _topk_mask(Array(x), k)
    y = x .* mask_gpu
    function topk_truncate_pullback(ȳ)
        return (ChainRulesCore.NoTangent(), ȳ .* mask_gpu, ChainRulesCore.NoTangent())
    end
    return y, topk_truncate_pullback
end

# NVTX profiling callbacks

function __init__()
    ParametricDFT._nvtx_push_fn[] = name -> (CUDA.NVTX.range_push(; message=name); nothing)
    ParametricDFT._nvtx_pop_fn[] = () -> (CUDA.NVTX.range_pop(); nothing)
end

function ParametricDFT.batched_matmul(A::CuArray{T,3}, B::CuArray{T,3}) where T
    d1, d2A, n = size(A)
    d2B, d3, n2 = size(B)
    @assert d2A == d2B "Inner dimensions must match: got $d2A and $d2B"
    @assert n == n2 "Batch sizes must match: got $n and $n2"
    C = similar(A, T, d1, d3, n)
    for k in 1:n
        C[:, :, k] .= A[:, :, k] * B[:, :, k]
    end
    return C
end

end # module
