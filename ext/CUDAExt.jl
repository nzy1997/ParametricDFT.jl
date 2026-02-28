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

function ParametricDFT.topk_truncate(x::CuArray{T}, k::Integer) where {T}
    m, n = size(x)
    k2 = min(Int(k), length(x))
    x_cpu = Array(x)

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
    mask = zeros(RT, m, n)
    for flat_idx in idx
        mask[flat_idx] = one(RT)
    end
    return x .* CuArray{RT}(mask)
end

# rrule for topk_truncate on GPU (gradient flows through kept elements)

function ChainRulesCore.rrule(::typeof(ParametricDFT.topk_truncate), x::CuArray{T}, k::Integer) where {T}
    m, n = size(x)
    k2 = min(Int(k), length(x))
    x_cpu = Array(x)

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
    mask_gpu = CuArray{RT}(mask_cpu)
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

end # module
