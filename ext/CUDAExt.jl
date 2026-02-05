module CUDAExt

using CUDA
using ParametricDFT
using LinearAlgebra
using ChainRulesCore

# ============================================================================
# Device Transfer
# ============================================================================

"""
    to_device(x, ::Val{:gpu})

Move array `x` to GPU using CUDA.jl. Converts to CuArray for GPU-accelerated computation.
Uses Float64 precision to maintain numerical accuracy consistent with CPU operations.
"""
function ParametricDFT.to_device(x::AbstractArray{T}, ::Val{:gpu}) where T
    # Use CuArray with explicit element type to preserve Float64 precision
    # CUDA.cu() defaults to Float32 which can cause type mismatches
    return CuArray{T}(x)
end

# Scalars and non-array types pass through unchanged
function ParametricDFT.to_device(x, ::Val{:gpu})
    return x
end

# ============================================================================
# GPU-Compatible Top-K Truncation
# ============================================================================

"""
    topk_truncate(x::CuArray, k::Integer)

GPU-compatible top-k truncation with frequency weighting.
Computes the mask on CPU (for sorting), applies it on GPU.
"""
function ParametricDFT.topk_truncate(x::CuArray{T}, k::Integer) where {T}
    m, n = size(x)
    k2 = min(Int(k), length(x))

    # Move to CPU for mask computation (sorting not efficient on GPU)
    x_cpu = Array(x)

    # Calculate frequency distances from center (DC component)
    center_i, center_j = (m + 1) ÷ 2, (n + 1) ÷ 2
    max_dist = sqrt((m/2)^2 + (n/2)^2)

    # Create frequency-weighted scores using broadcasting
    mags = abs.(x_cpu)
    scores = similar(mags, Float64)
    for j in 1:n, i in 1:m
        freq_dist = sqrt((i - center_i)^2 + (j - center_j)^2)
        freq_weight = 1.0 - (freq_dist / max_dist) * 0.5
        scores[i, j] = mags[i, j] * (1.0 + freq_weight)
    end

    # Select top k based on weighted scores
    scores_flat = vec(scores)
    idx = partialsortperm(scores_flat, 1:k2, rev=true)

    # Create mask on CPU
    mask = zeros(T, m, n)
    for flat_idx in idx
        mask[flat_idx] = one(T)
    end

    # Apply mask on GPU
    mask_gpu = CUDA.cu(mask)
    return x .* mask_gpu
end

# ============================================================================
# GPU-Compatible Riemannian Operations
# ============================================================================

"""
    project_tangent_unitary(U::CuMatrix, G::CuMatrix)

GPU-compatible projection of gradient onto unitary tangent space.
"""
function ParametricDFT.project_tangent_unitary(U::CuMatrix, G::CuMatrix)
    UhG = U' * G
    S = (UhG - UhG') / 2  # Skew-Hermitian part
    return U * S
end

"""
    retract_unitary_qr(U::CuMatrix, ξ::CuMatrix, α)

GPU-compatible QR retraction on unitary manifold.
Note: QR decomposition on GPU uses cuSOLVER.
"""
function ParametricDFT.retract_unitary_qr(U::CuMatrix, ξ::CuMatrix, α)
    Y = U + α * ξ
    # Use CUDA's QR decomposition
    Q, R = qr(Y)
    Q_mat = CuMatrix(Q)

    # Ensure correct orientation (handle sign ambiguity)
    # Get diagonal signs from R
    d = diag(R)
    signs = sign.(real.(d))
    # Multiply columns of Q by signs
    return Q_mat .* reshape(signs, 1, :)
end

# ============================================================================
# GPU Array Utilities
# ============================================================================

"""
    skew(A::CuMatrix)

GPU-compatible skew-Hermitian projection.
"""
ParametricDFT.skew(A::CuMatrix) = (A - A') / 2

# ============================================================================
# GPU-Compatible rrule for topk_truncate
# ============================================================================

"""
    rrule(::typeof(topk_truncate), x::CuArray, k::Integer)

GPU-compatible custom rrule for topk_truncate.
Computes the mask on CPU (for sorting), applies it on GPU using element-wise operations.
"""
function ChainRulesCore.rrule(::typeof(ParametricDFT.topk_truncate), x::CuArray{T}, k::Integer) where {T}
    m, n = size(x)
    k2 = min(Int(k), length(x))

    # Move to CPU for mask computation (sorting not efficient on GPU)
    x_cpu = Array(x)

    # Calculate frequency distances from center (DC component)
    center_i, center_j = (m + 1) ÷ 2, (n + 1) ÷ 2
    max_dist = sqrt((m/2)^2 + (n/2)^2)

    # Create frequency-weighted scores using broadcasting
    mags = abs.(x_cpu)
    scores = similar(mags, Float64)
    for j in 1:n, i in 1:m
        freq_dist = sqrt((i - center_i)^2 + (j - center_j)^2)
        freq_weight = 1.0 - (freq_dist / max_dist) * 0.5
        scores[i, j] = mags[i, j] * (1.0 + freq_weight)
    end

    # Select top k based on weighted scores
    scores_flat = vec(scores)
    idx = partialsortperm(scores_flat, 1:k2, rev=true)

    # Create mask on CPU
    mask_cpu = zeros(real(T), m, n)
    for flat_idx in idx
        mask_cpu[flat_idx] = one(real(T))
    end

    # Convert mask to GPU
    mask_gpu = CUDA.cu(mask_cpu)

    # Apply mask on GPU (element-wise, no scalar indexing)
    y = x .* mask_gpu

    function topk_truncate_pullback(ȳ)
        # Pullback: gradient flows through the kept elements
        # ȳ is on GPU, mask_gpu is on GPU - element-wise multiply
        x̄ = ȳ .* mask_gpu
        return (ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent())
    end

    return y, topk_truncate_pullback
end

end # module
