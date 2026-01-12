# ============================================================================
# Loss Functions for Sparse Basis Training
# ============================================================================
# This file provides loss functions for training parametric DFT circuits.

"""
    AbstractLoss

Abstract base type for loss functions. Custom loss functions should inherit from 
this type and implement `_loss_function(fft_result, input, loss)`.
"""
abstract type AbstractLoss end

"""
    L1Norm <: AbstractLoss

L1 norm loss: minimizes sum of absolute values in the transformed domain.
This encourages sparsity in the frequency representation.
"""
struct L1Norm <: AbstractLoss end

"""
    L2Norm <: AbstractLoss

L2 norm loss: minimizes sum of squared magnitudes in the transformed domain.
This encourages energy concentration (squared magnitude) with smoother gradients
compared to L1 norm, and less aggressive sparsity promotion than L1.
"""
struct L2Norm <: AbstractLoss end

"""
    MSELoss <: AbstractLoss

Mean Squared Error loss with truncation: minimizes reconstruction error after
forward transform, truncation (keeping top k elements), and inverse transform.

# Fields
- `k::Int`: Number of top elements to keep after truncation (by magnitude)

# Equation
L(θ) = Σᵢ ||xᵢ - T(θ)⁻¹(truncate(T(θ)(xᵢ), k))||²₂

This loss encourages the circuit to learn a representation where the top k
frequency components capture most of the signal information.
"""
struct MSELoss <: AbstractLoss
    k::Int
    MSELoss(k::Int) = k > 0 ? new(k) : error("k must be positive")
end

# ============================================================================
# Truncation Functions
# ============================================================================

"""
    topk_truncate(x::AbstractMatrix, k::Integer)

Return a matrix where frequency-dependent truncation is applied for image compression.
Low-frequency components (near center) are kept with higher priority, while high-frequency
components (away from center) are kept with lower priority.

This is more appropriate for image compression than global top-k selection, as it
preserves more low-frequency information which contains most of the image structure.
"""
function topk_truncate(x::AbstractMatrix{T}, k::Integer) where {T}
    m, n = size(x)
    k2 = min(Int(k), length(x))
    
    # Calculate frequency distances from center (DC component)
    center_i, center_j = (m + 1) ÷ 2, (n + 1) ÷ 2
    max_dist = sqrt((m/2)^2 + (n/2)^2)
    
    # Create frequency-weighted scores: combine magnitude with frequency position
    # Lower frequency (closer to center) gets higher weight
    scores = zeros(Float64, m, n)
    mags = abs.(x)
    
    @inbounds for j in 1:n, i in 1:m
        freq_dist = sqrt((i - center_i)^2 + (j - center_j)^2)
        # Weight: higher for low frequencies (smaller distance)
        # Use inverse distance weighting, normalized by max distance
        freq_weight = 1.0 - (freq_dist / max_dist) * 0.5  # Scale factor to balance magnitude vs frequency
        scores[i, j] = mags[i, j] * (1.0 + freq_weight)
    end
    
    # Select top k based on weighted scores
    scores_flat = vec(scores)
    idx = partialsortperm(scores_flat, 1:k2, rev=true)
    
    y = zeros(T, m, n)
    @inbounds for flat_idx in idx
        # Julia arrays are column-major: flat_idx = (j-1)*m + i
        i = ((flat_idx - 1) % m) + 1
        j = ((flat_idx - 1) ÷ m) + 1
        y[i, j] = x[i, j]
    end
    return y
end

function ChainRulesCore.rrule(::typeof(topk_truncate), x::AbstractMatrix{T}, k::Integer) where {T}
    m, n = size(x)
    k2 = min(Int(k), length(x))
    
    # Calculate frequency distances from center
    center_i, center_j = (m + 1) ÷ 2, (n + 1) ÷ 2
    max_dist = sqrt((m/2)^2 + (n/2)^2)
    
    # Create frequency-weighted scores
    scores = zeros(Float64, m, n)
    mags = abs.(x)
    
    @inbounds for j in 1:n, i in 1:m
        freq_dist = sqrt((i - center_i)^2 + (j - center_j)^2)
        freq_weight = 1.0 - (freq_dist / max_dist) * 0.5
        scores[i, j] = mags[i, j] * (1.0 + freq_weight)
    end
    
    # Select top k based on weighted scores
    scores_flat = vec(scores)
    idx = partialsortperm(scores_flat, 1:k2, rev=true)
    
    y = zeros(T, m, n)
    @inbounds for flat_idx in idx
        # Julia arrays are column-major: flat_idx = (j-1)*m + i
        i = ((flat_idx - 1) % m) + 1
        j = ((flat_idx - 1) ÷ m) + 1
        y[i, j] = x[i, j]
    end
    
    function pullback(ȳ)
        x̄ = zeros(T, m, n)
        @inbounds for flat_idx in idx
            # Julia arrays are column-major: flat_idx = (j-1)*m + i
            i = ((flat_idx - 1) % m) + 1
            j = ((flat_idx - 1) ÷ m) + 1
            x̄[i, j] = ȳ[i, j]
        end
        return (ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent())
    end
    return y, pullback
end

# ============================================================================
# Loss Function Computation
# ============================================================================

"""
    loss_function(tensors, m::Int, n::Int, optcode::AbstractEinsum, pic::Matrix, loss::AbstractLoss; inverse_code=nothing)

Compute the loss for current circuit parameters.

# Arguments
- `tensors::Vector`: Circuit parameters (unitary matrices)
- `m::Int`: Number of qubits for row dimension
- `n::Int`: Number of qubits for column dimension
- `optcode::AbstractEinsum`: Optimized einsum code (forward transform)
- `pic::Matrix`: Input signal (size must be 2^m × 2^n)
- `loss::AbstractLoss`: Loss function type
- `inverse_code::AbstractEinsum`: Optional inverse einsum code (required for MSELoss)

# Returns
- Loss value
"""
function loss_function(tensors::AbstractVector, m::Int, n::Int, optcode::OMEinsum.AbstractEinsum, pic::Matrix, loss::AbstractLoss; inverse_code=nothing)
    # Avoid splatting an AbstractVector during AD; Zygote may produce tuple tangents for varargs.
    # We delegate to the Tuple method for a stable tangent type.
    return loss_function(Tuple(tensors), m, n, optcode, pic, loss; inverse_code=inverse_code)
end

function loss_function(tensors::Tuple, m::Int, n::Int, optcode::OMEinsum.AbstractEinsum, pic::Matrix, loss::AbstractLoss; inverse_code=nothing)
    @assert (size(pic) == (2^m, 2^n)) "Input matrix size must be 2^m × 2^n"
    fft_pic = reshape(optcode(tensors..., reshape(pic, fill(2, m+n)...)), 2^m, 2^n)
    return _loss_function(fft_pic, pic, loss, tensors, m, n, inverse_code)
end

# Compute L1 norm: sum of absolute values
function _loss_function(fft_res, pic, loss::L1Norm, tensors, m, n, inverse_code)
    return sum(abs.(fft_res))
end

# Compute L2 norm: sum of squared magnitudes
function _loss_function(fft_res, pic, loss::L2Norm, tensors, m, n, inverse_code)
    return sum(abs2.(fft_res))
end

# Compute MSE with truncation: ||x - T⁻¹(truncate(T(x), k))||²₂
function _loss_function(fft_res, pic, loss::MSELoss, tensors, m, n, inverse_code)
    if inverse_code === nothing
        error("MSELoss requires inverse_code to be provided")
    end
    
    # Truncate: use frequency-dependent truncation for image compression
    # This keeps more low-frequency components and fewer high-frequency components
    k = min(loss.k, length(fft_res))
    fft_truncated = topk_truncate(fft_res, k)
    
    # Apply inverse transform
    reconstructed = reshape(inverse_code(conj.(tensors)..., reshape(fft_truncated, fill(2, m+n)...)), 2^m, 2^n)
    
    # Compute MSE: sum of squared differences
    return sum(abs2.(pic .- reconstructed))
end

