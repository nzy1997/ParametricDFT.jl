# ============================================================================
# Loss Functions for Sparse Basis Training
# ============================================================================
# This file provides loss functions for training parametric DFT circuits.

"""Abstract base type for loss functions. Subtypes implement `_loss_function`."""
abstract type AbstractLoss end

"""L1 norm loss: minimizes `sum(|T(x)|)` to encourage sparsity."""
struct L1Norm <: AbstractLoss end

"""L2 norm loss: minimizes `sum(|T(x)|^2)` for energy concentration."""
struct L2Norm <: AbstractLoss end

"""MSE loss with top-k truncation: `||x - T⁻¹(truncate(T(x), k))||²`. Field `k` is the number of kept coefficients."""
struct MSELoss <: AbstractLoss
    k::Int
    MSELoss(k::Int) = k > 0 ? new(k) : error("k must be positive")
end

# ============================================================================
# Truncation Functions
# ============================================================================

"""
    topk_truncate(x::AbstractMatrix, k::Integer)

Frequency-weighted top-k truncation: keeps `k` coefficients biased toward low frequencies.
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
    loss_function(tensors, m, n, optcode, pic::AbstractMatrix, loss; inverse_code=nothing)

Compute loss for a single image `pic` (2^m x 2^n) under the given circuit parameters.
"""
function loss_function(tensors::AbstractVector, m::Int, n::Int, optcode::OMEinsum.AbstractEinsum, pic::AbstractMatrix, loss::AbstractLoss; inverse_code=nothing)
    # Avoid splatting an AbstractVector during AD; Zygote may produce tuple tangents for varargs.
    # We delegate to the Tuple method for a stable tangent type.
    return loss_function(Tuple(tensors), m, n, optcode, pic, loss; inverse_code=inverse_code)
end

function loss_function(tensors::Tuple, m::Int, n::Int, optcode::OMEinsum.AbstractEinsum, pic::AbstractMatrix, loss::AbstractLoss; inverse_code=nothing)
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

# ============================================================================
# Batched Einsum Operations
# ============================================================================
# Batched einsum: extend OMEinsum contraction indices with a batch label so B images
# are processed in a single kernel call. Each image index (input/output) gets the batch
# label appended; the contraction order is then re-optimized for the batched tensor sizes.

using OMEinsum: getixsv, getiyv, uniquelabels

"""Add a batch label to image input/output indices. Returns `(batched_flat_code, batch_label)`."""
function make_batched_code(optcode, n_gates::Int)
    flat = OMEinsum.flatten(optcode)
    ixs = getixsv(flat)
    iy = getiyv(flat)

    # Batch label: one beyond all existing labels
    all_labels = uniquelabels(flat)
    batch_label = maximum(all_labels) + 1

    # Build new index lists: append batch_label to image input and output only
    new_ixs = Vector{Vector{Int}}(undef, length(ixs))
    for i in 1:n_gates
        new_ixs[i] = copy(ixs[i])  # gate indices unchanged
    end
    # Image input is the last entry (position n_gates + 1)
    new_ixs[n_gates + 1] = vcat(ixs[n_gates + 1], [batch_label])
    # Output gets batch dimension too
    new_iy = vcat(iy, [batch_label])

    batched_flat = DynamicEinCode(new_ixs, new_iy)
    return batched_flat, batch_label
end

"""Optimize contraction order for batched einsum. `batch_size` guides optimization
but result works at any runtime batch size."""
function optimize_batched_code(batched_flat, batch_label, batch_size::Int)
    all_labels = uniquelabels(batched_flat)
    size_dict = Dict{Int, Int}()
    for label in all_labels
        size_dict[label] = (label == batch_label) ? batch_size : 2
    end
    return optimize_code_cached(batched_flat, size_dict, TreeSA())
end

"""Apply circuit to B images in a single einsum call. Returns (2,...,2,B) tensor."""
function batched_forward(optcode_batched, tensors::Tuple, batch::Vector{<:AbstractMatrix}, m::Int, n::Int)
    qubit_dims = fill(2, m + n)
    # Stack B images into a single (2,2,...,2, B) tensor
    stacked = cat([reshape(img, qubit_dims...) for img in batch]...; dims=m + n + 1)
    return optcode_batched(tensors..., stacked)
end

"""Batched L1 loss: (1/B) * sum(|forward(images)|)."""
function batched_loss_l1(optcode_batched, tensors::Tuple, batch::Vector{<:AbstractMatrix}, m::Int, n::Int)
    result = batched_forward(optcode_batched, tensors, batch, m, n)
    return sum(abs.(result)) / length(batch)
end

"""Batched L2 loss: (1/B) * sum(|forward(images)|^2)."""
function batched_loss_l2(optcode_batched, tensors::Tuple, batch::Vector{<:AbstractMatrix}, m::Int, n::Int)
    result = batched_forward(optcode_batched, tensors, batch, m, n)
    return sum(abs2.(result)) / length(batch)
end

"""Batched MSE loss: batched forward, per-image topk_truncate + inverse."""
function batched_loss_mse(optcode_batched, tensors::Tuple, batch::Vector{<:AbstractMatrix}, m::Int, n::Int, k::Int, inverse_code)
    B = length(batch)
    qubit_dims = fill(2, m + n)

    # Batched forward pass — single einsum call
    fft_batched = batched_forward(optcode_batched, tensors, batch, m, n)

    # Per-image truncation + inverse (truncation mask is content-dependent)
    total_loss = zero(real(eltype(fft_batched)))
    conj_tensors = conj.(tensors)
    for i in 1:B
        # Extract per-image result and reshape to matrix for truncation
        fft_slice = reshape(selectdim(fft_batched, m + n + 1, i), 2^m, 2^n)
        fft_truncated = topk_truncate(fft_slice, k)
        # Inverse transform
        reconstructed = reshape(
            inverse_code(conj_tensors..., reshape(fft_truncated, qubit_dims...)),
            2^m, 2^n
        )
        total_loss += sum(abs2.(batch[i] .- reconstructed))
    end
    return total_loss / B
end

# Vector→Tuple wrapper methods to avoid Zygote vector-vs-tuple tangent mismatch
# when splatting. Same pattern as loss_function.
batched_loss_l1(oc, ts::AbstractVector, b, m, n) = batched_loss_l1(oc, Tuple(ts), b, m, n)
batched_loss_l2(oc, ts::AbstractVector, b, m, n) = batched_loss_l2(oc, Tuple(ts), b, m, n)
batched_loss_mse(oc, ts::AbstractVector, b, m, n, k, ic) = batched_loss_mse(oc, Tuple(ts), b, m, n, k, ic)

# ============================================================================
# Unified Batch Loss Interface
# ============================================================================

"""
    loss_function(tensors, m, n, optcode, pics::Vector{<:AbstractMatrix}, loss; inverse_code=nothing, batched_optcode=nothing)

Average loss over a batch of images. Uses batched einsum if `batched_optcode` is provided.
"""
function loss_function(tensors::AbstractVector, m::Int, n::Int, optcode::OMEinsum.AbstractEinsum,
                       pics::Vector{<:AbstractMatrix}, loss::AbstractLoss;
                       inverse_code=nothing, batched_optcode=nothing)
    return loss_function(Tuple(tensors), m, n, optcode, pics, loss;
                         inverse_code=inverse_code, batched_optcode=batched_optcode)
end

function loss_function(tensors::Tuple, m::Int, n::Int, optcode::OMEinsum.AbstractEinsum,
                       pics::Vector{<:AbstractMatrix}, loss::AbstractLoss;
                       inverse_code=nothing, batched_optcode=nothing)
    if batched_optcode !== nothing
        if loss isa L1Norm
            return batched_loss_l1(batched_optcode, tensors, pics, m, n)
        elseif loss isa L2Norm
            return batched_loss_l2(batched_optcode, tensors, pics, m, n)
        else  # MSELoss
            return batched_loss_mse(batched_optcode, tensors, pics, m, n, loss.k, inverse_code)
        end
    else
        n_imgs = length(pics)
        total = zero(real(eltype(tensors[1])))
        for img in pics
            total += loss_function(tensors, m, n, optcode, img, loss; inverse_code=inverse_code)
        end
        return total / n_imgs
    end
end
