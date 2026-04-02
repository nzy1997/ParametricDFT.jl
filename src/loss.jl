# ============================================================================
# Loss Functions for Sparse Basis Training
# ============================================================================
# This file provides loss functions for training parametric DFT circuits.

"""Abstract base type for loss functions. Subtypes implement `_loss_function`."""
abstract type AbstractLoss end

"""L1 norm loss: minimizes `sum(|T(x)|)` to encourage sparsity."""
struct L1Norm <: AbstractLoss end

"""
    L2Norm

L2 norm loss: `sum(|T(x)|^2)`.

!!! warning
    Unitary transforms preserve the L2 norm (Parseval's theorem), so this loss
    is constant with respect to the circuit parameters and its gradient is zero.
    It should NOT be used as a training objective. Use `L1Norm` or `MSELoss` instead.
    This type is retained for backward compatibility but may be removed in a future release.
"""
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

Magnitude-based top-k truncation: keeps the `k` coefficients with largest absolute value,
zeroing the rest. This is basis-agnostic — it does not assume any particular frequency layout.
"""
function topk_truncate(x::AbstractMatrix{T}, k::Integer) where {T}
    k2 = min(Int(k), length(x))
    k2 == length(x) && return copy(x)
    return x .* _topk_mask(x, k2)
end

"""
    _topk_mask(x::AbstractMatrix, k::Int) -> BitMatrix

Compute a boolean mask selecting the `k` elements of `x` with largest absolute value.
Uses quickselect (O(n) average) to find the threshold, then a broadcast comparison.
"""
function _topk_mask(x::AbstractMatrix, k::Int)
    magnitudes = abs.(x)
    # Quickselect to find the k-th largest magnitude — O(n) average, no perm allocation
    threshold = partialsort!(vec(copy(magnitudes)), k, rev=true)

    # Elements strictly above threshold are definitely kept
    mask = magnitudes .> threshold
    n_kept = count(mask)
    n_ties_needed = k - n_kept

    # Fill in ties at the threshold boundary until exactly k are selected
    if n_ties_needed > 0
        added = 0
        @inbounds for j in 1:size(x, 2), i in 1:size(x, 1)
            if !mask[i, j] && magnitudes[i, j] == threshold
                mask[i, j] = true
                added += 1
                added >= n_ties_needed && @goto done
            end
        end
        @label done
    end
    return mask
end

function ChainRulesCore.rrule(::typeof(topk_truncate), x::AbstractMatrix{T}, k::Integer) where {T}
    k2 = min(Int(k), length(x))
    if k2 == length(x)
        pullback_all(ȳ) = (ChainRulesCore.NoTangent(), ȳ, ChainRulesCore.NoTangent())
        return copy(x), pullback_all
    end

    # Compute mask once, reuse in both forward and pullback
    mask = _topk_mask(x, k2)
    y = x .* mask

    function pullback(ȳ)
        return (ChainRulesCore.NoTangent(), ȳ .* mask, ChainRulesCore.NoTangent())
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

"""Batched MSE loss: batched forward, per-image topk_truncate, batched inverse."""
function batched_loss_mse(optcode_batched, tensors::Tuple, batch::Vector{<:AbstractMatrix}, m::Int, n::Int, k::Int, inverse_code;
                          batched_inverse_code=nothing)
    B = length(batch)
    qubit_dims = fill(2, m + n)

    # Batched forward pass — single einsum call
    fft_batched = batched_forward(optcode_batched, tensors, batch, m, n)

    # Per-image truncation (mask is content-dependent, cannot be batched)
    # Use map instead of mutation to keep Zygote happy
    truncated_slices = map(1:B) do i
        fft_slice = reshape(selectdim(fft_batched, m + n + 1, i), 2^m, 2^n)
        reshape(topk_truncate(fft_slice, k), qubit_dims...)
    end

    # Batched inverse pass
    conj_tensors = conj.(tensors)
    if batched_inverse_code !== nothing
        # Single einsum call for all B inverse transforms
        stacked_trunc = cat(truncated_slices...; dims=m + n + 1)
        inv_batched = batched_inverse_code(conj_tensors..., stacked_trunc)
        total_loss = zero(real(eltype(fft_batched)))
        for i in 1:B
            reconstructed = reshape(selectdim(inv_batched, m + n + 1, i), 2^m, 2^n)
            total_loss += sum(abs2.(batch[i] .- reconstructed))
        end
    else
        # Fallback: per-image inverse
        total_loss = zero(real(eltype(fft_batched)))
        for i in 1:B
            reconstructed = reshape(
                inverse_code(conj_tensors..., truncated_slices[i]),
                2^m, 2^n
            )
            total_loss += sum(abs2.(batch[i] .- reconstructed))
        end
    end
    return total_loss / B
end

# Vector→Tuple wrapper methods to avoid Zygote vector-vs-tuple tangent mismatch
# when splatting. Same pattern as loss_function.
batched_loss_l1(oc, ts::AbstractVector, b, m, n) = batched_loss_l1(oc, Tuple(ts), b, m, n)
batched_loss_l2(oc, ts::AbstractVector, b, m, n) = batched_loss_l2(oc, Tuple(ts), b, m, n)
batched_loss_mse(oc, ts::AbstractVector, b, m, n, k, ic; kw...) = batched_loss_mse(oc, Tuple(ts), b, m, n, k, ic; kw...)

# ============================================================================
# Unified Batch Loss Interface
# ============================================================================

"""
    loss_function(tensors, m, n, optcode, pics::Vector{<:AbstractMatrix}, loss; inverse_code=nothing, batched_optcode=nothing)

Average loss over a batch of images. Uses batched einsum if `batched_optcode` is provided.
"""
function loss_function(tensors::AbstractVector, m::Int, n::Int, optcode::OMEinsum.AbstractEinsum,
                       pics::Vector{<:AbstractMatrix}, loss::AbstractLoss;
                       inverse_code=nothing, batched_optcode=nothing, batched_inverse_code=nothing)
    return loss_function(Tuple(tensors), m, n, optcode, pics, loss;
                         inverse_code=inverse_code, batched_optcode=batched_optcode,
                         batched_inverse_code=batched_inverse_code)
end

function loss_function(tensors::Tuple, m::Int, n::Int, optcode::OMEinsum.AbstractEinsum,
                       pics::Vector{<:AbstractMatrix}, loss::AbstractLoss;
                       inverse_code=nothing, batched_optcode=nothing, batched_inverse_code=nothing)
    if batched_optcode !== nothing
        if loss isa L1Norm
            return batched_loss_l1(batched_optcode, tensors, pics, m, n)
        elseif loss isa L2Norm
            return batched_loss_l2(batched_optcode, tensors, pics, m, n)
        else  # MSELoss
            return batched_loss_mse(batched_optcode, tensors, pics, m, n, loss.k, inverse_code;
                                    batched_inverse_code=batched_inverse_code)
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
