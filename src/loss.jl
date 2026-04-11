# ============================================================================
# Loss Functions for Sparse Basis Training
# ============================================================================
# This file provides loss functions for training parametric DFT circuits.

"""Abstract base type for loss functions. Subtypes implement `_loss_function`."""
abstract type AbstractLoss end

"""L1 norm loss: minimizes `sum(|T(x)|)` to encourage sparsity."""
struct L1Norm <: AbstractLoss end

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
# Tuple/Vector Normalization
# ============================================================================

"""Convert AbstractVector tensors to Tuple for stable Zygote AD tangent types."""
_ensure_tuple(t::Tuple) = t
_ensure_tuple(t::AbstractVector) = Tuple(t)

# ============================================================================
# Loss Function Computation
# ============================================================================

"""
    loss_function(tensors, m, n, optcode, pic::AbstractMatrix, loss; inverse_code=nothing)

Compute loss for a single image `pic` (2^m x 2^n) under the given circuit parameters.
"""
function loss_function(tensors, m::Int, n::Int, optcode::OMEinsum.AbstractEinsum, pic::AbstractMatrix, loss::AbstractLoss; inverse_code=nothing)
    ts = _ensure_tuple(tensors)
    @assert (size(pic) == (2^m, 2^n)) "Input matrix size must be 2^m × 2^n"
    fft_pic = reshape(optcode(ts..., reshape(pic, fill(2, m+n)...)), 2^m, 2^n)
    return _loss_function(fft_pic, pic, loss, ts, m, n, inverse_code)
end

# Compute L1 norm: sum of absolute values
function _loss_function(fft_res, pic, loss::L1Norm, tensors, m, n, inverse_code)
    return sum(abs.(fft_res))
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

"""Stack B images into a single `(2, ..., 2, B)` tensor for batched einsum."""
function stack_image_batch(batch::Vector{<:AbstractMatrix}, m::Int, n::Int)
    qubit_dims = fill(2, m + n)
    return cat([reshape(img, qubit_dims...) for img in batch]...; dims=m + n + 1)
end

"""Apply circuit to a pre-stacked image batch. Returns (2,...,2,B) tensor."""
function batched_forward(optcode_batched, tensors, stacked_batch::AbstractArray)
    return optcode_batched(_ensure_tuple(tensors)..., stacked_batch)
end

"""Apply circuit to B images in a single einsum call. Returns (2,...,2,B) tensor."""
function batched_forward(optcode_batched, tensors, batch::Vector{<:AbstractMatrix}, m::Int, n::Int)
    return batched_forward(optcode_batched, tensors, stack_image_batch(batch, m, n))
end

"""Batched L1 loss: (1/B) * sum(|forward(images)|)."""
function batched_loss_l1(optcode_batched, tensors, stacked_batch::AbstractArray)
    result = batched_forward(optcode_batched, tensors, stacked_batch)
    return sum(abs.(result)) / size(stacked_batch, ndims(stacked_batch))
end

function batched_loss_l1(optcode_batched, tensors, batch::Vector{<:AbstractMatrix}, m::Int, n::Int)
    return batched_loss_l1(optcode_batched, tensors, stack_image_batch(batch, m, n))
end

"""Batched MSE loss: batched forward, per-image topk_truncate, batched inverse."""
function batched_loss_mse(optcode_batched, tensors, batch_data, m::Int, n::Int, k::Int, inverse_code;
                          batched_inverse_code=nothing)
    ts = _ensure_tuple(tensors)
    stacked_batch = batch_data isa AbstractVector{<:AbstractMatrix} ? stack_image_batch(batch_data, m, n) : batch_data
    B = size(stacked_batch, ndims(stacked_batch))
    qubit_dims = fill(2, m + n)

    # Batched forward pass — single einsum call
    fft_batched = batched_forward(optcode_batched, ts, stacked_batch)

    # Per-image truncation (mask is content-dependent, cannot be batched)
    # Use map instead of mutation to keep Zygote happy
    truncated_slices = map(1:B) do i
        fft_slice = reshape(selectdim(fft_batched, m + n + 1, i), 2^m, 2^n)
        reshape(topk_truncate(fft_slice, k), qubit_dims...)
    end

    # Batched inverse pass
    conj_tensors = conj.(ts)
    if batched_inverse_code !== nothing
        # Single einsum call for all B inverse transforms
        stacked_trunc = cat(truncated_slices...; dims=m + n + 1)
        inv_batched = batched_inverse_code(conj_tensors..., stacked_trunc)
        total_loss = sum(abs2.(stacked_batch .- inv_batched))
    else
        # Fallback: per-image inverse
        total_loss = zero(real(eltype(fft_batched)))
        for i in 1:B
            reconstructed = reshape(
                inverse_code(conj_tensors..., truncated_slices[i]),
                2^m, 2^n
            )
            pic = reshape(selectdim(stacked_batch, m + n + 1, i), 2^m, 2^n)
            total_loss += sum(abs2.(pic .- reconstructed))
        end
    end
    return total_loss / B
end

# ============================================================================
# Unified Batch Loss Interface
# ============================================================================

"""
    loss_function(tensors, m, n, optcode, pics::Vector{<:AbstractMatrix}, loss; inverse_code=nothing, batched_optcode=nothing)

Average loss over a batch of images. Uses batched einsum if `batched_optcode` is provided.
"""
function loss_function(tensors, m::Int, n::Int, optcode::OMEinsum.AbstractEinsum,
                       pics::Vector{<:AbstractMatrix}, loss::AbstractLoss;
                       inverse_code=nothing, batched_optcode=nothing, batched_inverse_code=nothing)
    ts = _ensure_tuple(tensors)
    if batched_optcode !== nothing
        stacked = stack_image_batch(pics, m, n)
        return _loss_function_batched(ts, m, n, stacked, loss, inverse_code, batched_optcode, batched_inverse_code)
    else
        n_imgs = length(pics)
        total = zero(real(eltype(ts[1])))
        for img in pics
            total += loss_function(ts, m, n, optcode, img, loss; inverse_code=inverse_code)
        end
        return total / n_imgs
    end
end

function loss_function(tensors, m::Int, n::Int, optcode::OMEinsum.AbstractEinsum,
                       stacked_pics::AbstractArray, loss::AbstractLoss;
                       inverse_code=nothing, batched_optcode=nothing, batched_inverse_code=nothing)
    ts = _ensure_tuple(tensors)
    batched_optcode !== nothing || error("A pre-stacked batch requires batched_optcode")
    return _loss_function_batched(ts, m, n, stacked_pics, loss, inverse_code, batched_optcode, batched_inverse_code)
end

"""Dispatch batched loss computation to the appropriate loss-specific function."""
function _loss_function_batched(tensors::Tuple, m, n, stacked_pics, loss, inverse_code, batched_optcode, batched_inverse_code)
    if loss isa L1Norm
        return batched_loss_l1(batched_optcode, tensors, stacked_pics)
    elseif loss isa MSELoss
        return batched_loss_mse(batched_optcode, tensors, stacked_pics, m, n, loss.k, inverse_code;
                                batched_inverse_code=batched_inverse_code)
    else
        error("unsupported loss type for batched dispatch: $(typeof(loss))")
    end
end
