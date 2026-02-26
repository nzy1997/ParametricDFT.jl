# Batched einsum: add a batch dimension to einsum contractions so B images
# are processed in a single kernel call (B×K launches → K launches).

using OMEinsum: getixsv, getiyv, uniquelabels

"""Add batch dimension to image input/output indices of a flattened einsum code.
Returns `(batched_flat_code, batch_label)`."""
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
    return optimize_code(batched_flat, size_dict, TreeSA())
end

"""Apply circuit to B images in a single einsum call. Returns (2,...,2,B) tensor."""
function batched_forward(optcode_batched, tensors, batch::Vector{<:AbstractMatrix}, m::Int, n::Int)
    qubit_dims = fill(2, m + n)
    # Stack B images into a single (2,2,...,2, B) tensor
    stacked = cat([reshape(img, qubit_dims...) for img in batch]...; dims=m + n + 1)
    return optcode_batched(tensors..., stacked)
end

"""Batched L1 loss: (1/B) * sum(|forward(images)|)."""
function batched_loss_l1(optcode_batched, tensors, batch::Vector{<:AbstractMatrix}, m::Int, n::Int)
    result = batched_forward(optcode_batched, tensors, batch, m, n)
    return sum(abs.(result)) / length(batch)
end

"""Batched L2 loss: (1/B) * sum(|forward(images)|^2)."""
function batched_loss_l2(optcode_batched, tensors, batch::Vector{<:AbstractMatrix}, m::Int, n::Int)
    result = batched_forward(optcode_batched, tensors, batch, m, n)
    return sum(abs2.(result)) / length(batch)
end

"""Batched MSE loss: batched forward, per-image topk_truncate + inverse."""
function batched_loss_mse(optcode_batched, tensors, batch::Vector{<:AbstractMatrix}, m::Int, n::Int, k::Int, inverse_code)
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
