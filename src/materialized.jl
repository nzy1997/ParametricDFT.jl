# ============================================================================
# Materialized Unitary: Build Full Circuit Unitary for Fast Forward Passes
# ============================================================================
# Instead of contracting many tiny gate tensors via einsum (hundreds of GPU
# kernel launches), build the full D×D unitary matrix U once per optimizer step,
# then use U*X for forward passes (single cuBLAS GEMM).

using LinearAlgebra

"""
    build_circuit_unitary(batched_optcode, tensors::Tuple, m::Int, n::Int)

Build the full 2^(m+n) × 2^(m+n) unitary matrix from circuit gate tensors.

Applies the circuit to all D standard basis vectors in a single batched einsum
call. Column j of the result is the circuit applied to basis vector eⱼ.

# Arguments
- `batched_optcode`: Batched einsum code optimized for `batch_size = D`
- `tensors::Tuple`: Circuit gate tensors (2×2 unitary/phase matrices)
- `m::Int`: Number of row qubits
- `n::Int`: Number of column qubits

# Returns
- `Matrix{T}`: Full D×D unitary matrix where D = 2^(m+n)
"""
function build_circuit_unitary(batched_optcode, tensors::Tuple, m::Int, n::Int)
    D = 2^(m + n)
    T = eltype(tensors[1])
    # Identity matrix reshaped as (2,2,...,2, D) tensor = D basis vectors as "batch"
    I_mat = Matrix{T}(I, D, D)
    I_tensor = reshape(I_mat, fill(2, m + n)..., D)
    U_tensor = batched_optcode(tensors..., I_tensor)
    return reshape(U_tensor, D, D)
end

# Vector→Tuple wrapper for AD stability
function build_circuit_unitary(batched_optcode, tensors::AbstractVector, m::Int, n::Int)
    return build_circuit_unitary(batched_optcode, Tuple(tensors), m, n)
end

# ============================================================================
# Materialized Loss Functions
# ============================================================================

"""
    materialized_loss_l1(U, images, m, n)

L1 loss using materialized unitary: (1/B) * sum(|U * X|).
"""
function materialized_loss_l1(U::AbstractMatrix, images::Vector{<:AbstractMatrix}, m::Int, n::Int)
    B = length(images)
    X = _stack_image_columns(images)
    result = U * X
    return sum(abs.(result)) / B
end

"""
    materialized_loss_l2(U, images, m, n)

L2 loss using materialized unitary: (1/B) * sum(|U * X|²).
"""
function materialized_loss_l2(U::AbstractMatrix, images::Vector{<:AbstractMatrix}, m::Int, n::Int)
    B = length(images)
    X = _stack_image_columns(images)
    result = U * X
    return sum(abs2.(result)) / B
end

"""
    materialized_loss_mse(U, images, m, n, k)

MSE loss using materialized unitary: forward with U, truncate, inverse with U'.
"""
function materialized_loss_mse(U::AbstractMatrix, images::Vector{<:AbstractMatrix}, m::Int, n::Int, k::Int)
    B = length(images)

    # Forward: U * X
    X = _stack_image_columns(images)
    fft_batch = U * X  # D × B

    # Per-image truncation + inverse (truncation is content-dependent)
    total_loss = zero(real(eltype(U)))
    U_adj = U'
    for i in 1:B
        fft_col = reshape(fft_batch[:, i], 2^m, 2^n)
        fft_truncated = topk_truncate(fft_col, k)
        reconstructed = U_adj * vec(fft_truncated)
        total_loss += sum(abs2.(vec(images[i]) .- reconstructed))
    end
    return total_loss / B
end

"""Stack images as columns of a D × B matrix (mutation-free for AD)."""
function _stack_image_columns(images::Vector{<:AbstractMatrix})
    return reduce(hcat, vec.(images))
end

# ============================================================================
# Device Strategy Selection
# ============================================================================

"""
    select_device_strategy(m, n, batch_size, device)

Select optimal computation path based on problem size and device.

Returns:
- `:einsum_cpu` — Use standard einsum on CPU (default, unchanged behavior)
- `:materialized_gpu` — Build full unitary, use matmul on GPU (D ≥ 4096)
- `:einsum_gpu` — Use einsum on GPU with targeted fixes (D < 4096)
"""
function select_device_strategy(m::Int, n::Int, batch_size::Int, device::Symbol)
    if device == :cpu
        return :einsum_cpu
    end
    D = 2^(m + n)
    if D >= 4096  # 64×64+ images
        return :materialized_gpu
    else
        return :einsum_gpu
    end
end
