# Riemannian manifold abstraction with batched operations for GPU efficiency.
# Device-agnostic: works on both CPU arrays and GPU CuArrays via AbstractArray.

using LinearAlgebra

# ============================================================================
# Abstract Type and Interface
# ============================================================================

"""Abstract base type for Riemannian manifolds used in circuit optimization."""
abstract type AbstractRiemannianManifold end

"""
    project(m::AbstractRiemannianManifold, points, euclidean_grads)

Project Euclidean gradients onto the tangent space at `points`. Batched over last dim.
"""
function project end

"""
    retract(m::AbstractRiemannianManifold, points, tangent_vec, α)

Retract from `points` along `tangent_vec` with step size `α`. Batched over last dim.
"""
function retract end

"""
    transport(m::AbstractRiemannianManifold, old_points, new_points, vec)

Parallel transport `vec` from `old_points` to `new_points`. Batched over last dim.
"""
function transport end

# ============================================================================
# Generalized Batched Linear Algebra
# ============================================================================

"""
    batched_matmul(A::AbstractArray{T,3}, B::AbstractArray{T,3})

Batched matrix multiply: `C[:,:,k] = A[:,:,k] * B[:,:,k]` for each slice `k`.
"""
function batched_matmul(A::AbstractArray{T,3}, B::AbstractArray{T,3}) where T
    d1, d2A, n = size(A)
    d2B, d3, n2 = size(B)
    @assert d2A == d2B "Inner dimensions must match: got $d2A and $d2B"
    @assert n == n2 "Batch sizes must match: got $n and $n2"
    C = similar(A, T, d1, d3, n)
    for k in 1:n
        mul!(view(C, :, :, k), view(A, :, :, k), view(B, :, :, k))
    end
    return C
end

"""
    batched_adjoint(A::AbstractArray{T,3})

Batched conjugate transpose: `C[:,:,k] = A[:,:,k]'` for each slice `k`.
"""
function batched_adjoint(A::AbstractArray{T,3}) where T
    return permutedims(conj.(A), (2, 1, 3))
end

"""
    batched_inv(A::AbstractArray{T,3})

Batched matrix inverse: `C[:,:,k] = inv(A[:,:,k])` for each slice `k`.
Uses LU factorization for general matrices.
"""
function batched_inv(A::AbstractArray{T,3}) where T
    d1, d2, n = size(A)
    @assert d1 == d2 "Matrix must be square: got $d1 × $d2"
    C = similar(A)
    for k in 1:n
        C[:, :, k] = inv(A[:, :, k])
    end
    return C
end

# ============================================================================
# Manifold Classification Utilities
# ============================================================================

"""
    is_unitary_general(t::AbstractMatrix)

Check if a square matrix satisfies `U*U' ≈ I`. Returns `false` for non-square matrices.
"""
function is_unitary_general(t::AbstractMatrix{T}) where T
    n = size(t, 1)
    size(t, 2) == n || return false
    return isapprox(t * t', I, atol=1e-6)
end

"""Classify a tensor as `UnitaryManifold` or `PhaseManifold` based on unitarity."""
function classify_manifold end  # forward declaration

"""Group tensor indices by manifold type. Returns `Dict{AbstractRiemannianManifold, Vector{Int}}`."""
function group_by_manifold(tensors::Vector{<:AbstractMatrix})
    groups = Dict{AbstractRiemannianManifold, Vector{Int}}()
    for (i, t) in enumerate(tensors)
        push!(get!(groups, classify_manifold(t), Int[]), i)
    end
    return groups
end

# ============================================================================
# Batched Tensor Packing/Unpacking
# ============================================================================

"""Pack selected matrices into a batched `(d1, d2, n)` array."""
function stack_tensors(tensors::Vector{<:AbstractMatrix{T}}, indices::Vector{Int}) where T
    n = length(indices)
    n == 0 && return Array{T}(undef, 0, 0, 0)
    d1, d2 = size(tensors[indices[1]])
    batch = similar(tensors[indices[1]], T, d1, d2, n)
    for (k, idx) in enumerate(indices)
        copyto!(view(batch, :, :, k), tensors[idx])
    end
    return batch
end

"""In-place version: pack into pre-allocated `(d1, d2, n)` array."""
function stack_tensors!(batch::AbstractArray{T,3}, tensors::Vector{<:AbstractMatrix}, indices::Vector{Int}) where T
    for (k, idx) in enumerate(indices)
        copyto!(view(batch, :, :, k), tensors[idx])
    end
    return batch
end

"""Unpack `(d1, d2, n)` array back into individual matrices."""
function unstack_tensors!(tensors::Vector{<:AbstractMatrix}, batch::AbstractArray{T,3}, indices::Vector{Int}) where T
    d1, d2 = size(batch, 1), size(batch, 2)
    for (k, idx) in enumerate(indices)
        if isassigned(tensors, idx) && size(tensors[idx]) == (d1, d2)
            copyto!(tensors[idx], view(batch, :, :, k))
        else
            tensors[idx] = copy(view(batch, :, :, k))
        end
    end
end

# ============================================================================
# UnitaryManifold -- U(n) unitary group
# ============================================================================

"""U(n) unitary group manifold. Tensors are n x n unitary matrices."""
struct UnitaryManifold <: AbstractRiemannianManifold end

"""Batched U(n) Riemannian projection: `U * skew(U'G)` on (d,d,n) arrays."""
function project(::UnitaryManifold, U::AbstractArray{T,3}, G::AbstractArray{T,3}) where T
    UhG = batched_matmul(batched_adjoint(U), G)
    S = (UhG .- batched_adjoint(UhG)) ./ 2  # skew-Hermitian part
    return batched_matmul(U, S)
end

"""Batched Cayley retraction on U(n): `(I - α/2·W)⁻¹(I + α/2·W)·U` where `W = Ξ·U'`."""
function retract(::UnitaryManifold, U::AbstractArray{T,3}, Xi::AbstractArray{T,3}, α) where T
    RT = real(T)
    α_half = convert(RT, α) / 2
    d = size(U, 1)
    n = size(U, 3)

    # W = Xi * U' projected to skew-Hermitian (Lie algebra).
    # The projection ensures correctness even when Xi is not exactly tangent
    # (e.g. Adam's element-wise scaled direction).
    W_raw = batched_matmul(Xi, batched_adjoint(U))
    W = (W_raw .- batched_adjoint(W_raw)) ./ 2

    # Build batched identity
    I_batch = zeros(T, d, d, n)
    for k in 1:n
        for i in 1:d
            I_batch[i, i, k] = one(T)
        end
    end
    I_batch = convert(typeof(U), I_batch)

    lhs = I_batch .- α_half .* W   # I - α/2·W
    rhs = I_batch .+ α_half .* W   # I + α/2·W

    # (I - α/2·W)⁻¹ (I + α/2·W) U
    return batched_matmul(batched_matmul(batched_inv(lhs), rhs), U)
end

"""Batched U(n) parallel transport via re-projection."""
transport(::UnitaryManifold, U_old, U_new, v) = project(UnitaryManifold(), U_new, v)

# ============================================================================
# PhaseManifold -- U(1)^d product of unit complex numbers
# ============================================================================

"""U(1)^d manifold: each element is a unit complex number."""
struct PhaseManifold <: AbstractRiemannianManifold end

"""Batched U(1)^d projection: `im * imag(conj(z).*g) .* z`."""
function project(::PhaseManifold, Z::AbstractArray{T,3}, G::AbstractArray{T,3}) where T
    return T(im) .* imag.(conj.(Z) .* G) .* Z
end

"""Batched U(1)^d retraction: normalize `z + alpha*xi`."""
function retract(::PhaseManifold, Z::AbstractArray{T,3}, Xi::AbstractArray{T,3}, α) where T
    RT = real(T)
    α_typed = convert(RT, α)
    y = Z .+ α_typed .* Xi
    return y ./ T.(abs.(y))
end

"""Batched U(1)^d parallel transport via re-projection."""
transport(::PhaseManifold, Z_old, Z_new, v) = project(PhaseManifold(), Z_new, v)

# ============================================================================
# classify_manifold implementation (defined after concrete types)
# ============================================================================

function classify_manifold(t::AbstractMatrix)
    if is_unitary_general(t)
        return UnitaryManifold()
    else
        return PhaseManifold()
    end
end
