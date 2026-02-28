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

Project Euclidean gradients onto the tangent space at `points`.
Operates on batched arrays of shape `(d1, d2, ..., n)` where `n` is batch size.
"""
function project end

"""
    retract(m::AbstractRiemannianManifold, points, tangent_vec, α)

Retract from `points` along `tangent_vec` with step size `α`, returning
a new point on the manifold. Operates on batched arrays.
"""
function retract end

"""
    transport(m::AbstractRiemannianManifold, old_points, new_points, vec)

Parallel transport `vec` from `old_points` to `new_points`.
Operates on batched arrays.
"""
function transport end

# ============================================================================
# Generalized Batched Linear Algebra
# ============================================================================

"""
    batched_matmul(A::AbstractArray{T,3}, B::AbstractArray{T,3}) where T

Batched matrix multiply: `C[:,:,k] = A[:,:,k] * B[:,:,k]` for each slice `k`.
Works for arbitrary `(d1,d2,n) x (d2,d3,n) -> (d1,d3,n)`.
"""
function batched_matmul(A::AbstractArray{T,3}, B::AbstractArray{T,3}) where T
    d1, d2A, n = size(A)
    d2B, d3, n2 = size(B)
    @assert d2A == d2B "Inner dimensions must match: got $d2A and $d2B"
    @assert n == n2 "Batch sizes must match: got $n and $n2"
    C = similar(A, T, d1, d3, n)
    @inbounds for k in 1:n
        for j in 1:d3, i in 1:d1
            s = zero(T)
            for p in 1:d2A
                s += A[i, p, k] * B[p, j, k]
            end
            C[i, j, k] = s
        end
    end
    return C
end

"""
    batched_adjoint(A::AbstractArray{T,3}) where T

Batched conjugate transpose: `C[:,:,k] = A[:,:,k]'` for each slice `k`.
Works for arbitrary `(d1,d2,n) -> (d2,d1,n)`.
"""
function batched_adjoint(A::AbstractArray{T,3}) where T
    d1, d2, n = size(A)
    C = similar(A, T, d2, d1, n)
    @inbounds for k in 1:n, j in 1:d1, i in 1:d2
        C[i, j, k] = conj(A[j, i, k])
    end
    return C
end

# ============================================================================
# Manifold Classification Utilities
# ============================================================================

"""
    is_unitary_general(t::AbstractMatrix{T}) where T

Check if a square matrix satisfies `U*U' ~ I` (unitary manifold).
Returns `false` for non-square matrices.
Generalized version that works for any dimension (not just 2x2).
"""
function is_unitary_general(t::AbstractMatrix{T}) where T
    n = size(t, 1)
    size(t, 2) == n || return false
    return isapprox(t * t', Matrix{T}(I, n, n), atol=1e-6)
end

"""
    classify_manifold(t::AbstractMatrix) -> AbstractRiemannianManifold

Classify a single tensor into its manifold type based on unitarity check.
Must be called on CPU arrays (use `Array(t)` for GPU tensors).
"""
function classify_manifold end  # forward declaration

"""
    group_by_manifold(tensors::Vector{<:AbstractMatrix})

Group tensor indices by manifold type.
Returns `Dict{AbstractRiemannianManifold, Vector{Int}}`.
Calls `Array(t)` before classifying to handle GPU tensors.
"""
function group_by_manifold(tensors::Vector{<:AbstractMatrix})
    groups = Dict{AbstractRiemannianManifold, Vector{Int}}()
    for (i, t) in enumerate(tensors)
        m = classify_manifold(Array(t))
        if haskey(groups, m)
            push!(groups[m], i)
        else
            groups[m] = [i]
        end
    end
    return groups
end

# ============================================================================
# Batched Tensor Packing/Unpacking
# ============================================================================

"""
    stack_tensors(tensors::Vector{<:AbstractMatrix{T}}, indices::Vector{Int}) where T

Pack selected matrices into a batched `(d1, d2, n)` array.
"""
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

"""Batched QR retraction via modified Gram-Schmidt (pure broadcasting, no cuSOLVER).
Generalizes to arbitrary square matrices -- NOT hardcoded to 2 columns."""
function retract(::UnitaryManifold, U::AbstractArray{T,3}, Xi::AbstractArray{T,3}, α) where T
    RT = real(T)
    α_typed = convert(RT, α)
    d = size(U, 1)

    # Y = U + alpha * Xi
    Y = U .+ α_typed .* Xi

    # Modified Gram-Schmidt column by column
    Q = similar(Y)
    for col in 1:d
        v = Y[:, col:col, :]  # (d, 1, n)
        for prev in 1:(col-1)
            q_prev = Q[:, prev:prev, :]  # (d, 1, n)
            dot_val = sum(conj.(q_prev) .* v; dims=1)  # (1, 1, n)
            v = v .- dot_val .* q_prev
        end
        nrm = sqrt.(sum(abs2.(v); dims=1))
        nrm = max.(nrm, RT(1e-30))  # prevent division by zero
        copyto!(view(Q, :, col:col, :), v ./ nrm)
    end
    return Q
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

"""Classify a single tensor into its manifold type based on unitarity check."""
function classify_manifold(t::AbstractMatrix)
    if is_unitary_general(t)
        return UnitaryManifold()
    else
        return PhaseManifold()
    end
end
