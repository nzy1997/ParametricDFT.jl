# ============================================================================
# GPU-Compatible Riemannian Optimizer for Mixed Manifolds
# ============================================================================
# Custom implementation of Riemannian gradient descent that works with GPU arrays.
# This bypasses Manifolds.jl/Manopt.jl which don't support GPU.
#
# Supports two manifold types:
# 1. U(2) - Unitary matrices (for Hadamard-like gates)
#    - Tangent space at U: T_U = {U*S : S is skew-Hermitian}
#    - Riemannian gradient: proj_U(∇f) = U * skew(U' * ∇f)
#    - Retraction: QR-based (closed-form 2x2 Gram-Schmidt for batched)
#
# 2. U(1)^4 - Product of 4 unit circles (for controlled phase gates)
#    - Tangent space at z: T_z = {iθ*z : θ ∈ ℝ}
#    - Riemannian gradient: imaginary part of conj(z)*g
#    - Retraction: normalization z_new = z / |z|
#
# Performance: batched operations on (2,2,n) arrays reduce CUDA kernel launches
# from ~488/iter (50 tensors × ~10 ops) to ~20/iter (2 groups × ~10 batched ops)
# and eliminate all GPU→CPU transfers (closed-form QR, no cuSOLVER).

using LinearAlgebra

# ============================================================================
# NVTX Profiling Callbacks (set by CUDAExt at init time)
# ============================================================================

const _nvtx_push_fn = Ref{Any}(nothing)
const _nvtx_pop_fn = Ref{Any}(nothing)

"""NVTX range push. No-op unless CUDAExt sets the callback."""
_nvtx_range_push(name::String) = (_nvtx_push_fn[] !== nothing && _nvtx_push_fn[](name); nothing)

"""NVTX range pop. No-op unless CUDAExt sets the callback."""
_nvtx_range_pop() = (_nvtx_pop_fn[] !== nothing && _nvtx_pop_fn[](); nothing)

# ============================================================================
# Manifold Type Detection
# ============================================================================

"""
    is_unitary_tensor(t)

Check if tensor t is a unitary matrix (U(2) manifold).
Returns true for Hadamard-like gates, false for controlled phase gates.

Controlled phase gates have all elements with |z| ≈ 1 but the matrix itself
is not unitary (doesn't satisfy U*U' = I).
"""
function is_unitary_tensor(t::AbstractMatrix{T}) where T
    # Check if it's a 2x2 matrix
    size(t) == (2, 2) || return false
    # Check unitarity: U*U' ≈ I
    UUh = t * t'
    return isapprox(UUh, Matrix{T}(I, 2, 2), atol=1e-6)
end

"""
    classify_tensors_once(tensors)

Classify all tensors into U(2) unitary vs U(1)^4 manifold types ONCE.
Returns `(unitary_idx, u1_idx)` — integer index vectors for each group.

This moves each tensor to CPU exactly once for the unitarity check,
avoiding repeated `Array(t)` calls inside the optimization loop.
"""
function classify_tensors_once(tensors::Vector{<:AbstractMatrix})
    unitary_idx = Int[]
    u1_idx = Int[]
    for (i, t) in enumerate(tensors)
        if is_unitary_tensor(Array(t))
            push!(unitary_idx, i)
        else
            push!(u1_idx, i)
        end
    end
    return unitary_idx, u1_idx
end

# ============================================================================
# Batched Tensor Packing/Unpacking
# ============================================================================

"""
    stack_tensors(tensors, indices)

Pack selected 2×2 matrices into a batched (2,2,n) 3D array.
Uses `copyto!` with views to avoid scalar indexing on GPU.
"""
function stack_tensors(tensors::Vector{<:AbstractMatrix{T}}, indices::Vector{Int}) where T
    n = length(indices)
    n == 0 && return Array{T}(undef, 2, 2, 0)
    # Allocate a 3D array of same type as input tensors
    batch = similar(tensors[indices[1]], T, 2, 2, n)
    for (k, idx) in enumerate(indices)
        copyto!(view(batch, :, :, k), tensors[idx])
    end
    return batch
end

"""
    stack_tensors!(batch, tensors, indices)

Pack selected 2×2 matrices into an existing (2,2,n) 3D array.
Reuses the pre-allocated `batch` to avoid GPU memory allocation.
"""
function stack_tensors!(batch::AbstractArray{T,3}, tensors::Vector{<:AbstractMatrix}, indices::Vector{Int}) where T
    for (k, idx) in enumerate(indices)
        copyto!(view(batch, :, :, k), tensors[idx])
    end
    return batch
end

"""
    unstack_tensors!(tensors, batch, indices)

Unpack a (2,2,n) 3D array back into individual matrices.
If destination tensors already exist (correct size), reuses them via copyto!.
Otherwise allocates new matrices.
"""
function unstack_tensors!(tensors::Vector{<:AbstractMatrix}, batch::AbstractArray{T,3}, indices::Vector{Int}) where T
    for (k, idx) in enumerate(indices)
        if isassigned(tensors, idx) && size(tensors[idx]) == (2, 2)
            copyto!(tensors[idx], view(batch, :, :, k))
        else
            tensors[idx] = copy(view(batch, :, :, k))
        end
    end
end

# ============================================================================
# Batched 2×2 Matrix Operations
# ============================================================================

"""
    batched_matmul_2x2(A, B)

Closed-form batched 2×2 matrix multiply for (2,2,n) arrays.
Uses 4 fused broadcast kernels writing directly to views of a pre-allocated output.
Eliminates the 3 `cat` kernels + temporary allocations of the previous approach.
Result[i,j,:] = sum_k A[i,k,:] .* B[k,j,:]
"""
function batched_matmul_2x2(A::AbstractArray{T,3}, B::AbstractArray{T,3}) where T
    C = similar(A)
    @views begin
        C[1:1, 1:1, :] .= A[1:1, 1:1, :] .* B[1:1, 1:1, :] .+ A[1:1, 2:2, :] .* B[2:2, 1:1, :]
        C[1:1, 2:2, :] .= A[1:1, 1:1, :] .* B[1:1, 2:2, :] .+ A[1:1, 2:2, :] .* B[2:2, 2:2, :]
        C[2:2, 1:1, :] .= A[2:2, 1:1, :] .* B[1:1, 1:1, :] .+ A[2:2, 2:2, :] .* B[2:2, 1:1, :]
        C[2:2, 2:2, :] .= A[2:2, 1:1, :] .* B[1:1, 2:2, :] .+ A[2:2, 2:2, :] .* B[2:2, 2:2, :]
    end
    return C
end

"""
    batched_adjoint_2x2(A)

Batched conjugate transpose for (2,2,n) arrays.
Swaps rows/cols and conjugates. 4 fused broadcast kernels, no `cat` temporaries.
"""
function batched_adjoint_2x2(A::AbstractArray{T,3}) where T
    C = similar(A)
    @views begin
        C[1:1, 1:1, :] .= conj.(A[1:1, 1:1, :])
        C[1:1, 2:2, :] .= conj.(A[2:2, 1:1, :])  # transpose: swap (1,2) ↔ (2,1)
        C[2:2, 1:1, :] .= conj.(A[1:1, 2:2, :])
        C[2:2, 2:2, :] .= conj.(A[2:2, 2:2, :])
    end
    return C
end

# ============================================================================
# Batched Manifold Operations — U(2) Unitary
# ============================================================================

"""
    batched_project_unitary(U, G)

Batched Riemannian gradient projection for U(2):
  proj_U(G) = U * skew(U' * G)  where skew(A) = (A - A')/2

Operates on (2,2,n) batched arrays.
"""
function batched_project_unitary(U::AbstractArray{T,3}, G::AbstractArray{T,3}) where T
    UhG = batched_matmul_2x2(batched_adjoint_2x2(U), G)
    S = (UhG .- batched_adjoint_2x2(UhG)) ./ 2  # skew-Hermitian part
    return batched_matmul_2x2(U, S)
end

"""
    batched_retract_unitary_qr(U, Xi, α)

Closed-form 2×2 Gram-Schmidt QR retraction for batched (2,2,n) arrays.
No cuSOLVER calls, no CPU transfers — pure broadcast operations.

For Y = U + α*Xi, compute Q via modified Gram-Schmidt:
  q1 = Y[:,1] / ||Y[:,1]||
  q2_perp = Y[:,2] - <q1, Y[:,2]> * q1
  q2 = q2_perp / ||q2_perp||
  Q = [q1 q2]

Gram-Schmidt always produces R with positive real diagonal, so no sign
correction is needed (unlike the Householder QR used previously).
"""
function batched_retract_unitary_qr(U::AbstractArray{T,3}, Xi::AbstractArray{T,3}, α) where T
    RT = real(T)
    α_typed = convert(RT, α)

    # Y = U + α * Xi  — shape (2,2,n)
    Y = U .+ α_typed .* Xi

    # Column 1: shape (2,1,n)
    col1 = @view Y[:, 1:1, :]
    norm1 = sqrt.(sum(abs2.(col1); dims=1))  # (1,1,n)
    norm1 = max.(norm1, RT(1e-30))  # prevent division by zero
    q1 = col1 ./ norm1  # (2,1,n)

    # Column 2: orthogonalize against q1
    col2 = @view Y[:, 2:2, :]
    dot12 = sum(conj.(q1) .* col2; dims=1)  # (1,1,n)
    c2_perp = col2 .- dot12 .* q1  # (2,1,n)
    norm2 = sqrt.(sum(abs2.(c2_perp); dims=1))  # (1,1,n)
    norm2 = max.(norm2, RT(1e-30))
    q2 = c2_perp ./ norm2  # (2,1,n)

    # Q = [q1 q2] — shape (2,2,n), write directly to pre-allocated output
    Q = similar(Y)
    copyto!(view(Q, :, 1:1, :), q1)
    copyto!(view(Q, :, 2:2, :), q2)
    return Q
end

"""
    batched_transport_unitary(U_old, U_new, v)

Batched parallel transport on U(2) via re-projection.
"""
batched_transport_unitary(U_old, U_new, v) = batched_project_unitary(U_new, v)

# ============================================================================
# Batched Manifold Operations — U(1)^4
# ============================================================================

"""
    batched_project_u1(Z, G)

Batched Riemannian gradient projection for U(1)^4:
  proj_z(g) = im * imag(conj(z) .* g) .* z

Operates on (2,2,n) batched arrays. Pure broadcast — 1 fused kernel.
"""
function batched_project_u1(Z::AbstractArray{T,3}, G::AbstractArray{T,3}) where T
    return T(im) .* imag.(conj.(Z) .* G) .* Z
end

"""
    batched_retract_u1(Z, Xi, α)

Batched U(1)^4 retraction via normalization for (2,2,n) arrays.
  y = z + α*ξ; y_new = y / |y|

2 broadcast kernels.
"""
function batched_retract_u1(Z::AbstractArray{T,3}, Xi::AbstractArray{T,3}, α) where T
    RT = real(T)
    α_typed = convert(RT, α)
    y = Z .+ α_typed .* Xi
    return y ./ T.(abs.(y))
end

"""
    batched_transport_u1(Z_old, Z_new, v)

Batched parallel transport on U(1)^4 via re-projection.
"""
batched_transport_u1(Z_old, Z_new, v) = batched_project_u1(Z_new, v)

# ============================================================================
# U(1)^4 Manifold Operations (per-tensor, for backward compat)
# ============================================================================

"""
    project_tangent_u1_product(z, g)

Project Euclidean gradient g onto the tangent space of U(1)^4 at z.
For each element z[i], the tangent space is the imaginary axis scaled by z[i].
The projection of g[i] is: z[i] * im * imag(conj(z[i]) * g[i])

This is equivalent to the imaginary part of the product in the Lie algebra.
"""
function project_tangent_u1_product(z::AbstractMatrix{T}, g::AbstractMatrix{T}) where T
    # For each element on U(1), the Riemannian gradient is:
    # proj_{z_i}(g_i) = z_i * im * imag(conj(z_i) * g_i)
    # This simplifies to: im * imag(conj(z) .* g) .* z
    # Use T(im) to ensure type stability
    return T(im) .* imag.(conj.(z) .* g) .* z
end

"""
    retract_u1_product(z, ξ, α)

Retract on U(1)^4 product manifold.
For each element, move in direction ξ and project back to unit circle.
Ensures type stability by converting step size to match element type.
"""
function retract_u1_product(z::AbstractMatrix{T}, ξ::AbstractMatrix{T}, α) where T
    # Ensure α has the correct type to avoid Float32/Float64 mixing
    α_typed = convert(real(T), α)
    y = z .+ α_typed .* ξ
    # Normalize each element to stay on U(1)
    # Use explicit conversion to maintain type
    return y ./ T.(abs.(y))
end

# ============================================================================
# Unitary Manifold Operations (per-tensor, for backward compat)
# ============================================================================

"""
    skew(A)

Compute the skew-Hermitian part of matrix A: (A - A') / 2
"""
skew(A) = (A - A') / 2

"""
    project_tangent_unitary(U, G)

Project Euclidean gradient G onto the tangent space of U(n) at U.
Returns the Riemannian gradient.

For U ∈ U(n), the tangent space is T_U = {U*S : S skew-Hermitian}.
The projection is: proj_U(G) = U * skew(U' * G)
"""
function project_tangent_unitary(U, G)
    return U * skew(U' * G)
end

"""
    retract_unitary_qr(U, ξ, α)

QR-based retraction on the unitary manifold.
Moves from U in direction ξ with step size α.

Uses QR decomposition to ensure the result stays on U(n).
Ensures type stability by converting step size to match element type.
"""
function retract_unitary_qr(U::AbstractMatrix{T}, ξ::AbstractMatrix{T}, α) where T
    # Ensure α has the correct type to avoid Float32/Float64 mixing
    α_typed = convert(real(T), α)
    Y = U + α_typed * ξ
    Q, R = qr(Y)
    # Ensure we're on the same connected component (det = 1 for SU(n))
    # by adjusting signs based on diagonal of R
    Q_mat = Matrix{T}(Q)
    for i in axes(R, 1)
        if real(R[i, i]) < 0
            Q_mat[:, i] .*= -1
        end
    end
    return Q_mat
end

"""
    retract_unitary_cayley(U, ξ, α)

Cayley retraction on the unitary manifold.
More stable than QR for small steps.

Cayley(U, ξ) = (I - α/2 * W)^(-1) * (I + α/2 * W) * U
where W = ξ * U' - U * ξ' (skew-Hermitian)
"""
function retract_unitary_cayley(U, ξ, α)
    W = ξ * U' - U * ξ'
    n = size(U, 1)
    I_n = Matrix{eltype(U)}(I, n, n)
    return (I_n - (α/2) * W) \ ((I_n + (α/2) * W) * U)
end

# ============================================================================
# Parallel Transport (per-tensor, projection-based approximation)
# ============================================================================

"""
    parallel_transport_unitary(U_old, U_new, v)

Transport tangent vector `v` from tangent space at `U_old` to tangent space at `U_new`
on the unitary manifold U(n).

Uses projection-based transport: re-project `v` onto the tangent space at `U_new`.
This is a standard first-order approximation used in Riemannian adaptive optimizers.
"""
parallel_transport_unitary(U_old, U_new, v) = project_tangent_unitary(U_new, v)

"""
    parallel_transport_u1_product(z_old, z_new, v)

Transport tangent vector `v` from tangent space at `z_old` to tangent space at `z_new`
on the U(1)^4 product manifold.

Uses projection-based transport: re-project `v` onto the tangent space at `z_new`.
"""
parallel_transport_u1_product(z_old, z_new, v) = project_tangent_u1_product(z_new, v)

# ============================================================================
# GPU Speed Check
# ============================================================================

"""
    gpu_speed_check(tensors, loss_fn, grad_fn; warmup=3, measure=5)

Warmup and measure GPU iteration time. Warns if GPU appears slower than expected.
Returns `(avg_time_ms,)`.
"""
function gpu_speed_check(tensors, loss_fn, grad_fn; warmup::Int=3, measure::Int=5)
    # Warmup iterations (JIT compilation)
    for _ in 1:warmup
        g = grad_fn(tensors)
    end

    # Measure
    times = Float64[]
    for _ in 1:measure
        t0 = time_ns()
        g = grad_fn(tensors)
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)  # ms
    end
    avg_ms = sum(times) / length(times)
    return (avg_time_ms=avg_ms,)
end

# ============================================================================
# Batched Gradient Checking Helper
# ============================================================================

"""
    _check_grads_batched(grads)

Check if any gradient contains NaN or Inf. Returns true if bad gradients found.
Works with both Vector of matrices and batched 3D arrays.
"""
function _check_grads_batched(grads)
    for g in grads
        if any(isnan, g) || any(isinf, g)
            return true
        end
    end
    return false
end

# ============================================================================
# GPU-Compatible Gradient Descent (Batched)
# ============================================================================

"""
    riemannian_gradient_descent_gpu(
        tensors, loss_fn, grad_fn;
        lr=0.01, max_iter=100, tol=1e-6, verbose=false
    )

GPU-compatible Riemannian gradient descent for mixed manifold tensors.

Uses batched operations on (2,2,n) arrays to minimize CUDA kernel launches.
Tensor classification is done once before the loop. Gradient computation
via Zygote still operates on individual tensors (AD requirement).

# Arguments
- `tensors`: Vector of matrices (can be CuArrays)
- `loss_fn`: Function that computes loss given tensors
- `grad_fn`: Function that computes Euclidean gradients w.r.t. tensors

# Keyword Arguments
- `lr`: Initial learning rate (step size for line search)
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance for gradient norm
- `verbose`: Print progress
- `armijo_c`: Armijo sufficient decrease parameter (default: 1e-4)
- `armijo_tau`: Backtracking contraction factor (default: 0.5)
- `max_ls_steps`: Maximum line search steps per iteration (default: 10)

# Returns
- Optimized tensors (same type as input)
"""
function riemannian_gradient_descent_gpu(
    tensors::Vector{T},
    loss_fn::Function,
    grad_fn::Function;
    lr::Real = 0.01,
    max_iter::Int = 100,
    tol::Real = 1e-6,
    verbose::Bool = false,
    armijo_c::Real = 1e-4,
    armijo_tau::Real = 0.5,
    max_ls_steps::Int = 10
) where T <: AbstractMatrix

    current_tensors = copy.(tensors)
    n_tensors = length(tensors)
    ET = eltype(T)
    RT = real(ET)

    # Classify manifold types ONCE (not per-iteration)
    _nvtx_range_push("classify_tensors")
    unitary_idx, u1_idx = classify_tensors_once(tensors)
    _nvtx_range_pop()
    n_u = length(unitary_idx)
    n_u1 = length(u1_idx)

    if verbose
        println("  Manifold types: $n_u U(2), $n_u1 U(1)^4")
    end

    # Pre-loop: build persistent batched state + pre-allocate reusable buffers
    U_batch = n_u > 0 ? stack_tensors(current_tensors, unitary_idx) : Array{ET}(undef, 2, 2, 0)
    Z_batch = n_u1 > 0 ? stack_tensors(current_tensors, u1_idx) : Array{ET}(undef, 2, 2, 0)

    # Pre-allocate gradient batch buffers (reused every iteration via stack_tensors!)
    G_u_buf = n_u > 0 ? similar(U_batch) : Array{ET}(undef, 2, 2, 0)
    G_u1_buf = n_u1 > 0 ? similar(Z_batch) : Array{ET}(undef, 2, 2, 0)

    # Cache loss across iterations to avoid redundant forward passes.
    # After each accepted line search step, the candidate_loss IS the loss
    # at the new point — reuse it as current_loss in the next iteration.
    cached_loss = RT(NaN)
    if verbose
        cached_loss = RT(loss_fn(current_tensors))
        println("  Initial loss: ", round(cached_loss, digits=6))
    end

    for iter in 1:max_iter
        # Unstack persistent batches for Zygote (which needs individual tensors)
        if n_u > 0
            unstack_tensors!(current_tensors, U_batch, unitary_idx)
        end
        if n_u1 > 0
            unstack_tensors!(current_tensors, Z_batch, u1_idx)
        end

        # Compute Euclidean gradients (Zygote needs individual tensors)
        _nvtx_range_push("gradient")
        euclidean_grads_raw = grad_fn(current_tensors)
        euclidean_grads = euclidean_grads_raw isa Tuple ? collect(euclidean_grads_raw) : euclidean_grads_raw
        _nvtx_range_pop()

        # Check for NaN or Inf in gradients
        if _check_grads_batched(euclidean_grads)
            verbose && println("  WARNING: NaN or Inf in gradients at iter $iter")
            break
        end

        # === Batched projection (use persistent batches + pre-allocated grad buffers) ===
        _nvtx_range_push("projection")
        local rg_u_batch, rg_u1_batch

        if n_u > 0
            stack_tensors!(G_u_buf, euclidean_grads, unitary_idx)
            rg_u_batch = batched_project_unitary(U_batch, G_u_buf)
        end

        if n_u1 > 0
            stack_tensors!(G_u1_buf, euclidean_grads, u1_idx)
            rg_u1_batch = batched_project_u1(Z_batch, G_u1_buf)
        end
        _nvtx_range_pop()

        # Compute gradient norm directly from batched arrays (2 reductions, not ~50)
        grad_norm_sq = zero(RT)
        if n_u > 0
            grad_norm_sq += real(sum(abs2, rg_u_batch))
        end
        if n_u1 > 0
            grad_norm_sq += real(sum(abs2, rg_u1_batch))
        end
        grad_norm = sqrt(grad_norm_sq)

        # Check convergence
        if grad_norm < tol
            verbose && println("  Converged at iteration $iter (grad_norm = $grad_norm)")
            break
        end

        # === Armijo backtracking line search ===
        _nvtx_range_push("line_search")

        # Reuse cached loss from previous iteration's accepted step if available
        current_loss = isnan(cached_loss) ? RT(loss_fn(current_tensors)) : cached_loss

        if verbose && (iter % 10 == 0 || iter == 1)
            println("  Iter $iter: loss = $(round(current_loss, digits=6)), grad_norm = $(round(grad_norm, digits=6)), lr = $lr")
        end

        α = RT(lr)
        accepted = false
        for _ls in 1:max_ls_steps
            # Trial retraction at step size α
            if n_u > 0
                U_cand = batched_retract_unitary_qr(U_batch, .-rg_u_batch, α)
                unstack_tensors!(current_tensors, U_cand, unitary_idx)
            end
            if n_u1 > 0
                Z_cand = batched_retract_u1(Z_batch, .-rg_u1_batch, α)
                unstack_tensors!(current_tensors, Z_cand, u1_idx)
            end

            candidate_loss = RT(loss_fn(current_tensors))

            # Armijo sufficient decrease condition
            if candidate_loss <= current_loss - RT(armijo_c) * α * grad_norm_sq
                # Accept: update persistent batches and cache the loss
                if n_u > 0
                    U_batch = U_cand
                end
                if n_u1 > 0
                    Z_batch = Z_cand
                end
                cached_loss = candidate_loss
                accepted = true
                break
            end
            α *= RT(armijo_tau)
        end
        _nvtx_range_pop()

        if !accepted
            # Fall back: take the smallest step tried
            _nvtx_range_push("retraction_fallback")
            if n_u > 0
                U_batch = batched_retract_unitary_qr(U_batch, .-rg_u_batch, α)
            end
            if n_u1 > 0
                Z_batch = batched_retract_u1(Z_batch, .-rg_u1_batch, α)
            end
            cached_loss = RT(NaN)  # unknown after fallback
            _nvtx_range_pop()
        end
    end

    # Final unstack
    if n_u > 0
        unstack_tensors!(current_tensors, U_batch, unitary_idx)
    end
    if n_u1 > 0
        unstack_tensors!(current_tensors, Z_batch, u1_idx)
    end

    return current_tensors
end

# ============================================================================
# Riemannian Adam Optimizer (Batched)
# ============================================================================

"""
    riemannian_adam(
        tensors, loss_fn, grad_fn;
        lr=0.001, betas=(0.9, 0.999), eps=1e-8,
        max_iter=100, tol=1e-6, verbose=false
    )

Riemannian Adam optimizer for mixed manifold tensors (Becigneul & Ganea, 2019).

Uses batched (2,2,n) operations to minimize CUDA kernel launches and eliminate
GPU→CPU transfers. Adam state (m, v) is stored as batched 3D arrays per
manifold type. Bias correction, moment update, and direction computation all
use fused broadcasts.

# Arguments
- `tensors`: Vector of matrices (can be CuArrays)
- `loss_fn`: Function that computes loss given tensors
- `grad_fn`: Function that computes Euclidean gradients w.r.t. tensors

# Keyword Arguments
- `lr`: Learning rate (default: 0.001)
- `betas`: Tuple (beta1, beta2) for exponential moving averages (default: (0.9, 0.999))
- `eps`: Small constant for numerical stability (default: 1e-8)
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance for gradient norm
- `verbose`: Print progress

# Returns
- Optimized tensors (same type as input)

# Reference
Becigneul, G., & Ganea, O. (2019). Riemannian Adaptive Optimization Methods. ICLR 2019.
"""
function riemannian_adam(
    tensors::Vector{T},
    loss_fn::Function,
    grad_fn::Function;
    lr::Real = 0.001,
    betas::Tuple{Real,Real} = (0.9, 0.999),
    eps::Real = 1e-8,
    max_iter::Int = 100,
    tol::Real = 1e-6,
    verbose::Bool = false
) where T <: AbstractMatrix

    current_tensors = copy.(tensors)
    n_tensors = length(tensors)
    beta1, beta2 = betas
    ET = eltype(T)
    RT = real(ET)

    # Classify manifold types ONCE
    _nvtx_range_push("classify_tensors")
    unitary_idx, u1_idx = classify_tensors_once(tensors)
    _nvtx_range_pop()
    n_u = length(unitary_idx)
    n_u1 = length(u1_idx)

    if verbose
        println("  Manifold types: $n_u U(2), $n_u1 U(1)^4")
    end

    # Pre-loop: build persistent batched state + pre-allocate reusable buffers
    U_batch = n_u > 0 ? stack_tensors(current_tensors, unitary_idx) : Array{ET}(undef, 2, 2, 0)
    Z_batch = n_u1 > 0 ? stack_tensors(current_tensors, u1_idx) : Array{ET}(undef, 2, 2, 0)

    # Pre-allocate gradient batch buffers (reused every iteration via stack_tensors!)
    G_u_buf = n_u > 0 ? similar(U_batch) : Array{ET}(undef, 2, 2, 0)
    G_u1_buf = n_u1 > 0 ? similar(Z_batch) : Array{ET}(undef, 2, 2, 0)

    # Pre-allocate Adam direction buffer (reused for bias-corrected direction)
    dir_u_buf = n_u > 0 ? similar(U_batch) : Array{ET}(undef, 2, 2, 0)
    dir_u1_buf = n_u1 > 0 ? similar(Z_batch) : Array{ET}(undef, 2, 2, 0)

    # Initialize batched Adam state per manifold type
    # m_* : first moment (complex, same type as tensors)
    # v_* : second moment (real-valued)
    if n_u > 0
        proto = tensors[unitary_idx[1]]
        m_unitary = similar(proto, ET, 2, 2, n_u) .* false  # zeros on same device
        v_unitary = similar(proto, RT, 2, 2, n_u) .* false
    else
        m_unitary = Array{ET}(undef, 2, 2, 0)
        v_unitary = Array{RT}(undef, 2, 2, 0)
    end

    if n_u1 > 0
        proto = tensors[u1_idx[1]]
        m_u1 = similar(proto, ET, 2, 2, n_u1) .* false
        v_u1 = similar(proto, RT, 2, 2, n_u1) .* false
    else
        m_u1 = Array{ET}(undef, 2, 2, 0)
        v_u1 = Array{RT}(undef, 2, 2, 0)
    end

    if verbose
        initial_loss = loss_fn(current_tensors)
        println("  Initial loss: ", round(initial_loss, digits=6))
    end

    for iter in 1:max_iter
        # Unstack persistent batches for Zygote (which needs individual tensors)
        if n_u > 0
            unstack_tensors!(current_tensors, U_batch, unitary_idx)
        end
        if n_u1 > 0
            unstack_tensors!(current_tensors, Z_batch, u1_idx)
        end

        # Compute Euclidean gradients
        _nvtx_range_push("gradient")
        euclidean_grads_raw = grad_fn(current_tensors)
        euclidean_grads = euclidean_grads_raw isa Tuple ? collect(euclidean_grads_raw) : euclidean_grads_raw
        _nvtx_range_pop()

        if _check_grads_batched(euclidean_grads)
            verbose && println("  WARNING: NaN or Inf in gradients at iter $iter")
            break
        end

        # === Batched projection (use persistent batches + pre-allocated grad buffers) ===
        _nvtx_range_push("projection")

        local rg_unitary_batch, rg_u1_batch
        if n_u > 0
            stack_tensors!(G_u_buf, euclidean_grads, unitary_idx)
            rg_unitary_batch = batched_project_unitary(U_batch, G_u_buf)
        end

        if n_u1 > 0
            stack_tensors!(G_u1_buf, euclidean_grads, u1_idx)
            rg_u1_batch = batched_project_u1(Z_batch, G_u1_buf)
        end
        _nvtx_range_pop()

        # Gradient norm directly from batched arrays (2 reductions, not ~50)
        grad_norm_sq = zero(RT)
        if n_u > 0
            grad_norm_sq += real(sum(abs2, rg_unitary_batch))
        end
        if n_u1 > 0
            grad_norm_sq += real(sum(abs2, rg_u1_batch))
        end
        grad_norm = sqrt(grad_norm_sq)

        if verbose && (iter % 10 == 0 || iter == 1)
            loss = loss_fn(current_tensors)
            println("  Iter $iter: loss = $(round(loss, digits=6)), grad_norm = $(round(grad_norm, digits=6)), lr = $lr")
        end

        if grad_norm < tol
            verbose && println("  Converged at iteration $iter (grad_norm = $grad_norm)")
            break
        end

        # Bias correction factors
        bc1 = one(RT) - RT(beta1)^iter
        bc2 = one(RT) - RT(beta2)^iter

        # === Batched Adam update for unitary group ===
        _nvtx_range_push("retraction")
        if n_u > 0
            # In-place moment update (no allocation — fused broadcasts)
            @. m_unitary = RT(beta1) * m_unitary + RT(1 - beta1) * rg_unitary_batch
            @. v_unitary = RT(beta2) * v_unitary + RT(1 - beta2) * real(abs2(rg_unitary_batch))

            # Bias-corrected direction (in-place into dir_u_buf)
            @. dir_u_buf = (m_unitary / bc1) / (sqrt(v_unitary / bc2) + RT(eps))

            # Retract (use persistent U_batch directly, no re-stacking)
            U_old_batch = U_batch
            U_batch = batched_retract_unitary_qr(U_old_batch, .-dir_u_buf, lr)

            # Transport momentum (re-project onto new tangent space)
            m_unitary = batched_transport_unitary(U_old_batch, U_batch, m_unitary)
        end

        # === Batched Adam update for U(1) group ===
        if n_u1 > 0
            @. m_u1 = RT(beta1) * m_u1 + RT(1 - beta1) * rg_u1_batch
            @. v_u1 = RT(beta2) * v_u1 + RT(1 - beta2) * real(abs2(rg_u1_batch))

            @. dir_u1_buf = (m_u1 / bc1) / (sqrt(v_u1 / bc2) + RT(eps))

            Z_old_batch = Z_batch
            Z_batch = batched_retract_u1(Z_old_batch, .-dir_u1_buf, lr)

            m_u1 = batched_transport_u1(Z_old_batch, Z_batch, m_u1)
        end
        _nvtx_range_pop()
    end

    # Final unstack
    if n_u > 0
        unstack_tensors!(current_tensors, U_batch, unitary_idx)
    end
    if n_u1 > 0
        unstack_tensors!(current_tensors, Z_batch, u1_idx)
    end

    return current_tensors
end

# ============================================================================
# Stochastic Riemannian Gradient Descent
# ============================================================================

"""
    riemannian_sgd_gpu(
        tensors, batch_loss_fn, batch_grad_fn, data_batches;
        lr=0.01, epochs=10, verbose=false
    )

Stochastic Riemannian gradient descent for mini-batch training on GPU.

# Arguments
- `tensors`: Vector of unitary matrices (can be CuArrays)
- `batch_loss_fn`: Function (tensors, batch) -> loss
- `batch_grad_fn`: Function (tensors, batch) -> gradients
- `data_batches`: Iterator of data batches

# Keyword Arguments
- `lr`: Learning rate
- `epochs`: Number of passes through data
- `verbose`: Print progress

# Returns
- Optimized tensors
"""
function riemannian_sgd_gpu(
    tensors::Vector{T},
    batch_loss_fn::Function,
    batch_grad_fn::Function,
    data_batches;
    lr::Real = 0.01,
    epochs::Int = 10,
    steps_per_batch::Int = 1,
    verbose::Bool = false
) where T <: AbstractMatrix

    current_tensors = copy.(tensors)
    n_tensors = length(tensors)

    for epoch in 1:epochs
        epoch_loss = 0.0
        n_batches = 0

        for batch in data_batches
            # Multiple gradient steps per batch
            for step in 1:steps_per_batch
                # Compute Euclidean gradients for this batch
                euclidean_grads = batch_grad_fn(current_tensors, batch)

                # Project and update
                for i in 1:n_tensors
                    riemannian_grad = project_tangent_unitary(current_tensors[i], euclidean_grads[i])
                    current_tensors[i] = retract_unitary_qr(
                        current_tensors[i],
                        -riemannian_grad,
                        lr
                    )
                end
            end

            # Track loss
            epoch_loss += batch_loss_fn(current_tensors, batch)
            n_batches += 1
        end

        if verbose
            avg_loss = epoch_loss / n_batches
            println("  Epoch $epoch: avg_loss = $(round(avg_loss, digits=6))")
        end
    end

    return current_tensors
end

# ============================================================================
# Integration with Training Pipeline
# ============================================================================

"""
    _train_on_batch_gpu(
        batch, tensors, optcode, inverse_code, m, n, loss, steps;
        lr=0.01, optimizer=:gradient_descent
    )

GPU-compatible training on a batch of images.
Replaces _train_on_batch when device=:gpu or optimizer=:adam.

This function:
1. Uses Zygote for automatic differentiation
2. Applies the selected Riemannian optimizer
3. Returns optimized tensors (on same device as input)

# Supported optimizers
- `:gradient_descent` (default): Riemannian gradient descent
- `:adam`: Riemannian Adam (Becigneul & Ganea, 2019)
"""
function _train_on_batch_gpu(
    batch::Vector{<:AbstractMatrix},
    tensors::Vector{<:AbstractMatrix},
    optcode::OMEinsum.AbstractEinsum,
    inverse_code::OMEinsum.AbstractEinsum,
    m::Int, n::Int,
    loss::AbstractLoss,
    steps::Int;
    lr::Real = 0.01,
    optimizer::Symbol = :gradient_descent
)
    n_imgs = length(batch)

    # Loss function
    function loss_fn(ts)
        total = zero(real(eltype(ts[1])))
        for img in batch
            total += loss_function(ts, m, n, optcode, img, loss; inverse_code=inverse_code)
        end
        return total / n_imgs
    end

    # Gradient function using Zygote
    function grad_fn(ts)
        _, back = Zygote.pullback(loss_fn, ts)
        grads = back(one(real(eltype(ts[1]))))[1]
        return grads
    end

    # Dispatch to selected optimizer
    if optimizer === :adam
        optimized = riemannian_adam(
            tensors, loss_fn, grad_fn;
            lr=lr, max_iter=steps, tol=1e-8, verbose=false
        )
    else
        optimized = riemannian_gradient_descent_gpu(
            tensors, loss_fn, grad_fn;
            lr=lr, max_iter=steps, tol=1e-8, verbose=false
        )
    end

    return optimized
end

# ============================================================================
# Exports
# ============================================================================

# These are internal functions, exported through training.jl integration
