# Riemannian optimizers for mixed U(2)/U(1)^4 manifolds.
# Uses batched (2,2,n) operations to minimize CUDA kernel launches.

using LinearAlgebra

# NVTX profiling callbacks (set by CUDAExt at init time)
const _nvtx_push_fn = Ref{Any}(nothing)
const _nvtx_pop_fn = Ref{Any}(nothing)

"""NVTX range push. No-op unless CUDAExt sets the callback."""
_nvtx_range_push(name::String) = (_nvtx_push_fn[] !== nothing && _nvtx_push_fn[](name); nothing)

"""NVTX range pop. No-op unless CUDAExt sets the callback."""
_nvtx_range_pop() = (_nvtx_pop_fn[] !== nothing && _nvtx_pop_fn[](); nothing)

# Manifold Type Detection

"""Check if 2×2 matrix t satisfies U*U' ≈ I (U(2) manifold)."""
function is_unitary_tensor(t::AbstractMatrix{T}) where T
    size(t) == (2, 2) || return false
    return isapprox(t * t', Matrix{T}(I, 2, 2), atol=1e-6)
end

"""Classify tensors into U(2) vs U(1)^4 groups. Returns `(unitary_idx, u1_idx)`."""
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

# Batched Tensor Packing/Unpacking

"""Pack selected 2×2 matrices into a (2,2,n) batched array."""
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

"""In-place version: pack into pre-allocated (2,2,n) array."""
function stack_tensors!(batch::AbstractArray{T,3}, tensors::Vector{<:AbstractMatrix}, indices::Vector{Int}) where T
    for (k, idx) in enumerate(indices)
        copyto!(view(batch, :, :, k), tensors[idx])
    end
    return batch
end

"""Unpack (2,2,n) array back into individual matrices."""
function unstack_tensors!(tensors::Vector{<:AbstractMatrix}, batch::AbstractArray{T,3}, indices::Vector{Int}) where T
    for (k, idx) in enumerate(indices)
        if isassigned(tensors, idx) && size(tensors[idx]) == (2, 2)
            copyto!(tensors[idx], view(batch, :, :, k))
        else
            tensors[idx] = copy(view(batch, :, :, k))
        end
    end
end

# Batched 2×2 Matrix Operations

"""Batched 2×2 matrix multiply: C[i,j,:] = sum_k A[i,k,:] .* B[k,j,:]"""
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

"""Batched conjugate transpose for (2,2,n) arrays."""
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

# Batched Manifold Operations — U(2) Unitary

"""Batched U(2) Riemannian projection: U * skew(U'G) on (2,2,n) arrays."""
function batched_project_unitary(U::AbstractArray{T,3}, G::AbstractArray{T,3}) where T
    UhG = batched_matmul_2x2(batched_adjoint_2x2(U), G)
    S = (UhG .- batched_adjoint_2x2(UhG)) ./ 2  # skew-Hermitian part
    return batched_matmul_2x2(U, S)
end

"""Batched 2×2 QR retraction via Gram-Schmidt (pure broadcast, no cuSOLVER)."""
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

"""Batched U(2) parallel transport via re-projection."""
batched_transport_unitary(U_old, U_new, v) = batched_project_unitary(U_new, v)

# Batched Manifold Operations — U(1)^4

"""Batched U(1)^4 projection: im * imag(conj(z).*g) .* z on (2,2,n) arrays."""
function batched_project_u1(Z::AbstractArray{T,3}, G::AbstractArray{T,3}) where T
    return T(im) .* imag.(conj.(Z) .* G) .* Z
end

"""Batched U(1)^4 retraction: normalize(z + α*ξ) on (2,2,n) arrays."""
function batched_retract_u1(Z::AbstractArray{T,3}, Xi::AbstractArray{T,3}, α) where T
    RT = real(T)
    α_typed = convert(RT, α)
    y = Z .+ α_typed .* Xi
    return y ./ T.(abs.(y))
end

"""Batched U(1)^4 parallel transport via re-projection."""
batched_transport_u1(Z_old, Z_new, v) = batched_project_u1(Z_new, v)

# Per-Tensor Manifold Operations (used by tests and as reference implementations)

"""Project Euclidean gradient onto U(1)^4 tangent space at z."""
function project_tangent_u1_product(z::AbstractMatrix{T}, g::AbstractMatrix{T}) where T
    return T(im) .* imag.(conj.(z) .* g) .* z
end

"""Skew-Hermitian part: (A - A') / 2"""
skew(A) = (A - A') / 2

"""Project Euclidean gradient G onto U(n) tangent space at U."""
project_tangent_unitary(U, G) = U * skew(U' * G)

"""QR-based retraction on the unitary manifold."""
function retract_unitary_qr(U::AbstractMatrix{T}, ξ::AbstractMatrix{T}, α) where T
    α_typed = convert(real(T), α)
    Y = U + α_typed * ξ
    Q, R = qr(Y)
    Q_mat = Matrix{T}(Q)
    for i in axes(R, 1)
        if real(R[i, i]) < 0
            Q_mat[:, i] .*= -1
        end
    end
    return Q_mat
end

"""Parallel transport on U(n) via re-projection."""
parallel_transport_unitary(U_old, U_new, v) = project_tangent_unitary(U_new, v)

"""Parallel transport on U(1)^4 via re-projection."""
parallel_transport_u1_product(z_old, z_new, v) = project_tangent_u1_product(z_new, v)

# GPU-Compatible Gradient Descent (Batched)

"""Riemannian GD with Armijo line search on mixed U(2)/U(1)^4 manifold.
Uses batched (2,2,n) operations for GPU efficiency."""
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
        if any(g -> any(x -> isnan(x) || isinf(x), g), euclidean_grads)
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

# Riemannian Adam Optimizer (Batched)

"""Riemannian Adam (Becigneul & Ganea, 2019) on mixed U(2)/U(1)^4 manifold.
Uses batched (2,2,n) operations; Adam state stored as 3D arrays per manifold type."""
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

        if any(g -> any(x -> isnan(x) || isinf(x), g), euclidean_grads)
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

# Integration with Training Pipeline

"""Train on a batch using custom Riemannian optimizer (:gradient_descent or :adam).
When `batched_optcode` is provided, uses single batched einsum for forward pass."""
function _train_on_batch_gpu(
    batch::Vector{<:AbstractMatrix},
    tensors::Vector{<:AbstractMatrix},
    optcode::OMEinsum.AbstractEinsum,
    inverse_code::OMEinsum.AbstractEinsum,
    m::Int, n::Int,
    loss::AbstractLoss,
    steps::Int;
    lr::Real = 0.01,
    optimizer::Symbol = :gradient_descent,
    batched_optcode = nothing,
    batched_inverse_code = nothing
)
    n_imgs = length(batch)

    # Loss function — use batched einsum when available
    if batched_optcode !== nothing
        function batched_loss_fn(ts)
            if loss isa L1Norm
                return batched_loss_l1(batched_optcode, ts, batch, m, n)
            elseif loss isa L2Norm
                return batched_loss_l2(batched_optcode, ts, batch, m, n)
            else  # MSELoss
                return batched_loss_mse(batched_optcode, ts, batch, m, n, loss.k, inverse_code)
            end
        end
        loss_fn = batched_loss_fn
    else
        # Sequential fallback (batch_size=1 or no batched code)
        function sequential_loss_fn(ts)
            total = zero(real(eltype(ts[1])))
            for img in batch
                total += loss_function(ts, m, n, optcode, img, loss; inverse_code=inverse_code)
            end
            return total / n_imgs
        end
        loss_fn = sequential_loss_fn
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

