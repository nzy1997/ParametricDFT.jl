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
#    - Retraction: QR-based
#
# 2. U(1)^4 - Product of 4 unit circles (for controlled phase gates)
#    - Tangent space at z: T_z = {iθ*z : θ ∈ ℝ}
#    - Riemannian gradient: imaginary part of conj(z)*g
#    - Retraction: normalization z_new = z / |z|

using LinearAlgebra

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

# ============================================================================
# U(1)^4 Manifold Operations (for controlled phase gates)
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
# Unitary Manifold Operations
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
# Parallel Transport (projection-based approximation)
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
# GPU-Compatible Gradient Descent
# ============================================================================

"""
    riemannian_gradient_descent_gpu(
        tensors, loss_fn, grad_fn;
        lr=0.01, max_iter=100, tol=1e-6, verbose=false
    )

GPU-compatible Riemannian gradient descent for mixed manifold tensors.

Automatically detects manifold type for each tensor:
- Unitary matrices (U(2)): QR retraction
- U(1)^4 product (controlled phases): Normalization retraction

# Arguments
- `tensors`: Vector of matrices (can be CuArrays)
- `loss_fn`: Function that computes loss given tensors
- `grad_fn`: Function that computes Euclidean gradients w.r.t. tensors

# Keyword Arguments
- `lr`: Learning rate (step size)
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance for gradient norm
- `verbose`: Print progress

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
    verbose::Bool = false
) where T <: AbstractMatrix

    current_tensors = copy.(tensors)
    n_tensors = length(tensors)

    # Detect manifold type for each tensor (using CPU version for type check)
    is_unitary = [is_unitary_tensor(Array(t)) for t in tensors]

    if verbose
        n_unitary = sum(is_unitary)
        println("  Manifold types: $n_unitary U(2), $(n_tensors - n_unitary) U(1)^4")
    end

    # Debug: print initial loss
    if verbose
        initial_loss = loss_fn(current_tensors)
        println("  Initial loss: ", round(initial_loss, digits=6))
    end

    for iter in 1:max_iter
        # Compute Euclidean gradients
        euclidean_grads_raw = grad_fn(current_tensors)
        # Handle case where Zygote returns a tuple instead of vector
        euclidean_grads = euclidean_grads_raw isa Tuple ? collect(euclidean_grads_raw) : euclidean_grads_raw

        # Check for NaN or Inf in gradients
        has_bad_grad = false
        for g in euclidean_grads
            if any(isnan, g) || any(isinf, g)
                has_bad_grad = true
                break
            end
        end
        if has_bad_grad
            verbose && println("  WARNING: NaN or Inf in gradients at iter $iter")
            break
        end

        # Project to Riemannian gradients and compute norm
        riemannian_grads = Vector{T}(undef, n_tensors)
        grad_norm_sq = zero(real(eltype(T)))

        for i in 1:n_tensors
            # Project Euclidean gradient to tangent space based on manifold type
            if is_unitary[i]
                riemannian_grads[i] = project_tangent_unitary(current_tensors[i], euclidean_grads[i])
            else
                riemannian_grads[i] = project_tangent_u1_product(current_tensors[i], euclidean_grads[i])
            end
            grad_norm_sq += sum(abs2, riemannian_grads[i])
        end

        grad_norm = sqrt(grad_norm_sq)

        if verbose && (iter % 10 == 0 || iter == 1)
            loss = loss_fn(current_tensors)
            println("  Iter $iter: loss = $(round(loss, digits=6)), grad_norm = $(round(grad_norm, digits=6)), lr = $lr")
        end

        # Check convergence
        if grad_norm < tol
            verbose && println("  Converged at iteration $iter (grad_norm = $grad_norm)")
            break
        end

        # Update tensors using appropriate retraction
        for i in 1:n_tensors
            if is_unitary[i]
                # Use QR retraction for U(2)
                current_tensors[i] = retract_unitary_qr(
                    current_tensors[i],
                    -riemannian_grads[i],  # Negative gradient for descent
                    lr
                )
            else
                # Use normalization retraction for U(1)^4
                current_tensors[i] = retract_u1_product(
                    current_tensors[i],
                    -riemannian_grads[i],
                    lr
                )
            end
        end
    end

    return current_tensors
end

# ============================================================================
# Riemannian Adam Optimizer
# ============================================================================

"""
    riemannian_adam(
        tensors, loss_fn, grad_fn;
        lr=0.001, betas=(0.9, 0.999), eps=1e-8,
        max_iter=100, tol=1e-6, verbose=false
    )

Riemannian Adam optimizer for mixed manifold tensors (Bécigneul & Ganea, 2019).

Extends the standard Adam optimizer to Riemannian manifolds by:
- Using Riemannian gradients (projected onto tangent space)
- Retraction instead of Euclidean update
- Parallel transport of momentum between tangent spaces

Automatically detects manifold type for each tensor:
- Unitary matrices (U(2)): QR retraction, projection-based transport
- U(1)^4 product (controlled phases): Normalization retraction

# Arguments
- `tensors`: Vector of matrices (can be CuArrays)
- `loss_fn`: Function that computes loss given tensors
- `grad_fn`: Function that computes Euclidean gradients w.r.t. tensors

# Keyword Arguments
- `lr`: Learning rate (default: 0.001)
- `betas`: Tuple (β₁, β₂) for exponential moving averages (default: (0.9, 0.999))
- `eps`: Small constant for numerical stability (default: 1e-8)
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance for gradient norm
- `verbose`: Print progress

# Returns
- Optimized tensors (same type as input)

# Reference
Bécigneul, G., & Ganea, O. (2019). Riemannian Adaptive Optimization Methods. ICLR 2019.
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
    β₁, β₂ = betas

    # Detect manifold type for each tensor (using CPU version for type check)
    is_unitary = [is_unitary_tensor(Array(t)) for t in tensors]

    if verbose
        n_unitary = sum(is_unitary)
        println("  Manifold types: $n_unitary U(2), $(n_tensors - n_unitary) U(1)^4")
    end

    # Initialize optimizer state: first moment (m) and second moment (v)
    # m[i] has same type/shape as tensor i (complex, in tangent space)
    # v[i] has real element type, same shape (element-wise squared gradient norms)
    RT = real(eltype(T))
    m = [zero(t) for t in current_tensors]              # First moment (momentum)
    v = [zeros(RT, size(t)) for t in current_tensors]    # Second moment (real-valued)
    # Ensure v is on the same device as the tensors
    v = [similar(real.(t), RT) .* false for t in current_tensors]

    if verbose
        initial_loss = loss_fn(current_tensors)
        println("  Initial loss: ", round(initial_loss, digits=6))
    end

    for iter in 1:max_iter
        # Compute Euclidean gradients
        euclidean_grads_raw = grad_fn(current_tensors)
        euclidean_grads = euclidean_grads_raw isa Tuple ? collect(euclidean_grads_raw) : euclidean_grads_raw

        # Check for NaN or Inf in gradients
        has_bad_grad = false
        for g in euclidean_grads
            if any(isnan, g) || any(isinf, g)
                has_bad_grad = true
                break
            end
        end
        if has_bad_grad
            verbose && println("  WARNING: NaN or Inf in gradients at iter $iter")
            break
        end

        # Project to Riemannian gradients
        riemannian_grads = Vector{T}(undef, n_tensors)
        grad_norm_sq = zero(RT)

        for i in 1:n_tensors
            if is_unitary[i]
                riemannian_grads[i] = project_tangent_unitary(current_tensors[i], euclidean_grads[i])
            else
                riemannian_grads[i] = project_tangent_u1_product(current_tensors[i], euclidean_grads[i])
            end
            grad_norm_sq += sum(abs2, riemannian_grads[i])
        end

        grad_norm = sqrt(grad_norm_sq)

        if verbose && (iter % 10 == 0 || iter == 1)
            loss = loss_fn(current_tensors)
            println("  Iter $iter: loss = $(round(loss, digits=6)), grad_norm = $(round(grad_norm, digits=6)), lr = $lr")
        end

        # Check convergence
        if grad_norm < tol
            verbose && println("  Converged at iteration $iter (grad_norm = $grad_norm)")
            break
        end

        # Bias correction factors
        bc1 = one(RT) - RT(β₁)^iter
        bc2 = one(RT) - RT(β₂)^iter

        # Update each tensor
        for i in 1:n_tensors
            # Update first moment (momentum)
            m[i] = RT(β₁) .* m[i] .+ RT(1 - β₁) .* riemannian_grads[i]

            # Update second moment (element-wise squared gradient norms)
            v[i] = RT(β₂) .* v[i] .+ RT(1 - β₂) .* real.(abs2.(riemannian_grads[i]))

            # Bias-corrected estimates
            m_hat = m[i] ./ bc1
            v_hat = v[i] ./ bc2

            # Compute Adam update direction
            direction = m_hat ./ (sqrt.(v_hat) .+ RT(eps))

            # Save old point for parallel transport
            old_tensor = current_tensors[i]

            # Retract in negative direction (descent)
            if is_unitary[i]
                current_tensors[i] = retract_unitary_qr(
                    current_tensors[i],
                    -direction,
                    lr
                )
            else
                current_tensors[i] = retract_u1_product(
                    current_tensors[i],
                    -direction,
                    lr
                )
            end

            # Parallel transport momentum to new tangent space
            if is_unitary[i]
                m[i] = parallel_transport_unitary(old_tensor, current_tensors[i], m[i])
            else
                m[i] = parallel_transport_u1_product(old_tensor, current_tensors[i], m[i])
            end
        end
    end

    return current_tensors
end

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
- `:adam`: Riemannian Adam (Bécigneul & Ganea, 2019)
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
