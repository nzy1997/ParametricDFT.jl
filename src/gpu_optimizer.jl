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
function project_tangent_u1_product(z::AbstractMatrix, g::AbstractMatrix)
    # For each element on U(1), the Riemannian gradient is:
    # proj_{z_i}(g_i) = z_i * im * imag(conj(z_i) * g_i)
    # This simplifies to: im * imag(conj(z) .* g) .* z
    return im .* imag.(conj.(z) .* g) .* z
end

"""
    retract_u1_product(z, ξ, α)

Retract on U(1)^4 product manifold.
For each element, move in direction ξ and project back to unit circle.
"""
function retract_u1_product(z::AbstractMatrix, ξ::AbstractMatrix, α)
    y = z .+ α .* ξ
    # Normalize each element to stay on U(1)
    return y ./ abs.(y)
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
"""
function retract_unitary_qr(U, ξ, α)
    Y = U + α * ξ
    Q, R = qr(Y)
    # Ensure we're on the same connected component (det = 1 for SU(n))
    # by adjusting signs based on diagonal of R
    Q_mat = Matrix(Q)
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
        lr=0.01
    )

GPU-compatible training on a batch of images.
Replaces _train_on_batch when device=:gpu.

This function:
1. Moves tensors and data to GPU (if not already)
2. Uses Zygote for automatic differentiation
3. Applies Riemannian gradient descent with QR retraction
4. Returns optimized tensors (on same device as input)
"""
function _train_on_batch_gpu(
    batch::Vector{<:AbstractMatrix},
    tensors::Vector{<:AbstractMatrix},
    optcode::OMEinsum.AbstractEinsum,
    inverse_code::OMEinsum.AbstractEinsum,
    m::Int, n::Int,
    loss::AbstractLoss,
    steps::Int;
    lr::Real = 0.01
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

    # Run Riemannian gradient descent
    # Use the passed learning rate - don't override it
    # Debug: enable verbose for first batch only (controlled by global counter)
    optimized = riemannian_gradient_descent_gpu(
        tensors, loss_fn, grad_fn;
        lr=lr, max_iter=steps, tol=1e-8, verbose=true
    )

    return optimized
end

# ============================================================================
# Exports
# ============================================================================

# These are internal functions, exported through training.jl integration
