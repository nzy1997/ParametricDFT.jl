# Modular Riemannian optimizers using the AbstractRiemannianManifold API.
# Dispatches through project/retract/transport from manifolds.jl.

using LinearAlgebra

# ============================================================================
# Abstract Optimizer Type
# ============================================================================

"""Abstract base type for Riemannian optimizers."""
abstract type AbstractRiemannianOptimizer end

# ============================================================================
# Shared Infrastructure
# ============================================================================

"""
    _compute_gradients(grad_fn, tensors, verbose, iter)

Compute Euclidean gradients via `grad_fn`, converting Tuple to Vector if needed.
Returns `nothing` if NaN or Inf values are detected.
"""
function _compute_gradients(grad_fn, tensors, verbose::Bool, iter::Int)
    euclidean_grads_raw = grad_fn(tensors)
    raw = euclidean_grads_raw isa Tuple ? collect(euclidean_grads_raw) : euclidean_grads_raw

    # Build typed Vector, replacing any non-matrix ChainRules tangents (ZeroTangent, etc.)
    # with zero arrays. collect() first ensures the guard runs before typed conversion.
    euclidean_grads = AbstractMatrix[
        raw[i] isa AbstractMatrix ? raw[i] : zeros(eltype(tensors[i]), size(tensors[i]))
        for i in eachindex(raw)
    ]

    # Check for NaN/Inf in gradients
    if any(g -> any(x -> isnan(x) || isinf(x), g), euclidean_grads)
        verbose && println("  WARNING: NaN or Inf in gradients at iter $iter")
        return nothing
    end

    return euclidean_grads
end

"""
    _batched_project(manifold_groups, point_batches, grad_buf_batches, euclidean_grads)

Project Euclidean gradients onto manifold tangent spaces using batched operations.

For each `(manifold, indices)` in `manifold_groups`:
  - Stacks the relevant Euclidean gradients into the pre-allocated gradient buffer
  - Calls `project(manifold, point_batch, grad_batch)` to get Riemannian gradients

Returns `(rg_batches::Dict, grad_norm::Real)` where:
  - `rg_batches` maps each manifold to its projected Riemannian gradient batch
  - `grad_norm` is the L2 norm of the combined Riemannian gradient
"""
function _batched_project(
    manifold_groups::Dict{<:AbstractRiemannianManifold, Vector{Int}},
    point_batches::Dict,
    grad_buf_batches::Dict,
    euclidean_grads
)
    RT = real(eltype(first(values(point_batches))))
    rg_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
    grad_norm_sq = zero(RT)

    for (manifold, indices) in manifold_groups
        pb = point_batches[manifold]
        gb = grad_buf_batches[manifold]
        stack_tensors!(gb, euclidean_grads, indices)
        rg = project(manifold, pb, gb)
        rg_batches[manifold] = rg
        grad_norm_sq += real(sum(abs2, rg))
    end

    return rg_batches, sqrt(grad_norm_sq)
end

# ============================================================================
# RiemannianGD -- Gradient Descent with Armijo Line Search
# ============================================================================

"""
    RiemannianGD(; lr=0.01, armijo_c=1e-4, armijo_tau=0.5, max_ls_steps=10)

Riemannian gradient descent with Armijo backtracking line search.
"""
struct RiemannianGD <: AbstractRiemannianOptimizer
    lr::Float64
    armijo_c::Float64
    armijo_tau::Float64
    max_ls_steps::Int
end

RiemannianGD(; lr=0.01, armijo_c=1e-4, armijo_tau=0.5, max_ls_steps=10) =
    RiemannianGD(lr, armijo_c, armijo_tau, max_ls_steps)

"""
    optimize!(opt::RiemannianGD, tensors, loss_fn, grad_fn; max_iter, tol, verbose)

Run Riemannian gradient descent on `tensors` using the manifold API.
Uses batched operations and Armijo line search for step size selection.
"""
function optimize!(
    opt::RiemannianGD,
    tensors::Vector{T},
    loss_fn,
    grad_fn;
    max_iter::Int = 100,
    tol::Real = 1e-6,
    verbose::Bool = false
) where T <: AbstractMatrix

    current_tensors = copy.(tensors)
    ET = eltype(T)
    RT = real(ET)

    # Classify manifold types ONCE
    manifold_groups = group_by_manifold(tensors)

    if verbose
        for (m, idx) in manifold_groups
            println("  $(typeof(m)): $(length(idx)) tensors")
        end
    end

    # Pre-loop: build persistent batched state + pre-allocate reusable buffers
    point_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
    grad_buf_batches = Dict{AbstractRiemannianManifold, AbstractArray}()

    for (manifold, indices) in manifold_groups
        n_m = length(indices)
        if n_m > 0
            pb = stack_tensors(current_tensors, indices)
            point_batches[manifold] = pb
            grad_buf_batches[manifold] = similar(pb)
        end
    end

    # Cache loss across iterations to avoid redundant forward passes
    cached_loss = RT(NaN)
    if verbose
        cached_loss = RT(loss_fn(current_tensors))
        println("  Initial loss: ", round(cached_loss, digits=6))
    end

    for iter in 1:max_iter
        # Unstack persistent batches for Zygote (which needs individual tensors)
        for (manifold, indices) in manifold_groups
            unstack_tensors!(current_tensors, point_batches[manifold], indices)
        end

        # Compute Euclidean gradients
        euclidean_grads = _compute_gradients(grad_fn, current_tensors, verbose, iter)
        euclidean_grads === nothing && break

        # Batched projection
        rg_batches, grad_norm = _batched_project(
            manifold_groups, point_batches, grad_buf_batches, euclidean_grads
        )

        grad_norm_sq = grad_norm^2

        # Check convergence
        if grad_norm < tol
            verbose && println("  Converged at iteration $iter (grad_norm = $grad_norm)")
            break
        end

        # Armijo backtracking line search
        current_loss = isnan(cached_loss) ? RT(loss_fn(current_tensors)) : cached_loss

        if verbose && (iter % 10 == 0 || iter == 1)
            println("  Iter $iter: loss = $(round(current_loss, digits=6)), grad_norm = $(round(grad_norm, digits=6)), lr = $(opt.lr)")
        end

        alpha = RT(opt.lr)
        accepted = false

        for _ls in 1:opt.max_ls_steps
            # Trial retraction at step size alpha
            cand_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
            for (manifold, indices) in manifold_groups
                pb = point_batches[manifold]
                rg = rg_batches[manifold]
                cand = retract(manifold, pb, .-rg, alpha)
                cand_batches[manifold] = cand
                unstack_tensors!(current_tensors, cand, indices)
            end

            candidate_loss = RT(loss_fn(current_tensors))

            # Armijo sufficient decrease condition
            if candidate_loss <= current_loss - RT(opt.armijo_c) * alpha * grad_norm_sq
                # Accept: update persistent batches and cache the loss
                for (manifold, _) in manifold_groups
                    point_batches[manifold] = cand_batches[manifold]
                end
                cached_loss = candidate_loss
                accepted = true
                break
            end
            alpha *= RT(opt.armijo_tau)
        end

        if !accepted
            # Fall back: take the smallest step tried
            for (manifold, indices) in manifold_groups
                pb = point_batches[manifold]
                rg = rg_batches[manifold]
                point_batches[manifold] = retract(manifold, pb, .-rg, alpha)
            end
            cached_loss = RT(NaN)  # unknown after fallback
        end
    end

    # Final unstack
    for (manifold, indices) in manifold_groups
        unstack_tensors!(current_tensors, point_batches[manifold], indices)
    end

    return current_tensors
end

# ============================================================================
# RiemannianAdam -- Riemannian Adam Optimizer
# ============================================================================

"""
    RiemannianAdam(; lr=0.001, betas=(0.9, 0.999), eps=1e-8)

Riemannian Adam optimizer (Becigneul & Ganea, 2019) using batched manifold operations.
"""
struct RiemannianAdam <: AbstractRiemannianOptimizer
    lr::Float64
    beta1::Float64
    beta2::Float64
    eps::Float64
end

RiemannianAdam(; lr=0.001, betas=(0.9, 0.999), eps=1e-8) =
    RiemannianAdam(lr, betas[1], betas[2], eps)

"""
    optimize!(opt::RiemannianAdam, tensors, loss_fn, grad_fn; max_iter, tol, verbose)

Run Riemannian Adam on `tensors` using the manifold API.
Maintains per-manifold first and second moment estimates as batched arrays.
"""
function optimize!(
    opt::RiemannianAdam,
    tensors::Vector{T},
    loss_fn,
    grad_fn;
    max_iter::Int = 100,
    tol::Real = 1e-6,
    verbose::Bool = false
) where T <: AbstractMatrix

    current_tensors = copy.(tensors)
    ET = eltype(T)
    RT = real(ET)
    beta1, beta2 = RT(opt.beta1), RT(opt.beta2)

    # Classify manifold types ONCE
    manifold_groups = group_by_manifold(tensors)

    if verbose
        for (m, idx) in manifold_groups
            println("  $(typeof(m)): $(length(idx)) tensors")
        end
    end

    # Pre-loop: build persistent batched state + pre-allocate reusable buffers
    point_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
    grad_buf_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
    dir_buf_batches = Dict{AbstractRiemannianManifold, AbstractArray}()

    # Adam state: first moment (complex) and second moment (real) per manifold
    m_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
    v_batches = Dict{AbstractRiemannianManifold, AbstractArray}()

    for (manifold, indices) in manifold_groups
        n_m = length(indices)
        if n_m > 0
            pb = stack_tensors(current_tensors, indices)
            point_batches[manifold] = pb
            grad_buf_batches[manifold] = similar(pb)
            dir_buf_batches[manifold] = similar(pb)

            # Initialize moments: zeros on the same device as input
            proto = tensors[indices[1]]
            m_batches[manifold] = similar(proto, ET, size(pb)...) .* false  # zeros on same device
            v_batches[manifold] = similar(proto, RT, size(pb)...) .* false
        end
    end

    if verbose
        initial_loss = loss_fn(current_tensors)
        println("  Initial loss: ", round(initial_loss, digits=6))
    end

    for iter in 1:max_iter
        # Unstack persistent batches for Zygote (which needs individual tensors)
        for (manifold, indices) in manifold_groups
            unstack_tensors!(current_tensors, point_batches[manifold], indices)
        end

        # Compute Euclidean gradients
        euclidean_grads = _compute_gradients(grad_fn, current_tensors, verbose, iter)
        euclidean_grads === nothing && break

        # Batched projection
        rg_batches, grad_norm = _batched_project(
            manifold_groups, point_batches, grad_buf_batches, euclidean_grads
        )

        if verbose && (iter % 10 == 0 || iter == 1)
            loss = loss_fn(current_tensors)
            println("  Iter $iter: loss = $(round(loss, digits=6)), grad_norm = $(round(grad_norm, digits=6)), lr = $(opt.lr)")
        end

        if grad_norm < tol
            verbose && println("  Converged at iteration $iter (grad_norm = $grad_norm)")
            break
        end

        # Bias correction factors
        bc1 = one(RT) - beta1^iter
        bc2 = one(RT) - beta2^iter

        # Per-manifold Adam update
        for (manifold, indices) in manifold_groups
            rg = rg_batches[manifold]
            m_state = m_batches[manifold]
            v_state = v_batches[manifold]
            dir_buf = dir_buf_batches[manifold]

            # In-place moment update (fused broadcasts)
            @. m_state = beta1 * m_state + (one(RT) - beta1) * rg
            @. v_state = beta2 * v_state + (one(RT) - beta2) * real(abs2(rg))

            # Bias-corrected direction (in-place into dir_buf)
            @. dir_buf = (m_state / bc1) / (sqrt(v_state / bc2) + RT(opt.eps))

            # Retract (use persistent point_batch directly)
            old_batch = point_batches[manifold]
            new_batch = retract(manifold, old_batch, .-dir_buf, opt.lr)

            # Transport momentum (re-project onto new tangent space)
            m_batches[manifold] = transport(manifold, old_batch, new_batch, m_state)
            point_batches[manifold] = new_batch
        end
    end

    # Final unstack
    for (manifold, indices) in manifold_groups
        unstack_tensors!(current_tensors, point_batches[manifold], indices)
    end

    return current_tensors
end
