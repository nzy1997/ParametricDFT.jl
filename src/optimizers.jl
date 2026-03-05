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
    _compute_gradients(grad_fn, tensors)

Compute Euclidean gradients via `grad_fn`. Returns `nothing` on NaN/Inf.
"""
function _compute_gradients(grad_fn, tensors)
    euclidean_grads_raw = grad_fn(tensors)
    raw = euclidean_grads_raw isa Tuple ? collect(euclidean_grads_raw) : euclidean_grads_raw

    # Build typed Vector, replacing any non-matrix ChainRules tangents (ZeroTangent, etc.)
    # with zero arrays. collect() first ensures the guard runs before typed conversion.
    euclidean_grads = AbstractMatrix[
        raw[i] isa AbstractMatrix ? raw[i] : fill!(similar(tensors[i]), zero(eltype(tensors[i])))
        for i in eachindex(raw)
    ]

    # Check for NaN/Inf in gradients
    if any(g -> !all(isfinite, g), euclidean_grads)
        return nothing
    end

    return euclidean_grads
end

"""
    _batched_project(manifold_groups, point_batches, grad_buf_batches, euclidean_grads)

Batched Riemannian projection. Returns `(rg_batches, grad_norm)`.
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

"""Riemannian gradient descent with Armijo backtracking line search."""
struct RiemannianGD <: AbstractRiemannianOptimizer
    lr::Float64
    armijo_c::Float64
    armijo_tau::Float64
    max_ls_steps::Int
end

RiemannianGD(; lr=0.01, armijo_c=1e-4, armijo_tau=0.5, max_ls_steps=10) =
    RiemannianGD(lr, armijo_c, armijo_tau, max_ls_steps)

"""
    optimize!(opt::RiemannianGD, tensors, loss_fn, grad_fn; max_iter=100, tol=1e-6, loss_trace=nothing)

Run Riemannian gradient descent with Armijo line search. Returns optimized tensors.
When `loss_trace::Vector{Float64}` is provided, per-iteration losses are appended to it.
"""
function optimize!(
    opt::RiemannianGD,
    tensors::Vector{T},
    loss_fn,
    grad_fn;
    max_iter::Int = 100,
    tol::Real = 1e-6,
    loss_trace::Union{Nothing, Vector{Float64}} = nothing
) where T <: AbstractMatrix

    current_tensors = copy.(tensors)
    ET = eltype(T)
    RT = real(ET)

    # Classify manifold types ONCE
    manifold_groups = group_by_manifold(tensors)

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

    for iter in 1:max_iter
        # Unstack persistent batches for Zygote (which needs individual tensors)
        for (manifold, indices) in manifold_groups
            unstack_tensors!(current_tensors, point_batches[manifold], indices)
        end

        # Compute Euclidean gradients
        euclidean_grads = _compute_gradients(grad_fn, current_tensors)
        euclidean_grads === nothing && break

        # Batched projection
        rg_batches, grad_norm = _batched_project(
            manifold_groups, point_batches, grad_buf_batches, euclidean_grads
        )

        grad_norm_sq = grad_norm^2

        # Check convergence
        if grad_norm < tol
            break
        end

        # Armijo backtracking line search
        current_loss = isnan(cached_loss) ? RT(loss_fn(current_tensors)) : cached_loss

        alpha = RT(opt.lr)
        accepted = false
        last_cand_batches = Dict{AbstractRiemannianManifold, AbstractArray}()

        for _ls in 1:opt.max_ls_steps
            # Trial retraction at step size alpha
            last_cand_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
            for (manifold, indices) in manifold_groups
                pb = point_batches[manifold]
                rg = rg_batches[manifold]
                cand = retract(manifold, pb, .-rg, alpha)
                last_cand_batches[manifold] = cand
                unstack_tensors!(current_tensors, cand, indices)
            end

            candidate_loss = RT(loss_fn(current_tensors))

            # Armijo sufficient decrease condition
            if candidate_loss <= current_loss - RT(opt.armijo_c) * alpha * grad_norm_sq
                # Accept: update persistent batches and cache the loss
                for (manifold, _) in manifold_groups
                    point_batches[manifold] = last_cand_batches[manifold]
                end
                cached_loss = candidate_loss
                accepted = true
                break
            end
            alpha *= RT(opt.armijo_tau)
        end

        if !accepted
            # Fall back: use the last candidates (smallest step actually tried)
            for (manifold, _) in manifold_groups
                point_batches[manifold] = last_cand_batches[manifold]
            end
            cached_loss = RT(NaN)  # unknown after fallback
        end

        # Record per-iteration loss
        if loss_trace !== nothing
            if !isnan(cached_loss)
                push!(loss_trace, Float64(cached_loss))
            else
                # Need to unstack for loss evaluation
                for (manifold, indices) in manifold_groups
                    unstack_tensors!(current_tensors, point_batches[manifold], indices)
                end
                push!(loss_trace, Float64(loss_fn(current_tensors)))
            end
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

"""Riemannian Adam optimizer (Becigneul & Ganea, 2019) with batched manifold operations."""
struct RiemannianAdam <: AbstractRiemannianOptimizer
    lr::Float64
    beta1::Float64
    beta2::Float64
    eps::Float64
end

RiemannianAdam(; lr=0.001, betas=(0.9, 0.999), eps=1e-8) =
    RiemannianAdam(lr, betas[1], betas[2], eps)

"""
    optimize!(opt::RiemannianAdam, tensors, loss_fn, grad_fn; max_iter=100, tol=1e-6, loss_trace=nothing)

Run Riemannian Adam with momentum transport. Returns optimized tensors.
When `loss_trace::Vector{Float64}` is provided, per-iteration losses are appended to it.
"""
function optimize!(
    opt::RiemannianAdam,
    tensors::Vector{T},
    loss_fn,
    grad_fn;
    max_iter::Int = 100,
    tol::Real = 1e-6,
    loss_trace::Union{Nothing, Vector{Float64}} = nothing
) where T <: AbstractMatrix

    current_tensors = copy.(tensors)
    ET = eltype(T)
    RT = real(ET)
    beta1, beta2 = RT(opt.beta1), RT(opt.beta2)

    # Classify manifold types ONCE
    manifold_groups = group_by_manifold(tensors)

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

    for iter in 1:max_iter
        # Unstack persistent batches for Zygote (which needs individual tensors)
        for (manifold, indices) in manifold_groups
            unstack_tensors!(current_tensors, point_batches[manifold], indices)
        end

        # Compute Euclidean gradients
        euclidean_grads = _compute_gradients(grad_fn, current_tensors)
        euclidean_grads === nothing && break

        # Batched projection
        rg_batches, grad_norm = _batched_project(
            manifold_groups, point_batches, grad_buf_batches, euclidean_grads
        )

        if grad_norm < tol
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

        # Record per-iteration loss
        if loss_trace !== nothing
            for (manifold, indices) in manifold_groups
                unstack_tensors!(current_tensors, point_batches[manifold], indices)
            end
            push!(loss_trace, Float64(loss_fn(current_tensors)))
        end
    end

    # Final unstack
    for (manifold, indices) in manifold_groups
        unstack_tensors!(current_tensors, point_batches[manifold], indices)
    end

    return current_tensors
end

