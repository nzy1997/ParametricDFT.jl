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
    OptimizationState{ET, RT}

Bundles shared loop state built by `_common_setup`. Holds manifold groupings,
batched point/gradient buffers, the identity-batch cache for Cayley retraction,
and a per-tensor Euclidean-gradient buffer that is reused across iterations.
"""
struct OptimizationState{ET, RT}
    manifold_groups::Dict{AbstractRiemannianManifold, Vector{Int}}
    point_batches::Dict{AbstractRiemannianManifold, AbstractArray}
    grad_buf_batches::Dict{AbstractRiemannianManifold, AbstractArray}
    ibatch_cache::Dict{AbstractRiemannianManifold, AbstractArray}
    current_tensors::Vector{<:AbstractMatrix}
    # Pre-allocated Vector{AbstractMatrix} reused by `_compute_gradients!` each
    # iteration. Eliminates the per-iteration `AbstractMatrix[...]` comprehension.
    euclidean_grads_buf::Vector{AbstractMatrix}
end

_element_type(::OptimizationState{ET, RT}) where {ET, RT} = ET
_real_type(::OptimizationState{ET, RT}) where {ET, RT} = RT

"""
    _common_setup(tensors)

Build an `OptimizationState` from the initial tensor list. Groups tensors by
manifold, stacks into batched arrays, allocates gradient buffers, and creates
identity-batch caches for `UnitaryManifold` groups via `_make_identity_batch`.
"""
function _common_setup(tensors::Vector{T}) where T <: AbstractMatrix
    current_tensors = copy.(tensors)
    ET = eltype(T)
    RT = real(ET)

    manifold_groups = group_by_manifold(tensors)

    point_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
    grad_buf_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
    ibatch_cache = Dict{AbstractRiemannianManifold, AbstractArray}()

    for (manifold, indices) in manifold_groups
        n_m = length(indices)
        if n_m > 0
            pb = stack_tensors(current_tensors, indices)
            point_batches[manifold] = pb
            grad_buf_batches[manifold] = similar(pb)
            if manifold isa UnitaryManifold
                d = size(pb, 1)
                I_b = _make_identity_batch(ET, d, n_m)
                ibatch_cache[manifold] = convert(typeof(pb), I_b)
            end
        end
    end

    # Pre-allocate the Vector{AbstractMatrix} wrapper used by _compute_gradients!.
    # Entries are placeholders; they are overwritten each iteration.
    euclidean_grads_buf = AbstractMatrix[similar(t) for t in current_tensors]

    return OptimizationState{ET, RT}(
        manifold_groups, point_batches, grad_buf_batches, ibatch_cache,
        current_tensors, euclidean_grads_buf,
    )
end

"""
    _compute_gradients!(buf, grad_fn, tensors)

Compute Euclidean gradients via `grad_fn`, writing into the pre-allocated
`buf::Vector{AbstractMatrix}` (typically `state.euclidean_grads_buf`) to
avoid per-iteration wrapper allocation. Returns `buf` on success, `nothing`
on NaN/Inf (after logging which tensor carried the non-finite value).
"""
function _compute_gradients!(buf::Vector{AbstractMatrix}, grad_fn, tensors)
    euclidean_grads_raw = grad_fn(tensors)
    raw = euclidean_grads_raw isa Tuple ? collect(euclidean_grads_raw) : euclidean_grads_raw

    # Overwrite buf in place. For non-matrix ChainRules tangents (ZeroTangent,
    # NoTangent) substitute a zero array sized like the corresponding tensor.
    for i in eachindex(raw)
        buf[i] = raw[i] isa AbstractMatrix ? raw[i] :
                 fill!(similar(tensors[i]), zero(eltype(tensors[i])))
    end

    # Per-tensor NaN/Inf diagnostic. Walk once; on the first non-finite tensor,
    # report which index and the NaN/Inf counts, then stop.
    for (i, g) in enumerate(buf)
        if !all(isfinite, g)
            n_nan = count(isnan, g)
            n_inf = count(isinf, g)
            @warn "Non-finite gradient — optimizer will stop" tensor_index=i n_nan=n_nan n_inf=n_inf
            return nothing
        end
    end

    return buf
end

"""
    _compute_gradients(grad_fn, tensors)

Allocating wrapper used outside the main optimization loop. Prefer
`_compute_gradients!` inside the loop, where a buffer is already available.
"""
function _compute_gradients(grad_fn, tensors)
    buf = AbstractMatrix[similar(t) for t in tensors]
    return _compute_gradients!(buf, grad_fn, tensors)
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

# ============================================================================
# Per-Optimizer State Initialization
# ============================================================================

"""
    _init_optimizer_state(opt::AbstractRiemannianOptimizer, state::OptimizationState)

Initialize per-optimizer state. Returns `nothing` for GD, a NamedTuple of
moment/direction buffers for Adam.
"""
_init_optimizer_state(::RiemannianGD, ::OptimizationState) = nothing

function _init_optimizer_state(::RiemannianAdam, state::OptimizationState{ET, RT}) where {ET, RT}
    m_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
    v_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
    dir_buf_batches = Dict{AbstractRiemannianManifold, AbstractArray}()

    for (manifold, indices) in state.manifold_groups
        n_m = length(indices)
        if n_m > 0
            pb = state.point_batches[manifold]
            dir_buf_batches[manifold] = similar(pb)

            # Initialize moments: zeros on the same device as input
            proto = state.current_tensors[indices[1]]
            m_batches[manifold] = similar(proto, ET, size(pb)...) .* false
            v_batches[manifold] = similar(proto, RT, size(pb)...) .* false
        end
    end

    return (m_batches=m_batches, v_batches=v_batches, dir_buf_batches=dir_buf_batches)
end

# ============================================================================
# Per-Optimizer Update Steps
# ============================================================================

"""
    _update_step!(opt, state, rg_batches, loss_fn, grad_norm_sq, opt_state, iter; cached_loss)

Per-optimizer update dispatch. Returns `cached_loss::RT` (NaN when not evaluated).

- `RiemannianGD`: Armijo backtracking line search, evaluates loss multiple times.
  Returns the accepted candidate loss, or `RT(NaN)` after exhausting line-search steps.
- `RiemannianAdam`: moment update + retract, does not evaluate loss. Returns `RT(NaN)`.
"""
function _update_step!(
    opt::RiemannianGD,
    state::OptimizationState{ET, RT},
    rg_batches,
    loss_fn,
    grad_norm_sq,
    ::Nothing,  # GD has no opt_state
    iter;
    cached_loss::RT
) where {ET, RT}
    current_loss = isnan(cached_loss) ? RT(loss_fn(state.current_tensors)) : cached_loss

    alpha = RT(opt.lr)
    accepted = false
    # Reused across every line-search trial: keys are manifold types (stable
    # across trials), so overwriting entries is equivalent to fresh allocation.
    last_cand_batches = Dict{AbstractRiemannianManifold, AbstractArray}()

    for _ls in 1:opt.max_ls_steps
        for (manifold, indices) in state.manifold_groups
            pb = state.point_batches[manifold]
            rg = rg_batches[manifold]
            ib = get(state.ibatch_cache, manifold, nothing)
            cand = retract(manifold, pb, .-rg, alpha; I_batch=ib)
            last_cand_batches[manifold] = cand
            unstack_tensors!(state.current_tensors, cand, indices)
        end

        candidate_loss = RT(loss_fn(state.current_tensors))

        if candidate_loss <= current_loss - RT(opt.armijo_c) * alpha * grad_norm_sq
            for (manifold, _) in state.manifold_groups
                state.point_batches[manifold] = last_cand_batches[manifold]
            end
            return candidate_loss
        end
        alpha *= RT(opt.armijo_tau)
    end

    # Fallback: use last candidates (smallest step tried)
    for (manifold, _) in state.manifold_groups
        state.point_batches[manifold] = last_cand_batches[manifold]
    end
    return RT(NaN)
end

function _update_step!(
    opt::RiemannianAdam,
    state::OptimizationState{ET, RT},
    rg_batches,
    loss_fn,
    grad_norm_sq,
    opt_state,
    iter;
    cached_loss::RT
) where {ET, RT}
    beta1, beta2 = RT(opt.beta1), RT(opt.beta2)
    bc1 = one(RT) - beta1^iter
    bc2 = one(RT) - beta2^iter

    m_batches = opt_state.m_batches
    v_batches = opt_state.v_batches
    dir_buf_batches = opt_state.dir_buf_batches

    for (manifold, _indices) in state.manifold_groups
        rg = rg_batches[manifold]
        m_state = m_batches[manifold]
        v_state = v_batches[manifold]
        dir_buf = dir_buf_batches[manifold]

        # In-place moment update (fused broadcasts)
        @. m_state = beta1 * m_state + (one(RT) - beta1) * rg
        @. v_state = beta2 * v_state + (one(RT) - beta2) * real(abs2(rg))

        # Bias-corrected direction (in-place into dir_buf)
        @. dir_buf = (m_state / bc1) / (sqrt(v_state / bc2) + RT(opt.eps))

        # Retract
        old_batch = state.point_batches[manifold]
        ib = get(state.ibatch_cache, manifold, nothing)
        new_batch = retract(manifold, old_batch, .-dir_buf, opt.lr; I_batch=ib)

        # Transport momentum (re-project onto new tangent space)
        m_batches[manifold] = transport(manifold, old_batch, new_batch, m_state)
        state.point_batches[manifold] = new_batch
    end

    return RT(NaN)
end

# ============================================================================
# Shared Optimization Loop
# ============================================================================

"""
    _optimization_loop(opt, tensors, loss_fn, grad_fn; max_iter, tol, loss_trace)

Shared optimization loop. Delegates setup/gradient/convergence logic once,
then calls `_update_step!` for per-optimizer behavior each iteration.
Returns the optimized tensor vector.
"""
function _optimization_loop(
    opt::AbstractRiemannianOptimizer,
    tensors::Vector{<:AbstractMatrix},
    loss_fn,
    grad_fn;
    max_iter::Int = 100,
    tol::Real = 1e-6,
    loss_trace::Union{Nothing, Vector{Float64}} = nothing
)
    state = _common_setup(tensors)
    ET = _element_type(state)
    RT = _real_type(state)
    opt_state = _init_optimizer_state(opt, state)

    cached_loss = RT(NaN)

    for iter in 1:max_iter
        # Unstack persistent batches for Zygote (which needs individual tensors)
        for (manifold, indices) in state.manifold_groups
            unstack_tensors!(state.current_tensors, state.point_batches[manifold], indices)
        end

        # Compute Euclidean gradients (in-place into pre-allocated buffer)
        euclidean_grads = _compute_gradients!(
            state.euclidean_grads_buf, grad_fn, state.current_tensors
        )
        euclidean_grads === nothing && break

        # Batched projection
        rg_batches, grad_norm = _batched_project(
            state.manifold_groups, state.point_batches, state.grad_buf_batches, euclidean_grads
        )

        grad_norm_sq = grad_norm^2

        # Check convergence
        if grad_norm < tol
            break
        end

        # Per-optimizer update
        cached_loss = _update_step!(
            opt, state, rg_batches, loss_fn, grad_norm_sq, opt_state, iter;
            cached_loss=cached_loss
        )

        # Record per-iteration loss
        if loss_trace !== nothing
            if !isnan(cached_loss)
                push!(loss_trace, Float64(cached_loss))
            else
                # Need to unstack for loss evaluation
                for (manifold, indices) in state.manifold_groups
                    unstack_tensors!(state.current_tensors, state.point_batches[manifold], indices)
                end
                push!(loss_trace, Float64(loss_fn(state.current_tensors)))
            end
        end
    end

    # Final unstack
    for (manifold, indices) in state.manifold_groups
        unstack_tensors!(state.current_tensors, state.point_batches[manifold], indices)
    end

    return state.current_tensors
end

# ============================================================================
# Unified optimize! entry point
# ============================================================================

"""
    optimize!(opt::AbstractRiemannianOptimizer, tensors, loss_fn, grad_fn; max_iter=100, tol=1e-6, loss_trace=nothing)

Run Riemannian optimization on circuit `tensors`. Dispatches to `_optimization_loop`
which uses per-optimizer hooks (`_init_optimizer_state`, `_update_step!`).
Returns optimized tensors.

When `loss_trace::Vector{Float64}` is provided, per-iteration losses are appended to it.
"""
function optimize!(
    opt::AbstractRiemannianOptimizer,
    tensors::Vector{<:AbstractMatrix},
    loss_fn,
    grad_fn;
    max_iter::Int = 100,
    tol::Real = 1e-6,
    loss_trace::Union{Nothing, Vector{Float64}} = nothing
)
    return _optimization_loop(opt, tensors, loss_fn, grad_fn;
                              max_iter=max_iter, tol=tol, loss_trace=loss_trace)
end

