# GPU Optimizer Redesign: Clean Rewrite

**Date:** 2026-03-03
**Status:** Design (reviewed, corrections applied)
**Approach:** Delete existing optimizer/manifold/training flow, rebuild from scratch around materialized unitary + typed parameter groups.

## Problem Statement

The current optimizer has three performance bottlenecks on GPU:

1. **Einsum kernel overhead:** Each forward pass contracts ~40 small 2x2 tensors via separate GPU kernels. For a batch of B images, this is ~40B forward kernels + ~80B backward kernels through Zygote.
2. **Pack/unpack per iteration:** Tensors are unstacked from batched 3D arrays into individual matrices (for einsum) and re-stacked (for manifold ops) every iteration — ~80 small `copyto!` kernel launches.
3. **Stateless optimizer:** Adam momentum resets to zero each time `optimize!` is called (once per training batch), losing cross-batch momentum.

For m=n=5, batch=8, one optimizer step launches ~1,050 GPU kernels. Most are tiny and dominated by launch overhead (~5-10us each).

---

## What Gets Deleted vs Kept vs Rebuilt

### Kept Unchanged (do not modify)

These files are the foundation — they produce `optcode` and `tensors` that the new flow consumes:

| File | What it provides | Why kept |
|------|-----------------|----------|
| `src/qft.jl` | `qft_code(m,n)` → `(optcode, tensors)` | Circuit definition, no perf issue |
| `src/entangled_qft.jl` | `entangled_qft_code(m,n)` → `(optcode, tensors, n_entangle)` | Circuit definition |
| `src/tebd.jl` | `tebd_code(m,n)` → `(optcode, tensors, n_row, n_col)` | Circuit definition |
| `src/basis.jl` | `QFTBasis`, `EntangledQFTBasis`, `TEBDBasis` structs + `forward_transform`/`inverse_transform` | Public API, used by compression/serialization |
| `src/loss.jl` | Loss functions, `topk_truncate` + rrules, batched einsum helpers | Correct, has tested rrules |
| `src/serialization.jl` | `save_basis`/`load_basis` | Orthogonal to optimizer |
| `src/compression.jl` | `compress`/`recover` | Orthogonal to optimizer |
| `src/visualization.jl` | `plot_training_loss` etc. | Orthogonal to optimizer |
| `ext/CUDAExt.jl` | `batched_matmul`, `batched_inv`, `topk_truncate` GPU overrides | Already efficient, reused by new manifold ops |

### Deleted Entirely (remove all code)

| File | Why deleted |
|------|-------------|
| `src/manifolds.jl` | Rewritten from scratch: new `ManifoldParameterGroup` replaces Dict-based grouping, stack/unstack |
| `src/optimizers.jl` | Rewritten from scratch: stateful optimizer, takes `ManifoldParameters` |
| `src/training.jl` | Rewritten from scratch: new training loop built around materialized U |
| `src/materialized.jl` | Rewritten from scratch: expanded to primary path, simplified (no cache state) |

### Deleted Test Files (rewrite alongside new code)

| File | Why |
|------|-----|
| `test/optimizer_tests.jl` | Tests the old optimizer interface |
| `test/materialized_tests.jl` | Tests the old materialized helpers |

### Kept Test Files (may need minor include updates)

| File | Status |
|------|--------|
| `test/loss_tests.jl` | Keep — loss functions unchanged |
| `test/qft_tests.jl` | Keep — circuit generation unchanged |
| `test/entangled_qft_tests.jl` | Keep |
| `test/tebd_tests.jl` | Keep |
| `test/basis_tests.jl` | Keep |
| `test/serialization_tests.jl` | Keep |
| `test/compression_tests.jl` | Keep |
| `test/runtests.jl` | Update includes for new test files |

---

## The Interface Between Old and New

The circuit generators (`qft_code`, `entangled_qft_code`, `tebd_code`) return two things:

```julia
optcode, tensors = qft_code(m, n)
#  |         |
#  |         +-- Vector{Matrix{ComplexF64}} -- ~40 small 2x2 matrices
#  |            THE LEARNABLE PARAMETERS
#  |            (Hadamard gates, phase gates, entanglement gates)
#  |
#  +-- AbstractEinsum -- contraction tree
#     THE RECIPE: "contract tensor[3] with tensor[7] along index j, then..."
#     FIXED during training, never changes
```

The new optimizer flow consumes exactly these two outputs. Nothing else from the old code is needed.

---

## New Architecture

### Overview

```
Layer 1: ManifoldParameters          Layer 2: Materialized Unitary
(owns small tensors,                 (builds U from tensors each step,
 Riemannian geometry)                 fast forward/backward via U*X)

tensors --> ManifoldParameters       optcode + tensors --> build U
             |                                              |
             | project/retract/                             | U*X (1 GEMM)
             | transport on 2x2                             | Zygote through U*X
             | batched arrays                               |
             v                                              v
        updated tensors -----> next step ------> rebuild U from new tensors
```

**Key simplification from review:** No `CachedUnitary` with dirty flag. U is always rebuilt from tensors inside the loss function. This is necessary because:
1. Zygote must trace through `build_unitary` to compute dL/dtensors — the cache cannot be used in the AD path.
2. After retract, tensors change, so the cache would be dirty anyway.
3. Removing the cache eliminates a mutable state + conditional branch that adds complexity for negligible savings (~40 kernels once per GD step).

### Data Flow (one Adam optimizer step)

```
1. to_vector(params)                    Copy from batched 3D arrays to flat
                                        Vector{Matrix}. ~40 small copyto!
                                        calls (unavoidable: einsum needs
                                        individual matrices).
        |
        v
2. grad_fn(tensors_vec):                Inside Zygote trace:
   loss_fn(ts):
     U = build_unitary(optcode, ts)       a. einsum(ts..., I) -> U   [~40 kernels]
     loss = f(U * X)                      b. U * X                   [1 GEMM]
                                          c. f(result)               [~4 kernels]
   back(1.0):
     dL/d(U*X) -> dL/dU                  d. 1 GEMM
     dL/dU -> dL/dtensors                e. ~40 kernels (chain rule through einsum)
                                        Total: ~86 kernels
        |
        v
3. _normalize_gradients(grads)          Handle Zygote quirks:
                                        - Tuple -> Vector conversion
                                        - ZeroTangent -> zero matrix
                                        - NaN/Inf check -> return nothing
        |
        v
4. scatter_gradients!(params, grads)    Route per-tensor grads into group
                                        grad_bufs. ~40 small copyto!
        |
        v
5. For each ManifoldParameterGroup:     ~15 kernels per group
   rg = project(M, batch, grad_buf)       3 kernels (2x batched_matmul)
   m = beta1*m + (1-beta1)*rg             1 fused broadcast
   v = beta2*v + (1-beta2)*|rg|^2         1 fused broadcast
   dir = (m/bc1)/(sqrt(v/bc2)+eps)        1 fused broadcast
   new_batch = retract(M, batch, -dir, lr)  6 kernels (Cayley)
   m = transport(M, old, new, m)          3 kernels (re-projection)
        |
        v
6. Done. Next call to loss_fn will rebuild U from updated tensors.

Total per step: ~86 (grad) + ~40 (scatter) + ~15 (manifold) = ~141 kernels
(vs ~1,050 currently, ~7x reduction)
```

### Data Flow (one GD + Armijo step)

```
1-4. Same as Adam (compute grads + project)    ~129 kernels

5. current_loss = loss_fn(tensors_vec)         ~44 kernels (rebuild U + U*X)
   Note: U is rebuilt here. Cannot reuse from grad_fn because
   loss_fn is a pure function (Zygote requirement).

6. Armijo line search:
   saved_batches = [copy(b) for b in params.batches]  # save ONCE before loop

   For each trial (up to max_ls_steps):
     a. retract all groups from saved_batches   ~6 kernels/group
     b. to_vector(params)                       ~40 copyto!
     c. trial_loss = loss_fn(tensors_vec)       ~44 kernels (rebuild U + U*X)
     d. Check: trial_loss <= current - c*alpha*||g||^2
        If accept -> break
        If reject -> shrink alpha, continue

   Per trial: ~90 kernels
   3 trials: ~270 kernels

7. If no trial accepted: take smallest step from saved_batches.
   Restore saved_batches first, then retract with final alpha.

Total (3 trials): ~129 + 44 + 270 = ~443 kernels
(vs ~3,200 currently, ~7x reduction)
```

---

## Component Specifications

### Component 1: `src/manifolds.jl` (rewrite)

**Keep from old file:**
- `batched_matmul`, `batched_adjoint`, `batched_inv` — batched linear algebra primitives. CUDAExt overrides depend on their signatures.
- `AbstractRiemannianManifold`, `UnitaryManifold`, `PhaseManifold` types.
- `project`, `retract`, `transport` methods for both manifold types. The Riemannian geometry is correct.

**Delete from old file:**
- `classify_manifold`, `is_unitary_general`, `group_by_manifold` — replaced by `ManifoldParameterGroup` constructor.
- `stack_tensors`, `stack_tensors!`, `unstack_tensors!` — replaced by `to_vector` and `scatter_gradients!`.

**New code:**

```julia
# ============================================================================
# Typed Parameter Groups
# ============================================================================

"""A batch of tensors sharing the same manifold, stored as a 3D array."""
struct ManifoldParameterGroup{M <: AbstractRiemannianManifold}
    manifold::M
    indices::Vector{Int}    # positions in original tensor list
end

"""All circuit parameters grouped by manifold type, with buffers.
Parameterized on array type A to avoid type instability on GPU (CuArray vs Array)."""
mutable struct ManifoldParameters{T <: Number, A <: AbstractArray{T, 3}}
    groups::Vector{ManifoldParameterGroup}
    batches::Vector{A}        # (d1, d2, N) per group, concrete array type
    grad_bufs::Vector{A}      # pre-allocated gradient buffers, same type
    n_total::Int
end

"""
    ManifoldParameters(tensors::Vector{<:AbstractMatrix{T}})

Construct from flat tensor vector. Classifies each tensor by manifold type,
groups by type, and stacks into 3D batched arrays. Runs once at training init.
"""
function ManifoldParameters(tensors::Vector{<:AbstractMatrix{T}}) where T
    # Classify each tensor (on CPU to avoid GPU scalar indexing)
    manifold_map = Dict{AbstractRiemannianManifold, Vector{Int}}()
    for (i, t) in enumerate(tensors)
        m = _classify_manifold(t)
        push!(get!(manifold_map, m, Int[]), i)
    end

    # Sort groups by first index for deterministic ordering across Julia sessions
    sorted_pairs = sort(collect(manifold_map), by = p -> first(p[2]))

    groups = ManifoldParameterGroup[]
    batch_list = []
    grad_buf_list = []

    for (manifold, indices) in sorted_pairs
        push!(groups, ManifoldParameterGroup(manifold, indices))
        # Stack tensors into (d1, d2, N) array (same device as input)
        n = length(indices)
        d1, d2 = size(tensors[indices[1]])
        batch = similar(tensors[indices[1]], T, d1, d2, n)
        for (k, idx) in enumerate(indices)
            copyto!(view(batch, :, :, k), tensors[idx])
        end
        push!(batch_list, batch)
        push!(grad_buf_list, similar(batch))
    end

    # Convert to concrete-typed vectors
    A = typeof(batch_list[1])
    batches = convert(Vector{A}, batch_list)
    grad_bufs = convert(Vector{A}, grad_buf_list)

    return ManifoldParameters{T, A}(groups, batches, grad_bufs, length(tensors))
end

"""
    to_vector(mp::ManifoldParameters) -> Vector{<:AbstractMatrix}

Extract flat Vector of matrices from batched arrays. Allocates new matrices
each call (required: views would alias batch arrays that retract mutates).
~40 small copyto! calls for 2x2 matrices — unavoidable but cheap.

Returns matrices on the same device as the batches (CuMatrix on GPU, Matrix on CPU).
"""
function to_vector(mp::ManifoldParameters{T}) where T
    # Use the batch element type to determine matrix type (GPU-safe)
    proto = mp.batches[1]
    result = Vector{typeof(similar(proto, T, size(proto,1), size(proto,2)))}(undef, mp.n_total)
    for (g, group) in enumerate(mp.groups)
        batch = mp.batches[g]
        for (k, idx) in enumerate(group.indices)
            result[idx] = copy(view(batch, :, :, k))
        end
    end
    return result
end

"""
    scatter_gradients!(mp::ManifoldParameters, grads::Vector{<:AbstractMatrix})

Route per-tensor Euclidean gradients into group gradient buffers.
Single pass over indices, writes into pre-allocated grad_bufs.
"""
function scatter_gradients!(mp::ManifoldParameters, grads::Vector{<:AbstractMatrix})
    for (g, group) in enumerate(mp.groups)
        for (k, idx) in enumerate(group.indices)
            copyto!(view(mp.grad_bufs[g], :, :, k), grads[idx])
        end
    end
end

"""Classify a tensor as UnitaryManifold or PhaseManifold.
Runs on CPU to avoid GPU scalar indexing in isapprox.
Called once at training init, so CPU round-trip is negligible."""
function _classify_manifold(t::AbstractMatrix{T}) where T
    t_cpu = Array(t)  # move to CPU for safe element-wise comparison
    n = size(t_cpu, 1)
    if size(t_cpu, 2) == n && isapprox(t_cpu * t_cpu', I, atol=1e-6)
        return UnitaryManifold()
    else
        return PhaseManifold()
    end
end
```

**Design notes:**
- `to_vector` allocates new matrices each call. Views would alias `batches` which `retract` replaces, causing stale references. The cost is ~40 copies of 2x2 matrices = negligible.
- `ManifoldParameterGroup` does NOT store the batch — `ManifoldParameters` owns all batches. This keeps the group struct lightweight and avoids type parameter complexity.

### Component 2: `src/materialized.jl` (rewrite)

**Delete from old file:** Everything. Rebuild with simpler structure: no `CachedUnitary`, no dirty flag, no `select_device_strategy`.

**New code:**

```julia
"""
    build_unitary(optcode, tensors::Vector, m::Int, n::Int)

Build the full D x D unitary matrix from circuit gate tensors by applying the
circuit to all D standard basis vectors in a single batched einsum call.

Column j of U is the circuit applied to basis vector e_j.

IMPORTANT: This function must be called inside the Zygote-traced closure
so that dL/dtensors can be computed via chain rule through the einsum.
"""
function build_unitary(optcode, tensors::Vector, m::Int, n::Int)
    return build_unitary(optcode, Tuple(tensors), m, n)
end

function build_unitary(optcode, tensors::Tuple, m::Int, n::Int)
    T = eltype(tensors[1])
    D = 2^(m + n)
    # Build identity on CPU then move to same device as tensors.
    # This ensures OMEinsum receives consistent array types.
    I_cpu = Matrix{T}(I, D, D)
    I_device = convert(typeof(similar(tensors[1], T, D, D)), I_cpu)
    I_tensor = reshape(I_device, fill(2, m + n)..., D)
    U_tensor = optcode(tensors..., I_tensor)
    return reshape(U_tensor, D, D)
end

"""
    prepare_unitary_optcode(optcode, n_gates::Int, m::Int, n::Int)

Pre-optimize the batched einsum contraction code for building the D x D unitary.
Call once at training init; reuse the returned code for all build_unitary calls.
"""
function prepare_unitary_optcode(optcode, n_gates::Int, m::Int, n::Int)
    D = 2^(m + n)
    flat_batched, blabel = make_batched_code(optcode, n_gates)
    return optimize_batched_code(flat_batched, blabel, D)
end

# ============================================================================
# Materialized Loss Functions
# ============================================================================
# Dispatch on AbstractLoss type for clean extension.
# All functions are Zygote-safe: no mutation in the forward path.

"""Stack images as columns of a D x B matrix (mutation-free for AD).
Always returns a matrix, even for batch_size=1 (reduce(hcat, [v]) returns a vector)."""
function _stack_image_columns(images::Vector{<:AbstractMatrix})
    cols = vec.(images)
    return hcat(cols...)  # hcat(v) returns Dx1 matrix; reduce(hcat,[v]) returns vector
end

"""L1 loss: (1/B) * sum(|U * X|)"""
function materialized_loss(U::AbstractMatrix, images::Vector{<:AbstractMatrix},
                           ::L1Norm, m::Int, n::Int)
    X = _stack_image_columns(images)
    return sum(abs.(U * X)) / length(images)
end

"""L2 loss: (1/B) * sum(|U * X|^2)"""
function materialized_loss(U::AbstractMatrix, images::Vector{<:AbstractMatrix},
                           ::L2Norm, m::Int, n::Int)
    X = _stack_image_columns(images)
    return sum(abs2.(U * X)) / length(images)
end

"""
MSE loss: (1/B) * sum_i ||img_i - U' * truncate(U * img_i, k)||^2

Per-image truncation is required because topk_truncate selects different
indices for each image. The loop is Zygote-safe (scalar accumulation).

AD chain: tensors -> U (via build_unitary) -> U*X -> truncate -> U'*trunc -> loss
The topk_truncate rrule (defined in loss.jl, GPU override in CUDAExt.jl)
propagates gradients only through selected (kept) coefficients.
"""
function materialized_loss(U::AbstractMatrix, images::Vector{<:AbstractMatrix},
                           loss::MSELoss, m::Int, n::Int)
    B = length(images)
    X = _stack_image_columns(images)
    fft_batch = U * X       # D x B: all forward transforms at once
    U_adj = U'              # lazy adjoint, Zygote-differentiable

    total = zero(real(eltype(U)))
    for i in 1:B
        fft_col = reshape(fft_batch[:, i], 2^m, 2^n)
        fft_trunc = topk_truncate(fft_col, loss.k)
        recon = U_adj * vec(fft_trunc)
        total += sum(abs2.(vec(images[i]) .- recon))
    end
    return total / B
end

# ============================================================================
# Memory feasibility check
# ============================================================================

"""Check if materialized unitary fits in memory for given problem size."""
function materialized_feasible(m::Int, n::Int; max_memory_bytes::Int = 2^30)
    D = 2^(m + n)
    # D x D complex matrix: 16 bytes per element (ComplexF64)
    required = D * D * 16
    return required <= max_memory_bytes
end
```

**Design notes:**
- No `CachedUnitary` struct. The `build_unitary` function is pure — called inside the Zygote-traced closure every time. This is correct because Zygote must trace through it.
- `prepare_unitary_optcode` is called once at training init. It returns an optimized `AbstractEinsum` that is captured by the loss closure.
- The Vector -> Tuple conversion (`build_unitary(optcode, Tuple(tensors), m, n)`) is the same AD stability pattern used throughout the existing codebase (see `loss.jl:117-121`).
- `materialized_loss` dispatches on `AbstractLoss` subtype via multiple dispatch.
- MSE path: `U'` (lazy adjoint) is Zygote-differentiable. `topk_truncate` uses the existing custom rrule. The full chain `tensors -> U -> U*X -> truncate -> U'*trunc -> loss` is differentiable. Verified: the existing test `test/loss_tests.jl:230-254` confirms Zygote differentiates through `topk_truncate` in a similar chain.

### Component 3: `src/optimizers.jl` (rewrite)

**Delete from old file:** Everything.

**New code:**

```julia
abstract type AbstractRiemannianOptimizer end

# ============================================================================
# Gradient Normalization (handles Zygote tangent quirks)
# ============================================================================

"""
    _normalize_gradients(grads_raw, tensors::Vector{<:AbstractMatrix})

Normalize Zygote's raw gradient output to Vector{AbstractMatrix}.
Handles: Tuple -> Vector, ZeroTangent -> zero matrix, NaN/Inf -> nothing.

This is necessary because Zygote may return:
- A Tuple instead of Vector (when differentiating through splatted Tuple args)
- ZeroTangent for tensors that don't affect the loss
- NaN/Inf when the loss landscape has numerical issues
"""
function _normalize_gradients(grads_raw, tensors::Vector{<:AbstractMatrix})
    raw = grads_raw isa Tuple ? collect(grads_raw) : grads_raw

    euclidean_grads = AbstractMatrix[
        raw[i] isa AbstractMatrix ? raw[i] :
            similar(tensors[i]) .* false   # zero array on same device (no fill! mutation)
        for i in eachindex(raw)
    ]

    # NaN/Inf check (GPU-safe: all(isfinite.(g)) avoids scalar indexing on CuArray)
    if any(g -> !all(isfinite.(g)), euclidean_grads)
        return nothing
    end

    return euclidean_grads
end

# ============================================================================
# RiemannianAdam — Stateful optimizer
# ============================================================================

"""
Riemannian Adam optimizer (Becigneul & Ganea, 2019) with batched manifold ops.

State (moments, iteration counter) persists across optimize! calls,
so momentum carries across training batches. Call reset!() to clear state.
"""
mutable struct RiemannianAdam <: AbstractRiemannianOptimizer
    lr::Float64
    beta1::Float64
    beta2::Float64
    eps::Float64
    # Persistent state (lazily initialized on first optimize! call)
    # Uses Vector{Any} to allow Nothing -> concrete array type transition.
    # After init, elements are concrete (Array{T,3} or CuArray{T,3}).
    # Access is in a per-group loop (not inner hot loop), so the cost
    # of dynamic dispatch here is negligible vs the manifold ops.
    m_batches::Union{Nothing, Vector{Any}}   # first moments per group (complex)
    v_batches::Union{Nothing, Vector{Any}}    # second moments per group (real)
    iter::Int                                  # global step counter
end

RiemannianAdam(; lr=0.001, betas=(0.9, 0.999), eps=1e-8) =
    RiemannianAdam(lr, betas[1], betas[2], eps, nothing, nothing, 0)

"""Reset optimizer state (e.g., when starting a new training run)."""
function reset!(opt::RiemannianAdam)
    opt.m_batches = nothing
    opt.v_batches = nothing
    opt.iter = 0
end

"""
    optimize!(opt::RiemannianAdam, params, loss_fn, grad_fn; max_iter, tol)

Run Riemannian Adam. Mutates params.batches in-place. Returns params.

- loss_fn: (Vector{Matrix}) -> scalar. Not called directly by Adam (no line search).
- grad_fn: (Vector{Matrix}) -> Vector{AbstractMatrix} or nothing.
"""
function optimize!(
    opt::RiemannianAdam,
    params::ManifoldParameters{T},
    loss_fn,
    grad_fn;
    max_iter::Int = 100,
    tol::Real = 1e-6
) where T
    RT = real(T)
    beta1, beta2 = RT(opt.beta1), RT(opt.beta2)

    # Lazy init moments on first call
    if opt.m_batches === nothing
        opt.m_batches = [similar(b) .* false for b in params.batches]
        opt.v_batches = [similar(b, RT) .* false for b in params.batches]
    end

    for _ in 1:max_iter
        # 1. Extract flat tensor vector (allocates new matrices each call)
        tensors_vec = to_vector(params)

        # 2. Compute Euclidean gradients via Zygote
        euclidean_grads = grad_fn(tensors_vec)
        euclidean_grads === nothing && break

        # 3. Scatter into group gradient buffers
        scatter_gradients!(params, euclidean_grads)

        # 4. Increment iteration counter ONCE per step (not per group)
        opt.iter += 1
        bc1 = one(RT) - beta1 ^ opt.iter
        bc2 = one(RT) - beta2 ^ opt.iter

        # 5. Per-group Riemannian update
        grad_norm_sq = zero(RT)

        for (g, group) in enumerate(params.groups)
            M = group.manifold
            pb = params.batches[g]
            gb = params.grad_bufs[g]

            # Project Euclidean gradient onto tangent space
            rg = project(M, pb, gb)
            grad_norm_sq += real(sum(abs2, rg))

            # Moment updates (fused broadcasts, in-place)
            m_state = opt.m_batches[g]
            v_state = opt.v_batches[g]
            @. m_state = beta1 * m_state + (one(RT) - beta1) * rg
            @. v_state = beta2 * v_state + (one(RT) - beta2) * real(abs2(rg))

            # Bias-corrected direction
            dir_buf = @. (m_state / bc1) / (sqrt(v_state / bc2) + RT(opt.eps))

            # Retract along negative direction
            old_batch = pb
            new_batch = retract(M, old_batch, .-dir_buf, opt.lr)

            # Transport momentum to new tangent space
            opt.m_batches[g] = transport(M, old_batch, new_batch, m_state)
            params.batches[g] = new_batch
        end

        # Convergence check
        sqrt(grad_norm_sq) < tol && break
    end

    return params
end

# ============================================================================
# RiemannianGD — Gradient descent with Armijo line search (stateless)
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
    optimize!(opt::RiemannianGD, params, loss_fn, grad_fn; max_iter, tol)

Run Riemannian GD with Armijo line search. Mutates params.batches. Returns params.

- loss_fn: (Vector{Matrix}) -> scalar. Called for Armijo condition checks.
- grad_fn: (Vector{Matrix}) -> Vector{AbstractMatrix} or nothing.
"""
function optimize!(
    opt::RiemannianGD,
    params::ManifoldParameters{T},
    loss_fn,
    grad_fn;
    max_iter::Int = 100,
    tol::Real = 1e-6
) where T
    RT = real(T)

    for _ in 1:max_iter
        # 1. Compute gradients
        tensors_vec = to_vector(params)
        euclidean_grads = grad_fn(tensors_vec)
        euclidean_grads === nothing && break
        scatter_gradients!(params, euclidean_grads)

        # 2. Project onto tangent space, collect Riemannian gradients
        rg_list = AbstractArray[]
        grad_norm_sq = zero(RT)
        for (g, group) in enumerate(params.groups)
            rg = project(group.manifold, params.batches[g], params.grad_bufs[g])
            push!(rg_list, rg)
            grad_norm_sq += real(sum(abs2, rg))
        end
        sqrt(grad_norm_sq) < tol && break

        # 3. Current loss (rebuilds U via loss_fn)
        current_loss = RT(loss_fn(tensors_vec))

        # 4. Armijo line search
        # Save batches ONCE before the loop (not per trial)
        saved_batches = [copy(b) for b in params.batches]
        alpha = RT(opt.lr)
        accepted = false

        for _ in 1:opt.max_ls_steps
            # Trial: retract from SAVED batches (not accumulated)
            for (g, group) in enumerate(params.groups)
                params.batches[g] = retract(group.manifold, saved_batches[g],
                                            .-rg_list[g], alpha)
            end

            trial_vec = to_vector(params)
            trial_loss = RT(loss_fn(trial_vec))

            if trial_loss <= current_loss - RT(opt.armijo_c) * alpha * grad_norm_sq
                accepted = true
                break
            end

            alpha *= RT(opt.armijo_tau)
        end

        if !accepted
            # Take smallest step tried (alpha is already shrunk to minimum)
            for (g, group) in enumerate(params.groups)
                params.batches[g] = retract(group.manifold, saved_batches[g],
                                            .-rg_list[g], alpha)
            end
        end
    end

    return params
end
```

**Design notes:**
- `opt.iter` increments once per outer iteration, before the group loop. Bias correction `bc1`, `bc2` are computed once and shared across groups.
- `_normalize_gradients` is separate from `optimize!` — called in the `grad_fn` closure constructed by `training.jl`.
- Armijo saves batches once before the line search loop. Each trial retracts from `saved_batches` (not from the previous trial's result), so we always retract from the same base point with different step sizes.
- `to_vector` allocates new matrices each call. This is necessary because retract replaces `params.batches[g]`, which would invalidate views. Cost: ~40 copies of 2x2 matrices = negligible.

### Component 4: `src/training.jl` (rewrite)

**Delete from old file:** Everything.

**New code (structure):**

```julia
# ============================================================================
# Device Management
# ============================================================================

"""Move array to device. :gpu requires CUDA.jl via CUDAExt."""
to_device(x, ::Val{:cpu}) = x

"""Move array to CPU."""
to_cpu(x::AbstractArray) = Array(x)
to_cpu(x) = x

function to_device(x, device::Symbol)
    device === :cpu && return to_device(x, Val(:cpu))
    device === :gpu && hasmethod(to_device, Tuple{typeof(x), Val{:gpu}}) &&
        return to_device(x, Val(:gpu))
    device === :gpu && error("GPU requires CUDA.jl: `using CUDA`")
    error("Unknown device: $device")
end

# ============================================================================
# Core Training Loop
# ============================================================================

"""
Core training loop for all basis types.
Returns (final_tensors, best_val_loss, train_losses, val_losses, step_train_losses).

Key changes from old implementation:
1. Builds ManifoldParameters once at init (typed groups, no per-iter Dict lookups).
2. Forward/backward pass goes through materialized unitary U*X (1 GEMM)
   instead of per-image einsum contractions (~40 kernels per image).
3. Optimizer is stateful: Adam momentum persists across batches.
"""
function _train_basis_core(
    dataset::Vector{<:AbstractMatrix},
    optcode::OMEinsum.AbstractEinsum,
    inverse_code::OMEinsum.AbstractEinsum,
    initial_tensors::Vector,
    m::Int, n::Int,
    loss::AbstractLoss,
    epochs::Int,
    steps_per_image::Int,
    validation_split::Float64,
    shuffle::Bool,
    early_stopping_patience::Int,
    basis_name::String;
    save_loss_path::Union{Nothing, String} = nothing,
    optimizer::Union{Symbol, AbstractRiemannianOptimizer} = :gradient_descent,
    batch_size::Int = 1,
    device::Symbol = :cpu,
    checkpoint_interval::Int = 0,
    checkpoint_dir::Union{Nothing, String} = nothing,
    build_basis_fn::Union{Nothing, Function} = nothing
)
    # --- Setup (all ONCE) ---

    # Convert Symbol to optimizer type
    opt = if optimizer isa AbstractRiemannianOptimizer
        optimizer
    elseif optimizer === :adam
        RiemannianAdam(lr=0.001)
    elseif optimizer === :gradient_descent
        RiemannianGD(lr=0.01)
    else
        error("Unknown optimizer: $optimizer")
    end

    # Move tensors to device, build ManifoldParameters
    device_tensors = [to_device(Matrix{ComplexF64}(t), device) for t in initial_tensors]
    params = ManifoldParameters(device_tensors)

    # Pre-optimize the batched einsum code for building the unitary
    # (TreeSA optimization is expensive, do it once)
    unitary_optcode = prepare_unitary_optcode(optcode, length(initial_tensors), m, n)

    # Prepare dataset
    complex_dataset = [to_device(Complex{Float64}.(img), device) for img in dataset]
    n_images = length(complex_dataset)
    n_validation = max(1, round(Int, n_images * validation_split))
    indices = shuffle ? Random.shuffle(1:n_images) : collect(1:n_images)
    training_data = complex_dataset[indices[n_validation+1:end]]
    validation_data = complex_dataset[indices[1:n_validation]]
    batch_size = clamp(batch_size, 1, length(training_data))
    n_batches = ceil(Int, length(training_data) / batch_size)

    # Also prepare einsum-based batched code for validation loss
    # (validation doesn't need materialized path — runs once per epoch)
    # ... (same as current: per-image einsum for validation) ...

    # Tracking
    best_tensors = to_vector(params)
    best_val_loss = Inf
    patience_counter = 0
    train_losses = Float64[]
    val_losses = Float64[]
    step_train_losses = Float64[]
    loss_records = Dict{String, Any}[]
    global_step = 0

    # --- Training Loop ---

    for epoch in 1:epochs
        shuffle && epoch > 1 && Random.shuffle!(training_data)
        epoch_losses = Float64[]

        for batch_idx in 1:n_batches
            start_idx = (batch_idx - 1) * batch_size + 1
            end_idx = min(batch_idx * batch_size, length(training_data))
            batch = training_data[start_idx:end_idx]

            # Loss function: tensors -> build U -> materialized loss
            # This closure is traced by Zygote. build_unitary is INSIDE
            # the trace so dL/dtensors flows through the einsum.
            batch_loss_fn = ts -> begin
                U = build_unitary(unitary_optcode, ts, m, n)
                materialized_loss(U, batch, loss, m, n)
            end

            # Gradient function: wraps Zygote.pullback + normalization
            batch_grad_fn = ts -> begin
                _, back = Zygote.pullback(batch_loss_fn, ts)
                grads_raw = back(one(real(eltype(ts[1]))))[1]
                return _normalize_gradients(grads_raw, ts)
            end

            # Run optimizer (mutates params in-place)
            optimize!(opt, params, batch_loss_fn, batch_grad_fn;
                      max_iter=steps_per_image, tol=1e-8)

            # Track loss
            current_vec = to_vector(params)
            batch_loss = Float64(batch_loss_fn(current_vec))
            push!(epoch_losses, batch_loss)
            push!(step_train_losses, batch_loss)
            push!(loss_records, Dict("epoch" => epoch, "step" => batch_idx,
                                     "loss" => batch_loss))
            global_step += 1

            # Checkpoint if needed
            # ... (same pattern as current) ...
        end

        # Validation loss (per-image einsum, not materialized — runs once)
        val_loss = _compute_validation_loss(
            validation_data, to_vector(params), optcode, inverse_code, m, n, loss
        )
        avg_train = isempty(epoch_losses) ? Inf : sum(epoch_losses)/length(epoch_losses)
        push!(train_losses, avg_train)
        push!(val_losses, val_loss)

        # Early stopping
        if val_loss < best_val_loss
            best_val_loss = val_loss
            best_tensors = to_vector(params)
            patience_counter = 0
        else
            patience_counter += 1
            patience_counter >= early_stopping_patience && epoch > 1 && break
        end
    end

    final_tensors = [ComplexF64.(Array(t)) for t in best_tensors]

    if save_loss_path !== nothing
        _save_loss_history(save_loss_path, basis_name, loss_records,
                          train_losses, val_losses)
    end

    return final_tensors, best_val_loss, train_losses, val_losses, step_train_losses
end
```

**Design notes:**
- `batch_loss_fn` calls `build_unitary` inside the closure. Zygote traces through it every `grad_fn` call. This is ~40 einsum kernels per step, but only once (not per image).
- `batch_grad_fn` wraps `Zygote.pullback` + `_normalize_gradients` to handle tangent type quirks.
- `optimize!` no longer takes a `CachedUnitary` — the loss/grad functions are self-contained closures.
- Validation uses per-image einsum (not materialized). It runs once per epoch so performance doesn't matter.
- `train_basis` method signatures for `QFTBasis`, `EntangledQFTBasis`, `TEBDBasis` remain identical to current public API.

**CPU fallback:** When `device == :cpu`, the same code runs but with regular Arrays instead of CuArrays. The materialized path is still faster than per-image einsum on CPU for moderate D because it reduces the number of einsum calls from ~40*B to ~40 (build U) + 1 matmul. For very large D where the DxD matrix doesn't fit in cache, an einsum fallback can be added later.

---

## Kernel Count Summary (corrected)

### Adam, 1 step, batch=8, m=n=5

| Phase | Kernels | Source |
|-------|---------|--------|
| `to_vector` | ~40 | copyto! from batches |
| `build_unitary` (forward einsum) | ~40 | OMEinsum contraction tree |
| `U * X` forward | 1 | cuBLAS GEMM |
| Loss computation | ~4 | abs/abs2/sum broadcasts |
| Zygote backward: dL/d(U*X) -> dL/dU | ~2 | GEMM adjoints |
| Zygote backward: dL/dU -> dL/dtensors | ~40 | einsum backward |
| `scatter_gradients!` | ~40 | copyto! into grad_bufs |
| `project` (per group) | ~3 | batched matmul |
| Moment update (per group) | ~3 | fused broadcasts |
| `retract` (per group) | ~6 | Cayley: 3 matmul + 1 inv |
| `transport` (per group) | ~3 | re-projection |
| **Total** | **~182** | Assuming 2 manifold groups |

**vs current: ~1,050. Reduction: ~5.8x**

Note: The earlier estimate of ~91 was too optimistic — it undercounted `to_vector` and `scatter_gradients!` copy overhead. The actual improvement is still substantial.

### GD + Armijo (3 trials), batch=8

| Phase | Kernels |
|-------|---------|
| Gradient computation (same as Adam steps 1-7) | ~170 |
| `current_loss = loss_fn(tensors_vec)` | ~85 (rebuild U + U*X + loss) |
| Per trial: retract + to_vector + loss_fn | ~85 + ~6 + ~40 = ~131 |
| 3 trials | ~393 |
| **Total** | **~648** |

**vs current: ~3,200. Reduction: ~4.9x**

### Summary table

| Scenario | Current | New | Reduction |
|----------|---------|-----|-----------|
| Adam, 1 step, batch=8 | ~1,050 | ~182 | ~5.8x |
| GD + Armijo (3 trials), batch=8 | ~3,200 | ~648 | ~4.9x |
| Adam, 100 steps, batch=8 | ~105,000 | ~18,200 | ~5.8x |

Note: The main speedup comes from replacing per-image einsum (~40 kernels x B images)
with a single U*X GEMM. The pack/unpack and manifold ops are a smaller portion.
Actual wall-clock improvement depends on kernel launch overhead vs compute ratio.

---

## Memory Requirements

Materialized U is a D x D ComplexF64 matrix:

| Image size | D | U memory | Feasible? |
|-----------|---|----------|-----------|
| 4x4 (m=n=2) | 16 | 4 KB | Yes |
| 16x16 (m=n=4) | 256 | 1 MB | Yes |
| 32x32 (m=n=5) | 1,024 | 16 MB | Yes |
| 64x64 (m=n=6) | 4,096 | 256 MB | Yes |
| 128x128 (m=n=7) | 16,384 | 4 GB | Tight on consumer GPU |
| 256x256 (m=n=8) | 65,536 | 64 GB | No — need einsum fallback |

For the paper's target sizes (up to 64x64), materialized U fits comfortably.

---

## Known Risks and Mitigations

### Risk 1: MSE materialized gradient is untested

The existing tests verify L1 gradient through `build_circuit_unitary` + materialized loss. But MSE gradient through the full chain `tensors -> U -> U*X -> topk_truncate -> U'*trunc -> loss` has no dedicated test.

**Mitigation:** Add gradient correctness test (AD vs finite differences) for materialized MSE as the first test written.

### Risk 2: `topk_truncate` rrule in materialized chain

The `topk_truncate` rrule propagates gradients only through kept coefficients. In the materialized MSE path, this gradient must chain backward through `U'` and then through `build_unitary` to reach `tensors`. The rrule itself is correct, and Zygote's chain rule composition handles this — but the full chain is complex.

**Mitigation:** The finite-difference gradient test for MSE will catch any issues here.

### Risk 3: Adam behavioral change

Persistent momentum changes training dynamics. Loss curves will differ from the current implementation.

**Mitigation:** This is intentional and correct. Document the change. Run comparison experiments.

### Risk 4: Large D memory pressure

For D >= 16384 (128x128 images), the materialized U may not fit on GPU.

**Mitigation:** `materialized_feasible(m, n)` check. Fallback to einsum path for large D (can be added later, not needed for the paper's target sizes).

---

## Public API Compatibility

All `train_basis` signatures remain identical:

```julia
train_basis(QFTBasis, dataset; m, n, loss, epochs, steps_per_image,
            optimizer, batch_size, device, ...)
# Returns: (basis, history) -- unchanged
```

Symbol shortcuts `:gradient_descent`, `:adam` still work. Constructor `RiemannianGD()`, `RiemannianAdam()` still work (Adam now has additional internal state fields initialized to `nothing`).

**Breaking change:** `optimize!` now takes `ManifoldParameters` instead of `Vector{Matrix}`. Any code calling `optimize!` directly (not through `train_basis`) will need to wrap tensors in `ManifoldParameters` first. This is an internal API change — the public API (`train_basis`) is unaffected.

---

## Inherited Issues (known, deferred)

These exist in the current code and are inherited by the redesign. They are not blockers but should be addressed eventually:

1. **`retract` for `UnitaryManifold` allocates a CPU identity batch and transfers to GPU on every call.** For 2x2 matrices this is cheap (~0.01ms), but a cached device-resident identity batch would eliminate repeated CPU->GPU transfers. Fix: cache `I_batch` per (d, n, device) in a module-level dict, similar to `_freq_weight_cache` in CUDAExt.
2. **`Tuple(tensors)` allocation in `build_unitary` on every loss call.** Creates a ~40-element Tuple each time. Known pattern from existing codebase (loss.jl:120). Cost is small but present.

---

## Testing Strategy

1. **Materialized MSE gradient** (NEW, HIGH PRIORITY): AD vs finite differences through full `tensors -> U -> U*X -> truncate -> U'*trunc -> loss` chain.
2. **ManifoldParameters round-trip**: `to_vector(ManifoldParameters(tensors)) == tensors` (ordering preserved).
3. **scatter_gradients! correctness**: verify gradients land in correct group/position.
4. **Manifold properties**: unitarity after retract, tangent membership after project. (Same tests as current, adapted to new types.)
5. **Stateful Adam**: two consecutive `optimize!` calls — verify moments persist and `opt.iter` increments correctly.
6. **Adam iter counter**: verify `opt.iter` increments once per outer iteration, not once per group.
7. **Armijo rollback**: verify batches restored correctly when no trial is accepted.
8. **GPU/CPU parity**: same results on both devices (within floating-point tolerance).
9. **Convergence smoke test**: training loss decreases on a small problem.
10. **Benchmark**: kernel counts and wall-clock time via updated `profile_gpu.jl`.

---

## Git Workflow

### Step 1: Merge current work into main

```
main ◄── squash merge ── feature/riemannian-optimizers (42 commits)
```

Squash merge `feature/riemannian-optimizers` into `main` via PR.
This preserves the current working code (manifolds, optimizers, training,
materialized, CUDA extension, tests) as the baseline on `main`.

The 42 commits are squashed into a single clean commit on `main`.

### Step 2: Create redesign branch from main

```
main (with current code) ──► feature/gpu-optimizer-redesign
```

Branch from the updated `main`. The redesign work happens here.

### Step 3: Implement redesign in phases

Each phase is a logically complete unit with passing tests.
Commit after each phase so progress is never lost.

```
feature/gpu-optimizer-redesign
  ├── Phase 1: rewrite manifolds.jl + tests
  ├── Phase 2: rewrite materialized.jl + tests (including MSE gradient test)
  ├── Phase 3: rewrite optimizers.jl + tests
  ├── Phase 4: rewrite training.jl + integration tests
  ├── Phase 5: update exports, benchmarks, final verification
  └── PR back to main
```

### Step 4: Merge redesign into main

```
main ◄── PR ── feature/gpu-optimizer-redesign
```

Regular merge (not squash) to preserve the per-phase commit history,
which is useful for understanding the redesign if something breaks later.

---

## Implementation Order

1. `src/manifolds.jl` — rewrite with `ManifoldParameterGroup` (keep batched linalg + manifold geometry)
2. `src/materialized.jl` — rewrite with `build_unitary`, `materialized_loss`, no cache
3. `src/optimizers.jl` — rewrite with stateful Adam, GD + Armijo, `_normalize_gradients`
4. `src/training.jl` — rewrite `_train_basis_core` + all `train_basis` methods + helpers
5. `src/ParametricDFT.jl` — update exports
6. Tests: manifold group tests, materialized gradient tests (especially MSE), optimizer state tests, training integration tests
7. Update `examples/profile_gpu.jl` benchmark for new interface
