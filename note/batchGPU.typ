#import "@preview/cetz:0.4.2": canvas, draw
#show link: set text(blue)

// Tensor box helper for diagrams (same as main.typ / stepbystep.typ)
#let ngate(pos, n, name, text: none, width: 1, gap-y: 1, padding-y: 0.25) = {
  import draw: *
  let height = gap-y * (n - 1) + 2 * padding-y
  group(name: name, {
    rect((rel: (- width/2, -height/2), to: pos), (rel: (width/2, height/2), to: pos), fill: white, name: "body")
    if text != none { content("body", text) }
    for i in range(n){
      let y = height/2 - padding-y - i * gap-y
      anchor("i" + str(i), (rel: (-width/2, y), to: pos))
      anchor("o" + str(i), (rel: (width/2, y), to: pos))
    }
    anchor("b", (rel: (0, -height/2), to: pos))
    anchor("t", (rel: (0, height/2), to: pos))
  })
}

= Batched Riemannian Optimization and GPU Acceleration

This document describes how ParametricDFT.jl implements efficient Riemannian optimization for quantum circuit parameters. Three problems with the Manopt.jl baseline motivate the design: (1) `ProductManifold` iterates over tensors one-by-one, preventing batched operations; (2) Manopt lacks Riemannian Adam or any adaptive optimizer; (3) `ArrayPartition` is incompatible with CUDA, blocking GPU acceleration. The solutions --- batched manifold operations, custom `RiemannianGD`/`RiemannianAdam` optimizers, batched einsum contractions, and GPU dispatch via `CUDAExt.jl` --- are presented below.

= Notation

We use the following notation consistently:
- $bold(theta)_k in CC^(d times d)$: the $k$-th gate tensor (circuit parameter), $k = 1, dots, K$
- $bold(X) in CC^(2 times dots.c times 2)$: input image reshaped to $m+n$ qubit dimensions (each of size 2)
- $bold(Y)$: output of the forward transform
- $cal(L)$: loss function value
- $bold(g)_k$: Euclidean gradient of $cal(L)$ with respect to $bold(theta)_k$
- $tilde(bold(g))_k = "proj"(bold(theta)_k, bold(g)_k)$: Riemannian gradient (projection of $bold(g)_k$ onto the tangent space at $bold(theta)_k$)
- $"Retr"_(bold(theta))(bold(xi))$: retraction from $bold(theta)$ along tangent vector $bold(xi)$ (maps back to the manifold)
- $Gamma_(bold(theta) arrow.r bold(theta)')$: parallel transport from tangent space at $bold(theta)$ to tangent space at $bold(theta)'$
- $alpha$: step size (learning rate)
- $K$: total number of gate tensors in the circuit
- $B$: batch size (number of images processed in one einsum call)
- $d$: gate matrix dimension (typically $d = 2$ for qubit gates)

= Batched Manifold Operations (`manifolds.jl`)

== Key Insight

Circuit tensors are either $U(2)$ unitaries or $U(1)^4$ phase gates. Instead of applying manifold operations to each tensor individually (as Manopt.jl does), we group all same-type tensors into a single 3D array of shape $(d, d, K)$ and apply manifold operations to the whole batch:

```
Before (Manopt.jl):                After (custom):
  for each tensor:                   group_by_manifold(tensors)
    project(manifold_i, tensor_i)    stack_tensors → (d, d, K)
    retract(manifold_i, tensor_i)    project(manifold, batch)   # one call
                                     retract(manifold, batch)   # one call
                                     unstack_tensors!
```

The function `group_by_manifold(tensors)` returns a `Dict{AbstractRiemannianManifold, Vector{Int}}` mapping each manifold type to the indices of its tensors. Then `stack_tensors` packs those indices into a $(d, d, K)$ array, and `unstack_tensors!` writes the results back.

== Batched Linear Algebra

Three batched operations form the building blocks for manifold operations:

#table(
  columns: 3,
  [*Function*], [*Signature*], [*Operation*],
  [`batched_matmul(A, B)`],
  [$(d_1, d_2, n) times (d_2, d_3, n) arrow.r (d_1, d_3, n)$],
  [$bold(C)_(:,:,k) = bold(A)_(:,:,k) bold(B)_(:,:,k)$ for each slice $k$],
  [`batched_adjoint(A)`],
  [$(d_1, d_2, n) arrow.r (d_2, d_1, n)$],
  [$bold(C)_(:,:,k) = overline(bold(A)_(:,:,k))^top$ via `permutedims(conj.(), (2,1,3))`],
  [`batched_inv(A)`],
  [$(d, d, n) arrow.r (d, d, n)$],
  [$bold(C)_(:,:,k) = bold(A)_(:,:,k)^(-1)$ via LU factorization],
)

All three accept `AbstractArray{T,3}`, so they dispatch correctly to both CPU arrays and GPU `CuArray`s. See @gpu for GPU-specialized implementations.

== Tensor Packing and Unpacking

Two functions convert between individual matrices and batched 3D arrays:

#line(length: 100%)

```julia
# Pack: selected matrices → (d, d, K) batch
batch = stack_tensors(tensors, indices)     # allocating
stack_tensors!(batch, tensors, indices)     # in-place (reuses buffer)

# Unpack: (d, d, K) batch → individual matrices
unstack_tensors!(tensors, batch, indices)   # writes back in-place
```

#line(length: 100%)

The packing/unpacking boundary is critical: Zygote AD requires individual tensors (to compute per-tensor gradients), but manifold operations need batched arrays (for efficiency). The optimizer loop crosses this boundary every iteration.

== `UnitaryManifold` --- $U(n)$ Unitary Group

*Projection*: Map an Euclidean gradient $bold(G)$ at point $bold(U)$ to the tangent space $T_(bold(U)) U(n)$ (the space of matrices $bold(xi)$ such that $bold(U)^dagger bold(xi)$ is skew-Hermitian):

$ "proj"(bold(U), bold(G)) = bold(U) dot "skew"(bold(U)^dagger bold(G)), #h(2em) "skew"(bold(A)) = (bold(A) - bold(A)^dagger) / 2 $

#line(length: 100%)

```julia
function project(::UnitaryManifold, U, G)   # (d, d, K) arrays
    UhG = batched_matmul(batched_adjoint(U), G)
    S = (UhG .- batched_adjoint(UhG)) ./ 2  # skew-Hermitian part
    return batched_matmul(U, S)
end
```

#line(length: 100%)

*Cayley retraction*: Move from $bold(U)$ along tangent vector $bold(xi)$ by step $alpha$, landing back on $U(n)$:

$ "Retr"_(bold(U))(bold(xi)) = (bold(I) - alpha/2 dot bold(W))^(-1)(bold(I) + alpha/2 dot bold(W)) dot bold(U), #h(2em) bold(W) = "skew"(bold(xi) bold(U)^dagger) $

The skew-projection of $bold(W) = bold(xi) bold(U)^dagger$ ensures correctness even when $bold(xi)$ is not exactly tangent --- this matters for `RiemannianAdam`, where the bias-corrected direction involves element-wise scaling that can push the update slightly off the tangent space.

The `I_batch` keyword argument accepts a pre-allocated $(d, d, K)$ identity array to avoid repeated allocation in optimizer inner loops (e.g. the Armijo line search tries multiple step sizes):

#line(length: 100%)

```julia
function retract(::UnitaryManifold, U, Xi, α; I_batch=nothing)
    W_raw = batched_matmul(Xi, batched_adjoint(U))
    W = (W_raw .- batched_adjoint(W_raw)) ./ 2   # skew projection
    I_b = I_batch === nothing ? _make_identity_batch(T, d, n) : I_batch
    lhs = I_b .- (α/2) .* W
    rhs = I_b .+ (α/2) .* W
    return batched_matmul(batched_matmul(batched_inv(lhs), rhs), U)
end
```

#line(length: 100%)

*Parallel transport*: Transport vector $bold(v)$ from the tangent space at $bold(U)$ to $bold(U')$ via re-projection:

$ Gamma_(bold(U) arrow.r bold(U)') (bold(v)) = "proj"(bold(U)', bold(v)) $

== `PhaseManifold` --- $U(1)^d$ Product of Unit Circles

For diagonal phase gates, each element $z_j$ lives on the unit circle $U(1) = {z in CC : |z| = 1}$.

*Projection*: $ "proj"(bold(z), bold(g)) = i dot "Im"(overline(bold(z)) circle.stroked.tiny bold(g)) circle.stroked.tiny bold(z) $

*Retraction*: $ "Retr"_(bold(z))(bold(xi)) = (bold(z) + alpha bold(xi)) ⊘ |bold(z) + alpha bold(xi)| $ (element-wise normalize)

*Transport*: $ Gamma_(bold(z) arrow.r bold(z)') (bold(v)) = "proj"(bold(z)', bold(v)) $

All three are pure element-wise broadcasts --- no matrix multiplication needed.

== OptimizationState Data Flow

The following diagram shows how data moves between Zygote (which needs individual tensors for AD) and the batched optimizer:

#figure(canvas(length: 0.85cm, {
  import draw: *

  // Individual tensors (top-left)
  rect((-1.5, 1.5), (1.5, 2.5), fill: blue.lighten(85%), name: "tensors")
  content("tensors", text(8pt)[$bold(theta)_1, bold(theta)_2, dots, bold(theta)_K$])
  content((0, 3.0), text(8pt, weight: "bold")[Individual Tensors])

  // group_by_manifold
  rect((-1.5, -0.3), (1.5, 0.7), fill: green.lighten(85%), name: "group")
  content("group", text(7pt)[`group_by_manifold`])
  line((0, 1.5), (0, 0.7), stroke: gray, mark: (end: ">", size: 0.2))

  // Two batched groups (right side)
  rect((3.0, 1.0), (7.0, 2.0), fill: orange.lighten(85%), name: "ubatch")
  content("ubatch", text(7pt)[UnitaryManifold: $(d, d, K_U)$])
  rect((3.0, -0.5), (7.0, 0.5), fill: purple.lighten(85%), name: "pbatch")
  content("pbatch", text(7pt)[PhaseManifold: $(d, d, K_P)$])

  // stack_tensors arrows
  line((1.5, 0.5), (3.0, 1.5), stroke: gray, mark: (end: ">", size: 0.2))
  line((1.5, 0.0), (3.0, 0.0), stroke: gray, mark: (end: ">", size: 0.2))
  content((2.3, 1.3), text(6pt)[`stack`])
  content((2.3, -0.3), text(6pt)[`stack`])

  // Manifold ops
  rect((3.0, -2.5), (7.0, -1.5), fill: red.lighten(85%), name: "ops")
  content("ops", text(7pt)[`project` / `retract` / `transport`])
  line((5.0, -0.5), (5.0, -1.5), stroke: gray, mark: (end: ">", size: 0.2))

  // unstack back
  rect((-1.5, -2.5), (1.5, -1.5), fill: blue.lighten(85%), name: "tensors2")
  content("tensors2", text(8pt)[$bold(theta)'_1, bold(theta)'_2, dots, bold(theta)'_K$])
  line((3.0, -2.0), (1.5, -2.0), stroke: blue, mark: (end: ">", size: 0.2))
  content((2.3, -1.7), text(6pt, fill: blue)[`unstack`])

  // Labels
  content((-2.5, -0.8), text(7pt, fill: gray.darken(30%), style: "italic")[Zygote needs\ individual tensors])
  content((8.2, -0.8), text(7pt, fill: gray.darken(30%), style: "italic")[Manifold ops\ need batches])

}), caption: [OptimizationState data flow. `group_by_manifold` partitions tensor indices by type. `stack_tensors` packs each group into a $(d, d, K)$ batch for efficient manifold operations. `unstack_tensors!` writes updated values back for the next Zygote AD pass.])

= Custom Optimizers (`optimizers.jl`)

== `OptimizationState` --- Shared Loop State

All optimizer iterations share a single `OptimizationState` struct that bundles the batched arrays and bookkeeping:

#table(
  columns: 2,
  [*Field*], [*Purpose*],
  [`manifold_groups`\ `Dict{Manifold, Vector{Int}}`],
  [Maps each manifold type to the indices of its tensors. Created once by `group_by_manifold` and never changes.],
  [`point_batches`\ `Dict{Manifold, Array{T,3}}`],
  [Current parameter values as $(d, d, K)$ batched arrays, one per manifold type. Updated in-place each iteration by the optimizer.],
  [`grad_buf_batches`\ `Dict{Manifold, Array{T,3}}`],
  [Pre-allocated gradient buffers. `stack_tensors!` writes Euclidean gradients here each iteration to avoid allocation.],
  [`ibatch_cache`\ `Dict{Manifold, Array{T,3}}`],
  [Pre-allocated $(d, d, K)$ identity matrices for `UnitaryManifold` groups. Passed to `retract(...; I_batch=...)` to avoid allocation in the line search / Adam inner loop.],
  [`current_tensors`\ `Vector{Matrix}`],
  [Individual tensor copies that Zygote sees. Populated by `unstack_tensors!` each iteration.],
)

== Initialization: `_common_setup`

#line(length: 100%)

```julia
function _common_setup(tensors)
    current_tensors = copy.(tensors)
    manifold_groups = group_by_manifold(tensors)
    for (manifold, indices) in manifold_groups
        pb = stack_tensors(current_tensors, indices)   # (d, d, K_m)
        point_batches[manifold] = pb
        grad_buf_batches[manifold] = similar(pb)       # pre-allocate
        if manifold isa UnitaryManifold
            ibatch_cache[manifold] = _make_identity_batch(ET, d, K_m)
        end
    end
    return OptimizationState{ET, RT}(manifold_groups, point_batches,
                                      grad_buf_batches, ibatch_cache,
                                      current_tensors)
end
```

#line(length: 100%)

== Gradient Computation

`_compute_gradients(grad_fn, tensors)` calls Zygote to get Euclidean gradients, then handles edge cases:

1. *Tangent type handling*: Zygote may return `ZeroTangent` or `NoTangent` for tensors that don't affect the loss. These are replaced with zero arrays of the correct shape.
2. *NaN/Inf guard*: If any gradient element is non-finite, returns `nothing` to signal the optimizer to stop. This prevents divergent updates from corrupting parameters.

== Batched Projection

`_batched_project` converts Euclidean gradients to Riemannian gradients:

#line(length: 100%)

```julia
function _batched_project(manifold_groups, point_batches, grad_buf_batches, euclidean_grads)
    for (manifold, indices) in manifold_groups
        stack_tensors!(gb, euclidean_grads, indices)   # pack into buffer
        rg = project(manifold, pb, gb)                 # batched projection
        rg_batches[manifold] = rg
        grad_norm_sq += real(sum(abs2, rg))            # accumulate norm
    end
    return rg_batches, sqrt(grad_norm_sq)
end
```

#line(length: 100%)

The global gradient norm (across all manifold groups) is used for convergence checking: when $||tilde(bold(g))|| < "tol"$, the optimizer stops.

== `RiemannianGD` with Armijo Line Search

Gradient descent with adaptive step size. At each iteration, compute the Riemannian gradient $tilde(bold(g))$ and search for a step size $alpha$ satisfying the Armijo sufficient decrease condition:

$ cal(L)("Retr"_(bold(theta))(-alpha tilde(bold(g)))) <= cal(L)(bold(theta)) - c dot alpha dot ||tilde(bold(g))||^2 $

The algorithm tries $alpha, alpha tau, alpha tau^2, dots$ (backtracking with factor $tau in (0,1)$) until the condition holds or `max_ls_steps` is exhausted. All retraction trials reuse the cached `I_batch` to avoid per-trial identity matrix allocation:

#line(length: 100%)

```julia
for _ls in 1:opt.max_ls_steps
    for (manifold, indices) in state.manifold_groups
        ib = get(state.ibatch_cache, manifold, nothing)
        cand = retract(manifold, pb, .-rg, alpha; I_batch=ib)  # reuse I_batch
        unstack_tensors!(state.current_tensors, cand, indices)
    end
    candidate_loss = loss_fn(state.current_tensors)
    if candidate_loss <= current_loss - c * alpha * grad_norm_sq
        return candidate_loss   # accept
    end
    alpha *= tau   # backtrack
end
```

#line(length: 100%)

== `RiemannianAdam` with Momentum Transport

Extends Adam to manifolds (Bécigneul & Ganea, 2019). The key challenge: after retraction, the first moment $bold(m)_t$ lives in the _old_ tangent space and must be transported to the _new_ one.

Moment updates (with bias correction):

$ bold(m)_t = beta_1 bold(m)_(t-1) + (1 - beta_1) tilde(bold(g))_t, #h(2em) bold(v)_t = beta_2 bold(v)_(t-1) + (1 - beta_2) |tilde(bold(g))_t|^2 $

The second moment uses `real(abs2(rg))` because $|tilde(bold(g))_t|^2$ for complex gradients produces complex values, but the variance estimate must be real-valued (it scales the direction magnitude, not its phase).

Bias-corrected direction:

$ bold(d)_t = (bold(m)_t \/ (1 - beta_1^t)) / (sqrt(bold(v)_t \/ (1 - beta_2^t)) + epsilon) $

Update and transport:

$ bold(theta)_(t+1) = "Retr"_(bold(theta)_t)(-alpha bold(d)_t), #h(2em) bold(m)_t arrow.l Gamma_(bold(theta)_t arrow.r bold(theta)_(t+1))(bold(m)_t) $

The implementation uses fused broadcasts (`@.`) so each line compiles to a single GPU kernel:

#line(length: 100%)

```julia
@. m_state = β₁ * m_state + (1 - β₁) * rg             # 1 kernel
@. v_state = β₂ * v_state + (1 - β₂) * real(abs2(rg))  # 1 kernel
@. dir = (m_state / bc1) / (√(v_state / bc2) + ε)       # 1 kernel
new_batch = retract(manifold, old_batch, .-dir, lr; I_batch=ib)
m_state = transport(manifold, old_batch, new_batch, m_state)
```

#line(length: 100%)

Note: Adam does _not_ evaluate the loss at each step (unlike GD with Armijo). It trusts the bias-corrected moments. The optimization loop separately records the loss for tracing when requested.

== Cayley Retraction Geometry

The Cayley retraction maps a tangent vector $bold(xi) in T_(bold(U)) U(n)$ back onto the manifold $U(n)$. The curve $bold(U)(t) = (bold(I) - t/2 dot bold(W))^(-1)(bold(I) + t/2 dot bold(W)) dot bold(U)$ satisfies $bold(U)(0) = bold(U)$ and $bold(U)(t) in U(n)$ for all $t$:

#figure(canvas(length: 1.0cm, {
  import draw: *

  // Manifold surface (curved arc)
  bezier((-3, -0.5), (0, 1.5), (3, -0.5), stroke: (paint: blue.darken(30%), thickness: 1.5pt), name: "manifold")
  content((3.5, -0.3), text(9pt, fill: blue.darken(30%))[$U(n)$])

  // Point U on manifold
  circle((-1.5, 0.55), radius: 0.08, fill: black, name: "U")
  content((-1.8, 0.9), text(9pt)[$bold(U)$])

  // Tangent plane (dashed line through U)
  line((-3.0, -0.4), (0.5, 1.6), stroke: (dash: "dashed", paint: gray), name: "tangent")
  content((0.8, 1.8), text(8pt, fill: gray.darken(20%))[$T_(bold(U)) U(n)$])

  // Tangent vector xi
  line((-1.5, 0.55), (-0.2, 1.2), stroke: (paint: red, thickness: 1.2pt), mark: (end: ">", size: 0.25))
  content((0.1, 1.5), text(9pt, fill: red)[$bold(xi)$])

  // Cayley curve (from U to U')
  bezier((-1.5, 0.55), (-0.5, 1.2), (0.5, 1.0), (1.2, 0.3), stroke: (paint: orange, thickness: 1.0pt, dash: "dotted"))
  content((0.8, 1.1), text(7pt, fill: orange)[Cayley curve])

  // Retracted point U'
  circle((1.2, 0.3), radius: 0.08, fill: black, name: "U2")
  content((1.6, 0.6), text(9pt)[$bold(U')$])

  // Euclidean direction (would leave manifold)
  line((-1.5, 0.55), (0.0, 1.85), stroke: (paint: gray.lighten(40%), thickness: 0.8pt, dash: "loosely-dashed"), mark: (end: ">", size: 0.2))
  content((0.3, 2.15), text(7pt, fill: gray.lighten(20%))[off manifold])

}), caption: [Cayley retraction on $U(n)$. Starting at point $bold(U)$, a tangent vector $bold(xi)$ points along the tangent plane $T_(bold(U)) U(n)$. Following $bold(xi)$ in Euclidean space (gray dashed) would leave the manifold. The Cayley curve (orange dotted) stays on $U(n)$ and lands at the retracted point $bold(U') = "Retr"_(bold(U))(bold(xi))$.])

== The Optimization Loop

Each iteration of `_optimization_loop` follows this sequence:

+ *Unstack*: `unstack_tensors!` copies batched points into individual tensors for Zygote
+ *Gradients*: `_compute_gradients` calls Zygote AD to get Euclidean gradients $bold(g)_1, dots, bold(g)_K$
+ *Batched projection*: `_batched_project` packs gradients and projects onto tangent spaces, returning Riemannian gradients $tilde(bold(g))$ and the global gradient norm
+ *Convergence check*: if $||tilde(bold(g))|| < "tol"$, stop
+ *Optimizer step*: `_update_step!` dispatches to GD (Armijo line search) or Adam (moment update + retract)
+ *Loss trace*: record the current loss if requested
