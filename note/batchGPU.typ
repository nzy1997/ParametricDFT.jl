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

= Batched Einsum (`loss.jl`)

== Background: OMEinsum and the Forward Transform

The QFT circuit is represented as a tensor network contraction using #link("https://github.com/under-Peter/OMEinsum.jl")[OMEinsum.jl]. In OMEinsum, a contraction is specified by a `DynamicEinCode`: a list of input index lists and an output index list. Indices that appear in inputs but not in the output are summed over (contracted). For example, matrix multiplication $bold(C)_(i,k) = sum_j bold(A)_(i,j) bold(B)_(j,k)$ is written as:

#align(center)[`ein"ij, jk -> ik"(A, B)` #h(1em) or equivalently #h(1em) `DynamicEinCode([[1,2], [2,3]], [1,3])`]

where integer labels replace characters (OMEinsum uses `Int` labels internally).

The function `qft_code(m, n)` builds the einsum for the QFT circuit:
1. Construct the QFT circuit using `Yao.EasyBuild.qft_circuit` for $m$ row qubits and $n$ column qubits
2. Convert the circuit to an einsum via `yao2einsum` --- this assigns each gate and each qubit wire an integer index label
3. Optimize the contraction order using `TreeSA()` (done once, reused for all iterations)

*Concrete example* ($m = 2, n = 2$, i.e. $4 times 4$ images with 4 qubits). The circuit has $K = 6$ gate tensors (4 Hadamard gates $H$ and 2 controlled-phase gates $M_k$), plus the image tensor. After `yao2einsum`, the contraction looks like:

#align(center)[`DynamicEinCode([[5,1], [6,2], [7,3], [8,4], [5,2], [7,4], [5,6,7,8]], [1,2,3,4])`]

Reading this notation:
- Gate tensors: $bold(theta)_1$ has indices $[5,1]$, $bold(theta)_2$ has $[6,2]$, ..., $bold(theta)_6$ has $[7,4]$ --- each is a $2 times 2$ matrix
- Image tensor: indices $[5,6,7,8]$ --- these are the $m+n = 4$ qubit dimensions
- Output: indices $[1,2,3,4]$ --- the transformed image's qubit dimensions
- Contracted indices: $5,6,7,8$ appear in both gate and image inputs but not in the output, so they are summed over

The corresponding tensor contraction is:

$ bold(Y)_(j_1, j_2, j_3, j_4) = sum_(i_1, i_2, i_3, i_4) (bold(theta)_1)_(i_1, j_1) (bold(theta)_2)_(i_2, j_2) (bold(theta)_3)_(i_3, j_3) (bold(theta)_4)_(i_4, j_4) (bold(theta)_5)_(i_1, i_2) (bold(theta)_6)_(i_3, i_4) bold(X)_(i_1, i_2, i_3, i_4) $

== The Problem: Single-Image Contraction

Without batching, training on a dataset of $N$ images requires calling this einsum $N$ times per gradient step --- each call launches its own kernel(s).

The diagram below shows the tensor network for _one_ image. The gate tensors $bold(theta)_k$ (shared across all images) each have 2 legs that connect into the circuit, while the image tensor $bold(X)$ feeds $m+n$ qubit indices through the circuit to produce output $bold(Y)$:

#figure(canvas(length: 0.9cm, {
  import draw: *
  let n = 4
  let dy = 0.8

  // Image input tensor
  ngate((0, 0), n, "X", text: [$bold(X)$], gap-y: dy, width: 0.8)

  // Gate tensors along the qubit lines
  ngate((1.8, 1.2), 1, "t1", text: text(8pt)[$bold(theta)_1$], gap-y: dy, width: 0.6)
  ngate((2.8, 0.4), 1, "t2", text: text(8pt)[$bold(theta)_2$], gap-y: dy, width: 0.6)
  ngate((3.8, -0.4), 1, "t3", text: text(8pt)[$bold(theta)_3$], gap-y: dy, width: 0.6)
  content((4.8, -1.0), [$dots.c$])
  ngate((5.8, -1.2), 1, "tK", text: text(8pt)[$bold(theta)_K$], gap-y: dy, width: 0.6)

  // Output tensor
  ngate((7.5, 0), n, "Y", text: [$bold(Y)$], gap-y: dy, width: 0.8)

  // Connect image → gates → output (qubit lines)
  line("X.o0", "t1.i0", stroke: gray)
  line("t1.o0", "Y.i0", stroke: gray)
  line("X.o1", "t2.i0", stroke: gray)
  line("t2.o0", "Y.i1", stroke: gray)
  line("X.o2", "t3.i0", stroke: gray)
  line("t3.o0", "Y.i2", stroke: gray)
  line("X.o3", (4.5, -1.2), stroke: gray)
  line((5.2, -1.2), "tK.i0", stroke: gray)
  line("tK.o0", "Y.i3", stroke: gray)

  // Qubit labels
  for i in range(n) {
    content((-1.2, -i * dy + (n - 1) * dy / 2), [$q_#i$])
  }

  // "× N images" annotation
  rect((-1.5, -2.0), (8.5, 2.2), stroke: (dash: "dotted", paint: red.darken(20%)))
  content((3.5, -2.4), text(8pt, fill: red.darken(20%))[called $N$ times (once per image)])

}), caption: [Single-image einsum: gate tensors $bold(theta)_k$ sit on qubit lines between input $bold(X)$ and output $bold(Y)$. This contraction is called $N$ times, once per image.])

This is inefficient because the contraction tree is _identical_ for every image --- only the image tensor changes. The gate tensors $bold(theta)_1, dots, bold(theta)_K$ are shared.

== The Solution: Append a Batch Dimension

The key insight: add a new _batch label_ to the image input and output index lists, leaving gate indices unchanged. Since this label is not contracted (it appears in both the image input and the output), OMEinsum processes all $B$ images in a single kernel call.

Continuing the $m = 2, n = 2$ example, `make_batched_code` computes `batch_label = max(1,...,8) + 1 = 9` and appends it:

#align(center)[`DynamicEinCode([[5,1], [6,2], [7,3], [8,4], [5,2], [7,4],` *`[5,6,7,8,9]`*`],` *`[1,2,3,4,9]`*`)`]

Only the last two entries change (bold): the image input gains label 9, the output gains label 9. The contraction becomes:

$ bold(Y)_(j_1, j_2, j_3, j_4, b) = sum_(i_1, i_2, i_3, i_4) (bold(theta)_1)_(i_1, j_1) dots.c (bold(theta)_6)_(i_3, i_4) bold(X)_(i_1, i_2, i_3, i_4, b) $

where $b in {1, dots, B}$ indexes images in the batch.

#figure(canvas(length: 0.9cm, {
  import draw: *
  let n = 4
  let dy = 0.8

  // Batched image input tensor
  ngate((0, 0), n, "X", text: [$bold(X)$], gap-y: dy, width: 0.8)

  // Batch dimension leg (dangling down from X)
  line("X.b", (rel: (0, -0.6)), stroke: blue, name: "Xb")
  content((rel: (0.4, -0.1), to: "Xb.end"), text(8pt, fill: blue)[$b$])

  // Gate tensors along the qubit lines (same as single-image)
  ngate((1.8, 1.2), 1, "t1", text: text(8pt)[$bold(theta)_1$], gap-y: dy, width: 0.6)
  ngate((2.8, 0.4), 1, "t2", text: text(8pt)[$bold(theta)_2$], gap-y: dy, width: 0.6)
  ngate((3.8, -0.4), 1, "t3", text: text(8pt)[$bold(theta)_3$], gap-y: dy, width: 0.6)
  content((4.8, -1.0), [$dots.c$])
  ngate((5.8, -1.2), 1, "tK", text: text(8pt)[$bold(theta)_K$], gap-y: dy, width: 0.6)

  // Batched output tensor
  ngate((7.5, 0), n, "Y", text: [$bold(Y)$], gap-y: dy, width: 0.8)

  // Batch dimension leg (dangling down from Y)
  line("Y.b", (rel: (0, -0.6)), stroke: blue, name: "Yb")
  content((rel: (0.4, -0.1), to: "Yb.end"), text(8pt, fill: blue)[$b$])

  // Connect qubit lines
  line("X.o0", "t1.i0", stroke: gray)
  line("t1.o0", "Y.i0", stroke: gray)
  line("X.o1", "t2.i0", stroke: gray)
  line("t2.o0", "Y.i1", stroke: gray)
  line("X.o2", "t3.i0", stroke: gray)
  line("t3.o0", "Y.i2", stroke: gray)
  line("X.o3", (4.5, -1.2), stroke: gray)
  line((5.2, -1.2), "tK.i0", stroke: gray)
  line("tK.o0", "Y.i3", stroke: gray)

  // Qubit labels
  for i in range(n) {
    content((-1.2, -i * dy + (n - 1) * dy / 2), [$q_#i$])
  }

  // Annotation: batch dimension passes through
  line((0, -2.4), (7.5, -2.4), stroke: (dash: "dashed", paint: blue.darken(20%)))
  content((3.75, -2.9), text(8pt, fill: blue.darken(20%))[batch index $b$: passes through gates unchanged --- 1 call for all $B$ images])

}), caption: [Batched einsum: an extra batch index $b$ (blue) is added to $bold(X)$ and $bold(Y)$. Gate tensors $bold(theta)_k$ have no $b$ index, so the _same_ contraction processes all $B$ images in one call.])

The image tensor $bold(X)$ has shape $(2, 2, dots, 2, B) in CC^(2^(m+n) times B)$ --- $B$ images stacked along a new last dimension.

#line(length: 100%)

```julia
# Step 1: Append batch label to the einsum code
batched_flat, batch_label = make_batched_code(optcode, n_gates)

# Step 2: Re-optimize contraction order for batched tensor sizes
# size_dict: {1=>2, ..., 8=>2, 9=>B}  (all qubit dims = 2, batch dim = B)
batched_optcode = optimize_batched_code(batched_flat, batch_label, batch_size)

# Step 3: Stack B images and evaluate in one einsum call
stacked = stack_image_batch(batch, m, n)   # (2,...,2,B)
result = batched_optcode(tensors..., stacked)  # shape: (2,...,2,B)
```

#line(length: 100%)

== Einsum Contraction Cache (`einsum_cache.jl`)

TreeSA contraction order optimization can take minutes for large circuits. To amortize this cost across training runs, `optimize_code_cached` implements content-addressed disk caching:

1. *Key computation*: A SHA-256 hash of the einsum's index lists and size dictionary. This is stable across Julia sessions.
2. *Cache storage*: Serialized Julia objects in `~/.cache/ParametricDFT/einsum_codes/<hash>.jls`.
3. *Graceful fallback*: If deserialization fails (e.g. Julia version change), the cache file is deleted and TreeSA re-runs.

#figure(canvas(length: 0.85cm, {
  import draw: *

  // Input
  rect((-2, 2), (2, 3), fill: blue.lighten(85%), name: "input")
  content("input", text(8pt)[`(flat_code, size_dict)`])

  // Hash
  rect((-1.2, 0.5), (1.2, 1.5), fill: gray.lighten(85%), name: "hash")
  content("hash", text(8pt)[SHA-256 hash])
  line((0, 2), (0, 1.5), stroke: gray, mark: (end: ">", size: 0.2))

  // Decision diamond
  content((0, -0.8), text(8pt)[cache file\ exists?])
  rect((-1.2, -1.5), (1.2, -0.1), stroke: (dash: "dashed", paint: gray))
  line((0, 0.5), (0, -0.1), stroke: gray, mark: (end: ">", size: 0.2))

  // Hit path (left)
  rect((-4.5, -3.5), (-1.5, -2.5), fill: green.lighten(85%), name: "hit")
  content("hit", text(8pt)[`deserialize`])
  line((-0.6, -1.5), (-3.0, -2.5), stroke: green.darken(20%), mark: (end: ">", size: 0.2))
  content((-2.2, -1.7), text(7pt, fill: green.darken(20%))[hit])

  // Miss path (right)
  rect((1.5, -3.5), (4.5, -2.5), fill: orange.lighten(85%), name: "miss")
  content("miss", text(8pt)[TreeSA optimize])
  line((0.6, -1.5), (3.0, -2.5), stroke: orange.darken(20%), mark: (end: ">", size: 0.2))
  content((2.2, -1.7), text(7pt, fill: orange.darken(20%))[miss])

  // Serialize after miss
  rect((1.5, -5.2), (4.5, -4.2), fill: gray.lighten(85%), name: "save")
  content("save", text(8pt)[`serialize` to disk])
  line((3.0, -3.5), (3.0, -4.2), stroke: gray, mark: (end: ">", size: 0.2))

  // Both paths converge
  rect((-1.5, -7.0), (1.5, -6.0), fill: blue.lighten(85%), name: "result")
  content("result", text(8pt)[optimized einsum code])
  line((-3.0, -3.5), (-3.0, -6.5), stroke: gray)
  line((-3.0, -6.5), (-1.5, -6.5), stroke: gray, mark: (end: ">", size: 0.2))
  line((3.0, -5.2), (3.0, -6.5), stroke: gray)
  line((3.0, -6.5), (1.5, -6.5), stroke: gray, mark: (end: ">", size: 0.2))

}), caption: [Einsum cache flow. The key is a SHA-256 hash of the contraction structure and tensor sizes. On cache hit, deserialization is near-instant. On miss, TreeSA runs (potentially minutes) and the result is saved for next time.])

== Loss Dispatch by Type

The batching strategy differs by loss function:

*L1/L2*: Fully batched end-to-end. One einsum call produces all $B$ outputs as a $(2,dots,2,B)$ tensor, then a single reduction:

$ cal(L)_"L1" = 1/B sum_(b=1)^B sum_(j_1, dots, j_(m+n)) |bold(Y)_(j_1, dots, j_(m+n), b)|, #h(2em) cal(L)_"L2" = 1/B sum_(b=1)^B sum_(j_1, dots, j_(m+n)) |bold(Y)_(j_1, dots, j_(m+n), b)|^2 $

*MSE*: The forward pass is batched, but `topk_truncate` must run per-image because the truncation mask depends on each image's frequency content. The inverse transform is batched again when `batched_inverse_code` is available:

+ *Batched forward*: One einsum call $arrow.r$ all $B$ frequency-domain outputs
+ *Per-image truncation*: `map(1:B)` applies `topk_truncate` to each image slice (content-dependent mask, cannot batch)
+ *Batched inverse*: If `batched_inverse_code` is provided, the truncated slices are re-stacked and a single inverse einsum reconstructs all $B$ images. Otherwise, falls back to per-image inverse.

$ cal(L)_"MSE" = 1/B sum_(b=1)^B || bold(X)_b - "einsum"^(-1)(bold(theta)_1^*, dots, bold(theta)_K^*, "truncate"_k (bold(Y)_b)) ||^2 $

#line(length: 100%)

```julia
# Batched forward — single einsum call for all B images
fft_batched = batched_forward(optcode_batched, ts, stacked_batch)

# Per-image truncation (mask is content-dependent)
truncated_slices = map(1:B) do i
    fft_slice = reshape(selectdim(fft_batched, m + n + 1, i), 2^m, 2^n)
    reshape(topk_truncate(fft_slice, k), qubit_dims...)
end

# Batched inverse — single einsum call when batched_inverse_code available
if batched_inverse_code !== nothing
    stacked_trunc = cat(truncated_slices...; dims=m + n + 1)
    inv_batched = batched_inverse_code(conj.(tensors)..., stacked_trunc)
    total_loss = sum(abs2.(stacked_batch .- inv_batched))
end
```

#line(length: 100%)

== The Training Loop

The diagram below shows one iteration of the training loop. Gate tensors and the image batch feed into the batched einsum; the output flows through the loss; Zygote differentiates to get gradients; the Riemannian optimizer projects and retracts to update the tensors:

#figure(canvas(length: 0.85cm, {
  import draw: *

  // --- Forward pass (left to right) ---

  // Gate tensors
  ngate((0, 1.5), 1, "theta", text: [$bold(theta)$], width: 1.0)
  content((0, 2.3), text(8pt)[gate tensors])

  // Image batch
  ngate((0, -1.5), 1, "img", text: [$bold(X)_b$], width: 1.0)
  content((0, -2.3), text(8pt)[image batch])

  // Einsum
  ngate((3.0, 0), 2, "ein", text: text(8pt)[einsum], gap-y: 3, width: 1.2)

  // Arrows into einsum
  line("theta.o0", "ein.i0", stroke: gray, mark: (end: ">", size: 0.2))
  line("img.o0", "ein.i1", stroke: gray, mark: (end: ">", size: 0.2))

  // Loss
  ngate((5.5, 0), 1, "loss", text: [$cal(L)$], width: 0.8)
  content((5.5, 0.8), text(8pt)[loss])
  line("ein.o0", (4.3, 1.5), stroke: gray)
  line((4.3, 1.5), (4.3, 0), stroke: gray)
  line((4.3, 0), "loss.i0", stroke: gray, mark: (end: ">", size: 0.2))
  // Y label on the connection
  content((4.3, 1.9), text(8pt)[$bold(Y)_b$])

  // --- Backward pass (right to left, below) ---

  // Gradient
  ngate((5.5, -3.0), 1, "grad", text: text(8pt)[$nabla cal(L)$], width: 0.8)
  content((5.5, -3.8), text(8pt)[Zygote AD])
  line("loss.o0", (6.5, 0), stroke: gray)
  line((6.5, 0), (6.5, -3.0), stroke: gray)
  line((6.5, -3.0), "grad.o0", stroke: gray, mark: (end: ">", size: 0.2))

  // Optimizer: project + retract
  ngate((2.5, -3.0), 1, "opt", text: text(7pt)[project $arrow.r$ retract], width: 2.0)
  content((2.5, -3.8), text(8pt)[Riemannian optimizer])
  line("grad.i0", "opt.o0", stroke: gray, mark: (end: ">", size: 0.2))

  // Feedback loop: optimizer → gate tensors
  line("opt.i0", (-1.2, -3.0), stroke: (paint: blue, thickness: 1.2pt))
  line((-1.2, -3.0), (-1.2, 1.5), stroke: (paint: blue, thickness: 1.2pt))
  line((-1.2, 1.5), "theta.i0", stroke: (paint: blue, thickness: 1.2pt), mark: (end: ">", size: 0.25))
  content((-2.2, -0.8), text(8pt, fill: blue)[update $bold(theta)$])

}), caption: [One training iteration. Forward: gate tensors $bold(theta)$ and images $bold(X)_b$ contract via batched einsum to produce $bold(Y)_b$, then the loss $cal(L)$ is evaluated. Backward: Zygote computes $nabla cal(L)$, the Riemannian optimizer projects the gradient and retracts to update $bold(theta)$ on the manifold.])

= GPU Acceleration (`ext/CUDAExt.jl`) <gpu>

== Device Abstraction

`to_device(x, device)` moves arrays between CPU and GPU using Julia's extension mechanism:

#line(length: 100%)

```julia
# Core (training.jl) — always available
to_device(x, Val(:cpu)) = x       # CPU is identity
to_cpu(x::AbstractArray) = Array(x)

# Extension (CUDAExt.jl) — loaded when CUDA.jl is available
to_device(x::AbstractArray{T}, Val(:gpu)) = CuArray{T}(x)
to_device(x, Val(:gpu)) = x       # scalars pass through
```

#line(length: 100%)

The `Val{:gpu}` dispatch avoids runtime type instability: if `CUDA.jl` is loaded, the extension adds the method; otherwise `to_device(x, :gpu)` raises an informative error.

== GPU Dispatch Table

On GPU, the batched operations dispatch to optimized kernels:

#table(
  columns: 3,
  [*Operation*], [*CPU (`manifolds.jl`)*], [*GPU (`CUDAExt.jl`)*],
  [`batched_matmul`], [Per-slice `mul!` loop], [`CUBLAS.gemm_strided_batched!` (1 kernel)],
  [`batched_inv`], [Per-slice `inv()` loop], [Broadcast $2 times 2$ formula (1 kernel)],
  [`batched_adjoint`], [`permutedims(conj.())`], [Same (GPU broadcast)],
  [Einsum contraction], [CPU OMEinsum], [CuArray dispatch (GPU contraction)],
  [`topk_truncate`], [CPU `partialsort!` + mask], [Frequency-weighted GPU sort + mask],
)

== CUBLAS Strided Batched GEMM

For $K$ matrix multiplications of $d times d$ matrices, the CPU loops $K$ times. On GPU, `CUBLAS.gemm_strided_batched!` processes all $K$ slices in a single kernel launch:

#line(length: 100%)

```julia
function batched_matmul(A::CuArray{T,3}, B::CuArray{T,3})
    C = similar(A, T, d1, d3, n)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), A, B, zero(T), C)
    return C
end
```

#line(length: 100%)

This eliminates $K - 1$ kernel launches. For $2 times 2$ matrices (our typical case), each individual multiplication is too small to saturate a GPU, but the batched call amortizes the $~10 mu s$ launch overhead across all $K$ gates.

== Closed-Form $2 times 2$ Inverse

For $2 times 2$ matrices, the analytical inverse $mat(a,b;c,d)^(-1) = (a d - b c)^(-1) mat(d,-b;-c,a)$ is implemented as pure broadcasting --- a single GPU kernel for all $K$ slices:

#line(length: 100%)

```julia
function batched_inv(A::CuArray{T,3})
    if d == 2
        a, b, c, dd = A[1:1,1:1,:], A[1:1,2:2,:], A[2:2,1:1,:], A[2:2,2:2,:]
        det = a .* dd .- b .* c
        inv_det = one(T) ./ det
        return cat(
            cat(dd .* inv_det, -(b .* inv_det); dims=2),
            cat(-(c .* inv_det), a .* inv_det; dims=2);
            dims=1)
    else
        # Fallback: per-slice LU inverse
    end
end
```

#line(length: 100%)

The `1:1` indexing (instead of scalar indexing) keeps everything as `CuArray` slices, avoiding `@allowscalar` and enabling broadcasting.

== Frequency-Aware Top-$k$ Truncation

On GPU, `topk_truncate` uses frequency-weighted scores instead of raw magnitudes:

$ s_(i,j) = |bold(Y)_(i,j)| dot (1 + w_(i,j)), #h(2em) w_(i,j) = 1 - d_(i,j) / (2 d_"max") $

where $d_(i,j) = sqrt((i - i_c)^2 + (j - j_c)^2)$ is the distance from the center (DC component) and $d_"max"$ is the maximum distance. This weighting favors low-frequency components (structural information) over high-frequency ones (fine details) of similar magnitude.

The implementation caches frequency weights per $(m, n)$ pair to avoid per-call GPU allocations. The threshold is extracted via 1-element slice `sorted[k:k]` (not `sorted[k]`) to avoid `@allowscalar`, which would block GPU kernel fusion.

== When GPU Wins vs CPU

- *Small circuits* ($< 10$ tensors): CPU faster due to GPU transfer overhead
- *Large circuits or batches*: GPU wins from batched parallelism
- *L1/L2 loss*: fully batched, GPU-friendly
- *MSE loss*: `topk_truncate` is per-image, limiting GPU parallelism (though the inverse is now batched)

= The Training Pipeline (`training.jl`)

The `_train_basis_core` function orchestrates the full training loop, composing the batched einsum, Riemannian optimizers, and device management described in previous sections.

== Data Preparation

#line(length: 100%)

```julia
# Convert images to Complex{Float64} and move to device
complex_dataset = [to_device(Complex{Float64}.(img), device) for img in dataset]

# Split into training/validation sets
n_validation = clamp(round(Int, n_images * validation_split), 0, n_images - 1)
indices = shuffle ? Random.shuffle(1:n_images) : collect(1:n_images)
training_data = complex_dataset[indices[n_validation+1:end]]
validation_data = complex_dataset[indices[1:n_validation]]

# Clamp batch_size and move tensors to device
batch_size = clamp(batch_size, 1, length(training_data))
device_tensors = [to_device(Matrix{ComplexF64}(t), device) for t in initial_tensors]
```

#line(length: 100%)

== Batched Einsum Pre-Computation

When `batch_size > 1`, the batched einsum codes are optimized _once_ at startup and reused for all epochs and batches:

#line(length: 100%)

```julia
if batch_size > 1
    flat_batched, blabel = make_batched_code(optcode, n_gates)
    batched_optcode = optimize_batched_code(flat_batched, blabel, batch_size)
    # Also batch the inverse code for MSELoss
    if inverse_code !== nothing && loss isa MSELoss
        flat_batched_inv, blabel_inv = make_batched_code(inverse_code, n_gates)
        batched_inverse_code = optimize_batched_code(flat_batched_inv, blabel_inv, batch_size)
    end
end
```

#line(length: 100%)

This is critical for efficiency: TreeSA optimization can take minutes for large circuits, but `optimize_code_cached` (see the einsum cache in the previous section) ensures it runs at most once per unique circuit structure.

== Batch Iteration Scaling

The number of optimizer iterations per batch is scaled by the batch size:

$ "batch\_max\_iter" = "steps\_per\_image" times |"batch"| $

Without this scaling, `batch_size=16` would do $16 times$ fewer total optimization steps on the aggregated data. The scaling ensures that total optimizer iterations per image remains constant regardless of how images are grouped into batches.

== Loss and Gradient Construction

For each batch, closures are constructed that capture the batched einsum codes:

#line(length: 100%)

```julia
# Batched path: single einsum call for all B images
batch_loss_fn = ts -> loss_function(ts, m, n, optcode, stacked_batch, loss;
                                     batched_optcode=batched_optcode,
                                     batched_inverse_code=batched_inverse_code)

# Gradient via Zygote pullback
batch_grad_fn = ts -> begin
    _, back = Zygote.pullback(batch_loss_fn, ts)
    return back(one(real(eltype(ts[1]))))[1]
end

# Run optimizer (iterations scaled by batch size)
current_tensors = optimize!(opt, current_tensors, batch_loss_fn, batch_grad_fn;
                            max_iter=steps_per_image * length(batch), tol=1e-8)
```

#line(length: 100%)

== Checkpointing

When `checkpoint_interval > 0` and `checkpoint_dir` is set, the training loop saves the current tensors and loss history every `checkpoint_interval` global steps. This allows resuming long training runs without losing progress.

== Early Stopping

Validation loss is computed after each epoch. If it improves, the best tensors are snapshotted. If it does not improve for `early_stopping_patience` consecutive epochs, training stops:

#line(length: 100%)

```julia
if val_loss < best_val_loss
    best_val_loss = val_loss
    best_tensors = copy.(current_tensors)
    patience_counter = 0
else
    patience_counter += 1
    patience_counter >= early_stopping_patience && epoch > 1 && break
end
```

#line(length: 100%)

== Finalization

After training, the best tensors are moved back to CPU and converted to `ComplexF64` for serialization:

```julia
final_tensors = [ComplexF64.(Array(t)) for t in best_tensors]
```

This ensures consistent types regardless of whether training ran on CPU or GPU.
