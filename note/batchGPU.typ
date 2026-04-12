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
