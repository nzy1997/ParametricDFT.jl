#import "@preview/cetz:0.4.2": canvas, draw
#show link: set text(blue)

// Tensor box helper for diagrams (same as main.typ ngate)
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

= Prerequisite

1. Understand image processing in Julia, please check YouTube Video: #link("https://www.youtube.com/watch?app=desktop&v=DGojI9xcCfg", [Working With Images in Julia | Week 1, lecture 3 | 18.S191 MIT Fall 2020 | Grant Sanderson]).
2. Undertand fast Fourier transformation for image processing, please check the following videos by Steve Brunton:
  - #link("https://www.youtube.com/watch?v=E8HeD-MUrjY", [The Fast Fourier Transform (FFT)]). It may require some knowledge about complex numbers.
  - #link("https://www.youtube.com/watch?v=gGEBUdM0PVc", [Image Compression and the FFT])
3. Understand tensor network, please check the following repository: #link("https://github.com/GiggleLiu/tutorial-tensornetwork", [Tutorial on Tensor Networks]).
4. Understand basic optimization theory, please check:
  - The 3blue1brown video: #link("https://youtu.be/IHZwWFHWa-w?si=8MWIX_0JHnDYkCSE")[Gradient descent, how neural networks learn | Deep Learning Chapter 2]
  - Manifold optimization: YouTube video #link("https://www.youtube.com/watch?v=dJz1klEutRY", [Manopt.jl: Optimisation on Riemannian Manifolds | Ronny Bergmann | JuliaCon 2022]) and Julia package #link("https://github.com/JuliaManifolds/Manopt.jl", [Manopt.jl]).

= Get started
1. Go through the code in `examples/img_process.jl`. It may require some knowledge about manifold optimization, please check the documentation page of #link("https://github.com/JuliaManifolds/Manifolds.jl", [Manifolds.jl]).
2. Read the note `note/main.typ` to understand the theory underlying the code.

= Code Structure

The codebase evolves through three stages, each solving problems from the previous one. The file tree below shows which files belong to which stage:

```
src/
├── ParametricDFT.jl          # Main module: exports, include() ordering
├── qft.jl                    # QFT circuit → einsum contraction code
├── loss.jl                   # Loss functions: L1Norm, L2Norm, MSELoss
├── basis.jl                  # AbstractSparseBasis: QFTBasis,
│                             #   EntangledQFTBasis, TEBDBasis
├── entangled_qft.jl          # Entangled QFT circuit (XY correlation)
├── tebd.jl                   # TEBD circuit (brickwork topology)
├── manifolds.jl              # Riemannian manifolds: UnitaryManifold,
│                             #   PhaseManifold, batched linear algebra
├── optimizers.jl             # RiemannianGD (Armijo), RiemannianAdam
├── training.jl               # Training pipeline, checkpointing,
│                             #   early stopping
├── compression.jl            # Image compression using learned bases
├── serialization.jl          # Basis save/load (JSON3)
└── visualization.jl          # Training loss plots (CairoMakie)

ext/
└── CUDAExt.jl                # GPU: batched CUBLAS, 2×2 inverse formula

test/
├── runtests.jl               # Test runner
├── basis_tests.jl            # Basis construction + transform tests
├── loss_tests.jl             # Loss function correctness
├── manifold_tests.jl         # Manifold projection/retraction/transport
├── optimizer_tests.jl        # Optimizer convergence tests
├── training_tests.jl         # Training pipeline integration
├── compression_tests.jl      # Compression round-trip tests
├── entangled_qft_tests.jl    # Entangled QFT tests
├── tebd_tests.jl             # TEBD circuit tests
├── serialization_tests.jl    # Save/load round-trip tests
└── cuda_tests.jl             # GPU-specific tests (requires CUDA)
```

= Stage 1: Manopt.jl Baseline

== The Approach

Use #link("https://github.com/JuliaManifolds/Manopt.jl", [Manopt.jl]) to optimize QFT circuit parameters on a product manifold. All circuit tensors are $2 times 2$ unitary matrices, so we model each as a point on the Stiefel manifold `Stiefel(2, 2, ℂ)` (which equals $U(2)$). The tensors are packed into a `ProductManifold` and optimized via `gradient_descent`.

Note: Manopt.jl is _not_ used in the main library (`src/`). It is only used in `examples/manopt_baseline.jl` as a comparison benchmark. The main training pipeline (`src/training.jl`) always uses the custom `RiemannianGD` / `RiemannianAdam` optimizers.

#line(length: 100%)

```julia
# Build product manifold: all tensors on Stiefel(2,2,ℂ) ≅ U(2)
# Note: Stiefel is used instead of UnitaryMatrices to avoid
# a local_metric MethodError in Manopt's Armijo linesearch.
S = Stiefel(2, 2, ℂ)
M = ProductManifold(ntuple(_ -> S, length(tensors))...)

# Convert between tensors and manifold points
p0 = ArrayPartition(tensors...)          # tensors → point
ts = collect(p.x)                        # point → tensors

# Loss: average over dataset, Riemannian gradient via Zygote AD
f(M, p) = mean(loss_function(collect(p.x), m, n, optcode,
                              img, loss) for img in images)
grad_f(M, p) = ManifoldDiff.gradient(M, x -> f(M, x), p,
    RiemannianProjectionBackend(AutoZygote()))

# Run optimization
result = gradient_descent(M, f, grad_f, p0;
    stopping_criterion = StopAfterIteration(steps) |
                          StopWhenGradientNormLess(1e-5))
```

#line(length: 100%)

== What Manopt.jl Provides

Manopt.jl is a mature library with 35+ solvers. For our use case, the relevant features are:
- `gradient_descent` with *Armijo line search by default*, plus Wolfe-Powell, cubic bracketing, nonmonotone line search options
- Direction update rules: momentum, Nesterov acceleration, gradient averaging
- Stochastic gradient descent (single-sample updates)
- Automatic differentiation via ManifoldDiff.jl + Zygote

== Problems Encountered

#table(
  columns: 2,
  [*Problem*], [*Why It Happens*],
  [Slow: per-tensor manifold operations],
  [Manopt.jl's `ProductManifold` iterates over each component manifold one-by-one. For a circuit with $K$ tensors, this means $K$ separate projection/retraction calls --- no way to batch same-type tensors into a single operation.],
  [Cannot use GPU],
  [`ProductManifold` uses `ArrayPartition` with heterogeneous element types. CUDA requires homogeneous `CuArray`. Additionally, individual $2 times 2$ matrix operations are too small for GPU kernels (launch overhead ~10$mu$s dominates).],
  [No Riemannian Adam],
  [Manopt.jl does not include Adam or any adaptive learning rate optimizer. Its gradient-based methods are limited to GD with momentum/Nesterov variants. For our problem, Adam's per-parameter adaptive rates converge faster than GD.],
)

These problems motivate Stage 2: building custom Riemannian optimizers with batched operations.

= Stage 2: Custom Riemannian Optimizers

== Notation

We use the following notation consistently in this section:
- $bold(theta)_k in CC^(d times d)$: the $k$-th gate tensor (circuit parameter), $k = 1, dots, K$
- $bold(X) in CC^(2 times dots.c times 2)$: input image reshaped to $m+n$ qubit dimensions (each of size 2)
- $bold(Y)$: output of the forward transform
- $cal(L)$: loss function value
- $bold(g)_k$: Euclidean gradient of $cal(L)$ with respect to $bold(theta)_k$
- $tilde(bold(g))_k = "proj"(bold(theta)_k, bold(g)_k)$: Riemannian gradient (projection of $bold(g)_k$ onto the tangent space at $bold(theta)_k$)
- $"Retr"_(bold(theta))(bold(xi))$: retraction from $bold(theta)$ along tangent vector $bold(xi)$ (maps back to the manifold)
- $Gamma_(bold(theta) arrow.r bold(theta)')$: parallel transport from tangent space at $bold(theta)$ to tangent space at $bold(theta)'$
- $alpha$: step size (learning rate)

== Problem $arrow.r$ Solution Overview

#table(
  columns: 3,
  [*Problem from Stage 1*], [*Solution*], [*File*],
  [Per-tensor manifold ops],
  [Group same-type tensors into $(d,d,K)$ batches, apply manifold ops once per type],
  [`manifolds.jl`],
  [No Riemannian Adam],
  [Custom `RiemannianAdam` with momentum transport + `RiemannianGD` with Armijo line search],
  [`optimizers.jl`],
  [No GPU support],
  [Batched linear algebra dispatching to CUBLAS on GPU; device abstraction via `to_device`],
  [`ext/CUDAExt.jl`],
  [Single-image einsum contraction],
  [Batched einsum: append batch label to the contraction so $B$ images share one kernel call],
  [`loss.jl`],
)

== Solution 1: Batched Manifold Operations (`manifolds.jl`)

*Key insight*: Circuit tensors are either $U(2)$ unitaries or $U(1)^4$ phase gates. Group all same-type tensors into a single 3D array and apply manifold operations to the whole batch.

```
Before (Manopt.jl):                After (custom):
  for each tensor:                   group_by_manifold(tensors)
    project(manifold_i, tensor_i)    stack_tensors → (d, d, K)
    retract(manifold_i, tensor_i)    project(manifold, batch)   # one call
                                     retract(manifold, batch)   # one call
                                     unstack_tensors!
```

Two manifold types implemented:

*`UnitaryManifold`* --- $U(n)$ unitary group:
- Projection: $"proj"(bold(U), bold(G)) = bold(U) dot "skew"(bold(U)^dagger bold(G))$, #h(1em) where $"skew"(bold(A)) = (bold(A) - bold(A)^dagger) slash 2$
- Retraction (Cayley): $"Retr"_(bold(U))(bold(xi)) = (bold(I) - alpha/2 dot bold(W))^(-1)(bold(I) + alpha/2 dot bold(W)) dot bold(U)$, #h(1em) $bold(W) = bold(xi) bold(U)^dagger$
- Transport: $Gamma_(bold(U) arrow.r bold(U)') (bold(v)) = "proj"(bold(U)', bold(v))$ (re-project onto new tangent space)

*`PhaseManifold`* --- $U(1)^d$ product of unit circles:
- Projection: $"proj"(bold(z), bold(g)) = upright(i) dot "Im"(overline(bold(z)) dot.o bold(g)) dot.o bold(z)$  #h(1em) ($upright(i) = sqrt(-1)$)
- Retraction: $"Retr"_(bold(z))(bold(xi)) = (bold(z) + alpha bold(xi)) ⊘ |bold(z) + alpha bold(xi)|$ (element-wise normalize)
- Transport: $Gamma_(bold(z) arrow.r bold(z)') (bold(v)) = "proj"(bold(z)', bold(v))$

#line(length: 100%)

```julia
# Classify and batch tensors ONCE before training loop
manifold_groups = group_by_manifold(tensors)  # Dict{Manifold, Vector{Int}}
for (manifold, indices) in manifold_groups
    pb = stack_tensors(tensors, indices)       # (d, d, K) batch
    rg = project(manifold, pb, grad_batch)     # batched projection
    new_pb = retract(manifold, pb, .-rg, α)    # batched retraction
    unstack_tensors!(tensors, new_pb, indices)  # write back
end
```

#line(length: 100%)

== Solution 2: Custom Optimizers (`optimizers.jl`)

=== `RiemannianGD` with Armijo Line Search

Gradient descent with adaptive step size. At each iteration, compute the Riemannian gradient $tilde(bold(g))$ and search for a step size $alpha$ satisfying the Armijo sufficient decrease condition:

$cal(L)("Retr"_(bold(theta))(-alpha tilde(bold(g)))) <= cal(L)(bold(theta)) - c dot alpha dot ||tilde(bold(g))||^2$

The algorithm tries $alpha, alpha tau, alpha tau^2, dots$ (backtracking with factor $tau in (0,1)$) until the condition holds. All retraction trials use batched operations.

=== `RiemannianAdam` with Momentum Transport

Extends Adam to manifolds (Bécigneul & Ganea, 2019). The key challenge: after retraction, the first moment $bold(m)_t$ lives in the _old_ tangent space and must be transported to the _new_ one.

$bold(m)_t = beta_1 bold(m)_(t-1) + (1 - beta_1) tilde(bold(g))_t$, #h(2em) $bold(v)_t = beta_2 bold(v)_(t-1) + (1 - beta_2) |tilde(bold(g))_t|^2$

Bias-corrected direction: $bold(d)_t = (bold(m)_t \/ (1 - beta_1^t)) / (sqrt(bold(v)_t \/ (1 - beta_2^t)) + epsilon)$

Update and transport:
$bold(theta)_(t+1) = "Retr"_(bold(theta)_t)(-alpha bold(d)_t)$, #h(2em) $bold(m)_t arrow.l Gamma_(bold(theta)_t arrow.r bold(theta)_(t+1))(bold(m)_t)$

#line(length: 100%)

```julia
# Per-manifold Adam update (fused broadcasts over (d,d,K) arrays)
@. m_state = β₁ * m_state + (1 - β₁) * rg        # moment update
@. v_state = β₂ * v_state + (1 - β₂) * abs2(rg)   # variance update
@. dir = (m_state / bc1) / (√(v_state / bc2) + ε)  # bias-corrected
new_batch = retract(manifold, old_batch, .-dir, lr) # retract
m_state = transport(manifold, old_batch, new_batch, m_state)  # transport
```

#line(length: 100%)

== Solution 3: Batched Multi-Image Einsum (`loss.jl`)

=== Background: OMEinsum and the Forward Transform

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

where we used the label mapping: $5 arrow.r i_1, 6 arrow.r i_2, 7 arrow.r i_3, 8 arrow.r i_4$ (contracted) and $1 arrow.r j_1, 2 arrow.r j_2, 3 arrow.r j_3, 4 arrow.r j_4$ (output). In general, for $m$ row qubits and $n$ column qubits:

$ bold(Y)_(j_1, dots, j_(m+n)) = sum_(i_1, dots, i_(m+n)) product_(k=1)^K (bold(theta)_k)_(a_k, b_k) dot bold(X)_(i_1, dots, i_(m+n)) $

where each gate $bold(theta)_k$ connects a pair of indices $(a_k, b_k)$ determined by the circuit structure, and all $i_ell$ that do not appear in the output are summed over.

=== The Problem: Single-Image Contraction

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

This is inefficient because the contraction tree is _identical_ for every image --- only the image tensor (the last entry `[5,6,7,8]`) changes. The gate tensors $bold(theta)_1, dots, bold(theta)_K$ are shared across all images, yet `optcode(tensors..., reshape(img, ...))` must be called $N$ times.

=== The Solution: Append a Batch Dimension

The key insight: add a new _batch label_ to the image input and output index lists, leaving gate indices unchanged. Since this label is not contracted (it appears in both the image input and the output), OMEinsum processes all $B$ images in a single kernel call.

Continuing the $m = 2, n = 2$ example, `make_batched_code` computes `batch_label = max(1,...,8) + 1 = 9` and appends it:

#align(center)[`DynamicEinCode([[5,1], [6,2], [7,3], [8,4], [5,2], [7,4],` *`[5,6,7,8,9]`*`],` *`[1,2,3,4,9]`*`)`]

Only the last two entries change (bold): the image input gains label 9, the output gains label 9. All gate index lists are unchanged. The contraction becomes:

$ bold(Y)_(j_1, j_2, j_3, j_4, b) = sum_(i_1, i_2, i_3, i_4) (bold(theta)_1)_(i_1, j_1) dots.c (bold(theta)_6)_(i_3, i_4) bold(X)_(i_1, i_2, i_3, i_4, b) $

where $b in {1, dots, B}$ indexes images in the batch. Because $b$ is a free index (not summed over), the gate contractions are identical for every image --- OMEinsum handles this in one call. In general:

$ bold(Y)_(j_1, dots, j_(m+n), b) = sum_(i_1, dots, i_(m+n)) product_(k=1)^K (bold(theta)_k)_(a_k, b_k) dot bold(X)_(i_1, dots, i_(m+n), b) $

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

The implementation in `loss.jl` has three steps:

#line(length: 100%)

```julia
# Step 1: Append batch label to the einsum code
# make_batched_code finds max label (=8), sets batch_label = 9,
# appends it to image input [5,6,7,8] → [5,6,7,8,9]
# and output [1,2,3,4] → [1,2,3,4,9]
batched_flat, batch_label = make_batched_code(optcode, n_gates)

# Step 2: Re-optimize contraction order for batched tensor sizes
# size_dict: {1=>2, 2=>2, ..., 8=>2, 9=>B}  (all qubit dims = 2, batch dim = B)
batched_optcode = optimize_batched_code(batched_flat, batch_label, batch_size)

# Step 3: Stack B images and evaluate in one einsum call
# Each image (2^m × 2^n) is reshaped to (2,2,...,2) then stacked along dim m+n+1
stacked = cat([reshape(img, fill(2,m+n)...) for img in batch]...; dims=m+n+1)
result = batched_optcode(tensors..., stacked)  # shape: (2,2,...,2,B)
```

#line(length: 100%)

=== Loss Dispatch by Type

The batching strategy differs by loss function:

- *L1/L2*: Fully batched end-to-end. One einsum call produces all $B$ outputs as a $(2,dots,2,B)$ tensor, then a single `sum(abs.())` or `sum(abs2.())` reduction:

$cal(L)_"L1" = 1/B sum_(b=1)^B sum_(j_1, dots, j_(m+n)) |bold(Y)_(j_1, dots, j_(m+n), b)|$, #h(2em) $cal(L)_"L2" = 1/B sum_(b=1)^B sum_(j_1, dots, j_(m+n)) |bold(Y)_(j_1, dots, j_(m+n), b)|^2$

- *MSE*: The forward pass is batched (one einsum call), but `topk_truncate` must run per-image because the truncation mask depends on each image's frequency content. The inverse transform also runs per-image. So MSE is "half-batched": batched forward, sequential truncation + inverse:

$cal(L)_"MSE" = 1/B sum_(b=1)^B || bold(X)_b - bold(Y)^(-1)_b ||^2$, #h(2em) where $bold(Y)^(-1)_b = "einsum"^(-1)(bold(theta)_1^*, dots, bold(theta)_K^*, "truncate"_k (bold(Y)_b))$

Here $"einsum"^(-1)$ denotes the inverse transform (using conjugated tensors $bold(theta)_k^*$), and $"truncate"_k$ keeps the top-$k$ frequency-weighted coefficients of $bold(Y)_b$.

=== How Everything Fits Together: The Training Loop

The diagram below shows one iteration of the training loop. The gate tensors and image batch feed into the batched einsum; the output flows through the loss, then Zygote differentiates back to get gradients, which the Riemannian optimizer uses to update the tensors on the manifold:

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

== Remaining Problem

The batched operations run on CPU with per-slice loops:

```
batched_matmul: for k in 1:K; mul!(C[:,:,k], A[:,:,k], B[:,:,k]); end
batched_inv:    for k in 1:K; C[:,:,k] = inv(A[:,:,k]); end
```

For $2 times 2$ matrices, each operation takes nanoseconds but each kernel launch takes ~10$mu$s. On GPU, $K$ separate launches would be slower than CPU. This motivates Stage 3.

= Stage 3: GPU Acceleration

== Problem $arrow.r$ Solution Overview

#table(
  columns: 3,
  [*Problem from Stage 2*], [*Solution*], [*File*],
  [$K$ separate kernel launches for $K$ matrix ops],
  [`CUBLAS.gemm_strided_batched!` --- single kernel for all $K$ multiplications],
  [`ext/CUDAExt.jl`],
  [Per-slice `inv()` too small for GPU],
  [Closed-form $2 times 2$ inverse: $mat(a,b;c,d)^(-1) = (a d - b c)^(-1) mat(d,-b;-c,a)$],
  [`ext/CUDAExt.jl`],
  [CPU-only einsum],
  [`CuArray` dispatch --- OMEinsum automatically uses GPU contraction],
  [`ext/CUDAExt.jl`],
  [Device management boilerplate],
  [`to_device(x, :gpu/:cpu)` abstraction],
  [`ext/CUDAExt.jl`],
)

== GPU Dispatch Table

On GPU, the batched operations dispatch to optimized kernels:

#table(
  columns: 3,
  [*Operation*], [*CPU (manifolds.jl)*], [*GPU (CUDAExt.jl)*],
  [`batched_matmul`], [Per-slice `mul!` loop], [`CUBLAS.gemm_strided_batched!` (1 kernel)],
  [`batched_inv`], [Per-slice `inv()` loop], [Broadcast $2 times 2$ formula (1 kernel)],
  [`batched_adjoint`], [`permutedims(conj.())`], [Same (GPU broadcast)],
  [Einsum], [CPU OMEinsum], [CuArray dispatch (GPU contraction)],
)

== When GPU Wins vs CPU

- *Small circuits* ($< 10$ tensors): CPU faster due to GPU transfer overhead
- *Large circuits or batches*: GPU wins from batched parallelism
- *L1/L2 loss*: fully batched, GPU-friendly
- *MSE loss*: `topk_truncate` is per-image, limiting GPU parallelism

= Sparse Basis Types

Three basis types are implemented, each a subtype of `AbstractSparseBasis`:

#table(
  columns: 3,
  [*Basis Type*], [*Circuit Structure*], [*Parameters*],
  [`QFTBasis`],
  [Standard QFT: Hadamard $H$ + controlled-phase $M_k$ gates, independent row/column qubits],
  [$H in U(2)$, $M_k in U(1)^4$],
  [`EntangledQFTBasis`],
  [QFT + entanglement gates $E_k$ between row/column qubits (captures XY correlation)],
  [QFT params + $n$ additional phases $phi_k$],
  [`TEBDBasis`],
  [Ring topology of Hadamard + controlled-phase gates (brickwork pattern)],
  [$H in U(2)$, $T_k in U(1)^4$],
)

Frequency-dependent truncation (`topk_truncate`) for compression. The score for coefficient $(i,j)$ is:

$s_(i,j) = |bold(Y)_(i,j)| dot (1 + w_(i,j))$, #h(2em) $w_(i,j) = 1 - d_(i,j) / (2 d_"max")$

where $d_(i,j) = sqrt((i - i_c)^2 + (j - j_c)^2)$ is the distance from the DC component (center), and $d_"max"$ is the maximum such distance. The top-$k$ coefficients by score $s_(i,j)$ are kept; the rest are zeroed. This weighting favors low-frequency components (structural information) over high-frequency ones (fine details) of similar magnitude.

= Running Examples

== Optimizer Benchmark

Run `examples/optimizer_benchmark.jl` to compare:
- Manopt.jl GD (CPU baseline)
- PDFT `RiemannianGD` (CPU/GPU)
- PDFT `RiemannianAdam` (CPU/GPU)

== Image Compression

Run `examples/basis_demo.jl` for a complete image compression workflow:
1. Load image and create a sparse basis (`QFTBasis`, `EntangledQFTBasis`, or `TEBDBasis`)
2. Train the basis on the image using `train_basis!`
3. Compress the image using `compress_image`
4. Compare reconstruction quality with the standard Fourier basis

= Tasks
- Create an image dataset $cal(D) = {bold(X)_b}_(b=1)^N$
- Train the tensor network on the dataset and compare with Fourier basis
- Use GPU to speed up training (see `ext/CUDAExt.jl`)
- Experiment with different loss functions (L1, MSE, Hybrid)
- Experiment with different basis types (QFT, Entangled QFT, TEBD)
- Add edge detection features
- Compare compression quality at different compression ratios (60%, 70%, 95%)
