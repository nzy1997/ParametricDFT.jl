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
├── entangled_qft.jl          # Entangled QFT circuit (XY correlation)
├── tebd.jl                   # TEBD circuit (ring topology)
├── mera.jl                   # MERA-inspired circuit (hierarchical topology)
├── loss.jl                   # Loss functions: L1Norm, L2Norm, MSELoss
├── basis.jl                  # AbstractSparseBasis: QFTBasis,
│                             #   EntangledQFTBasis, TEBDBasis, MERABasis
├── manifolds.jl              # Riemannian manifolds: UnitaryManifold,
│                             #   PhaseManifold, batched linear algebra
├── optimizers.jl             # RiemannianGD (Armijo), RiemannianAdam
├── training.jl               # Training pipeline, checkpointing,
│                             #   early stopping
├── compression.jl            # Image compression using learned bases
├── serialization.jl          # Basis save/load (JSON3)
├── visualization.jl          # Training loss plots (CairoMakie)
└── circuit_visualization.jl  # Circuit diagram generation (plot_circuit)

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
├── mera_tests.jl             # MERA circuit tests
├── circuit_visualization_tests.jl  # Circuit visualization tests
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

These problems motivate building custom batched Riemannian optimizers with GPU support.

= Beyond the Manopt baseline

The batched, GPU-accelerated implementation that replaces the Manopt baseline is documented in two places:

- *Mathematical formulation.* `note/main.typ` section 6 (Riemannian Optimization) covers the manifold structure of $U(2)$ and $U(1)^4$, the Riemannian gradient projection, the Cayley retraction (with comparison to QR), Riemannian gradient descent with Armijo backtracking line search, Riemannian Adam with parallel transport of momentum, and the batched einsum formulation with its gradient computation.
- *Implementation and performance.* The GitHub Pages site at #link("https://nzy1997.github.io/ParametricDFT.jl/") hosts the API reference (Training page, autogenerated from `loss.jl`, `manifolds.jl`, `optimizers.jl`, `training.jl`, `einsum_cache.jl`) and the Performance page, which reports measured wall-clock times against a Manopt.jl baseline with a batch-size decision rule.

= Sparse Basis Types

Four basis types are implemented, each a subtype of `AbstractSparseBasis`:

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
  [Ring topology of Hadamard + controlled-phase gates],
  [$H in U(2)$, $T_k in U(1)^4$],
  [`MERABasis`],
  [MERA-inspired hierarchical topology of Hadamard + controlled-phase gates (multi-scale)],
  [$H in U(2)$, $D_k, W_k in U(1)^4$],
)

== MERA-inspired vs TEBD: Connectivity Patterns

The key difference between TEBD and the MERA-inspired basis is the _connectivity pattern_ of controlled-phase gates. Both use the same gate parameterization ($"diag"(1, 1, 1, e^(i phi_k))$), but they wire gates to different qubit pairs:

- *TEBD (ring topology)*: Each qubit connects to its nearest neighbor in a ring: $(1,2), (2,3), dots, (n-1,n), (n,1)$. This gives $n$ gates per dimension with $O(n)$ depth. The ring captures _local_ correlations with periodic boundary conditions.

- *MERA-inspired (hierarchical topology)*: Gates are organized in $log_2 n$ layers with increasing stride $s = 2^(l-1)$. Layer 1 connects nearby qubits (fine-scale correlations), layer 2 connects qubits at distance 2 (medium-scale), and so on. This gives $2(n-1)$ gates per dimension with $O(log n)$ depth. The hierarchy naturally captures _multi-scale_ features --- analogous to how image features exist at different scales (edges $arrow.r$ textures $arrow.r$ objects). Unlike true MERA, all gates are unitary controlled-phase gates rather than coarse-graining isometries, since image compression requires an invertible transform.

Both basis types require power-of-2 qubit counts per dimension. For 2D images, row and column dimensions are processed independently (separable structure).

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
1. Load image and create a sparse basis (`QFTBasis`, `EntangledQFTBasis`, `TEBDBasis`, or `MERABasis`)
2. Train the basis on the image using `train_basis!`
3. Compress the image using `compress_image`
4. Compare reconstruction quality with the standard Fourier basis

= Tasks
- Create an image dataset $cal(D) = {bold(X)_b}_(b=1)^N$
- Train the tensor network on the dataset and compare with Fourier basis
- Use GPU to speed up training (see `ext/CUDAExt.jl`)
- Experiment with different loss functions (L1, MSE, Hybrid)
- Experiment with different basis types (QFT, Entangled QFT, TEBD, MERA-inspired)
- Add edge detection features
- Compare compression quality at different compression ratios (60%, 70%, 95%)
