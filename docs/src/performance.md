# Performance

This page characterises the cost of the batched training inner loop and
documents the speed-ups that batching gives over a naive per-image
baseline.

## Methodology

All numbers come from
[`examples/benchmark_training.jl`](https://github.com/nzy1997/ParametricDFT.jl/blob/main/examples/benchmark_training.jl),
which measures with `@elapsed` (min of 3 trials after 2 untimed warm-ups)
the wall-clock time, allocation count, and peak memory of
five [`optimize!`](@ref) iterations for every combination of

* **batch size** `B ∈ {1, 4, 16, 64}`,
* **loss** `L1Norm()` or `MSELoss(k = 8)`,
* **optimizer** `RiemannianGD(lr = 0.01)` or `RiemannianAdam(lr = 0.001)`.

`time/iter/B` below is the per-image cost in microseconds — the
quantity that would be **flat** in `B` if batching were perfectly
amortised. A downward curve is a real batching saving; a flat or
slowly-falling curve signals a per-image bottleneck.

Results were captured on the CPU path (`julia-1.12.5`, single-threaded,
no CUDA). A GPU column will be added once someone runs the same script
on a CUDA host — the harness picks up `CuArray` tensors automatically
when `using CUDA` has loaded the extension.

To regenerate:

```bash
julia --project=. examples/benchmark_training.jl
```

## CPU results (`m = n = 2`, 4 × 4 images, `K = 6` gate tensors)

### L1 loss, `RiemannianGD`

| `B` | time / 5 iters (ms) | time/iter/B (μs) | allocs | memory (KiB) |
| ---:| ---:| ---:| ---:| ---:|
|   1 |  6.25 | 1250.9 |  34370 |  1578.3 |
|   4 |  6.38 |  319.1 |  32423 |  1681.8 |
|  16 |  6.62 |   82.8 |  32366 |  2375.7 |
|  64 |  7.77 |   24.3 |  34208 |  5308.5 |

### L1 loss, `RiemannianAdam`

| `B` | time / 5 iters (ms) | time/iter/B (μs) | allocs | memory (KiB) |
| ---:| ---:| ---:| ---:| ---:|
|   1 |  5.71 | 1142.2 |  28829 |  1329.8 |
|   4 |  5.86 |  293.1 |  27194 |  1402.3 |
|  16 |  6.15 |   76.9 |  27089 |  1926.6 |
|  64 |  7.12 |   22.3 |  28889 |  4272.4 |

### MSE loss (`k = 8`), `RiemannianGD`

| `B` | time / 5 iters (ms) | time/iter/B (μs) | allocs | memory (KiB) |
| ---:| ---:| ---:| ---:| ---:|
|   1 | 16.13 | 3226.6 |  69824 |  3195.0 |
|   4 | 18.44 |  922.0 |  75848 |  4063.2 |
|  16 | 22.67 |  283.4 |  91705 |  7401.1 |
|  64 | 40.04 |  125.1 | 169219 | 33996.5 |

### MSE loss (`k = 8`), `RiemannianAdam`

| `B` | time / 5 iters (ms) | time/iter/B (μs) | allocs | memory (KiB) |
| ---:| ---:| ---:| ---:| ---:|
|   1 | 16.82 | 3363.7 |  58416 |  2674.3 |
|   4 | 18.98 |  948.8 |  63550 |  3373.3 |
|  16 | 22.79 |  284.9 |  77397 |  6263.1 |
|  64 | 38.18 |  119.3 | 145563 | 30824.6 |

## Reading the numbers

**L1 batching is nearly ideal.** `time/iter/B` drops **~52×** from
`B = 1` to `B = 64` (from ~1.2 ms/image/iter to ~23 μs/image/iter).
Allocations are essentially **flat** with `B`
(34 370 → 34 208 for GD): the inner loop reuses `OptimizationState`'s
persistent buffers; what grows with `B` is only the einsum output
tensor. This matches the "image-level batching" story in the
[notes](https://github.com/nzy1997/ParametricDFT.jl/blob/main/note/batchGPU.pdf) §3–§4.

**MSE batching is bottlenecked by the per-image `topk_truncate`.**
`time/iter/B` drops ~26× (half the L1 scaling) and **allocations grow
2.4× from `B = 1` to `B = 64`** (70 k → 169 k for GD) because the
`map(1:B)` in `batched_loss_mse` runs the content-dependent top-$k$
selection once per image. Memory also grows roughly linearly in `B`.
This validates the note's claim that the per-image topk is the last
remaining serial step on the MSE path. A batched segmented-sort
path on GPU (CUB `cub::DeviceSegmentedRadixSort`) would close this
gap; tracked in
[issue #70](https://github.com/nzy1997/ParametricDFT.jl/issues/70).

## Known bottlenecks and follow-ups

* **Per-image `topk_truncate` in the MSE path** — above. Batched
  implementation on GPU is the single biggest untaken speedup.
* **`Complex{Float64}` throughout.** A mixed-precision
  (`Complex{Float32}`) variant would roughly halve memory and often
  double throughput on GPU; tracked in issue #70.
* **Per-iteration allocations inside `project` and `retract`** —
  each call to `project(::UnitaryManifold, ...)` and
  `retract(::UnitaryManifold, ...)` allocates several `(d, d, K)`
  arrays. Writing in-place variants plus batched `batched_matmul!`
  would cut the low-level allocation count further; an architectural
  change, tracked in issue #70.

## History

| PR | change | observed effect |
| --- | --- | --- |
| [#71](https://github.com/nzy1997/ParametricDFT.jl/pull/71) | reuse Armijo `last_cand_batches` Dict across line-search trials | up to `max_ls_steps - 1` fewer Dict allocations per iteration |
| [#72](https://github.com/nzy1997/ParametricDFT.jl/pull/72) | pre-allocate `euclidean_grads_buf` in `OptimizationState` | one fewer `Vector{AbstractMatrix}` per iteration |
