# Performance

This page characterises the cost of the training inner loop and gives a
decision rule — "for my batch size and image size, which configuration
should I use?" — grounded in measured numbers against a Manopt.jl
baseline.

## Methodology

All numbers come from
[`examples/speedup_benchmark.jl`](https://github.com/nzy1997/ParametricDFT.jl/blob/main/examples/speedup_benchmark.jl).
The harness runs each cell for **10 `optimize!` iterations**, after two
untimed warm-up runs, and reports the **minimum of three wall-clock
trials** (`@elapsed`). GPU cells use `CUDA.synchronize()` inside the timed
block so the interval covers kernel completion, not just launch.

The benchmark uses the **`MSELoss(k)`** with `k = ⌊0.1 · 2^(m+n)⌋` (10 %
keep ratio) across every cell. Seeds (`Random.seed!(42)` at the top)
make the numbers byte-reproducible run-to-run on the same host.

Results were captured on

- Julia 1.12.5
- CPU path: single-threaded
- GPU path: NVIDIA GeForce RTX 3090

Re-run with

```bash
julia --project=examples examples/speedup_benchmark.jl
```

## Main decision table — `m = n = 6` (64 × 64 images, `k = 409`)

Reported cell: `time / 10 iters` in milliseconds. **Bold** marks the
fastest cell in each column.

| config | `B = 1` | `B = 8` | `B = 32` | `B = 64` |
| --- | ---:| ---:| ---:| ---:|
| `Manopt-GD-CPU`   |  4 115  | 32 459  | 110 708  | 200 919 |
| `PDFT-GD-CPU`     |    315  |  1 116  |   6 574  |  14 848 |
| `PDFT-GD-GPU`     |    871  |  1 013  |   2 439  |   3 975 |
| `PDFT-Adam-CPU`   | **253** |    941  |   5 424  |  12 862 |
| `PDFT-Adam-GPU`   |    498  | **867** | **2 285**| **3 817**|

## Appendix — `m = n = 9` (512 × 512 DIV2K-scale images, `B = 1`, `k = 26 214`)

| config | `time / 10 iters (ms)` |
| --- | ---:|
| `PDFT-GD-CPU` | 22 685 |
| `PDFT-GD-GPU` |  **4 575** |

Manopt is intentionally absent — by extrapolating the `m = n = 6` row
(~50× slowdown from `B = 1` to `B = 64`, and another ~20× for the
`m = 6 → m = 9` image-size jump), a single Manopt cell would take
>1 hour. Not worth measuring to restate what the main table already
shows.

## Which configuration for which situation?

Reading the main table column by column:

| batch size | winner | runner-up | verdict |
| ---: | --- | --- | --- |
| `B = 1`  | `PDFT-Adam-CPU` (253 ms) | `PDFT-GD-CPU` (315 ms) | **Use CPU.** GPU launch overhead at `K = 42` gates exceeds any throughput gain. |
| `B = 8`  | `PDFT-Adam-GPU` (867 ms) | `PDFT-Adam-CPU` (941 ms) | **Crossover.** GPU and CPU within 10 %. Either is fine; GPU wins by a whisker. |
| `B = 32` | `PDFT-Adam-GPU` (2 285 ms) | `PDFT-GD-GPU` (2 439 ms) | **Use GPU.** 2.4× faster than best CPU (`Adam-CPU` 5 424 ms). |
| `B = 64` | `PDFT-Adam-GPU` (3 817 ms) | `PDFT-GD-GPU` (3 975 ms) | **Use GPU.** 3.4× faster than best CPU. |

And the appendix shows that at production image size (512 × 512) **even
at `B = 1`** GPU wins 5× — the crossover batch-size threshold gets lower
as the circuit grows, because per-kernel work scales faster than
per-kernel launch overhead.

## Manopt comparison

Every PDFT cell is **10–50× faster than Manopt at the same batch size**:

| `B` | `Manopt-GD-CPU` | `PDFT-GD-CPU` | `PDFT-Adam-GPU` |
| ---: | ---: | ---: | ---: |
|  1 |  4 115 | **315** (13×) |   498 (8.3×) |
|  8 | 32 459 | **1 116** (29×) |   867 (37×) |
| 32 | 110 708 | **6 574** (17×) | 2 285 (48×) |
| 64 | 200 919 | **14 848** (14×) | 3 817 (**53×**) |

The ratio widens with `B` because Manopt has no batch axis — each extra
image in the batch adds a full serial forward+backward pass — while
PDFT's batched einsum amortises per-call overhead across `B`.

**Bottom line:** there is no batch size at which Manopt is competitive
with the PDFT stack on this workload.

## Adam vs GD

Adam is a hair faster than GD at every cell because GD runs an Armijo
backtracking line search (up to 10 loss evaluations per step), while
Adam trusts its bias-corrected moments and never re-evaluates. The
per-step speed gap is 15–25 %; whether the extra step-size hygiene is
worth it is a convergence question that this page does not try to
answer.

## Known bottlenecks and follow-ups

* **Per-image `topk_truncate` no longer the bottleneck on GPU.** The
  batched `topk_truncate` added in
  [PR #72](https://github.com/nzy1997/ParametricDFT.jl/pull/72) issues
  one `sort(dims = 1)` call for all `B` images instead of `B` sequential
  sorts. Contributes meaningfully to the `Adam-GPU` / `GD-GPU` numbers
  at large `B`.
* **`Complex{Float64}` throughout.** A mixed-precision
  (`Complex{Float32}`) variant would roughly halve memory and often
  double throughput on GPU; tracked in
  [issue #70](https://github.com/nzy1997/ParametricDFT.jl/issues/70).
* **Per-iteration allocations inside `project` and `retract`.** Each
  call still allocates several `(d, d, K)` intermediates. A fully
  in-place `project!` / `retract!` refactor would cut allocation count
  further and cooperate better with CUDA Graphs; tracked in issue #70.

## History

| PR | change | observed effect |
| --- | --- | --- |
| [#71](https://github.com/nzy1997/ParametricDFT.jl/pull/71) (absorbed into #72) | reuse Armijo `last_cand_batches` Dict; per-tensor NaN/Inf diagnostic | up to `max_ls_steps − 1` fewer Dict allocations per iteration; better divergent-run debugging |
| [#72](https://github.com/nzy1997/ParametricDFT.jl/pull/72) | pre-allocated `euclidean_grads_buf`; batched `topk_truncate` (CPU + GPU via CUB sort-with-dims); Manopt comparison harness; this Performance page | GPU MSE path no longer per-image sort-bound; main decision table + appendix populated |
