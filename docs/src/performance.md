# Performance

Characterises the training inner loop and gives a decision rule — "for
my batch size and image size, which configuration should I use?" —
grounded in measured numbers against a Manopt.jl baseline.

## Methodology

Numbers come from
[`examples/speedup_benchmark.jl`](https://github.com/nzy1997/ParametricDFT.jl/blob/main/examples/speedup_benchmark.jl).
Each cell runs 10 `optimize!` iterations, after two untimed warm-ups,
reporting the minimum of three `@elapsed` trials. GPU cells use
`CUDA.synchronize()` inside the timed block so the interval covers
kernel completion, not just launch. Loss is `MSELoss(k)` with
`k = ⌊0.1 · 2^(m + n)⌋` (10 % keep ratio). Seed fixed at
`Random.seed!(42)`.

Host: Julia 1.12.5 on a single CPU thread + NVIDIA GeForce RTX 3090.
Re-run with

```bash
julia --project=examples examples/speedup_benchmark.jl
```

## Main table — `m = n = 6` (64 × 64 images, `k = 409`, 10 iters)

| config | `B = 1` | `B = 8` | `B = 32` | `B = 64` |
| --- | ---:| ---:| ---:| ---:|
| `Manopt-GD-CPU`   |  4 115 | 32 459 | 110 708 | 200 919 |
| `PDFT-GD-CPU`     |    315 |  1 116 |   6 574 |  14 848 |
| `PDFT-GD-GPU`     |    871 |  1 013 |   2 439 |   3 975 |
| `PDFT-Adam-CPU`   |    253 |    941 |   5 424 |  12 862 |
| `PDFT-Adam-GPU`   |    498 |    867 |   2 285 |   3 817 |

## Appendix — `m = n = 9` (512 × 512, `B = 1`, `k = 26 214`)

| config | `time / 10 iters (ms)` |
| --- | ---:|
| `PDFT-GD-CPU` | 22 685 |
| `PDFT-GD-GPU` |  4 575 |

## Decision rule

| batch size | use this | runner-up (ms) | verdict |
| ---: | --- | --- | --- |
| `B = 1`  | `PDFT-Adam-CPU` (253 ms) | `PDFT-GD-CPU` (315) | GPU launch overhead still dominates; stay on CPU. |
| `B = 8`  | `PDFT-Adam-GPU` (867 ms) | `PDFT-Adam-CPU` (941) | Crossover — GPU and CPU within 10 %. |
| `B = 32` | `PDFT-Adam-GPU` (2 285 ms) | `PDFT-GD-GPU` (2 439) | GPU wins, 2.4× over best CPU. |
| `B = 64` | `PDFT-Adam-GPU` (3 817 ms) | `PDFT-GD-GPU` (3 975) | GPU wins, 3.4× over best CPU. |

The appendix confirms the threshold drops as the circuit grows: at
512 × 512, GPU wins 5× even at `B = 1`.

- At `B = 64`, `PDFT-Adam-GPU` is **52.6× faster** than `Manopt-GD-CPU` —
  Manopt has no batch axis, so its wall-clock grows ~linearly in `B`
  while PDFT amortises batching. No batch size makes Manopt competitive.
- Adam is 15–25 % faster than GD at every cell because GD re-evaluates
  the loss in its Armijo line search; whether the step-size hygiene
  is worth the cost is a convergence question this page does not
  answer.

## Known bottlenecks

* **`topk_truncate` no longer per-image on GPU.** Batched version added
  in [#72](https://github.com/nzy1997/ParametricDFT.jl/pull/72) issues
  one `sort(dims = 1)` call for all `B` images instead of `B` sequential
  sorts.
* **`Complex{Float64}` throughout.** A `Complex{Float32}` variant would
  roughly halve memory and often double GPU throughput. Tracked in
  [#70](https://github.com/nzy1997/ParametricDFT.jl/issues/70).
* **Allocations inside `project` and `retract`** — each call still
  allocates several `(d, d, K)` intermediates. Tracked in #70.

---

_Numbers on this page captured by PR [#72](https://github.com/nzy1997/ParametricDFT.jl/pull/72)
(which absorbed [#71](https://github.com/nzy1997/ParametricDFT.jl/pull/71))._
