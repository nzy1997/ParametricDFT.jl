# Remove Materialized Unitary Path

**Date:** 2026-03-03
**Status:** Approved

## Context

The materialized unitary approach (`src/materialized.jl`) builds a full D×D unitary matrix from circuit gate tensors, then uses `U*X` (single GEMM) instead of per-image einsum contractions. This was designed to reduce GPU kernel launch overhead for small images.

Benchmark results on RTX 3090 (`examples/gpu_benchmark_output.log`) showed:

- **32×32 (D=1024):** Materialized gradient is 175x slower than einsum (1465ms vs 8.4ms CPU)
- **64×64 (D=4096):** OOM on GPU during Zygote backward pass
- **256×256+ images:** Einsum-on-GPU already gives 5-7x speedup over CPU without materialization

The materialized path is counterproductive because building U via `optcode(tensors..., I_D)` applies the entire circuit to D basis vectors — far more expensive than applying it to a small batch of images.

## Decision

Delete `materialized.jl` entirely. The einsum path is the correct approach for all image sizes:
- Small images (≤128×128): CPU is faster than GPU regardless of approach
- Large images (256×256+): GPU einsum already saturates GPU cores, no materialization needed

## Changes

| Action | File | Details |
|--------|------|---------|
| Delete | `src/materialized.jl` | Entire file (124 lines) |
| Delete | `test/materialized_tests.jl` | Entire file (264 lines) |
| Edit | `src/ParametricDFT.jl` | Remove `include("materialized.jl")` |
| Edit | `src/training.jl` | Remove strategy selection + materialized loss branch |
| Edit | `test/runtests.jl` | Remove `include("materialized_tests.jl")` |
| Edit | `examples/profile_gpu.jl` | Remove materialized benchmarks (sections 13-17) |
| Edit | `examples/gpu_benchmark.jl` | Remove Part 4 (materialized vs einsum) |

### Training loop simplification

Before (3 code paths):
```
if strategy == :materialized_gpu      → DELETE
elseif batched_optcode !== nothing    → KEEP (batched einsum)
else                                  → KEEP (per-image einsum)
```

After (2 code paths): batched einsum (batch_size > 1) and per-image einsum (batch_size = 1).

### No API changes

`train_basis` public API unchanged. All removed functions were internal (not exported).

## Impact on GPU Optimizer Redesign

The `docs/plans/2026-03-03-gpu-optimizer-redesign.md` design doc is superseded for the materialized components. The ManifoldParameterGroup + stateful Adam improvements described there are orthogonal and can proceed independently if desired.
