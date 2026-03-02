# PR #35 Review Comments Resolution — Design

**Date:** 2026-03-02
**PR:** https://github.com/nzy1997/ParametricDFT.jl/pull/35
**Review:** https://github.com/nzy1997/ParametricDFT.jl/pull/35#pullrequestreview-3870483431

## Goal

Address 15 remaining reviewer comments on PR #35. Five comments were already resolved by prior file consolidation (batched_einsum.jl, riemannian_optimizers.jl, test files merged).

## Decisions

- **Verbose/println**: Remove entirely from all src/ files. Users inspect the returned `history` object.
- **NVTX profiling**: Delete from core. CUDAExt can add GPU profiling later if needed.
- **Docstrings**: Trim to one-liner purpose + concise inline param descriptions. No separate Arguments/Returns headings.
- **Batched einsum structure**: Keep current decomposition (correct separation of concerns). Add inline comments explaining OMEinsum batching strategy.
- **`fft_with_training`**: Delete entirely — superseded by `train_basis`.
- **`to_device`**: Short-circuit with ternary operator.
- **`optimizer_crosscheck.jl`**: Merge into `basis_demo.jl` with comparison plot and table for GPU and CPU batched versions.
- **`group_by_manifold`**: Simplify with `get!` pattern.

## Changes by File

### `src/optimizers.jl`
1. Delete NVTX profiling code: `_nvtx_push_fn`, `_nvtx_pop_fn`, `_nvtx_range_push`, `_nvtx_range_pop`, and all call sites
2. Remove `verbose` parameter from `optimize!(::RiemannianGD)` and `optimize!(::RiemannianAdam)`
3. Delete all `println` inside both optimizers
4. Trim docstrings to concise format

### `src/training.jl`
5. Remove `verbose` parameter and all `println` from `_train_basis_core`
6. Short-circuit `to_device` with ternary one-liner
7. Add comment on `copy.` for early stopping snapshot
8. Delete verbose helper / optimizer name formatting block

### `src/loss.jl`
9. Trim docstrings to concise format
10. Add brief inline comments explaining OMEinsum batching strategy

### `src/qft.jl`
11. Delete `fft_with_training` function entirely

### `src/ParametricDFT.jl`
12. Remove `fft_with_training` from exports

### `src/manifolds.jl`
13. Simplify `group_by_manifold` with `get!` pattern
14. Trim docstrings

### `ext/CUDAExt.jl`
15. Remove NVTX callback injection code

### `examples/`
16. Merge `optimizer_crosscheck.jl` into `basis_demo.jl` with comparison plot and table for both GPU and CPU batched versions
17. Delete `optimizer_crosscheck.jl`

### `test/runtests.jl`
18. Remove `"fft with training"` testset

### `test/training_tests.jl` + `test/optimizer_tests.jl`
19. Update all callers to remove `verbose` kwarg

## Not Changed (already resolved)
- `src/batched_einsum.jl` → merged into `loss.jl`
- `src/riemannian_optimizers.jl` → renamed to `optimizers.jl`
- `test/batched_einsum_tests.jl`, `test/batched_ops_tests.jl` → merged into `loss_tests.jl`, `manifold_tests.jl`
