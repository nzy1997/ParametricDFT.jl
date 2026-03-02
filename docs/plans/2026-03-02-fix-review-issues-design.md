# Fix Plan: Code Review Issues on `feature/riemannian-optimizers`

**Date:** 2026-03-02
**Branch:** `feature/riemannian-optimizers`
**Scope:** Critical and Important issues (1–7) from the post-PR code quality review.

---

## Background

A code quality review of the Riemannian optimizers feature branch identified 10 issues.
This plan addresses the 7 Critical and Important issues, grouped into 3 logical commits.

---

## Chunk 1 — GPU/Core Correctness (Issues 1 & 2)

**Files:** `src/manifolds.jl`, `ext/CUDAExt.jl`
**Commit:** `fix: GPU-compatible batched ops and topk_mask DRY refactor`

### Issue 1: `batched_matmul` and `batched_adjoint` are GPU-incompatible

Both functions use triple-nested scalar-indexing loops over array elements. CUDA.jl
forbids scalar indexing on `CuArray` by default, so GPU training with unitary tensors
errors immediately at `UnitaryManifold.project`.

**Fix:**
- `batched_adjoint`: replace loop with `permutedims(conj.(A), (2, 1, 3))` — a single
  broadcast+permute that works on any `AbstractArray` including `CuArray`.
- `batched_matmul` (CPU): replace loops with a `for k in 1:n; mul!(view(C,:,:,k), ...)`
  pattern using BLAS `mul!`.
- `batched_matmul` (GPU): add a `CuArray`-specialised method in `CUDAExt.jl` that
  uses a per-slice page-copy approach (`C[:,:,k] .= A[:,:,k] * B[:,:,k]` for each `k`).
  This approach was chosen over `NNlib.batched_mul` for three reasons:
  1. `NNlib` is not a listed dependency and adding it would require `Project.toml` changes,
     introducing an undesirable new dependency for a single operation.
  2. The page-copy approach dispatches through CUBLAS for each slice via the standard
     CUDA.jl `*` overload for `CuMatrix`, which is correct and avoids the new dependency.
  3. `mul!(view(CuArray,...), view(CuArray,...), view(CuArray,...))` may also route
     through CUBLAS in future Julia/CUDA.jl versions, but the explicit CUDAExt method
     ensures correct (non-scalar-indexing) behavior now.

### Issue 2: Score-computation logic duplicated in `CUDAExt.jl`

The block that builds `scores`, calls `partialsortperm`, and constructs the binary mask
is copy-pasted between `topk_truncate` and its `rrule`. A bug fix in one won't propagate
to the other, risking forward/backward mask mismatch — an AD correctness bug.

**Fix:** Extract a private `_topk_mask(x_cpu, k, m, n) -> mask_gpu` helper. Both
`topk_truncate` and its `rrule` call this helper.

---

## Chunk 2 — Type Stability + AD (Issues 3 & 7)

**Files:** `src/optimizers.jl`
**Commit:** `fix: typed optimizer state dicts and stable gradient vector collection`

### Issue 3: `Dict{AbstractRiemannianManifold, Any}` causes type instability

`point_batches`, `grad_buf_batches`, `rg_batches`, `dir_buf_batches`, `m_batches`, and
`v_batches` all use `Any` as the value type. Every dict access in the hot optimizer loop
forces boxing and prevents compiler inference.

**Fix:** Change all optimizer state dictionaries to
`Dict{AbstractRiemannianManifold, AbstractArray}`. Every stored value is a batched array;
this is the correct concrete bound.

### Issue 7: `collect` on Tuple tangents may produce `Vector{Any}`

In `_compute_gradients`, `collect(euclidean_grads_raw)` on a heterogeneous `Tuple` falls
back to `Vector{Any}`, propagating type instability to all downstream gradient indexing.

**Fix:** Replace with `AbstractArray[euclidean_grads_raw[i] for i in eachindex(euclidean_grads_raw)]`,
producing a stable `Vector{AbstractArray}`.

---

## Chunk 3 — Test Quality (Issues 4, 5, 6)

**Files:** `test/training_tests.jl`, `test/loss_tests.jl`
**Commit:** `test: fix misnamed test, add loss-reduction assertions, add batched MSE gradient check`

### Issue 4: Misnamed test uses wrong optimizer

`"gradient_descent with batch_size > 1"` in `training_tests.jl` passes `optimizer=:adam`,
testing Adam again instead of the GD batch path.

**Fix:** Change `optimizer=:adam` to `optimizer=:gradient_descent` in that testset.

### Issue 5: Batched training tests are smoke-only

All three new batched training subtests only assert `basis isa QFTBasis`. They pass even
if the optimizer fails to reduce loss.

**Fix:** Compute average dataset loss before and after `train_basis` in each subtest,
then assert `final_loss < initial_loss`.

### Issue 6: No gradient test for batched MSE

`batched_loss_mse` — the most complex AD path (`selectdim` → `topk_truncate` → inverse
einsum) — has no Zygote gradient test.

**Fix:** Add `@testset "Zygote gradient through batched MSE"` inside the existing
`"Batched Einsum"` testset. Compute `Zygote.gradient` through `batched_loss_mse` and
compare against a finite-difference gradient to validate the full backward path.

---

## Success Criteria

- All 3 commits pass `julia --project=. -e 'using Pkg; Pkg.test()'`
- GPU training with unitary tensors no longer errors (verified via `test/cuda_tests.jl`
  on a CUDA machine, or by confirming no scalar indexing warnings with `CUDA.allowscalar(false)`)
- No `Vector{Any}` in gradient accumulation (verified by `@code_warntype` on `_compute_gradients`)
