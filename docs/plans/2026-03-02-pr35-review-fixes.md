# PR #35 Review Comments Resolution — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address all remaining reviewer comments on PR #35 by removing verbose output, NVTX profiling, dead code, and simplifying docstrings/patterns.

**Architecture:** Remove all println/verbose from src/ files (users inspect returned history). Delete NVTX profiling from core (GPU-only concern). Delete `fft_with_training` (superseded by `train_basis`). Merge `optimizer_crosscheck.jl` into `basis_demo.jl`.

**Tech Stack:** Julia, OMEinsum, Zygote, CairoMakie (for plots in basis_demo.jl)

---

### Task 1: Remove NVTX profiling from core and CUDAExt

**Files:**
- Modify: `src/optimizers.jl` (lines 7-22 NVTX section, plus all `_nvtx_range_push/pop` call sites)
- Modify: `ext/CUDAExt.jl` (lines 57-60 `__init__` function)

**Step 1: Remove NVTX declarations and helpers from optimizers.jl**

Delete lines 7-22 (the entire NVTX section):
```julia
# ============================================================================
# NVTX Profiling Callbacks
# ============================================================================
# No-ops by default; CUDAExt sets real callbacks at init time.

const _nvtx_push_fn = Ref{Union{Nothing, Function}}(nothing)
const _nvtx_pop_fn = Ref{Union{Nothing, Function}}(nothing)

"""Push an NVTX range (no-op unless CUDAExt loaded)."""
_nvtx_range_push(name::String) = (_nvtx_push_fn[] !== nothing && _nvtx_push_fn[](name); nothing)

"""Pop the current NVTX range."""
_nvtx_range_pop() = (_nvtx_pop_fn[] !== nothing && _nvtx_pop_fn[](); nothing)
```

Then delete every `_nvtx_range_push(...)` and `_nvtx_range_pop()` call throughout the file. These occur at approximately lines: 38, 41, 78, 93, 138, 140, 192, 228, 232, 239, 292, 294, 361, 383.

**Step 2: Remove NVTX callback injection from CUDAExt.jl**

In `ext/CUDAExt.jl`, delete the `__init__` function (lines 57-60):
```julia
function __init__()
    ParametricDFT._nvtx_push_fn[] = name -> CUDA.NVTX.range_push(name)
    ParametricDFT._nvtx_pop_fn[] = () -> CUDA.NVTX.range_pop()
end
```

**Step 3: Run tests**

Run: `julia --project=. -e 'using Test, ParametricDFT, Random, LinearAlgebra, Zygote, Statistics, OMEinsum; include("test/optimizer_tests.jl")'`
Expected: All optimizer tests pass (no NVTX references in tests).

**Step 4: Commit**

```bash
git add src/optimizers.jl ext/CUDAExt.jl
git commit -m "refactor: remove NVTX profiling from core and CUDAExt"
```

---

### Task 2: Remove verbose/println from optimize! functions

**Files:**
- Modify: `src/optimizers.jl` — both `optimize!` methods
- Modify: `test/optimizer_tests.jl` — remove `verbose=false` kwargs

**Step 1: Remove verbose from RiemannianGD optimize! (lines ~130-250)**

Remove the `verbose::Bool = false` parameter from the function signature. Delete all lines containing `verbose &&` or `if verbose`. Specifically:
- Delete manifold classification printing (lines ~142-146)
- Delete initial loss printing (lines ~163-166)
- Delete iteration progress printing (lines ~196-198)
- Delete convergence message printing (line ~187)

Keep all non-verbose logic intact (the `return current_tensors` path, loss computation, line search, retraction).

**Step 2: Remove verbose from RiemannianAdam optimize! (lines ~287-393)**

Same treatment:
- Remove `verbose::Bool = false` from signature
- Delete manifold classification printing (lines ~296-300)
- Delete initial loss printing (lines ~326-329)
- Delete iteration progress printing (lines ~346-349)
- Delete convergence message (line ~352)

**Step 3: Remove verbose from _compute_gradients (line 37)**

Remove `verbose::Bool` parameter and the `verbose && println(...)` NaN/Inf warning (line ~52). Keep the `return nothing` for NaN/Inf — just don't print.

Update all call sites of `_compute_gradients` within the file to remove the `verbose` argument.

**Step 4: Update optimizer tests**

In `test/optimizer_tests.jl`, remove all `verbose=false` kwargs from `optimize!` calls. These are at approximately lines: 48, 60, 75, 80, 98, 116, 192, 199.

**Step 5: Run tests**

Run: `julia --project=. -e 'using Test, ParametricDFT, Random, LinearAlgebra, Zygote, Statistics, OMEinsum; include("test/optimizer_tests.jl")'`
Expected: All pass.

**Step 6: Commit**

```bash
git add src/optimizers.jl test/optimizer_tests.jl
git commit -m "refactor: remove verbose/println from optimize! functions"
```

---

### Task 3: Remove verbose/println from _train_basis_core and train_basis

**Files:**
- Modify: `src/training.jl` — `_train_basis_core` and all 3 `train_basis` methods
- Modify: `test/training_tests.jl` — remove `verbose=false` kwargs

**Step 1: Remove verbose from _train_basis_core**

In `src/training.jl`:
- Remove `verbose::Bool` parameter from function signature (line 42)
- Delete the entire verbose block (lines 85-104): device names, optimizer name formatting, all println
- Delete `verbose && println(...)` for batched einsum info (line 113)
- Delete `verbose && println(...)` for checkpoint interval (line 136)
- Delete `verbose && println(...)` for epoch number (line 142)
- Delete `verbose && print(...)` for batch loss (line 191)
- Delete `verbose && println(...)` for checkpoint saved (line 201)
- Delete `verbose && println()` blank line (line 205)
- Delete `verbose && println(...)` for avg train/val loss (lines 215-218 — the entire `if verbose` block)
- Delete `verbose && println(...)` for validation improved/no improvement (lines 223, 229)
- Delete `verbose && println(...)` for early stopping (line 232)
- Delete `verbose && println(...)` for training completed (line 241)
- Delete `verbose && println(...)` for loss history saved (line 246)
- Keep the `verbose=false` that is passed to `optimize!` — but since Task 2 removes that parameter, change the `optimize!` call (line ~178) to remove `verbose=false`.

**Step 2: Remove verbose from all 3 train_basis methods**

Remove `verbose::Bool = true` parameter from:
- `train_basis(::Type{QFTBasis}, ...)` (line ~271)
- `train_basis(::Type{EntangledQFTBasis}, ...)` (check similar location)
- `train_basis(::Type{TEBDBasis}, ...)` (check similar location)

Remove `verbose` from the call to `_train_basis_core` in each method (the positional arg at line ~296 and equivalents).

**Step 3: Update training tests**

In `test/training_tests.jl`, remove all `verbose=false` kwargs from `train_basis` calls. These are at approximately 17 locations throughout the file.

**Step 4: Run tests**

Run: `julia --project=. -e 'using Test, ParametricDFT, Random, LinearAlgebra, Zygote, Statistics, OMEinsum; include("test/training_tests.jl")'`
Expected: All 38 pass.

**Step 5: Commit**

```bash
git add src/training.jl test/training_tests.jl
git commit -m "refactor: remove verbose/println from training pipeline"
```

---

### Task 4: Delete fft_with_training

**Files:**
- Modify: `src/qft.jl` — delete function (lines 41-91)
- Modify: `src/ParametricDFT.jl` — remove from exports (line 20)
- Modify: `test/runtests.jl` — delete testset (lines 23-29)

**Step 1: Delete fft_with_training from qft.jl**

Delete lines 41-91 (the entire docstring + function). This includes the docstring starting at line 41 and function body through line 91.

**Step 2: Remove from exports**

In `src/ParametricDFT.jl` line 20, change:
```julia
export fft_with_training, qft_code, ft_mat, ift_mat
```
to:
```julia
export qft_code, ft_mat, ift_mat
```

Also update the comment on line 63 from:
```julia
# 2. QFT circuit (uses loss_function from loss.jl for fft_with_training)
```
to:
```julia
# 2. QFT circuit (uses loss.jl)
```

**Step 3: Delete testset from runtests.jl**

Delete lines 23-29:
```julia
@testset "fft with training" begin
    Random.seed!(1234)
    m, n = 2, 2
    pic = rand(ComplexF64, 2^m, 2^n)
    theta = ParametricDFT.fft_with_training(m, n, pic, ParametricDFT.L1Norm(); steps=10)
    @test theta isa ArrayPartition
end
```

**Step 4: Check if RecursiveArrayTools is still needed**

`fft_with_training` was the only user of `ArrayPartition` from `RecursiveArrayTools`. Check if anything else uses it. If not, remove from `using` in `src/ParametricDFT.jl` (line 6) and from `test/runtests.jl` (line 6).

**Step 5: Run tests**

Run: `julia --project=. -e 'using Test, ParametricDFT, Random, LinearAlgebra, Zygote, Statistics, OMEinsum; include("test/runtests.jl")'` (or just the qft testset inline)
Expected: "fft with training" testset is gone, "qft" and "fft and ifft are inverses" still pass.

**Step 6: Commit**

```bash
git add src/qft.jl src/ParametricDFT.jl test/runtests.jl
git commit -m "refactor: delete fft_with_training (superseded by train_basis)"
```

---

### Task 5: Simplify to_device with short-circuit

**Files:**
- Modify: `src/training.jl` (lines 10-23)

**Step 1: Replace to_device if-else with short-circuit**

Replace lines 10-23:
```julia
function to_device(x, device::Symbol)
    if device === :cpu
        return to_device(x, Val(:cpu))
    elseif device === :gpu
        # Check if the CUDAExt has added the Val{:gpu} method
        if hasmethod(to_device, Tuple{typeof(x), Val{:gpu}})
            return to_device(x, Val(:gpu))
        else
            error("GPU support requires CUDA.jl. Install and load CUDA.jl first: `using CUDA`")
        end
    else
        error("Unknown device: $device. Supported: :cpu, :gpu")
    end
end
```

With:
```julia
function to_device(x, device::Symbol)
    device === :cpu && return to_device(x, Val(:cpu))
    device === :gpu && hasmethod(to_device, Tuple{typeof(x), Val{:gpu}}) && return to_device(x, Val(:gpu))
    device === :gpu && error("GPU support requires CUDA.jl. Install and load CUDA.jl first: `using CUDA`")
    error("Unknown device: $device. Supported: :cpu, :gpu")
end
```

**Step 2: Add comment on copy. usage**

At line ~225 (or wherever `best_tensors = copy.(current_tensors)` appears after prior edits), add a comment:
```julia
# Snapshot current tensors for early stopping
best_tensors = copy.(current_tensors)
```

**Step 3: Run tests**

Run: `julia --project=. -e 'using Test, ParametricDFT, Random, LinearAlgebra, Zygote, Statistics, OMEinsum; include("test/training_tests.jl")'`
Expected: All pass.

**Step 4: Commit**

```bash
git add src/training.jl
git commit -m "refactor: short-circuit to_device, add copy. comment"
```

---

### Task 6: Trim docstrings across src/ files

**Files:**
- Modify: `src/loss.jl` — trim docstrings + add batched einsum comments
- Modify: `src/optimizers.jl` — trim docstrings
- Modify: `src/manifolds.jl` — trim docstrings + simplify group_by_manifold

**Step 1: Trim loss.jl docstrings**

For `loss_function` (lines ~147-163), replace with concise format:
```julia
"""
    loss_function(tensors, m, n, optcode, image, loss; inverse_code, batched_optcode)

Compute sparsity loss for `image` under circuit `tensors`. Dispatches to per-image
or batched einsum path based on `batched_optcode`.
"""
```

For `make_batched_code` (line ~212), `optimize_batched_code` (line ~236), `batched_forward` (line ~248) — trim each to 1-2 line purpose statement. Add a brief inline comment at the top of the batched einsum section:
```julia
# Batched einsum: extend OMEinsum contraction indices with a batch label so B images
# are processed in a single kernel call. Each image index (input/output) gets the batch
# label appended; the contraction order is then re-optimized for the batched tensor sizes.
```

**Step 2: Trim optimizers.jl docstrings**

For `_compute_gradients`, `_batched_project`, `RiemannianGD`, `RiemannianAdam`, and both `optimize!` methods — trim to 1-2 line purpose + concise param list. Example for `optimize!`:
```julia
"""
    optimize!(opt::RiemannianGD, tensors, loss_fn, grad_fn; max_iter=100, tol=1e-6)

Run Riemannian gradient descent with Armijo line search. Returns optimized tensors.
"""
```

**Step 3: Trim manifolds.jl docstrings and simplify group_by_manifold**

Trim `classify_manifold`, `group_by_manifold`, `project`, `retract`, `transport`, `batched_matmul`, `batched_adjoint`, `is_unitary_general`, `stack_tensors`, `unstack_tensors!` docstrings to concise format.

Replace `group_by_manifold` body (lines 101-112):
```julia
function group_by_manifold(tensors::Vector{<:AbstractMatrix})
    groups = Dict{AbstractRiemannianManifold, Vector{Int}}()
    for (i, t) in enumerate(tensors)
        push!(get!(groups, classify_manifold(Array(t)), Int[]), i)
    end
    return groups
end
```

**Step 4: Run tests**

Run: `julia --project=. -e 'using Test, ParametricDFT, Random, LinearAlgebra, Zygote, Statistics, OMEinsum; include("test/manifold_tests.jl"); include("test/loss_tests.jl"); include("test/optimizer_tests.jl")'`
Expected: All pass (docstring changes don't affect behavior).

**Step 5: Commit**

```bash
git add src/loss.jl src/optimizers.jl src/manifolds.jl
git commit -m "docs: trim docstrings, add batched einsum comments, simplify group_by_manifold"
```

---

### Task 7: Merge optimizer_crosscheck.jl into basis_demo.jl

**Files:**
- Modify: `examples/basis_demo.jl` — add optimizer comparison section with GPU/CPU plots and tables
- Delete: `examples/optimizer_crosscheck.jl`

**Step 1: Read and understand both files**

Read `examples/basis_demo.jl` fully and `examples/optimizer_crosscheck.jl` fully. Understand the structure and where the crosscheck content should be inserted.

**Step 2: Add optimizer comparison section to basis_demo.jl**

After the existing training section, add a new section that:
- Runs QFTBasis training with both `RiemannianGD` and `RiemannianAdam` on CPU
- If CUDA is available, also runs both optimizers on GPU with batched mode
- Generates a comparison table (optimizer, device, batch_size, final_loss, time)
- Generates a comparison plot using CairoMakie showing loss curves for all configurations
- Saves the plot to a file

The key requirements from the user:
- Must generate a comparison plot
- Must generate a comparison table
- Must cover both GPU and CPU batched versions

**Step 3: Delete optimizer_crosscheck.jl**

```bash
rm examples/optimizer_crosscheck.jl
```

**Step 4: Verify the example runs**

Run: `julia --project=. examples/basis_demo.jl` (with a small dataset, or just verify syntax)
Expected: No errors on CPU path.

**Step 5: Commit**

```bash
git add examples/basis_demo.jl
git rm examples/optimizer_crosscheck.jl
git commit -m "refactor: merge optimizer crosscheck into basis_demo with comparison plot/table"
```

---

### Task 8: Final cleanup and push

**Step 1: Run full modified test suite**

Run: `julia --project=. -e 'using Test, ParametricDFT, Random, LinearAlgebra, Zygote, Statistics, OMEinsum; include("test/optimizer_tests.jl"); include("test/training_tests.jl"); include("test/loss_tests.jl"); include("test/manifold_tests.jl")'`
Expected: All tests pass.

**Step 2: Push to PR**

```bash
git push origin feature/riemannian-optimizers
```

**Step 3: Delete the design and plan docs**

```bash
rm docs/plans/2026-03-02-pr35-review-fixes-design.md docs/plans/2026-03-02-pr35-review-fixes.md
git add docs/plans/
git commit -m "chore: remove implementation plan docs"
git push origin feature/riemannian-optimizers
```
