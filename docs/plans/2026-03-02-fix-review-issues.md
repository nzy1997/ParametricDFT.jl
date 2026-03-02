# Fix Code Review Issues Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 7 Critical/Important issues from the post-review of `feature/riemannian-optimizers` in three logical commits.

**Architecture:** Three independent commits — (1) GPU correctness in `manifolds.jl` + `CUDAExt.jl`, (2) type stability in `optimizers.jl`, (3) test quality in test files. Each commit leaves the test suite fully green.

**Tech Stack:** Julia 1.x, CUDA.jl, ChainRulesCore.jl, Zygote.jl, OMEinsum.jl

---

## Commit 1: GPU/Core Correctness (Issues 1 & 2)

### Task 1: Fix `batched_adjoint` — replace scalar loop with broadcast

**Files:**
- Modify: `src/manifolds.jl:71-78`

**Step 1: Run existing test to establish baseline**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -E "batched_adjoint|PASSED|FAILED"
```

Expected: tests pass (the function is correct on CPU today; we're replacing implementation only).

**Step 2: Replace the loop implementation**

In `src/manifolds.jl`, replace lines 71–78:

```julia
# OLD (scalar indexing — breaks on CuArray):
function batched_adjoint(A::AbstractArray{T,3}) where T
    d1, d2, n = size(A)
    C = similar(A, T, d2, d1, n)
    @inbounds for k in 1:n, j in 1:d1, i in 1:d2
        C[i, j, k] = conj(A[j, i, k])
    end
    return C
end
```

with:

```julia
# NEW (broadcast + permute — GPU-native, no scalar indexing):
function batched_adjoint(A::AbstractArray{T,3}) where T
    return permutedims(conj.(A), (2, 1, 3))
end
```

**Step 3: Run manifold tests**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -A2 "batched_adjoint"
```

Expected: all subtests in `@testset "batched_adjoint generalized"` and `"batched_adjoint rectangular"` still pass.

---

### Task 2: Fix `batched_matmul` — replace scalar loop with BLAS-dispatching implementation

**Files:**
- Modify: `src/manifolds.jl:47-63`
- Modify: `ext/CUDAExt.jl` (add GPU-specific method)

**Step 1: Replace CPU implementation in `src/manifolds.jl` lines 47–63**

```julia
# OLD (scalar indexing — breaks on CuArray):
function batched_matmul(A::AbstractArray{T,3}, B::AbstractArray{T,3}) where T
    d1, d2A, n = size(A)
    d2B, d3, n2 = size(B)
    @assert d2A == d2B "Inner dimensions must match: got $d2A and $d2B"
    @assert n == n2 "Batch sizes must match: got $n and $n2"
    C = similar(A, T, d1, d3, n)
    @inbounds for k in 1:n
        for j in 1:d3, i in 1:d1
            s = zero(T)
            for p in 1:d2A
                s += A[i, p, k] * B[p, j, k]
            end
            C[i, j, k] = s
        end
    end
    return C
end
```

with:

```julia
# NEW (mul! per slice — dispatches to BLAS on CPU):
function batched_matmul(A::AbstractArray{T,3}, B::AbstractArray{T,3}) where T
    d1, d2A, n = size(A)
    d2B, d3, n2 = size(B)
    @assert d2A == d2B "Inner dimensions must match: got $d2A and $d2B"
    @assert n == n2 "Batch sizes must match: got $n and $n2"
    C = similar(A, T, d1, d3, n)
    for k in 1:n
        mul!(view(C, :, :, k), view(A, :, :, k), view(B, :, :, k))
    end
    return C
end
```

**Step 2: Add GPU-specific method at the bottom of `ext/CUDAExt.jl` (before the final `end`)**

In `ext/CUDAExt.jl`, add before line 83 (`end # module`):

```julia
# GPU batched matrix multiply: per-slice copy to CuMatrix, CUBLAS dispatch, copy back
function ParametricDFT.batched_matmul(A::CuArray{T,3}, B::CuArray{T,3}) where T
    d1, d2A, n = size(A)
    d2B, d3, n2 = size(B)
    @assert d2A == d2B "Inner dimensions must match: got $d2A and $d2B"
    @assert n == n2 "Batch sizes must match: got $n and $n2"
    C = similar(A, T, d1, d3, n)
    for k in 1:n
        # A[:,:,k] copies the k-th page to a contiguous CuMatrix (not scalar indexing).
        # `*` on CuMatrix dispatches to CUBLAS gemm.
        C[:, :, k] .= A[:, :, k] * B[:, :, k]
    end
    return C
end
```

**Step 3: Run manifold tests**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -E "batched_matmul|PASSED|FAILED"
```

Expected: all subtests in `@testset "batched_matmul generalized"` and `"batched_matmul rectangular"` still pass.

---

### Task 3: Extract `_topk_mask` helper in `ext/CUDAExt.jl` (DRY fix)

**Files:**
- Modify: `ext/CUDAExt.jl:18-74`

**Step 1: Extract helper and refactor both callers**

Replace the entire block from line 16 to line 74 in `ext/CUDAExt.jl` with:

```julia
# GPU-compatible top-k truncation with frequency weighting

"""Compute the binary mask for top-k frequency-weighted selection on CPU arrays.
Returns a GPU CuArray mask of element type `real(T)`."""
function _topk_mask(x_cpu::Array{T}, k::Integer) where T
    m, n = size(x_cpu)
    k2 = min(Int(k), length(x_cpu))
    center_i, center_j = (m + 1) ÷ 2, (n + 1) ÷ 2
    max_dist = sqrt((m/2)^2 + (n/2)^2)

    scores = similar(x_cpu, Float64)
    mags = abs.(x_cpu)
    for j in 1:n, i in 1:m
        freq_dist = sqrt((i - center_i)^2 + (j - center_j)^2)
        freq_weight = 1.0 - (freq_dist / max_dist) * 0.5
        scores[i, j] = mags[i, j] * (1.0 + freq_weight)
    end

    idx = partialsortperm(vec(scores), 1:k2, rev=true)
    RT = real(T)
    mask_cpu = zeros(RT, m, n)
    for flat_idx in idx
        mask_cpu[flat_idx] = one(RT)
    end
    return CuArray{RT}(mask_cpu)
end

function ParametricDFT.topk_truncate(x::CuArray{T}, k::Integer) where {T}
    mask_gpu = _topk_mask(Array(x), k)
    return x .* mask_gpu
end

# rrule for topk_truncate on GPU (gradient flows through kept elements)

function ChainRulesCore.rrule(::typeof(ParametricDFT.topk_truncate), x::CuArray{T}, k::Integer) where {T}
    mask_gpu = _topk_mask(Array(x), k)
    y = x .* mask_gpu

    function topk_truncate_pullback(ȳ)
        return (ChainRulesCore.NoTangent(), ȳ .* mask_gpu, ChainRulesCore.NoTangent())
    end
    return y, topk_truncate_pullback
end
```

**Step 2: Run full test suite**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all tests pass. The refactor is behavior-preserving — same mask logic, just shared.

**Step 3: Commit chunk 1**

```bash
git add src/manifolds.jl ext/CUDAExt.jl
git commit -m "$(cat <<'EOF'
fix: GPU-compatible batched ops and topk_mask DRY refactor

- batched_adjoint: replace scalar loop with permutedims(conj.(A),(2,1,3))
- batched_matmul: replace scalar loop with mul! per slice (CPU BLAS);
  add CuArray method in CUDAExt using page-copy + CUBLAS dispatch
- Extract _topk_mask helper in CUDAExt to eliminate topk_truncate/rrule
  duplication — forward and backward pass now share identical mask logic

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

Expected: commit succeeds, `git log --oneline -1` shows the new commit.

---

## Commit 2: Type Stability + AD (Issues 3 & 7)

### Task 4: Fix typed state dicts and `collect` on Tuple tangents

**Files:**
- Modify: `src/optimizers.jl:40`, `82`, `150-151`, `206`, `304-310`

**Step 1: Fix `collect` in `_compute_gradients` (line 40)**

Replace line 40 in `src/optimizers.jl`:

```julia
# OLD:
euclidean_grads = euclidean_grads_raw isa Tuple ? collect(euclidean_grads_raw) : euclidean_grads_raw
```

with:

```julia
# NEW: typed comprehension avoids Vector{Any} for heterogeneous Tuples
euclidean_grads = euclidean_grads_raw isa Tuple ?
    AbstractArray[euclidean_grads_raw[i] for i in eachindex(euclidean_grads_raw)] :
    euclidean_grads_raw
```

**Step 2: Fix `rg_batches` dict in `_batched_project` (line 82)**

Replace line 82 in `src/optimizers.jl`:

```julia
# OLD:
rg_batches = Dict{AbstractRiemannianManifold, Any}()
```

with:

```julia
# NEW:
rg_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
```

**Step 3: Fix `point_batches` and `grad_buf_batches` in `optimize!(::RiemannianGD, ...)` (lines 150–151)**

Replace lines 150–151 in `src/optimizers.jl`:

```julia
# OLD:
point_batches = Dict{AbstractRiemannianManifold, Any}()
grad_buf_batches = Dict{AbstractRiemannianManifold, Any}()
```

with:

```julia
# NEW:
point_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
grad_buf_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
```

**Step 4: Fix `cand_batches` inside the line search loop (line 206)**

Replace line 206 in `src/optimizers.jl`:

```julia
# OLD:
cand_batches = Dict{AbstractRiemannianManifold, Any}()
```

with:

```julia
# NEW:
cand_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
```

**Step 5: Fix the five state dicts in `optimize!(::RiemannianAdam, ...)` (lines 304–310)**

Replace lines 304–310 in `src/optimizers.jl`:

```julia
# OLD:
point_batches = Dict{AbstractRiemannianManifold, Any}()
grad_buf_batches = Dict{AbstractRiemannianManifold, Any}()
dir_buf_batches = Dict{AbstractRiemannianManifold, Any}()

# Adam state: first moment (complex) and second moment (real) per manifold
m_batches = Dict{AbstractRiemannianManifold, Any}()
v_batches = Dict{AbstractRiemannianManifold, Any}()
```

with:

```julia
# NEW:
point_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
grad_buf_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
dir_buf_batches = Dict{AbstractRiemannianManifold, AbstractArray}()

# Adam state: first moment (complex) and second moment (real) per manifold
m_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
v_batches = Dict{AbstractRiemannianManifold, AbstractArray}()
```

**Step 6: Run full test suite**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all optimizer tests pass. The dict value type change is backward-compatible — all stored arrays were already `AbstractArray`.

**Step 7: Commit chunk 2**

```bash
git add src/optimizers.jl
git commit -m "$(cat <<'EOF'
fix: typed optimizer state dicts and stable gradient vector collection

- Replace Dict{AbstractRiemannianManifold, Any} with AbstractArray value
  type in all six state dicts across RiemannianGD and RiemannianAdam —
  allows the compiler to infer element types through hot loop dict access
- Replace collect(euclidean_grads_raw) with typed comprehension in
  _compute_gradients to avoid Vector{Any} for heterogeneous Tuple tangents

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Commit 3: Test Quality (Issues 4, 5, 6)

### Task 5: Fix misnamed test — wrong optimizer (Issue 4)

**Files:**
- Modify: `test/training_tests.jl:239`

**Step 1: Run the test to confirm it currently passes with wrong optimizer**

```bash
julia --project=. -e '
  using Pkg; Pkg.test(test_args=["training_tests"])
' 2>&1 | grep "gradient_descent with batch"
```

**Step 2: Fix `optimizer=:adam` → `optimizer=:gradient_descent` at line 239**

In `test/training_tests.jl`, inside `@testset "gradient_descent with batch_size > 1"` (line 227), change line 239:

```julia
# OLD:
            optimizer=:adam,
```

to:

```julia
# NEW:
            optimizer=:gradient_descent,
```

**Step 3: Run training tests**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -E "gradient_descent|PASS|FAIL"
```

Expected: test still passes (gradient descent does reduce loss).

---

### Task 6: Add loss-reduction assertions to batched training tests (Issue 5)

**Files:**
- Modify: `test/training_tests.jl:204-265`

**Step 1: Add initial-loss computation and final-loss assertion to all three subtests**

The three subtests to update are:
1. `"adam with batch_size > 1 uses batched einsum"` (line 206)
2. `"gradient_descent with batch_size > 1"` (line 227, now using `:gradient_descent`)
3. `"batch_size=1 fallback still works"` (line 246)

For each subtest, add before the `train_basis` call:

```julia
# Compute initial loss before training
optcode_init, tensors_init = ParametricDFT.qft_code(m, n)
loss_obj = ParametricDFT.L1Norm()  # match the loss used in train_basis call
initial_loss = sum(
    ParametricDFT.loss_function(tensors_init, m, n, optcode_init, img, loss_obj)
    for img in dataset
) / length(dataset)
```

And after the `train_basis` call, replace the bare `@test basis isa QFTBasis` block with:

```julia
@test basis isa QFTBasis
@test basis.m == m
@test basis.n == n

# Verify training actually reduced loss
optcode_trained, _ = ParametricDFT.qft_code(m, n)
final_loss = sum(
    ParametricDFT.loss_function(basis.tensors, m, n, optcode_trained, img, loss_obj)
    for img in dataset
) / length(dataset)
@test final_loss < initial_loss
```

**Note on loss object per subtest:**
- `"adam with batch_size > 1"` uses `L1Norm()` → `loss_obj = ParametricDFT.L1Norm()`
- `"gradient_descent with batch_size > 1"` uses `L2Norm()` → `loss_obj = ParametricDFT.L2Norm()`
- `"batch_size=1 fallback"` uses `L1Norm()` → `loss_obj = ParametricDFT.L1Norm()`

**Step 2: Run batched training tests**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -A5 "batched training"
```

Expected: all three subtests pass including the new `final_loss < initial_loss` assertion.

---

### Task 7: Add Zygote gradient test for `batched_loss_mse` (Issue 6)

**Files:**
- Modify: `test/loss_tests.jl` — add inside the `@testset "Batched Einsum"` block, after the existing `"different runtime batch size"` subtest (the last subtest before the closing `end`).

**Step 1: Write the new test**

Add the following subtest at the end of the `@testset "Batched Einsum"` block (before its closing `end`):

```julia
@testset "Zygote gradient through batched MSE" begin
    Random.seed!(42)
    m, n = 2, 2
    optcode, tensors_raw = ParametricDFT.qft_code(m, n)
    optcode_inv, _ = ParametricDFT.qft_code(m, n; inverse=true)
    tensors = [ComplexF64.(t) for t in tensors_raw]
    n_gates = length(tensors)
    B, k = 3, 5
    batch = [rand(ComplexF64, 2^m, 2^n) for _ in 1:B]

    batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
    batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, B)

    # Zygote gradient through batched MSE
    zg = Zygote.gradient(
        ts -> ParametricDFT.batched_loss_mse(batched_opt, ts, batch, m, n, k, optcode_inv),
        tensors
    )[1]

    @test zg isa Vector
    @test length(zg) == n_gates
    @test all(g isa AbstractMatrix for g in zg)

    # Finite-difference check on real part of element [1,1] of first tensor
    ε = 1e-5
    ts_p = deepcopy(tensors)
    ts_m = deepcopy(tensors)
    ts_p[1][1, 1] += ε
    ts_m[1][1, 1] -= ε

    fd = (
        ParametricDFT.batched_loss_mse(batched_opt, ts_p, batch, m, n, k, optcode_inv) -
        ParametricDFT.batched_loss_mse(batched_opt, ts_m, batch, m, n, k, optcode_inv)
    ) / (2ε)

    # Zygote returns the conjugate Wirtinger gradient; real part matches finite diff
    @test isapprox(real(zg[1][1, 1]), fd, rtol=1e-4)
end
```

**Step 2: Run loss tests**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | grep -E "batched MSE|PASS|FAIL"
```

Expected: the new `"Zygote gradient through batched MSE"` test passes, confirming the `selectdim → topk_truncate → inverse einsum` AD path is correct.

**Step 3: Commit chunk 3**

```bash
git add test/training_tests.jl test/loss_tests.jl
git commit -m "$(cat <<'EOF'
test: fix misnamed test, add loss-reduction assertions, add batched MSE gradient check

- Fix "gradient_descent with batch_size > 1" test: was using optimizer=:adam,
  now correctly uses :gradient_descent
- Add initial/final loss comparison in all three batched training subtests
  so they verify the optimizer actually reduces loss, not just runs
- Add Zygote finite-difference gradient check for batched_loss_mse,
  covering the selectdim → topk_truncate → inverse einsum backward path

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

**Step 4: Final verification — full test suite**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all tests pass. Three new commits on `feature/riemannian-optimizers`.
