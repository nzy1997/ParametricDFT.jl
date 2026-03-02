# GPU Acceleration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make GPU competitive with CPU across image sizes by materializing the circuit unitary for forward passes and fixing GPU-hostile patterns in manifold/loss code.

**Architecture:** Two-pronged: (1) Build full unitary U from gates via batched einsum with identity input, use U*X as forward pass; (2) Fix `batched_matmul` to use cuBLAS, replace Gram-Schmidt QR retraction with Cayley, eliminate `@allowscalar` in topk. Smart device dispatch selects materialized path for large images.

**Tech Stack:** Julia, CUDA.jl (CUBLAS), OMEinsum, Zygote, ChainRulesCore

---

## Task 1: Fix `batched_matmul` to Use cuBLAS Batched GEMM

**Files:**
- Modify: `ext/CUDAExt.jl:56-66`
- Modify: `test/manifold_tests.jl` (add GPU-mirrored test if CUDA available)

**Context:** The current GPU `batched_matmul` loops over the batch dimension with `C[:,:,k] .= A[:,:,k] * B[:,:,k]`, launching N separate cuBLAS kernels. CUDA.jl's CUBLAS module provides `gemm_strided_batched!` which does all slices in one kernel launch.

**Step 1: Write the failing test**

Add to `test/manifold_tests.jl` inside the `"Manifold Abstraction"` testset, after the existing `batched_matmul` tests:

```julia
@testset "batched_matmul strided batched path" begin
    # Verify batched_matmul produces correct results for the sizes
    # used in the optimizer (2x2 gates batched)
    Random.seed!(53)
    for d in [2, 3, 4]
        n = 20  # larger batch to exercise batched path
        A = randn(ComplexF64, d, d, n)
        B = randn(ComplexF64, d, d, n)
        C = ParametricDFT.batched_matmul(A, B)
        @test size(C) == (d, d, n)
        for k in 1:n
            @test C[:, :, k] ≈ A[:, :, k] * B[:, :, k] atol=1e-12
        end
    end
end
```

**Step 2: Run test to verify it passes with CPU (baseline)**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: PASS (this test should pass on CPU already — it validates the interface)

**Step 3: Implement cuBLAS batched GEMM in CUDAExt**

Replace `ext/CUDAExt.jl:56-66` with:

```julia
function ParametricDFT.batched_matmul(A::CuArray{T,3}, B::CuArray{T,3}) where T
    d1, d2A, n = size(A)
    d2B, d3, n2 = size(B)
    @assert d2A == d2B "Inner dimensions must match: got $d2A and $d2B"
    @assert n == n2 "Batch sizes must match: got $n and $n2"
    C = similar(A, T, d1, d3, n)
    # Use CUBLAS strided batched GEMM — single kernel launch for all slices
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', one(T), A, B, zero(T), C)
    return C
end
```

**Step 4: Run tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: PASS

**Step 5: Commit**

```bash
git add ext/CUDAExt.jl test/manifold_tests.jl
git commit -m "perf: replace batched_matmul loop with cuBLAS gemm_strided_batched"
```

---

## Task 2: Add Cayley Retraction for UnitaryManifold

**Files:**
- Modify: `src/manifolds.jl:142-164` (replace retract body)
- Modify: `test/manifold_tests.jl` (existing retract test already validates unitarity)

**Context:** The current Gram-Schmidt QR retraction iterates column-by-column with many small GPU kernel launches. The Cayley retraction computes `(I - α/2·W)⁻¹(I + α/2·W)·U` where `W = Ξ·U'`, requiring only a few batched operations. For 2×2 matrices, the inverse uses the explicit formula `[a b; c d]⁻¹ = (1/(ad-bc))[d -b; -c a]`.

**Step 1: Write the failing test for Cayley retraction**

Add to `test/manifold_tests.jl` after the existing `"UnitaryManifold retract"` testset:

```julia
@testset "UnitaryManifold Cayley retract" begin
    Random.seed!(54)
    um = ParametricDFT.UnitaryManifold()
    for d in [2, 3, 4]
        n = 5
        U = Array{ComplexF64}(undef, d, d, n)
        for k in 1:n
            Q, _ = qr(randn(ComplexF64, d, d))
            U[:, :, k] = Matrix{ComplexF64}(Q)
        end
        G = randn(ComplexF64, d, d, n)
        Xi = ParametricDFT.project(um, U, G)

        # Cayley retraction should produce unitary result
        Q_cay = ParametricDFT.retract(um, U, Xi, 0.1)
        @test size(Q_cay) == (d, d, n)
        for k in 1:n
            @test Q_cay[:, :, k]' * Q_cay[:, :, k] ≈ Matrix{ComplexF64}(I, d, d) atol=1e-10
        end

        # Small step should stay close to U
        Q_small = ParametricDFT.retract(um, U, Xi, 1e-8)
        for k in 1:n
            @test Q_small[:, :, k] ≈ U[:, :, k] atol=1e-6
        end
    end
end
```

**Step 2: Run test to verify existing retract passes (baseline)**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: PASS

**Step 3: Add `batched_inv` helper and replace retract**

Add `batched_inv` after `batched_adjoint` in `src/manifolds.jl` (after line 62):

```julia
"""
    batched_inv(A::AbstractArray{T,3})

Batched matrix inverse: `C[:,:,k] = inv(A[:,:,k])` for each slice `k`.
Uses LU factorization for general matrices.
"""
function batched_inv(A::AbstractArray{T,3}) where T
    d1, d2, n = size(A)
    @assert d1 == d2 "Matrix must be square: got $d1 × $d2"
    C = similar(A)
    for k in 1:n
        C[:, :, k] = inv(A[:, :, k])
    end
    return C
end
```

Replace the `retract` function for `UnitaryManifold` (lines 142–164) with:

```julia
"""Batched Cayley retraction on U(n): (I - α/2·W)⁻¹(I + α/2·W)·U where W = Ξ·U'."""
function retract(::UnitaryManifold, U::AbstractArray{T,3}, Xi::AbstractArray{T,3}, α) where T
    RT = real(T)
    α_half = convert(RT, α) / 2
    d = size(U, 1)
    n = size(U, 3)

    # W = Xi * U' (skew-Hermitian in Lie algebra)
    W = batched_matmul(Xi, batched_adjoint(U))

    # Build I ± (α/2)*W
    I_batch = zeros(T, d, d, n)
    for k in 1:n
        for i in 1:d
            I_batch[i, i, k] = one(T)
        end
    end
    lhs = I_batch .- α_half .* W   # I - α/2·W
    rhs = I_batch .+ α_half .* W   # I + α/2·W

    # (I - α/2·W)⁻¹ (I + α/2·W) U
    lhs_inv = batched_inv(lhs)
    return batched_matmul(batched_matmul(lhs_inv, rhs), U)
end
```

**Step 4: Run tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: PASS — both old and new retraction tests should pass since the Cayley retraction also preserves unitarity.

**Step 5: Add cuBLAS batched inverse to CUDAExt**

Add to `ext/CUDAExt.jl` after the `batched_matmul` override:

```julia
function ParametricDFT.batched_inv(A::CuArray{T,3}) where T
    d, _, n = size(A)
    if d == 2
        # Explicit 2×2 inverse: (1/(ad-bc)) * [d -b; -c a]
        # Pure broadcasting — single GPU kernel
        a = A[1:1, 1:1, :]
        b = A[1:1, 2:2, :]
        c = A[2:2, 1:1, :]
        dd = A[2:2, 2:2, :]
        det = a .* dd .- b .* c
        inv_det = one(T) ./ det
        return cat(
            cat(dd .* inv_det, -(b .* inv_det); dims=2),
            cat(-(c .* inv_det), a .* inv_det; dims=2);
            dims=1
        )
    else
        # General case: use CUBLAS batched LU + inverse
        C = similar(A)
        for k in 1:n
            C[:, :, k] .= inv(A[:, :, k])
        end
        return C
    end
end
```

**Step 6: Run tests again**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: PASS

**Step 7: Commit**

```bash
git add src/manifolds.jl ext/CUDAExt.jl test/manifold_tests.jl
git commit -m "perf: replace Gram-Schmidt QR retraction with Cayley retraction

Cayley retraction (I - α/2·W)⁻¹(I + α/2·W)·U uses batched matmul and
batched inverse instead of column-by-column Gram-Schmidt. For 2×2 GPU
matrices, the inverse uses an explicit formula (single broadcast kernel)."
```

---

## Task 3: Fix GPU `topk_truncate` — Eliminate `@allowscalar` and Cache Weights

**Files:**
- Modify: `ext/CUDAExt.jl:18-43` (rewrite `_topk_mask_gpu`)
- Modify: `src/loss.jl` (add `FreqWeightsCache` struct after topk, before loss_function)
- Test: existing `test/loss_tests.jl` topk tests validate correctness

**Context:** Current GPU topk creates temporary `CuArray(Float64.(1:m))` every call, does a full sort, and uses `CUDA.@allowscalar` to read a threshold value — forcing CPU-GPU synchronization. Fix: cache frequency weights and avoid scalar indexing.

**Step 1: Write failing test for cached frequency weights**

Add to `test/loss_tests.jl` inside the `"topk_truncate"` testset:

```julia
@testset "topk_truncate deterministic across calls" begin
    Random.seed!(55)
    x = rand(ComplexF64, 8, 8)
    y1 = ParametricDFT.topk_truncate(x, 10)
    y2 = ParametricDFT.topk_truncate(x, 10)
    @test y1 ≈ y2 atol=1e-15
end
```

**Step 2: Run test (baseline)**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: PASS

**Step 3: Rewrite `_topk_mask_gpu` in CUDAExt**

Replace `ext/CUDAExt.jl:18-38` with:

```julia
# Cached frequency weights to avoid per-call GPU allocations
const _freq_weight_cache = Dict{Tuple{Int,Int}, CuArray{Float64,2}}()

function _get_freq_weights_gpu(m::Int, n::Int)
    key = (m, n)
    if !haskey(_freq_weight_cache, key)
        center_i = (m + 1) / 2
        center_j = (n + 1) / 2
        max_dist = sqrt((m / 2)^2 + (n / 2)^2)
        is = Float64.(1:m)
        js = Float64.(1:n)'
        freq_dists = sqrt.((is .- center_i) .^ 2 .+ (js .- center_j) .^ 2)
        freq_weights = 1.0 .- (freq_dists ./ max_dist) .* 0.5
        _freq_weight_cache[key] = CuArray(freq_weights)
    end
    return _freq_weight_cache[key]
end

function _topk_mask_gpu(x::CuArray{T}, k::Integer) where T
    m, n = size(x)
    k2 = min(Int(k), m * n)
    RT = real(T)

    freq_weights = _get_freq_weights_gpu(m, n)
    scores = abs.(x) .* (1.0 .+ freq_weights)

    # Sort on GPU, extract threshold without @allowscalar
    sorted = sort(vec(scores); rev=true)
    # Use a 1-element slice instead of scalar indexing
    threshold_arr = sorted[k2:k2]
    mask = RT.(vec(scores) .>= repeat(threshold_arr, m * n))
    return reshape(mask, m, n)
end
```

**Step 4: Update rrule and topk_truncate accordingly**

No changes needed — `_topk_mask_gpu` is called by the existing `topk_truncate` and `rrule` functions which remain the same.

**Step 5: Run tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: PASS

**Step 6: Commit**

```bash
git add ext/CUDAExt.jl test/loss_tests.jl
git commit -m "perf: cache GPU frequency weights and eliminate @allowscalar in topk

Frequency weight arrays are now computed once per image size and cached.
Threshold extraction uses array slicing instead of CUDA.@allowscalar
to avoid CPU-GPU synchronization."
```

---

## Task 4: Implement Materialized Unitary — `build_circuit_unitary`

**Files:**
- Create: `src/materialized.jl`
- Modify: `src/ParametricDFT.jl` (add include + export)
- Create: `test/materialized_tests.jl`
- Modify: `test/runtests.jl` (include new test file)

**Context:** The core of the GPU acceleration. Build the full D×D unitary from circuit gates by passing D=2^(m+n) basis vectors through the batched einsum as a single call. The resulting matrix U enables `U*X` forward passes — one cuBLAS GEMM instead of hundreds of tiny einsum contractions.

**Step 1: Write the failing test**

Create `test/materialized_tests.jl`:

```julia
@testset "Materialized Unitary" begin

    @testset "build_circuit_unitary correctness" begin
        Random.seed!(60)
        m, n = 3, 3  # 8×8 images for fast testing
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)

        # Build batched einsum for D basis vectors
        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)

        U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors), m, n)
        @test size(U) == (D, D)

        # Verify U is unitary
        @test U * U' ≈ Matrix{ComplexF64}(I, D, D) atol=1e-8

        # Verify each column matches per-image einsum
        for j in 1:D
            e_j = zeros(ComplexF64, D)
            e_j[j] = 1.0
            expected = vec(optcode(tensors..., reshape(e_j, fill(2, m + n)...)))
            @test U[:, j] ≈ expected atol=1e-10
        end
    end

    @testset "build_circuit_unitary entangled QFT" begin
        Random.seed!(61)
        m, n = 3, 3
        optcode, tensors_raw, _ = ParametricDFT.entangled_qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)

        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)

        U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors), m, n)
        @test size(U) == (D, D)
        @test U * U' ≈ Matrix{ComplexF64}(I, D, D) atol=1e-8
    end

    @testset "materialized_forward matches einsum forward" begin
        Random.seed!(62)
        m, n = 3, 3
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)

        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)
        U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors), m, n)

        # Compare for random images
        for _ in 1:5
            img = randn(ComplexF64, 2^m, 2^n)
            einsum_result = reshape(optcode(tensors..., reshape(img, fill(2, m + n)...)), 2^m, 2^n)
            matmul_result = reshape(U * vec(img), 2^m, 2^n)
            @test matmul_result ≈ einsum_result atol=1e-10
        end
    end

    @testset "materialized_forward batched" begin
        Random.seed!(63)
        m, n = 3, 3
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)

        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)
        U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors), m, n)

        # Batched forward: U * [img1 | img2 | ... | imgB]
        B = 4
        images = [randn(ComplexF64, 2^m, 2^n) for _ in 1:B]
        X = hcat([vec(img) for img in images]...)
        result = U * X  # D × B

        for i in 1:B
            einsum_result = vec(reshape(optcode(tensors..., reshape(images[i], fill(2, m + n)...)), D))
            @test result[:, i] ≈ einsum_result atol=1e-10
        end
    end

end
```

**Step 2: Add include to runtests.jl**

Add after the last include in `test/runtests.jl` (after `include("optimizer_tests.jl")`):

```julia
include("materialized_tests.jl")
```

**Step 3: Run test to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: FAIL with `UndefVarError: build_circuit_unitary not defined`

**Step 4: Implement `build_circuit_unitary`**

Create `src/materialized.jl`:

```julia
# ============================================================================
# Materialized Unitary: Build Full Circuit Unitary for Fast Forward Passes
# ============================================================================
# Instead of contracting many tiny gate tensors via einsum (hundreds of GPU
# kernel launches), build the full D×D unitary matrix U once per optimizer step,
# then use U*X for forward passes (single cuBLAS GEMM).

"""
    build_circuit_unitary(batched_optcode, tensors::Tuple, m::Int, n::Int)

Build the full 2^(m+n) × 2^(m+n) unitary matrix from circuit gate tensors.

Applies the circuit to all D standard basis vectors in a single batched einsum
call. Column j of the result is the circuit applied to basis vector eⱼ.

# Arguments
- `batched_optcode`: Batched einsum code optimized for `batch_size = D`
- `tensors::Tuple`: Circuit gate tensors (2×2 unitary/phase matrices)
- `m::Int`: Number of row qubits
- `n::Int`: Number of column qubits

# Returns
- `Matrix{T}`: Full D×D unitary matrix where D = 2^(m+n)
"""
function build_circuit_unitary(batched_optcode, tensors::Tuple, m::Int, n::Int)
    D = 2^(m + n)
    T = eltype(tensors[1])
    # Identity matrix reshaped as (2,2,...,2, D) tensor = D basis vectors as "batch"
    I_mat = Matrix{T}(I, D, D)
    I_tensor = reshape(I_mat, fill(2, m + n)..., D)
    U_tensor = batched_optcode(tensors..., I_tensor)
    return reshape(U_tensor, D, D)
end

# Vector→Tuple wrapper for AD stability
function build_circuit_unitary(batched_optcode, tensors::AbstractVector, m::Int, n::Int)
    return build_circuit_unitary(batched_optcode, Tuple(tensors), m, n)
end
```

**Step 5: Add include and export in main module**

In `src/ParametricDFT.jl`, add after line 75 (`include("training.jl")`):

```julia
include("materialized.jl")
```

Add `build_circuit_unitary` to the exports (in the export block).

**Step 6: Run tests to verify they pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: PASS

**Step 7: Commit**

```bash
git add src/materialized.jl src/ParametricDFT.jl test/materialized_tests.jl test/runtests.jl
git commit -m "feat: add build_circuit_unitary for materialized forward pass

Builds full D×D unitary from circuit gates via batched einsum with identity
input. Enables single cuBLAS GEMM forward passes instead of hundreds of
tiny einsum kernel launches."
```

---

## Task 5: Implement Materialized Loss Functions

**Files:**
- Modify: `src/materialized.jl` (add loss functions)
- Modify: `test/materialized_tests.jl` (add loss equivalence tests)

**Context:** New loss functions that use U*X instead of einsum for forward/inverse passes. Must produce identical results to existing einsum-based losses for numerical equivalence. These are the functions the optimizer will call in the materialized path.

**Step 1: Write failing tests**

Add to `test/materialized_tests.jl`:

```julia
@testset "Materialized Loss Functions" begin

    @testset "materialized_loss L1 matches einsum" begin
        Random.seed!(64)
        m, n = 3, 3
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)
        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)
        U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors), m, n)

        images = [randn(ComplexF64, 2^m, 2^n) for _ in 1:4]
        loss_mat = ParametricDFT.materialized_loss_l1(U, images, m, n)
        loss_ein = ParametricDFT.batched_loss_l1(
            ParametricDFT.optimize_batched_code(flat_batched, blabel, 4),
            Tuple(tensors), images, m, n)
        @test loss_mat ≈ loss_ein atol=1e-8
    end

    @testset "materialized_loss L2 matches einsum" begin
        Random.seed!(65)
        m, n = 3, 3
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)
        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)
        U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors), m, n)

        images = [randn(ComplexF64, 2^m, 2^n) for _ in 1:4]
        loss_mat = ParametricDFT.materialized_loss_l2(U, images, m, n)
        loss_ein = ParametricDFT.batched_loss_l2(
            ParametricDFT.optimize_batched_code(flat_batched, blabel, 4),
            Tuple(tensors), images, m, n)
        @test loss_mat ≈ loss_ein atol=1e-8
    end

    @testset "materialized_loss MSE matches einsum" begin
        Random.seed!(66)
        m, n = 3, 3
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        inverse_code, _ = ParametricDFT.qft_code(m, n; inverse=true)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)
        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)
        U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors), m, n)

        images = [randn(ComplexF64, 2^m, 2^n) for _ in 1:4]
        k = 10
        loss_mat = ParametricDFT.materialized_loss_mse(U, images, m, n, k)
        loss_ein = ParametricDFT.batched_loss_mse(
            ParametricDFT.optimize_batched_code(flat_batched, blabel, 4),
            Tuple(tensors), images, m, n, k, inverse_code)
        @test loss_mat ≈ loss_ein atol=1e-8
    end

end
```

**Step 2: Run tests to verify they fail**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: FAIL with `UndefVarError: materialized_loss_l1 not defined`

**Step 3: Implement materialized loss functions**

Add to `src/materialized.jl`:

```julia
# ============================================================================
# Materialized Loss Functions
# ============================================================================
# Forward pass: U * X where X = [vec(img₁) | ... | vec(imgB)]
# Inverse pass: U' * X (adjoint)

"""
    materialized_loss_l1(U, images, m, n)

L1 loss using materialized unitary: (1/B) * sum(|U * X|).
"""
function materialized_loss_l1(U::AbstractMatrix, images::Vector{<:AbstractMatrix}, m::Int, n::Int)
    B = length(images)
    D = 2^(m + n)
    X = _stack_image_columns(images, D)
    result = U * X
    return sum(abs.(result)) / B
end

"""
    materialized_loss_l2(U, images, m, n)

L2 loss using materialized unitary: (1/B) * sum(|U * X|²).
"""
function materialized_loss_l2(U::AbstractMatrix, images::Vector{<:AbstractMatrix}, m::Int, n::Int)
    B = length(images)
    D = 2^(m + n)
    X = _stack_image_columns(images, D)
    result = U * X
    return sum(abs2.(result)) / B
end

"""
    materialized_loss_mse(U, images, m, n, k)

MSE loss using materialized unitary: forward with U, truncate, inverse with U'.
"""
function materialized_loss_mse(U::AbstractMatrix, images::Vector{<:AbstractMatrix}, m::Int, n::Int, k::Int)
    B = length(images)
    D = 2^(m + n)
    X = _stack_image_columns(images, D)

    # Forward: U * X
    fft_batch = U * X  # D × B

    # Per-image truncation + inverse (truncation is content-dependent)
    total_loss = zero(real(eltype(U)))
    U_inv = U'  # Adjoint for inverse
    for i in 1:B
        fft_col = reshape(fft_batch[:, i], 2^m, 2^n)
        fft_truncated = topk_truncate(fft_col, k)
        reconstructed = U_inv * vec(fft_truncated)
        total_loss += sum(abs2.(vec(images[i]) .- reconstructed))
    end
    return total_loss / B
end

"""Stack images as columns of a D × B matrix."""
function _stack_image_columns(images::Vector{<:AbstractMatrix}, D::Int)
    B = length(images)
    T = eltype(images[1])
    X = similar(images[1], T, D, B)
    for i in 1:B
        X[:, i] = vec(images[i])
    end
    return X
end
```

**Step 4: Run tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: PASS

**Step 5: Commit**

```bash
git add src/materialized.jl test/materialized_tests.jl
git commit -m "feat: add materialized loss functions (L1, L2, MSE)

Forward pass uses U*X (single matmul), inverse uses U' for MSE.
Results match einsum-based losses to numerical precision."
```

---

## Task 6: Add AD Tests for Materialized Path

**Files:**
- Modify: `test/materialized_tests.jl` (add gradient tests)

**Context:** The materialized path must produce correct gradients through `build_circuit_unitary` → `U * X` → loss. Verify by comparing against finite differences.

**Step 1: Write the gradient test**

Add to `test/materialized_tests.jl`:

```julia
@testset "Materialized AD Gradients" begin

    @testset "gradient through materialized L1" begin
        Random.seed!(67)
        m, n = 2, 2  # Small for fast finite diff
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)
        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)

        images = [randn(ComplexF64, 2^m, 2^n) for _ in 1:2]

        loss_fn = ts -> begin
            U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(ts), m, n)
            ParametricDFT.materialized_loss_l1(U, images, m, n)
        end

        grads = Zygote.gradient(loss_fn, tensors)[1]
        @test grads !== nothing
        @test length(grads) == length(tensors)

        # Finite difference check for first gate
        eps = 1e-6
        for idx in 1:min(2, length(tensors))
            for i in 1:2, j in 1:2
                ts_plus = deepcopy(tensors)
                ts_minus = deepcopy(tensors)
                ts_plus[idx][i, j] += eps
                ts_minus[idx][i, j] -= eps
                fd_grad = (loss_fn(ts_plus) - loss_fn(ts_minus)) / (2 * eps)
                @test real(grads[idx][i, j]) ≈ real(fd_grad) atol=1e-4
            end
        end
    end

    @testset "materialized gradient matches einsum gradient" begin
        Random.seed!(68)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)
        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)
        batch_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, 2)

        images = [randn(ComplexF64, 2^m, 2^n) for _ in 1:2]

        # Materialized gradient
        mat_loss_fn = ts -> begin
            U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(ts), m, n)
            ParametricDFT.materialized_loss_l1(U, images, m, n)
        end
        grads_mat = Zygote.gradient(mat_loss_fn, tensors)[1]

        # Einsum gradient
        ein_loss_fn = ts -> ParametricDFT.batched_loss_l1(batch_optcode, Tuple(ts), images, m, n)
        grads_ein = Zygote.gradient(ein_loss_fn, tensors)[1]

        # Gradients should match
        for idx in 1:length(tensors)
            @test grads_mat[idx] ≈ grads_ein[idx] atol=1e-6
        end
    end

end
```

**Step 2: Run tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: PASS

**Step 3: Commit**

```bash
git add test/materialized_tests.jl
git commit -m "test: add AD gradient checks for materialized unitary path"
```

---

## Task 7: Integrate Materialized Path into Training Pipeline

**Files:**
- Modify: `src/training.jl:75-82, 122-133` (add materialized code path)
- Modify: `src/materialized.jl` (add `select_device_strategy`)
- Modify: `test/materialized_tests.jl` (add integration test)

**Context:** The training pipeline needs to detect when the materialized path should be used (based on image size and device) and construct loss functions accordingly. The optimizer remains unchanged — it just receives a different loss function.

**Step 1: Write the integration test**

Add to `test/materialized_tests.jl`:

```julia
@testset "Training Integration" begin

    @testset "select_device_strategy" begin
        @test ParametricDFT.select_device_strategy(3, 3, 4, :cpu) == :einsum_cpu
        @test ParametricDFT.select_device_strategy(6, 6, 4, :gpu) == :materialized_gpu
        @test ParametricDFT.select_device_strategy(3, 3, 4, :gpu) == :einsum_gpu
    end

    @testset "materialized path produces valid training" begin
        Random.seed!(69)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        images = [randn(Float64, 2^m, 2^n) for _ in 1:4]
        D = 2^(m + n)
        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)

        # Build materialized loss
        loss_fn = ts -> begin
            U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(ts), m, n)
            complex_images = [Complex{Float64}.(img) for img in images]
            ParametricDFT.materialized_loss_l1(U, complex_images, m, n)
        end

        grad_fn = ts -> begin
            _, back = Zygote.pullback(loss_fn, ts)
            back(1.0)[1]
        end

        opt = ParametricDFT.RiemannianAdam(lr=0.01)
        initial_loss = loss_fn(tensors)
        result = ParametricDFT.optimize!(opt, tensors, loss_fn, grad_fn; max_iter=10)
        final_loss = loss_fn(result)

        # Loss should decrease (optimization works through materialized path)
        @test final_loss < initial_loss
    end

end
```

**Step 2: Run tests to verify they fail**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: FAIL with `UndefVarError: select_device_strategy not defined`

**Step 3: Add `select_device_strategy` to materialized.jl**

Add to `src/materialized.jl`:

```julia
# ============================================================================
# Device Strategy Selection
# ============================================================================

"""
    select_device_strategy(m, n, batch_size, device)

Select optimal computation path based on problem size and device.

Returns:
- `:einsum_cpu` — Use standard einsum on CPU (default, unchanged behavior)
- `:materialized_gpu` — Build full unitary, use matmul on GPU (D ≥ 4096)
- `:einsum_gpu` — Use einsum on GPU with targeted fixes (D < 4096)
"""
function select_device_strategy(m::Int, n::Int, batch_size::Int, device::Symbol)
    if device == :cpu
        return :einsum_cpu
    end
    D = 2^(m + n)
    if D >= 4096  # 64×64+ images
        return :materialized_gpu
    else
        return :einsum_gpu
    end
end
```

**Step 4: Integrate into `_train_basis_core`**

In `src/training.jl`, after the existing batched einsum pre-computation (line 82), add the materialized path setup:

```julia
    # Pre-compute materialized unitary code for GPU acceleration of large images
    strategy = select_device_strategy(m, n, batch_size, device)
    unitary_optcode = nothing
    if strategy == :materialized_gpu
        n_gates = length(initial_tensors)
        D = 2^(m + n)
        flat_u, blabel_u = make_batched_code(optcode, n_gates)
        unitary_optcode = optimize_batched_code(flat_u, blabel_u, D)
    end
```

Modify the loss function construction (lines 122–133) to include the materialized path:

```julia
            batch_loss_fn = if strategy == :materialized_gpu && unitary_optcode !== nothing
                ts -> begin
                    U = build_circuit_unitary(unitary_optcode, ts, m, n)
                    if loss isa L1Norm
                        materialized_loss_l1(U, batch, m, n)
                    elseif loss isa L2Norm
                        materialized_loss_l2(U, batch, m, n)
                    else  # MSELoss
                        materialized_loss_mse(U, batch, m, n, loss.k)
                    end
                end
            elseif batched_optcode !== nothing
                ts -> loss_function(ts, m, n, optcode, batch, loss;
                                    inverse_code=inverse_code, batched_optcode=batched_optcode)
            else
                ts -> begin
                    total = zero(real(eltype(ts[1])))
                    for img in batch
                        total += loss_function(ts, m, n, optcode, img, loss; inverse_code=inverse_code)
                    end
                    return total / length(batch)
                end
            end
```

**Step 5: Run tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: PASS

**Step 6: Commit**

```bash
git add src/materialized.jl src/training.jl test/materialized_tests.jl
git commit -m "feat: integrate materialized unitary path into training pipeline

Auto-selects materialized GPU path for D >= 4096 (64x64+ images).
Optimizer receives a loss function that internally builds U and uses
matmul, transparent to the optimizer."
```

---

## Task 8: Update Profile Script and Benchmark

**Files:**
- Modify: `examples/profile_gpu.jl` (add materialized benchmarks)

**Context:** Add benchmarks comparing materialized vs. einsum forward/backward pass to validate the performance improvement.

**Step 1: Add materialized benchmarks to profile_gpu.jl**

Append to `examples/profile_gpu.jl` after the existing benchmarks:

```julia
println("\n" * "="^70)
println("  Materialized Unitary Benchmarks")
println("="^70)
println()

# Build materialized unitary
D = 2^(m + n)
flat_u, blabel_u = ParametricDFT.make_batched_code(optcode, n_gates)
unitary_optcode = ParametricDFT.optimize_batched_code(flat_u, blabel_u, D)

println("--- Build circuit unitary (D=$D) ---")
t_cpu = bench(() -> ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors_cpu), m, n))
t_gpu = bench_gpu(() -> ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors_gpu), m, n))
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

U_cpu = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors_cpu), m, n)
U_gpu = CuArray(U_cpu)

println("\n--- Materialized forward (single image) ---")
img_vec_cpu = vec(images_cpu[1])
img_vec_gpu = CuArray(img_vec_cpu)
t_cpu = bench(() -> U_cpu * img_vec_cpu)
t_gpu = bench_gpu(() -> U_gpu * img_vec_gpu)
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

println("\n--- Materialized forward (batch=8) ---")
X_cpu = hcat([vec(img) for img in images_cpu]...)
X_gpu = CuArray(X_cpu)
t_cpu = bench(() -> U_cpu * X_cpu)
t_gpu = bench_gpu(() -> U_gpu * X_gpu)
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

println("\n--- Materialized L1 loss + gradient ---")
mat_fn_cpu = ts -> begin
    U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(ts), m, n)
    ParametricDFT.materialized_loss_l1(U, images_cpu, m, n)
end
mat_fn_gpu = ts -> begin
    U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(ts), m, n)
    ParametricDFT.materialized_loss_l1(U, images_gpu, m, n)
end
t_cpu = bench(() -> Zygote.gradient(mat_fn_cpu, tensors_cpu))
t_gpu = bench_gpu(() -> Zygote.gradient(mat_fn_gpu, tensors_gpu))
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)
```

**Step 2: Run the profile script (requires CUDA GPU)**

Run: `julia --project=examples examples/profile_gpu.jl`
Expected: New benchmark sections show materialized path timings

**Step 3: Commit**

```bash
git add examples/profile_gpu.jl
git commit -m "bench: add materialized unitary benchmarks to profile_gpu.jl"
```

---

## Summary

| Task | Component | Key Files | Est. Complexity |
|------|-----------|-----------|----------------|
| 1 | cuBLAS batched_matmul | `ext/CUDAExt.jl` | Low |
| 2 | Cayley retraction | `src/manifolds.jl`, `ext/CUDAExt.jl` | Medium |
| 3 | Fix GPU topk_truncate | `ext/CUDAExt.jl` | Medium |
| 4 | `build_circuit_unitary` | `src/materialized.jl` (new) | Medium |
| 5 | Materialized loss functions | `src/materialized.jl` | Medium |
| 6 | AD gradient tests | `test/materialized_tests.jl` | Low |
| 7 | Training pipeline integration | `src/training.jl` | Medium |
| 8 | Profile benchmarks | `examples/profile_gpu.jl` | Low |

**Dependency order:** Tasks 1–3 are independent (can be parallelized). Task 4 depends on nothing. Tasks 5–6 depend on Task 4. Task 7 depends on Tasks 4–5. Task 8 depends on all.
