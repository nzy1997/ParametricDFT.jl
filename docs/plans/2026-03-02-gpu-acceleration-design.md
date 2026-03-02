# GPU Acceleration via Materialized Unitary + Targeted Fixes

**Date:** 2026-03-02
**Status:** Approved
**Branch:** feature/gpu-acceleration (to be created)

## Problem

GPU code is significantly slower than CPU for all image sizes tested (32x32). The root cause is **architecture mismatch**: quantum circuit gates are 2x2 matrices, and OMEinsum decomposes each circuit contraction into many tiny operations. Each GPU kernel launch costs ~5-10us but the actual computation per gate is ~30-100ns. GPU spends 99.9% of its time on launch overhead.

### Bottleneck Breakdown

| Bottleneck | Location | Severity | Root Cause |
|-----------|----------|----------|------------|
| Einsum kernel launches | `loss.jl:125`, entire einsum pipeline | CRITICAL | Hundreds of tiny kernels per forward pass |
| `CUDA.@allowscalar` in topk | `ext/CUDAExt.jl:35` | CRITICAL | CPU-GPU sync per image in MSELoss |
| Loop-based `batched_matmul` | `ext/CUDAExt.jl:56-66` | HIGH | N separate kernels instead of 1 cuBLAS call |
| Gram-Schmidt QR retraction | `manifolds.jl:152-161` | HIGH | 64+ kernels per retraction, called thousands of times |
| Temporary GPU allocations | Throughout | MEDIUM | Per-call allocations in hot loops |

## Solution: Two-Pronged Approach

### Prong 1: Materialized Unitary Forward Pass

**Core idea:** Build the full unitary matrix U from circuit gates once per optimizer step, then use U for all forward/backward passes as a single dense matmul.

**Method:** Use the existing batched einsum infrastructure to apply the circuit to all D=2^(m+n) standard basis vectors at once. This produces the full unitary matrix.

```julia
function build_circuit_unitary(batched_optcode, tensors, m, n)
    D = 2^(m + n)
    T = eltype(tensors[1])
    I_tensor = reshape(Matrix{T}(I, D, D), fill(2, m + n)..., D)
    U_tensor = batched_optcode(tensors..., I_tensor)
    return reshape(U_tensor, D, D)
end
```

**Why this solves the GPU problem:**
- Single batched einsum call with batch_size=D makes each internal gate contraction operate on D times more data per kernel
- For D=1024 (32x32): ~1M elements per kernel instead of ~1K -> GPU utilization goes from <0.01% to ~10%
- For D=4096 (64x64): ~16M elements per kernel -> GPU well-utilized
- Forward pass becomes `U * X` (single cuBLAS GEMM, perfectly GPU-optimized)
- Inverse pass becomes `U' * X` (adjoint matmul)

**Key properties preserved:**
- Quantum circuit structure: gates remain individual 2x2 parameters optimized on per-gate manifolds
- AD compatibility: Zygote differentiates through batched einsum + matmul chain naturally
- No topology extraction needed: uses existing einsum code with identity input

**Requirements:**
- A batched einsum code optimized for batch_size=D (built once at circuit setup)
- The optimizer builds U at each step, uses it for loss evaluation

**Expected GPU performance by image size:**

| Image size | D | Build U (GPU) | Forward U*X | CPU baseline | GPU faster? |
|-----------|-----|---------------|-------------|-------------|-------------|
| 32x32 | 1024 | ~1.5ms | ~0.02ms | ~0.5ms/step | ~3x slower |
| 64x64 | 4096 | ~5ms | ~0.1ms | ~50ms/step | ~5-10x faster |
| 128x128 | 16384 | ~15ms | ~0.5ms | ~1500ms/step | ~50-100x faster |

### Prong 2: Targeted GPU Fixes

These fixes reduce GPU overhead in the optimizer and loss pipeline, independent of the materialized unitary.

#### Fix 1: batched_matmul -> cuBLAS Batched GEMM

Replace the for-loop in `ext/CUDAExt.jl:56-66` with `NNlib.batched_mul` (single cuBLAS call).

```julia
function ParametricDFT.batched_matmul(A::CuArray{T,3}, B::CuArray{T,3}) where T
    return NNlib.batched_mul(A, B)
end
```

#### Fix 2: Cayley Retraction (replaces Gram-Schmidt QR)

Replace the column-by-column Gram-Schmidt in `manifolds.jl:142-164` with Cayley retraction.

For U(n): retract from U along tangent Xi with step alpha:
```
W = Xi * U'                     (skew-Hermitian)
retract(U, Xi, alpha) = (I - alpha/2 * W)^{-1} * (I + alpha/2 * W) * U
```

For 2x2 matrices: explicit inverse formula `[a b; c d]^{-1} = (1/(ad-bc)) * [d -b; -c a]` -> pure element-wise broadcasting, one GPU kernel.

Reduces 64+ kernels per retraction to ~5 batched operations.

#### Fix 3: GPU-native topk_truncate

1. **Cache frequency weight arrays** - compute once at setup, not per call (currently `CuArray(Float64.(1:m))` allocated every call)
2. **Eliminate `CUDA.@allowscalar`** - use GPU-native threshold computation
3. **Replace full sort** with more efficient k-th element selection

#### Fix 4: Pre-allocated Optimizer Buffers

Store intermediate arrays in the optimizer state, allocated once at `optimize!` start. Avoids per-iteration GPU memory allocations for manifold operations.

### Smart Device Dispatch

Auto-select computation path based on problem size:

```julia
function select_device_strategy(m, n, batch_size, device)
    D = 2^(m + n)
    if device == :cpu
        return :einsum_cpu          # Current behavior, unchanged
    elseif D >= 4096                # 64x64+ images
        return :materialized_gpu    # Build U, use matmul
    else
        return :einsum_gpu          # Einsum with targeted fixes
    end
end
```

The optimizer remains strategy-agnostic - it receives a differentiable loss function that internally uses either materialized U or direct einsum.

## File Changes

### New Files
- `src/materialized.jl` - `build_circuit_unitary`, `materialized_loss`, materialized forward/inverse helpers
- `test/materialized_tests.jl` - Tests for materialized unitary path

### Modified Files
- `src/ParametricDFT.jl` - Include new file, exports
- `src/manifolds.jl` - Add Cayley retraction alongside QR (QR kept as fallback)
- `src/training.jl` - Device strategy dispatch, integration with materialized path
- `src/loss.jl` - Cached frequency weights for topk
- `ext/CUDAExt.jl` - Fixed batched_matmul, improved topk_truncate, batched_solve_2x2
- `test/runtests.jl` - Include new test file

### Unchanged
- Abstract types and interfaces
- Basis types (QFTBasis, EntangledQFTBasis, TEBDBasis)
- Optimizer interface (optimize!)
- Serialization, compression, visualization
- CPU code paths (no regression risk)

## Testing Strategy

1. **Correctness:** Verify `build_circuit_unitary` produces the same U as direct einsum application to each basis vector
2. **AD correctness:** Verify gradients through materialized path match gradients through einsum path (finite differences)
3. **Retraction correctness:** Verify Cayley retraction preserves unitarity (U*U' approx I)
4. **Loss equivalence:** Verify materialized loss matches einsum loss to numerical precision
5. **Performance:** Benchmark profile_gpu.jl with new code to measure actual speedups
