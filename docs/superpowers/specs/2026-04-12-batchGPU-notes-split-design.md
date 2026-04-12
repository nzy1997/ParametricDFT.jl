# Design Spec: Split stepbystep.typ and Create batchGPU.typ

**Date:** 2026-04-12
**Status:** Approved

## Goal

Split `note/stepbystep.typ` into two standalone Typst documents:

1. **`stepbystep.typ`** — Conceptual overview and onboarding guide (prerequisites, code structure, Manopt baseline, basis types, examples)
2. **`batchGPU.typ`** — All batching and GPU content (batched manifold ops, custom optimizers, batched einsum, GPU acceleration, training pipeline)

Both documents are standalone with their own notation sections. Additionally, update all content to match the latest code.

## Changes to `stepbystep.typ`

After the split, `stepbystep.typ` retains:

1. **Prerequisites** — unchanged (video links, background knowledge)
2. **Get Started** — unchanged (pointers to examples and main.typ)
3. **Code Structure** — unchanged (file tree)
4. **Notation** — copy of the notation section (standalone)
5. **Stage 1: Manopt.jl Baseline** — unchanged (approach, what Manopt provides, problems encountered)
6. **Bridge paragraph** — NEW: 1-2 sentences pointing to `note/batchGPU.typ` for Stages 2-3
7. **Sparse Basis Types** — unchanged (QFT, Entangled QFT, TEBD, MERA table + comparison + frequency-weighted truncation)
8. **Running Examples** — unchanged (optimizer benchmark, image compression)
9. **Tasks** — unchanged

Removed content: Stage 2 (Solutions 1-3, training loop), Stage 3 (GPU Acceleration), and the "Remaining Problem" bridge between them.

## Structure of `batchGPU.typ`

### Section 1: Introduction

- Title: "Batched Riemannian Optimization and GPU Acceleration"
- 1 paragraph motivation: recap the 3 Manopt problems (per-tensor ops, no Adam, no GPU) and state this document presents the solutions
- Self-contained, no dependency on `stepbystep.typ`

### Section 2: Notation

- Same notation block as `stepbystep.typ`: θ_k, X, Y, L, g̃, Retr, Γ, α
- Additional symbols:
  - K: number of gate tensors
  - B: batch size (number of images)
  - d: matrix dimension (typically 2)
  - OptimizationState field names mapped to math symbols

### Section 3: Batched Manifold Operations (`manifolds.jl`)

Content:
- Key insight: group same-type tensors into (d, d, K) batches
- Batched linear algebra: `batched_matmul`, `batched_adjoint`, `batched_inv` — signatures, batch dimension convention
- Tensor packing/unpacking: `stack_tensors` / `unstack_tensors!`
- UnitaryManifold: projection formula, Cayley retraction with skew-projection of W = ξU† (correctness for non-tangent inputs like Adam's scaled directions), transport as re-projection
- PhaseManifold: element-wise projection, normalization retraction
- `classify_manifold` / `group_by_manifold`: tensor routing

New diagram — **OptimizationState data flow:**
Shows how `group_by_manifold` partitions tensor indices → `stack_tensors` packs into batches → manifold ops process batches → `unstack_tensors!` writes back. Arrows between Zygote (needs individual tensors) and optimizer (needs batched arrays).

### Section 4: Custom Optimizers (`optimizers.jl`)

Content:
- **`OptimizationState` struct** (NEW): 5 fields (`manifold_groups`, `point_batches`, `grad_buf_batches`, `ibatch_cache`, `current_tensors`) with explanations
- **`_common_setup`**: initialization flow — copy tensors → group → stack → allocate grad buffers → cache identity batches
- **Gradient computation**: `_compute_gradients` — Zygote call, tangent type handling (ZeroTangent/NoTangent → zero arrays), NaN/Inf guard
- **Batched projection**: `_batched_project` — pack Euclidean grads → project per manifold → accumulate global gradient norm
- **RiemannianGD with Armijo**: algorithm, Armijo condition formula, identity batch reuse across line search steps (I_batch kwarg optimization — NEW), code snippet
- **RiemannianAdam with Momentum Transport**: moment update formulas (bias correction), `real(abs2(rg))` for complex second moment (explain why `real()`), fused broadcasts via `@.`, momentum transport after retraction, code snippet
- **Optimization loop**: unstack → Zygote → batched project → convergence check → optimizer step → loss trace

New diagram — **Cayley retraction geometry:**
Point U on U(n), tangent vector ξ, Cayley curve, retracted point. Illustrates manifold constraint.

### Section 5: Batched Einsum (`loss.jl`)

Content:
- **OMEinsum background**: `DynamicEinCode`, contraction semantics, `TreeSA` optimization (moved from current notes)
- **Single-image contraction**: the problem + existing tensor network diagram (moved)
- **Batch dimension trick**: `make_batched_code` — append batch label + existing batched diagram (moved)
- **Contraction order optimization**: `optimize_batched_code` with size dict
- **Einsum cache** (NEW): `einsum_cache.jl` — content-addressed SHA256 caching, disk storage in `~/.cache/ParametricDFT/einsum_codes/`, graceful fallback on corruption
- **Loss dispatch by type**:
  - L1/L2: fully batched (unchanged)
  - MSE: UPDATED — forward batched, truncation per-image (content-dependent mask), inverse now **batched** when `batched_inverse_code` available
- **Training loop diagram** (moved from current notes)

New diagram — **Einsum cache hit/miss flow:**
Decision diamond: cache hit → deserialize → use; cache miss → TreeSA optimize → serialize → use.

### Section 6: GPU Acceleration (`ext/CUDAExt.jl`)

Content:
- **Device abstraction**: `to_device(x, :gpu/:cpu)`, `to_cpu`, Val dispatch extension mechanism
- **GPU dispatch table** (moved): batched_matmul → CUBLAS, batched_inv → 2×2 formula, etc.
- **CUBLAS `gemm_strided_batched!`**: why 1 kernel beats K separate launches
- **Closed-form 2×2 inverse**: formula, pure broadcasting (single kernel), fallback for larger matrices
- **Frequency-aware top-k truncation**: `_get_freq_weights_gpu`, scoring formula, GPU sort + 1-element slice trick (no `@allowscalar`)
- **When GPU wins vs CPU** (moved)

### Section 7: The Training Pipeline (`training.jl`)

Content (NEW — not in current notes at this detail level):
- **Data preparation**: complex conversion, train/val split, device transfer, batch size clamping
- **Batched einsum pre-computation**: `make_batched_code` + `optimize_batched_code` done once at startup
- **Batch iteration scaling**: `batch_max_iter = steps_per_image * length(batch)` — preserves per-image sample complexity
- **Loss function construction**: `batch_loss_fn` and `batch_grad_fn` closure construction per batch
- **Checkpointing**: periodic save of tensors + metadata
- **Early stopping**: validation loss tracking, patience counter, best-tensor bookkeeping
- **Finalization**: CPU transfer, ComplexF64 conversion of best tensors

## New Diagrams Summary

| Diagram | Location | Description |
|---------|----------|-------------|
| OptimizationState data flow | Section 3 | Tensor indices → stack → manifold ops → unstack; Zygote ↔ optimizer boundary |
| Cayley retraction geometry | Section 4 | Point on U(n), tangent vector, Cayley curve, retracted point |
| Einsum cache hit/miss | Section 5 | Decision flow: SHA256 key → cache check → hit/miss paths |

Existing diagrams moved from `stepbystep.typ`:
- Single-image tensor network diagram → Section 5
- Batched tensor network diagram → Section 5
- Training loop flow diagram → Section 5

## Code Updates to Reflect

1. **`OptimizationState` struct** — document all 5 fields (not in current notes)
2. **Identity batch caching** — `I_batch` kwarg in Cayley retraction (not in current notes)
3. **Einsum cache** — `einsum_cache.jl` content-addressed caching (not in current notes)
4. **Batched MSE inverse** — `batched_inverse_code` path makes inverse batched too (current notes say MSE is "half-batched")
5. **Training pipeline** — `batch_max_iter` scaling, checkpoint support, full data prep flow (not in current notes at this detail)

## Conventions

- Both files use CeTZ 0.4.2 for diagrams
- Both files use the same `ngate` helper function for tensor box diagrams
- Code snippets show Julia with comments mapping to the math
- Typst math mode for formulas; code blocks for implementation
