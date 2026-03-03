# Remove Materialized Unitary Path — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Delete `src/materialized.jl` and all references, simplifying the training loop to einsum-only.

**Architecture:** The materialized path (build D×D unitary, use U*X GEMM) was benchmarked and found to be 175x slower than einsum at 32×32 and OOM at 64×64. The einsum-on-GPU path already gives 5-7x speedup for large images. We remove all materialized code and simplify the training loop from 3 code paths to 2.

**Tech Stack:** Julia, OMEinsum, Zygote

---

### Task 1: Remove materialized.jl from module

**Files:**
- Delete: `src/materialized.jl`
- Modify: `src/ParametricDFT.jl:76-77`

**Step 1: Delete the source file**

```bash
rm src/materialized.jl
```

**Step 2: Remove the include from the module**

In `src/ParametricDFT.jl`, delete lines 76-77:
```julia
# 7b. Materialized unitary (uses loss functions, batched einsum)
include("materialized.jl")
```

**Step 3: Verify module loads**

Run: `julia --project=. -e 'using ParametricDFT; println("OK")'`
Expected: `OK`

**Step 4: Commit**

```bash
git add -u src/materialized.jl src/ParametricDFT.jl
git commit -m "remove materialized.jl from module"
```

---

### Task 2: Remove materialized branch from training loop

**Files:**
- Modify: `src/training.jl:84-92` (strategy selection)
- Modify: `src/training.jl:132-142` (materialized loss branch)

**Step 1: Remove strategy selection block**

In `src/training.jl`, delete lines 84-92:
```julia
    # Pre-compute materialized unitary code for GPU acceleration of large images
    strategy = select_device_strategy(m, n, batch_size, device)
    unitary_optcode = nothing
    if strategy == :materialized_gpu
        n_gates_u = length(initial_tensors)
        D = 2^(m + n)
        flat_u, blabel_u = make_batched_code(optcode, n_gates_u)
        unitary_optcode = optimize_batched_code(flat_u, blabel_u, D)
    end
```

**Step 2: Simplify the loss function construction**

Replace lines 132-154 (the 3-branch if/elseif/else) with a 2-branch version:

```julia
            # Construct loss function for this batch
            batch_loss_fn = if batched_optcode !== nothing
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

**Step 3: Run tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: All tests pass (materialized tests will fail in next task — that's expected since we haven't removed them yet). If the test runner errors on missing `materialized_tests.jl` include, proceed to Task 3 first.

**Step 4: Commit**

```bash
git add src/training.jl
git commit -m "remove materialized branch from training loop"
```

---

### Task 3: Remove materialized tests

**Files:**
- Delete: `test/materialized_tests.jl`
- Modify: `test/runtests.jl:49`

**Step 1: Delete the test file**

```bash
rm test/materialized_tests.jl
```

**Step 2: Remove the include from runtests.jl**

In `test/runtests.jl`, delete line 49:
```julia
include("materialized_tests.jl")
```

**Step 3: Run full test suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: All tests pass. Zero failures.

**Step 4: Commit**

```bash
git add -u test/materialized_tests.jl test/runtests.jl
git commit -m "remove materialized tests"
```

---

### Task 4: Clean up examples

**Files:**
- Modify: `examples/profile_gpu.jl:163-232` (remove materialized benchmarks)
- Modify: `examples/gpu_benchmark.jl` (remove Part 4)

**Step 1: Remove materialized benchmarks from profile_gpu.jl**

Delete lines 163-232 (from `println("\n" * "="^70)` / "Materialized Unitary Benchmarks" through the end of the Analysis println). Replace with a simpler analysis section:

```julia
println("\n" * "="^70)
println("  Analysis")
println("="^70)
println("""
  Key insight: For 32x32 images with 2x2 gate tensors, each GPU kernel
  launch (~5-10us overhead) costs more than the actual computation.
  The einsum contracts many tiny tensors, each becoming a separate kernel.
  Zygote's AD tape multiplies this: backward pass launches ~2x more kernels.

  GPU wins when: larger images (256x256+) where per-kernel compute
  dominates launch overhead, or batched einsum with batch_size >= 2.
""")
```

**Step 2: Remove Part 4 from gpu_benchmark.jl**

Delete the entire "Part 4: Materialized U vs Einsum" section (lines 233-303 approximately) and the materialized memory table / summary text that references materialized. Update the summary to reflect einsum-only findings.

**Step 3: Verify examples parse**

Run: `julia --project=examples -e 'include("examples/profile_gpu.jl")' 2>&1 | head -5` — check for syntax errors (don't need to run full benchmark).

Actually, just verify syntax:
```bash
julia --project=examples -e 'include_string(Main, read("examples/profile_gpu.jl", String))' 2>&1 | head -3
julia --project=examples -e 'include_string(Main, read("examples/gpu_benchmark.jl", String))' 2>&1 | head -3
```

**Step 4: Commit**

```bash
git add examples/profile_gpu.jl examples/gpu_benchmark.jl
git commit -m "remove materialized benchmarks from examples"
```

---

### Task 5: Final verification

**Step 1: Run full test suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: All tests pass.

**Step 2: Verify no dangling references**

Run: `grep -r "materialized\|build_circuit_unitary\|select_device_strategy" src/ test/ --include="*.jl"`
Expected: No matches (only docs/plans/ and examples/ log files may have references, which is fine).

**Step 3: Commit if any fixups needed, otherwise done**
