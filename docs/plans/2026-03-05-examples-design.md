# Examples Cleanup — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace sprawling example scripts with three clean, focused examples that verify library performance and demonstrate the API.

**Architecture:** Start from PR#49's `feat/examples` branch. Keep `optimizer_benchmark.jl` as-is, rewrite `circuit_visualization.jl` without the abstract specification layer, and add a new minimal `basis_demo.jl`.

**Tech Stack:** Julia, ParametricDFT.jl, CairoMakie, Manopt.jl, CUDA.jl

---

### Task 1: Set up branch from feat/examples

**Files:** None (git operations only)

**Step 1: Create working branch from feat/examples**

```bash
git checkout origin/feat/examples -b examples-cleanup
```

**Step 2: Verify starting state**

```bash
ls examples/
```
Expected: `Project.toml`, `.gitignore`, `circuit_visualization.jl`, `optimizer_benchmark.jl` (plus possibly old files)

**Step 3: Remove old example files that shouldn't be in the final version**

```bash
git rm -f examples/entangle_position_demo.jl examples/verify_tebd.jl examples/cat.png 2>/dev/null
git rm -rf examples/BasisDemo examples/BasisDemo_QuickDraw 2>/dev/null
```

**Step 4: Commit**

```bash
git add -A examples/
git commit -m "chore: remove old example files"
```

---

### Task 2: Create basis_demo.jl

**Files:**
- Create: `examples/basis_demo.jl`

**Step 1: Write basis_demo.jl**

The script should (~80 lines):
1. `using ParametricDFT, Random, LinearAlgebra`
2. Set `Random.seed!(42)` for reproducibility
3. Define small basis sizes: `m = 3, n = 3` (8×8 images)
4. Create a random 8×8 test image: `img = rand(ComplexF64, 2^m, 2^n)`
5. For each basis type (`QFTBasis`, `EntangledQFTBasis`, `TEBDBasis`):
   a. Construct the basis: `basis = QFTBasis(m, n)` etc.
   b. Print basis info: `println("$(typeof(basis)): $(num_parameters(basis)) parameters, image_size=$(image_size(basis))")`
   c. Forward transform: `coeffs = forward_transform(basis, img)`
   d. Full inverse (no truncation): `recovered = inverse_transform(basis, coeffs)`
   e. Check round-trip error: `err = norm(img - recovered) / norm(img)`
   f. Print: `println("  Round-trip error: $err")`
   g. Truncated reconstruction: `truncated = topk_truncate(coeffs, k)` where `k = round(Int, length(coeffs) * 0.5)`
   h. `approx = inverse_transform(basis, truncated)`
   i. Print truncated error: `println("  50% truncation error: $(norm(img - approx) / norm(img))")`
6. Print summary table

**Step 2: Run to verify it works**

```bash
julia --project=. examples/basis_demo.jl
```
Expected: prints basis info, near-zero round-trip errors, moderate truncation errors

**Step 3: Commit**

```bash
git add examples/basis_demo.jl
git commit -m "feat: add minimal basis_demo.jl example"
```

---

### Task 3: Rewrite circuit_visualization.jl

**Files:**
- Rewrite: `examples/circuit_visualization.jl` (891 lines → ~250 lines)

**Step 1: Write the simplified circuit_visualization.jl**

The new version removes all abstract types (`AbstractCircuitSpec`, `QFTCircuitSpec`, etc.) and the generic dispatch layer. Instead, use direct plotting functions:

Structure (~250 lines):
1. Imports: `using ParametricDFT, CairoMakie, OMEinsum`
2. `CairoMakie.activate!(type = "png", px_per_unit = 2)`
3. Constants (keep PR#49's color scheme):
   ```julia
   const COLORS = (
       wire = "#333333", hadamard = "#4285F4", phase_gate = "#34A853",
       entangle_gate = "#EA4335", tebd_gate = "#9C27B0", wrap_gate = "#FF5722",
       text = "#333333", label = "#666666"
   )
   const GATE_SIZE = 0.6
   const WIRE_WIDTH = 2
   const QUBIT_SPACING = 1.0
   const GATE_SPACING = 1.2
   ```
4. Helper functions (~30 lines):
   - `draw_wire!(ax, y, x_start, x_end)` — draw a horizontal wire
   - `draw_gate!(ax, x, y, label, color)` — draw a colored gate box with label
   - `draw_two_qubit_gate!(ax, x, y1, y2, label, color)` — two-qubit gate with vertical line
5. `draw_qft!(ax, m; label_prefix="x")` (~40 lines):
   - Draw `m` horizontal wires
   - For each qubit: Hadamard gate, then phase gates R₂...Rₘ connecting to subsequent qubits
   - Label qubits on left
6. `draw_2d_qft!(ax, m, n)` (~20 lines):
   - Draw row QFT block (m qubits) then column QFT block (n qubits) with separator
7. `draw_entangled_qft!(ax, m, n; entangle_position=:back)` (~30 lines):
   - Draw 2D QFT with entanglement gates at specified position
   - Entanglement gates connect row qubit i to col qubit i (for i = 1..min(m,n))
8. `draw_tebd!(ax, m, n)` (~30 lines):
   - Draw m+n qubits
   - Row ring: nearest-neighbor two-qubit gates including wrap-around (xₘ,x₁)
   - Column ring: same for column qubits
9. `main()` function (~40 lines):
   - Create output directory
   - For each circuit type, create a Figure, call the draw function, save PNG
   - Circuits to draw: QFT 1D (m=4), QFT 2D (m=3, n=3), Entangled QFT (m=3, n=3), TEBD (m=3, n=3)

Key simplification: no `generate_circuit_operations`, no `render_circuit!` dispatcher, no abstract spec types. Each `draw_*!` function directly places gates on the axis.

**Step 2: Run to verify it generates PNGs**

```bash
julia --project=examples examples/circuit_visualization.jl
ls examples/CircuitDiagrams/
```
Expected: PNG files for each circuit type

**Step 3: Commit**

```bash
git add examples/circuit_visualization.jl
git commit -m "refactor: simplify circuit_visualization.jl — remove abstract spec layer"
```

---

### Task 4: Review and refine optimizer_benchmark.jl

**Files:**
- Review: `examples/optimizer_benchmark.jl` (keep as-is from PR#49, only minor cleanup)

**Step 1: Review the file for any issues**

Check for:
- Correct API usage (does `train_basis` signature match current src?)
- No hardcoded paths that won't work
- GPU fallback behavior (currently `@assert CUDA.functional()` — consider making GPU configs optional)

**Step 2: Minor cleanup if needed**

If `train_basis` API has changed, update the call. Otherwise keep as-is.

**Step 3: Commit if changes were made**

```bash
git add examples/optimizer_benchmark.jl
git commit -m "refactor: minor cleanup in optimizer_benchmark.jl"
```

---

### Task 5: Final verification and cleanup

**Files:**
- Review: `examples/Project.toml`, `examples/.gitignore`

**Step 1: Verify Project.toml has all needed deps**

Ensure it includes: ParametricDFT (path=".."), CUDA, CairoMakie, Manopt, Manifolds, ManifoldDiff, Images, ImageQualityIndexes, JSON3, FileIO, OMEinsum, RecursiveArrayTools, Zygote, ADTypes, LinearAlgebra (for basis_demo).

**Step 2: Verify .gitignore covers output dirs**

Ensure CircuitDiagrams/ is in .gitignore (for circuit_visualization output).

**Step 3: Run basis_demo.jl as a smoke test**

```bash
julia --project=. examples/basis_demo.jl
```
Expected: clean output with near-zero round-trip errors

**Step 4: Final commit**

```bash
git add examples/
git commit -m "chore: finalize examples — Project.toml and .gitignore"
```

---

### Summary

| Task | Script | Action | ~Lines |
|------|--------|--------|--------|
| 1 | — | Branch setup, remove old files | — |
| 2 | basis_demo.jl | Write new minimal showcase | ~80 |
| 3 | circuit_visualization.jl | Rewrite without abstract layer | ~250 |
| 4 | optimizer_benchmark.jl | Review, minor cleanup | ~700 |
| 5 | — | Project.toml, .gitignore, smoke test | — |
