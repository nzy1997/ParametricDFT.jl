# CLAUDE.md

## Project Overview
Julia library building quantum circuit for finding sparse basis in image compression.

## Skills
- [check-code-quality](skills/check-code-quality/SKILL.md) — Review code changes for design principles (DRY, KISS, SOLID, HC/LC), Julia-specific quality (type stability, AD safety, performance), and test quality.
- [issue-pr](skills/issue-pr/SKILL.md) — Create an issue and a pull request from the current branch.

## Commands
```bash
julia --project=. -e 'using Pkg; Pkg.test()'   # Run all tests
julia --project=. -e 'using Pkg; Pkg.status()'  # Check dependencies
```

## Git Safety
- **NEVER force push** (`git push --force`, `git push -f`, `git push --force-with-lease`). This is an absolute rule with no exceptions. Force push can silently destroy other people's work and stashed changes.
- **NEVER use `GIT_OBJECT_DIRECTORY` env var** to work around commit failures. If `git add` or `git commit` fails with "insufficient permission for adding an object to repository database", fix the root cause by running `git repack -a -d` to consolidate objects into an owned pack file.
- **Verify after commit** — Run `git fsck --connectivity-only` after each commit to catch object corruption immediately, rather than discovering it sessions later.
- **NEVER manipulate `.git` internals directly** — Don't copy, move, or create files inside `.git/objects/` manually. Use `git repack` to fix object storage issues.

## Architecture

### Core Modules
- `src/ParametricDFT.jl` — Main module, exports, dependency ordering via `include()`
- `src/loss.jl` — Loss functions (`L1Norm`, `L2Norm`, `MSELoss`) with batched evaluation
- `src/basis.jl` — Sparse basis abstractions (`QFTBasis`, `EntangledQFTBasis`, `TEBDBasis`, `MERABasis`)
- `src/manifolds.jl` — Riemannian manifold abstraction (`UnitaryManifold`, `PhaseManifold`), batched linear algebra
- `src/optimizers.jl` — Riemannian optimizers (`RiemannianGD`, `RiemannianAdam`) with unified `optimize!` interface
- `src/training.jl` — Training pipeline (`_train_basis_core`), device management, checkpointing, early stopping
- `src/qft.jl` — QFT circuit code generation
- `src/entangled_qft.jl` — Entangled QFT circuit implementation
- `src/tebd.jl` — TEBD circuit implementation
- `src/mera.jl` — MERA circuit implementation
- `src/serialization.jl` — Basis save/load (JSON3)
- `src/compression.jl` — Image compression using learned bases
- `src/visualization.jl` — Training loss visualization (CairoMakie)
- `src/circuit_visualization.jl` — Circuit diagram generation (`plot_circuit` for all basis types)
- `ext/CUDAExt.jl` — GPU extension module for CUDA support

### Abstract Hierarchy Tree
```
AbstractSparseBasis
├── QFTBasis              # Quantum Fourier Transform basis
├── EntangledQFTBasis     # QFT with entanglement gates
├── TEBDBasis             # Time-Evolving Block Decimation
└── MERABasis             # Multi-scale Entanglement Renormalization Ansatz

AbstractLoss
├── L1Norm
├── L2Norm
└── MSELoss

AbstractRiemannianManifold
├── UnitaryManifold       # U(n) unitary group
└── PhaseManifold         # U(1)^d product manifold

AbstractRiemannianOptimizer
├── RiemannianGD          # Gradient descent with Armijo line search
└── RiemannianAdam        # Adam with momentum transport
```

### Key Patterns
- **Multiple dispatch extensibility**: Add new behavior by defining new subtypes of abstract types and dispatching, not by modifying existing code.
- **Batched GPU operations**: `batched_matmul`, `batched_adjoint` reduce kernel launches. Batched einsum codes process multiple images in a single call.
- **Zygote AD compatibility**: No mutation in differentiated paths. Custom `rrule` for `topk_truncate`. Convert `Tuple` to `Vector` for stable AD tangent types.
- **OMEinsum tensor contractions**: Pre-optimized Einstein summation codes for forward/inverse transforms.
- **Device abstraction**: `to_device(x, :gpu/:cpu)` with CUDAExt extending functions for GPU arrays.
- **Circuit parameters**: Stored as `Vector{Matrix{ComplexF64}}`, optimized on Riemannian manifolds.

## Conventions

### File Naming
- Source files: `src/<feature>.jl` — one feature per file
- Test files: `test/<feature>_tests.jl`
- Main module: `src/ParametricDFT.jl` with explicit `include()` ordering
- Examples: `examples/<demo_name>.jl`

### Naming Conventions
- **Types**: PascalCase — `QFTBasis`, `UnitaryManifold`, `RiemannianAdam`
- **Abstract types**: `Abstract` prefix — `AbstractSparseBasis`, `AbstractRiemannianManifold`
- **Functions**: snake_case — `forward_transform`, `topk_truncate`, `stack_tensors`
- **Internal functions**: underscore prefix — `_loss_function`, `_compute_gradients`, `_batched_project`
- **Constants**: UPPER_CASE or CamelCase for singleton-like values

### Code Style
- Section separators: `# ============================================================================`
- Broadcast operators for array ops: `.+`, `.*`, `.^`
- Type annotations in public function signatures
- Docstrings with triple-quoted strings (`"""..."""`) including purpose, arguments, returns
- `@assert` for input validation preconditions
- `@inbounds` with explicit loops for performance-critical sections

## Testing Requirements

### Coverage

New code must have >95% test coverage.

### Naming
- Test sets: `@testset "FeatureName"` or `@testset "function_name"`
- Descriptive test names inside testsets

### Key Testing Patterns
- `Random.seed!(42)` at the start of stochastic tests for reproducibility
- Numerical accuracy: `@test result ≈ expected atol=1e-10` (never use `==` for floats)
- Mathematical properties: verify unitarity (`U * U' ≈ I`), manifold membership, loss monotonicity
- Shape checks paired with value checks — never shape-only
- Adversarial cases: boundary conditions, degenerate inputs, edge cases
- Gradient checks: compare AD gradients against finite differences

### File Organization
- `test/runtests.jl` — main test runner, includes all test files
- `test/<feature>_tests.jl` — one test file per source module
- `test/cuda_tests.jl` — GPU-specific tests (separate, requires CUDA)
- `test/benchmark_test.jl` — performance benchmarks

## Documentation Locations
- `docs/` — General documentation
- `docs/plans/` — Design documents and implementation plans
- `examples/` — Example scripts and demos
- `note/` — Theoretical notes and prerequisites (Typst)

## Documentation Requirements
- Public functions must have docstrings with arguments, returns, and examples
- Design documents go in `docs/plans/YYYY-MM-DD-<topic>.md`
- New abstract type subtypes must document which interface methods they implement
