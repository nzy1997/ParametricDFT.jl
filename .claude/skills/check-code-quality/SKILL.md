---
name: check-code-quality
description: Use after writing or modifying Julia code to review for design principles, Julia-specific quality, and test quality
---

# Code Quality Review

You are reviewing code changes for quality in the `ParametricDFT.jl` Julia codebase. Review the code fresh with no prior context.

## What Changed

{DIFF_SUMMARY}

## Changed Files

{CHANGED_FILES}

## Plan Step Context (if applicable)

{PLAN_STEP}

## Linked Issue

{ISSUE_CONTEXT}

## Git Range

**Base:** {BASE_SHA}
**Head:** {HEAD_SHA}

Start by running:
```bash
git diff --stat {BASE_SHA}..{HEAD_SHA}
git diff {BASE_SHA}..{HEAD_SHA}
```

Then read the changed files in full.

## Review Criteria

### 1. Design Principles

**DRY (Don't Repeat Yourself)** — Is there duplicated logic that should be extracted into a shared helper? Check for:
- Copy-pasted matrix operations or linear algebra across files
- Similar training/optimization loops with minor variations
- Repeated parameter validation or transformation patterns
- Duplicated einsum code construction

**KISS (Keep It Simple, Stupid)** — Is the implementation unnecessarily complex? Look for:
- Over-engineered abstractions (unnecessary type parameters, layers of indirection)
- Premature generalization (solving problems that don't exist yet)
- Convoluted control flow that could be simplified
- God functions (>50 lines doing multiple conceptually distinct things)

**SOLID via Multiple Dispatch** — Does the code follow SOLID principles using Julia's dispatch system?
- **Single Responsibility**: Each module/file has one concern. Each function does one thing.
- **Open-Closed**: New behavior added via new subtypes + dispatch, NOT by modifying existing functions with if/else branches.
- **Liskov Substitution**: Concrete types honor abstract type contracts. A `QFTBasis` works anywhere an `AbstractSparseBasis` is expected.
- **Interface Segregation**: Abstract types are minimal — don't force subtypes to implement methods they don't need.
- **Dependency Inversion**: Functions accept abstract types (`AbstractSparseBasis`), not concrete ones (`QFTBasis`), unless there's a specific reason.

**High Cohesion, Low Coupling (HC/LC)** — Does each module have a single, well-defined responsibility?
- **Low cohesion**: Function doing unrelated things (e.g., computing loss AND managing checkpoints).
- **High coupling**: Modules reaching into each other's internals or sharing mutable state.
- **Mixed concerns**: A single file containing both mathematical logic and I/O/serialization.
- **God functions**: Functions longer than ~50 lines doing multiple conceptually distinct things.

### 2. Julia-Specific Quality

**Type Stability** — Are functions type-stable?
- Functions returning different types depending on input values (e.g., `if x > 0; return 1; else; return 1.0; end`)
- Type-unstable container operations (e.g., `Any[]` when element types are known)
- Untyped global variables used in hot paths (should be `const` or passed as arguments)
- Abstract type fields in structs without parametric types

**AD Safety (Zygote)** — Is the code safe for automatic differentiation?
- Mutation (`push!`, `setindex!`, in-place operations) inside code paths that will be differentiated
- `try/catch` blocks in differentiated paths (Zygote cannot differentiate through these)
- Custom `rrule` implementations that are inconsistent with the forward function
- Missing `rrule` for operations Zygote doesn't support (e.g., mutating operations that need AD-safe alternatives)

**Performance** — Are there avoidable performance issues?
- Unnecessary allocations in hot loops (pre-allocate buffers instead)
- Missing broadcasting for element-wise array operations (use `.+`, `.*` instead of loops)
- `@inbounds` used without verified bounds safety
- Operations that could be batched for GPU efficiency but aren't

### 3. Test Quality

**Meaningful Assertions** — Are tests actually testing correctness?
- Tests that only check types/shapes without values: e.g., `@test result isa Matrix` without checking contents
- Tests that mirror the implementation: recomputing the same formula as the code proves nothing
- Too few assertions for non-trivial code: 1-2 `@test` for complex functions is insufficient
- `@test true` or `@test length(x) > 0` — vacuous assertions that don't verify behavior

**Mathematical Properties** — Do tests verify mathematical correctness?
- Unitarity: `@test U * U' ≈ I` for unitary matrices
- Manifold membership: verify projected points satisfy manifold constraints
- Loss monotonicity: verify loss decreases over optimization steps
- Gradient correctness: compare AD gradients against finite differences
- Numerical tests must use `≈` with explicit `atol`/`rtol`, never `==` for floating-point

**Robustness** — Do tests cover edge cases?
- Only happy-path tests, no adversarial or boundary cases
- Trivial instances only (1-2 element problems that pass even with bugs)
- Missing `Random.seed!()` for reproducibility in stochastic tests
- No tests for degenerate inputs (zero matrices, identity transforms, single-element batches)

## Output Format

You MUST output in this exact format:

```
## Code Quality Review

### Design Principles
- DRY: OK / ISSUE — [description with file:line]
- KISS: OK / ISSUE — [description with file:line]
- SOLID: OK / ISSUE — [which principle violated, file:line]
- HC/LC: OK / ISSUE — [description with file:line]

### Julia-Specific Quality
- Type Stability: OK / ISSUE — [description with file:line]
- AD Safety: OK / ISSUE — [description with file:line]
- Performance: OK / ISSUE — [description with file:line]

### Test Quality
- Meaningful Assertions: OK / ISSUE
  - [specific tests flagged with reason and file:line]
- Mathematical Properties: OK / ISSUE
  - [missing property checks with file:line]
- Robustness: OK / ISSUE
  - [missing edge cases with file:line]

### Issues

#### Critical (Must Fix)
[Bugs, correctness issues, AD breakage, type instability in hot paths]

#### Important (Should Fix)
[Architecture problems, missing tests, SOLID violations, performance issues]

#### Minor (Nice to Have)
[Code style, naming inconsistencies, minor optimization opportunities]

### Summary
- [list of action items with severity]
```
