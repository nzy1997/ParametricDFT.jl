---
description: Julia coding conventions for the ParametricDFT.jl codebase — auto-activated when editing Julia files
globs: ["*.jl"]
---

# Julia Conventions for ParametricDFT.jl

Follow these conventions when writing or modifying Julia code in this project.

## Naming

- **Types**: PascalCase — `QFTBasis`, `UnitaryManifold`, `RiemannianAdam`
- **Abstract types**: `Abstract` prefix — `AbstractSparseBasis`, `AbstractRiemannianManifold`
- **Functions**: snake_case — `forward_transform`, `topk_truncate`, `stack_tensors`
- **Internal functions**: underscore prefix — `_loss_function`, `_compute_gradients`
- **Constants**: UPPER_CASE or CamelCase for singleton-like values
- **Type parameters**: single uppercase letters — `T`, `M`, `O`

## Type Hierarchy

- New types extending this library MUST subtype the appropriate abstract type
- Implement all required interface methods for the abstract type (check existing subtypes for the contract)
- Use multiple dispatch for specialization — NEVER use `if typeof(x) == ConcreteType` branching
- Keep abstract types minimal: don't add methods unless all subtypes need them
- Example: a new basis → `struct MyBasis <: AbstractSparseBasis ... end`, then define `forward_transform`, `inverse_transform`, etc.

## Function Design

- Keep functions under 50 lines; extract sub-operations into well-named helpers
- Public API functions: type annotations in signatures, docstrings with `"""..."""`
- Preconditions: `@assert condition "message"` at function entry
- Fatal errors: `error("descriptive message")`
- Return types should be inferrable from input types (type stability)

## AD Compatibility (Zygote)

These rules apply to ANY code that will be differentiated through (loss functions, forward/inverse transforms, training steps):

- **No mutation**: Never use `push!`, `setindex!`, `fill!`, or `x .= y` in differentiated paths. Construct new arrays instead.
- **No try/catch**: Zygote cannot differentiate through exception handling. Use conditionals.
- **Stable tangent types**: Convert `Tuple` to `Vector` when Zygote produces tuple tangents. See `_compute_gradients` in `training.jl` for the pattern.
- **Custom rrules**: When Zygote doesn't support an operation, define `rrule` in ChainRulesCore. The rrule must be mathematically consistent with the forward function.

## Performance

- **Broadcasting**: Use `.+`, `.*`, `.^` for element-wise array operations, not explicit loops (unless loop is needed for control flow).
- **Pre-allocation**: In hot loops, pre-allocate output buffers and reuse them across iterations.
- **`@inbounds`**: Only use when bounds are verified. Always pair with a bounds check or `@assert` outside the loop.
- **Batching for GPU**: Reduce kernel launches by batching operations. Use `batched_matmul` and batched einsum codes.
- **Avoid untyped globals**: Global variables in hot paths must be `const`. Pass configuration as function arguments instead.

## Testing

- Every new public function gets a `@testset` block
- Use `Random.seed!(42)` for reproducibility in stochastic tests
- Numerical comparisons: `@test result ≈ expected atol=1e-10` — NEVER use `==` for floating-point
- Verify mathematical properties:
  - Unitarity: `@test U * U' ≈ I`
  - Manifold membership after projection/retraction
  - Loss monotonicity during optimization
- Include adversarial cases: degenerate inputs, boundary conditions, single-element batches
- Test files: `test/<feature>_tests.jl`, included from `test/runtests.jl`
- >95% test coverage required for new code
