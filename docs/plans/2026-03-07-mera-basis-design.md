# MERA Basis Design

**Date:** 2026-03-07
**Issue:** [#21 — MERA circuit structure](https://github.com/nzy1997/ParametricDFT.jl/issues/21)

## Summary

Add `MERABasis <: AbstractSparseBasis` implementing a Multi-scale Entanglement Renormalization Ansatz (MERA) circuit for image compression. The MERA provides a hierarchical multi-scale structure that captures correlations at different length scales.

## Design Decisions

1. **Isometry approximation:** Use full 2-qubit unitary gates (controlled-phase) instead of true St(2,4) isometries. This fits the existing `UnitaryManifold`/`PhaseManifold` infrastructure without requiring new manifold types.

2. **2D treatment:** Apply MERA independently to row and column qubits (separable), matching the QFT/TEBD pattern. Row MERA on qubits `1:m`, column MERA on qubits `m+1:m+n`.

3. **Gate type:** All disentanglers and isometries use controlled-phase gates (learnable phase parameters), consistent with QFT/TEBD. Combined with a Hadamard layer on all qubits.

4. **Multi-layer connectivity:** Hierarchical pairing with doubling stride. Layer l has stride `2^(l-1)`. This captures the true multi-scale MERA structure rather than simple alternating even/odd layers.

5. **Code structure:** Per-dimension builder (`_mera_single_dim`) composed by `mera_code(m, n)`, avoiding duplication for row/col.

## Circuit Structure

For `n_qubits = 2^k` qubits (one dimension), the MERA has `k` layers:

### Hadamard Layer
- H gates on all qubits (creates frequency/superposition basis)

### MERA Layers
Each layer `l` (l = 1, ..., k) has stride `s = 2^(l-1)`:

1. **Disentanglers** — controlled-phase gates on shifted pairs:
   - Pairs: `(s, 2s)`, `(3s, 4s)`, ...
   - Count: `2^(k-l)` per layer

2. **Isometries** — controlled-phase gates on block-aligned pairs:
   - Pairs: `(1, s+1)`, `(2s+1, 3s+1)`, ...
   - Count: `2^(k-l)` per layer

### Gate Counts
- Total phase gates per dimension: `2 * (2^k - 1)` = `2 * (n_qubits - 1)`
- For 2D with m row qubits, n col qubits: `2*(2^m - 1) + 2*(2^n - 1)` phase gates
- Plus `m + n` Hadamard gates

### Example: 8 qubits (k=3)
- Layer 1 (stride 1): disentanglers (2,3),(4,5),(6,7),(8,1); isometries (1,2),(3,4),(5,6),(7,8)
- Layer 2 (stride 2): disentanglers (2,4),(6,8); isometries (1,3),(5,7)
- Layer 3 (stride 4): disentangler (2,6); isometry (1,5)
- Total: 14 phase gates + 8 Hadamards

## Architecture

### New Files
- `src/mera.jl` — Circuit construction
- `test/mera_tests.jl` — Tests

### Modified Files
- `src/basis.jl` — `MERABasis` struct + interface methods
- `src/ParametricDFT.jl` — Exports and `include("mera.jl")`
- `src/training.jl` — `train_basis(::Type{MERABasis}, ...)` dispatch
- `src/serialization.jl` — `MERABasisJSON` + save/load
- `test/runtests.jl` — Include `mera_tests.jl`

### `src/mera.jl` Functions

```julia
_mera_single_dim(n_qubits, qubit_offset, total_qubits, phases) -> Yao chain block
mera_code(m, n; phases=nothing, inverse=false) -> (optcode, tensors, n_row_gates, n_col_gates)
get_mera_gate_indices(tensors, n_gates) -> Vector{Int}
extract_mera_phases(tensors, gate_indices) -> Vector{Float64}
```

### `MERABasis` Struct

```julia
struct MERABasis <: AbstractSparseBasis
    m::Int
    n::Int
    tensors::Vector
    optcode::AbstractEinsum
    inverse_code::AbstractEinsum
    n_row_gates::Int       # 2*(2^m - 1)
    n_col_gates::Int       # 2*(2^n - 1)
    phases::Vector{Float64}
end
```

### Interface Methods
- `forward_transform(basis::MERABasis, image)` — einsum contraction
- `inverse_transform(basis::MERABasis, freq_domain)` — conjugated einsum
- `image_size(basis::MERABasis)` — `(2^m, 2^n)`
- `num_parameters(basis::MERABasis)` — sum of tensor element counts
- `basis_hash(basis::MERABasis)` — SHA-256 of parameters
- `num_gates(basis::MERABasis)` — `n_row_gates + n_col_gates`
- `get_phases(basis::MERABasis)` — copy of phase vector

## Testing Plan

1. Circuit construction — correct types, tensor counts
2. Gate count verification — e.g. m=3,n=3: 14+14=28 phase gates
3. Unitarity — forward/inverse roundtrip recovery
4. Phase extraction — roundtrip with custom initial phases
5. Basis interface — all required methods work
6. Inverse code — `mera_code(m, n; inverse=true)` validity
7. Edge cases — m=1,n=1 (2x2 images); asymmetric m != n
8. Serialization — save/load roundtrip
9. Training smoke test — `train_basis(MERABasis, ...; epochs=1)` runs

All tests use `Random.seed!(42)` and `atol=1e-10`.
