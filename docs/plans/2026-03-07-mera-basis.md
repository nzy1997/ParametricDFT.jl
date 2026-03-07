# MERA Basis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `MERABasis <: AbstractSparseBasis` with hierarchical multi-scale MERA circuit for image compression.

**Architecture:** Per-dimension MERA builder (`_mera_single_dim`) composed into `mera_code(m, n)`. MERA applied independently to row and column qubits. All gates are controlled-phase (learnable phases) + Hadamards, fitting the existing manifold/optimizer infrastructure.

**Tech Stack:** Yao.jl (circuit construction), OMEinsum (tensor network), Zygote (AD), JSON3 (serialization)

**Design doc:** `docs/plans/2026-03-07-mera-basis-design.md`

**Working directory:** `/tmp/ParametricDFT-work` on branch `feature/mera-basis`

---

### Task 1: Create `src/mera.jl` — Circuit Construction

**Files:**
- Create: `src/mera.jl`

**Step 1: Write `_mera_single_dim` and `mera_code`**

Create `src/mera.jl` with the following content:

```julia
# ============================================================================
# MERA (Multi-scale Entanglement Renormalization Ansatz) Circuit Construction
# ============================================================================
# This file provides the MERA circuit construction with hierarchical multi-scale
# structure. For n = 2^k qubits per dimension, the MERA has k layers.
# Each layer l has stride s = 2^(l-1) and contains:
#   1. Disentanglers: controlled-phase gates on shifted pairs at stride s
#   2. Isometries: controlled-phase gates on block-aligned pairs at stride s
# MERA is applied independently to row and column qubits (separable 2D).

"""
    _n_mera_gates(n_qubits::Int)

Calculate the number of MERA phase gates for a single dimension.
For n_qubits = 2^k, returns 2*(n_qubits - 1).

# Arguments
- `n_qubits::Int`: Number of qubits (must be a power of 2)

# Returns
- `Int`: Number of phase gates
"""
function _n_mera_gates(n_qubits::Int)
    return 2 * (n_qubits - 1)
end

"""
    _mera_single_dim(n_qubits::Int, qubit_offset::Int, total_qubits::Int, phases::Vector{<:Real})

Build a MERA circuit for a single dimension (row or column qubits).

For n_qubits = 2^k, builds k layers. Each layer l (1-indexed) has stride s = 2^(l-1):
- Disentanglers on shifted pairs: (s, 2s), (3s, 4s), ...
- Isometries on block-aligned pairs: (1, s+1), (2s+1, 3s+1), ...

Qubit indices are offset by `qubit_offset` and the circuit is built on a `total_qubits` register.

# Arguments
- `n_qubits::Int`: Number of qubits for this dimension (must be power of 2, >= 2)
- `qubit_offset::Int`: Offset for qubit indices (0 for row, m for col)
- `total_qubits::Int`: Total number of qubits in full circuit
- `phases::Vector{<:Real}`: Phase parameters for all gates in this dimension.
  Length must equal `2*(n_qubits - 1)`.

# Returns
- `AbstractBlock`: Yao chain block for this dimension's MERA
"""
function _mera_single_dim(n_qubits::Int, qubit_offset::Int, total_qubits::Int, phases::Vector{<:Real})
    @assert n_qubits >= 2 "n_qubits must be >= 2, got $n_qubits"
    @assert ispow2(n_qubits) "n_qubits must be a power of 2, got $n_qubits"
    expected_n_gates = _n_mera_gates(n_qubits)
    @assert length(phases) == expected_n_gates "phases must have length $expected_n_gates, got $(length(phases))"

    k = Int(log2(n_qubits))
    qc = chain(total_qubits)
    phase_idx = 1

    for l in 1:k
        s = 2^(l - 1)
        n_pairs = n_qubits ÷ (2 * s)  # 2^(k-l) pairs

        # Disentanglers: shifted pairs (s, 2s), (3s, 4s), ...
        for p in 1:n_pairs
            q1 = qubit_offset + (2 * p - 1) * s
            q2 = qubit_offset + 2 * p * s
            # Wrap q2 around if it exceeds n_qubits
            q2_wrapped = qubit_offset + mod1(2 * p * s, n_qubits)
            push!(qc, control(total_qubits, q1, q2_wrapped => shift(phases[phase_idx])))
            phase_idx += 1
        end

        # Isometries: block-aligned pairs (1, s+1), (2s+1, 3s+1), ...
        for p in 1:n_pairs
            q1 = qubit_offset + (2 * (p - 1)) * s + 1
            q2 = qubit_offset + (2 * (p - 1)) * s + s + 1
            push!(qc, control(total_qubits, q1, q2 => shift(phases[phase_idx])))
            phase_idx += 1
        end
    end

    return qc
end

"""
    mera_code(m::Int, n::Int; phases=nothing, inverse=false)

Generate an optimized tensor network representation of a 2D MERA circuit.

The circuit consists of:
1. Hadamard layer: H gates on all m+n qubits
2. Row MERA: hierarchical multi-scale layers on qubits 1:m
3. Column MERA: hierarchical multi-scale layers on qubits m+1:m+n

For m row qubits and n col qubits (both must be powers of 2, >= 2):
- Row gates: 2*(2^m - 1) = 2*(m_dim - 1) controlled-phase gates
- Col gates: 2*(2^n - 1) = 2*(n_dim - 1) controlled-phase gates

Note: m and n here are the number of qubits, so image dimensions are 2^m x 2^n.

# Arguments
- `m::Int`: Number of row qubits (must be >= 1; if 1, no MERA layers for rows, only Hadamard)
- `n::Int`: Number of column qubits (must be >= 1; if 1, no MERA layers for cols, only Hadamard)
- `phases::Union{Nothing, Vector{<:Real}}`: Initial phases for MERA gates.
  If nothing, defaults to zeros. Length must equal `2*(2^m - 1) + 2*(2^n - 1)`.
  However, if m or n is 1 (single qubit), that dimension has 0 gates.
- `inverse::Bool`: If true, generate inverse transform code

# Returns
- `optcode::AbstractEinsum`: Optimized einsum contraction code
- `tensors::Vector`: Circuit parameters (Hadamard gates + MERA phase gates)
- `n_row_gates::Int`: Number of row MERA phase gates
- `n_col_gates::Int`: Number of column MERA phase gates

# Example
```julia
# Create MERA for 8×8 images (m=3, n=3)
optcode, tensors, n_row, n_col = mera_code(3, 3)

# Create with custom phases
n_gates = 2*(2^3 - 1) + 2*(2^3 - 1)  # 28 total
phases = rand(n_gates) * 2π
optcode, tensors, n_row, n_col = mera_code(3, 3; phases=phases)
```
"""
function mera_code(m::Int, n::Int; phases::Union{Nothing, Vector{<:Real}}=nothing, inverse=false)
    @assert m >= 1 "m must be >= 1, got $m"
    @assert n >= 1 "n must be >= 1, got $n"

    total = m + n
    n_row_gates = m >= 2 ? _n_mera_gates(2^m) : 0  # No MERA for single qubit
    n_col_gates = n >= 2 ? _n_mera_gates(2^n) : 0
    n_gates = n_row_gates + n_col_gates

    # Default phases to zeros if not provided
    if phases === nothing
        phases = zeros(n_gates)
    end
    @assert length(phases) == n_gates "phases must have length $n_gates ($(n_row_gates) row + $(n_col_gates) col), got $(length(phases))"

    row_phases = phases[1:n_row_gates]
    col_phases = phases[n_row_gates+1:end]

    # Build circuit
    qc = chain(total)

    # Layer 1: Hadamard gates on all qubits
    for i in 1:total
        push!(qc, put(total, i => H))
    end

    # Layer 2: Row MERA (qubits 1:m) — only if m >= 2
    if m >= 2
        # _mera_single_dim expects n_qubits as number of qubits in that dimension
        # but here m is log2 of that. The actual qubit count in the register is m.
        # Wait — m IS the number of qubits. 2^m is the image dimension.
        # So we pass m as n_qubits to _mera_single_dim? No — _mera_single_dim
        # expects n_qubits to be a power of 2 and builds log2(n_qubits) layers.
        # Since m qubits means 2^m image dim, we should pass m as n_qubits
        # only if m itself is a power of 2... but that's wrong.
        #
        # Actually re-reading the design: for n_qubits = 2^k qubits, MERA has k layers.
        # But m IS the number of qubits (not 2^k). So n_qubits = m, k = log2(m).
        # This means m must be a power of 2.
        #
        # But wait — looking at the design doc example: "8 qubits (k=3)" with
        # n_row_gates = 2*(8-1) = 14. That's 8 qubits = 2^3.
        # But in the codebase, m=3 means 3 qubits, 2^3=8 image dim.
        # The gate count formula says n_row_gates = 2*(2^m - 1).
        # For m=3: n_row_gates = 2*(8-1) = 14. But we only have 3 qubits!
        #
        # This is inconsistent. With only m=3 qubits, the MERA stride structure
        # would have: k=log2(3) which is not integer.
        #
        # The design must mean: m qubits, and m must be such that we can build
        # a hierarchical structure. The number of MERA layers = floor(log2(m)).
        # But the gate count 2*(2^m - 1) doesn't match m qubits.
        #
        # Let me re-read: "n_row_gates = 2*(2^m - 1)" — this is huge for m=3 (14 gates
        # on 3 qubits). That can't be right.
        #
        # Actually: looking at it again, the design says the MERA is applied to
        # "m row qubits" and the circuit has stride doubling. With m qubits:
        # - Layer 1 (stride 1): pairs (1,2),(3,4),... → floor(m/2) disentanglers + isometries
        # - Layer 2 (stride 2): pairs (1,3),(5,7),... → floor(m/4) each
        # - etc.
        # Total gates = 2 * sum_{l=1}^{floor(log2(m))} floor(m / 2^l)
        #
        # For m=3: Layer 1 has 1 disentangler + 1 isometry = 2 gates. Done (stride 2 > m/2).
        # For m=4: Layer 1 has 2+2=4, Layer 2 has 1+1=2 = 6 gates = 2*(4-1) ✓
        # For m=8: 2*(8-1) = 14 ✓
        #
        # So the formula 2*(m-1) works when m is a power of 2.
        # For non-power-of-2, we need a different count.
        #
        # The simplest approach: require m to be a power of 2 (>= 2) for MERA dimensions.
        # This is consistent with the design doc which says "n_qubits = 2^k".
        # Since m is the number of qubits, m itself must be a power of 2.
        #
        # But existing bases (QFT, TEBD) work with any m. Let's keep it simple
        # and require m to be a power of 2 for MERABasis.
        @assert ispow2(m) "m must be a power of 2 for MERA, got $m"
        row_mera = _mera_single_dim(m, 0, total, row_phases)
        push!(qc, row_mera)
    end

    # Layer 3: Column MERA (qubits m+1:m+n) — only if n >= 2
    if n >= 2
        @assert ispow2(n) "n must be a power of 2 for MERA, got $n"
        col_mera = _mera_single_dim(n, m, total, col_phases)
        push!(qc, col_mera)
    end

    # Convert to tensor network
    tn = yao2einsum(qc; optimizer=nothing)

    # Reorder tensors: Hadamard gates first, then phase gates
    perm_vec = sortperm(tn.tensors, by=x -> !(x ≈ mat(H)))
    ixs = tn.code.ixs[perm_vec]
    tensors = tn.tensors[perm_vec]

    if inverse
        code_reorder = DynamicEinCode([ixs..., tn.code.iy[total+1:end]], tn.code.iy[1:total])
    else
        code_reorder = DynamicEinCode([ixs..., tn.code.iy[1:total]], tn.code.iy[total+1:end])
    end
    optcode = optimize_code(code_reorder, uniformsize(tn.code, 2), TreeSA())

    return optcode, tensors, n_row_gates, n_col_gates
end

"""
    get_mera_gate_indices(tensors, n_gates::Int)

Identify which tensors correspond to MERA phase gates.

# Arguments
- `tensors::Vector`: Circuit tensors
- `n_gates::Int`: Number of MERA phase gates

# Returns
- `Vector{Int}`: Indices of MERA phase gate tensors
"""
function get_mera_gate_indices(tensors, n_gates::Int)
    function is_ctrl_phase(x)
        size(x) != (2, 2) && return false
        tol = 0.15
        return isapprox(abs(x[1,1]), 1, atol=tol) &&
               isapprox(abs(x[1,2]), 1, atol=tol) &&
               isapprox(abs(x[2,1]), 1, atol=tol) &&
               isapprox(abs(x[2,2]), 1, atol=tol)
    end

    ctrl_phase_indices = findall(is_ctrl_phase, tensors)
    if length(ctrl_phase_indices) >= n_gates
        return ctrl_phase_indices[end-n_gates+1:end]
    else
        return ctrl_phase_indices
    end
end

"""
    extract_mera_phases(tensors, gate_indices::Vector{Int})

Extract the phase parameters from MERA gate tensors.

# Arguments
- `tensors::Vector`: Circuit tensors
- `gate_indices::Vector{Int}`: Indices of MERA phase gates

# Returns
- `Vector{Float64}`: Phase parameters for each MERA gate
"""
function extract_mera_phases(tensors, gate_indices::Vector{Int})
    phases = Float64[]
    for idx in gate_indices
        push!(phases, angle(tensors[idx][2, 2]))
    end
    return phases
end
```

**IMPORTANT NOTE on `_mera_single_dim`:** The function receives `n_qubits` which is the actual number of qubits in the dimension (e.g., m=4 means 4 qubits, image size 2^4=16). For MERA, `n_qubits` must be a power of 2 (>= 2). The number of layers is `log2(n_qubits)`. The gate count per dimension is `2*(n_qubits - 1)`.

**IMPORTANT NOTE on gate count:** The design doc says `n_row_gates = 2*(2^m - 1)`. This is WRONG — it should be `2*(m - 1)` where m is the number of qubits. The design doc confused `m` (number of qubits) with `2^m` (image dimension). The correct formula: for m qubits, `n_row_gates = 2*(m - 1)`. For m=4 (16-pixel row): 6 gates. For m=8 (256-pixel row): 14 gates. The implementation above in `_n_mera_gates` takes `n_qubits` directly, so `_n_mera_gates(m) = 2*(m-1)` and `_n_mera_gates(2^m)` would be wrong. **Use `_n_mera_gates(m)` not `_n_mera_gates(2^m)`.**

Fix the `mera_code` function's gate count lines to:
```julia
n_row_gates = m >= 2 ? _n_mera_gates(m) : 0
n_col_gates = n >= 2 ? _n_mera_gates(n) : 0
```

**Step 2: Verify `_mera_single_dim` gate pairing manually**

Before writing tests, verify the gate pairs for m=4 (4 qubits, k=2):
- Layer 1 (s=1): disentanglers (1,2),(3,4) → 2 pairs; isometries (1,2),(3,4) → 2 pairs
  Wait, that gives the same pairs. Let me re-derive from the design:
  - Disentanglers: `(s, 2s), (3s, 4s)` = (1,2),(3,4) with offset
  - Isometries: `(1, s+1), (2s+1, 3s+1)` = (1,2),(3,4) with offset
  These are the same pairs! The disentanglers should be on the *shifted* pairs.

  Looking at the 8-qubit example from design:
  - Layer 1 (s=1): disentanglers (2,3),(4,5),(6,7),(8,1); isometries (1,2),(3,4),(5,6),(7,8)

  So disentanglers start at qubit `s+1` pairing with `s+2`, stride by `2s`:
  - Disentangler pairs: `(s+1, 2s), (3s+1, 4s), ...` — no, that doesn't match (2,3).

  For 8 qubits, s=1: disentanglers at (2,3),(4,5),(6,7),(8,1).
  Pattern: start at qubit 2, pairs of stride 1, stepping by 2: (2,3), (4,5), (6,7), (8,1→wrap).

  Isometries: (1,2),(3,4),(5,6),(7,8).
  Pattern: start at qubit 1, pairs of stride 1, stepping by 2.

  Layer 2 (s=2): disentanglers (2,4),(6,8); isometries (1,3),(5,7).
  Pattern: disentanglers start at 2 with stride 2, stepping by 4.
  Isometries start at 1 with stride 2, stepping by 4.

  Layer 3 (s=4): disentangler (2,6); isometry (1,5).

  General pattern for layer l, stride s = 2^(l-1), with n_qubits qubits:
  - n_pairs = n_qubits / (2*s)
  - Disentanglers: for p = 0, 1, ..., n_pairs-1: (2*p*s + s + 1, 2*p*s + 2*s) with wrapping
    Simplified: (2*p*s + 2, 2*p*s + s + 1) — no, let me just use the examples.

  For s=1, n=8: pairs at (2,3),(4,5),(6,7),(8,1)
    p=0: (2, 3) = (0*2+2, 0*2+3) = (2, 3)
    p=1: (4, 5) = (1*2+2, 1*2+3) = (4, 5)
    p=2: (6, 7) = (2*2+2, 2*2+3) = (6, 7)
    p=3: (8, 1) = (3*2+2, 3*2+3=9→1 wrap) = (8, 1)

    Formula: q1 = 2*p*s + s + 1 = 2p+2, q2 = 2*p*s + 2*s = 2p+2 → no.
    q1 = 2*p + 2, q2 = 2*p + 3 (with mod n_qubits wrapping on q2)

  For s=2, n=8: pairs (2,4),(6,8)
    p=0: (2, 4) = (0*4+2, 0*4+4)
    p=1: (6, 8) = (1*4+2, 1*4+4)
    Formula: q1 = 2*p*s + 2 = 4p+2, q2 = 2*p*s + 2*s = 4p+4
    General: q1 = 2*p*s + s, q2 = 2*p*s + 2*s — with 1-based: q1 = 2*p*s + s, q2 = 2*p*s + 2*s
    s=1: q1 = 2p+1, q2 = 2p+2 → (1,2),(3,4),(5,6),(7,8) — that's isometries not disentanglers!

  Let me be precise. With 1-based indexing:
  - **Isometries** at layer l (stride s): q1 = 2*p*s + 1, q2 = 2*p*s + s + 1 for p = 0..n_pairs-1
    s=1: (1,2),(3,4),(5,6),(7,8) ✓
    s=2: (1,3),(5,7) ✓
    s=4: (1,5) ✓

  - **Disentanglers** at layer l (stride s): q1 = 2*p*s + s + 1, q2 = mod1(2*(p+1)*s + 1, n_qubits) for p = 0..n_pairs-1
    s=1: q1 = 2p+2, q2 = mod1(2p+3, 8) → (2,3),(4,5),(6,7),(8,1) ✓
    s=2: q1 = 4p+3, q2 = mod1(4p+5, 8) → (3,5),(7,1) — but design says (2,4),(6,8)!

  That doesn't match. Let me re-examine:
  s=2, n=8: disentanglers should be (2,4),(6,8).
    q1 = 2*p*s + s = 4p+2, q2 = 2*p*s + 2*s = 4p+4
    p=0: (2,4) ✓, p=1: (6,8) ✓

  s=1, n=8: q1 = 2p+1, q2 = 2p+2
    p=0: (1,2), p=1: (3,4), p=2: (5,6), p=3: (7,8) — that's isometries again!

  Hmm, so for s=1 disentanglers (2,3),(4,5),(6,7),(8,1):
    q1 = 2*p*s + s + 1 = 2p+2, q2 = mod1(2*p*s + s + 2, 8) = mod1(2p+3, 8)
    p=0: (2,3), p=1: (4,5), p=2: (6,7), p=3: (8, mod1(9,8)=1) ✓

  For s=2: q1 = 2*p*s + s + 1 = 4p+3, q2 = mod1(4p+4, 8)
    p=0: (3,4), p=1: (7,8) — NOT (2,4),(6,8)!

  The design example must be wrong for layer 2, OR the indexing is different.

  Let me look at this differently. The disentanglers should connect the "boundary" qubits between isometry blocks. If isometries at stride s pair (1,s+1),(2s+1,3s+1),..., then disentanglers at stride s should pair (s+1,2s+1) boundaries — the qubit at the end of one block with the qubit at the start of the next:

  s=1: isometry blocks (1,2),(3,4),(5,6),(7,8). Boundaries: (2,3),(4,5),(6,7),(8,1) ✓
  s=2: isometry blocks (1,3),(5,7). Boundaries: (3,5),(7,1) — these are stride-2 gaps.

  But the design says s=2 disentanglers are (2,4),(6,8). These are NOT boundaries. They look like within-block pairs at the even positions.

  I think the design's layer 2 example may have a different convention. Let me just go with the standard binary MERA convention from the literature, which is the "boundary" pattern:

  **Disentanglers** connect adjacent blocks: q1 = 2*p*s + s + 1, q2 = mod1(2*(p+1)*s + 1, n_qubits) — but that's a gap of s, not adjacent.

  Actually the standard MERA has disentanglers BEFORE isometries within each layer. The disentanglers remove short-range entanglement between neighboring sites that will be grouped by the isometry. Let me use the convention:

  **Layer l, stride s = 2^(l-1):**
  1. Disentanglers on "odd" pairs: (q, q+s) where q = s, 3s, 5s, ... with wrapping
  2. Isometries on "even" pairs: (q, q+s) where q = 1, 2s+1, 4s+1, ...

  This gives for n=8:
  Layer 1 (s=1): Dis: (1,2),(3,4),(5,6),(7,8); Iso: (2,3),(4,5),(6,7),(8,1)
  — but that reverses what the design says.

  **Decision:** The exact gate pairing affects expressiveness but not correctness (all are learnable phases). Use the standard MERA convention from Vidal's paper where disentanglers come first and act on the inter-block boundaries, isometries act on within-block pairs. The implementation should be verified by the roundtrip test (forward+inverse = identity) which confirms unitarity regardless of pairing convention.

**Corrected `_mera_single_dim`:**

```julia
function _mera_single_dim(n_qubits::Int, qubit_offset::Int, total_qubits::Int, phases::Vector{<:Real})
    @assert n_qubits >= 2 "n_qubits must be >= 2, got $n_qubits"
    @assert ispow2(n_qubits) "n_qubits must be a power of 2, got $n_qubits"
    expected_n_gates = _n_mera_gates(n_qubits)
    @assert length(phases) == expected_n_gates "phases must have length $expected_n_gates, got $(length(phases))"

    k = Int(log2(n_qubits))
    qc = chain(total_qubits)
    phase_idx = 1

    for l in 1:k
        s = 2^(l - 1)
        n_pairs = n_qubits ÷ (2 * s)

        # Disentanglers: inter-block boundary pairs
        for p in 0:(n_pairs - 1)
            q1 = qubit_offset + mod1(2 * p * s + s, n_qubits)
            q2 = qubit_offset + mod1(2 * p * s + s + 1, n_qubits)
            push!(qc, control(total_qubits, q1, q2 => shift(phases[phase_idx])))
            phase_idx += 1
        end

        # Isometries: within-block pairs
        for p in 0:(n_pairs - 1)
            q1 = qubit_offset + 2 * p * s + 1
            q2 = qubit_offset + 2 * p * s + s + 1
            push!(qc, control(total_qubits, q1, q2 => shift(phases[phase_idx])))
            phase_idx += 1
        end
    end

    return qc
end
```

For n=8 this gives:
- Layer 1 (s=1, 4 pairs each):
  Dis: (1,2),(3,4),(5,6),(7,8); Iso: (1,2),(3,4),(5,6),(7,8) — same pairs!

That's wrong too. The issue is these controlled-phase gates commute when on the same qubits, so having two on the same pair just doubles the phase — it's equivalent to one gate with double the phase. We need distinct pairs.

**Final corrected approach — use the design doc's convention directly:**

Looking at the design doc's 8-qubit example one more time:
- Layer 1 (s=1): dis (2,3),(4,5),(6,7),(8,1); iso (1,2),(3,4),(5,6),(7,8)
- Layer 2 (s=2): dis (2,4),(6,8); iso (1,3),(5,7)
- Layer 3 (s=4): dis (2,6); iso (1,5)

Pattern for **isometries**: q1 = 2*p*s + 1, q2 = 2*p*s + s + 1, p = 0..n_pairs-1
- s=1: (1,2),(3,4),(5,6),(7,8) ✓
- s=2: (1,3),(5,7) ✓
- s=4: (1,5) ✓

Pattern for **disentanglers**: q1 = 2*p*s + 2, q2 = mod1(2*p*s + s + 2, n_qubits), p = 0..n_pairs-1
- s=1: (2,3),(4,5),(6,7),(8,mod1(9,8)=1) → (2,3),(4,5),(6,7),(8,1) ✓
- s=2: (2,4),(6,8) ✓  — (2, mod1(4,8)=4), (6, mod1(8,8)=8)
- s=4: (2,6) ✓ — (2, mod1(6,8)=6)

**This works!** Disentangler formula: q1 = 2*p*s + 2, q2 = mod1(2*p*s + s + 2, n_qubits)

But wait — with qubit_offset, the formulas need adjustment. Let me write it cleanly:

```julia
# Disentanglers
for p in 0:(n_pairs - 1)
    local_q1 = 2 * p * s + 2
    local_q2 = mod1(2 * p * s + s + 2, n_qubits)
    q1 = qubit_offset + local_q1
    q2 = qubit_offset + local_q2
    push!(qc, control(total_qubits, q1, q2 => shift(phases[phase_idx])))
    phase_idx += 1
end

# Isometries
for p in 0:(n_pairs - 1)
    local_q1 = 2 * p * s + 1
    local_q2 = 2 * p * s + s + 1
    q1 = qubit_offset + local_q1
    q2 = qubit_offset + local_q2
    push!(qc, control(total_qubits, q1, q2 => shift(phases[phase_idx])))
    phase_idx += 1
end
```

For m=4 (4 qubits, k=2):
- Layer 1 (s=1, 2 pairs): dis (2,3),(4,1); iso (1,2),(3,4) → 4 gates
- Layer 2 (s=2, 1 pair): dis (2,4); iso (1,3) → 2 gates
- Total: 6 = 2*(4-1) ✓

For m=2 (2 qubits, k=1):
- Layer 1 (s=1, 1 pair): dis (2,mod1(3,2)=1); iso (1,2) → 2 gates
- Total: 2 = 2*(2-1) ✓

**Step 3: Commit `src/mera.jl`**

```bash
cd /tmp/ParametricDFT-work
git add src/mera.jl
git commit -m "feat: add MERA circuit construction (src/mera.jl)"
git fsck --connectivity-only
```

---

### Task 2: Write initial tests for MERA circuit

**Files:**
- Create: `test/mera_tests.jl`
- Modify: `test/runtests.jl` (add `include("mera_tests.jl")`)

**Step 1: Write test file**

```julia
# ============================================================================
# Tests for MERA Circuit (mera.jl)
# ============================================================================

using Yao: mat, H

@testset "MERA Circuit" begin

    @testset "mera_code basic construction" begin
        Random.seed!(42)
        m, n = 2, 2  # 2 row qubits + 2 col qubits = 4 total
        total = m + n

        optcode, tensors, n_row, n_col = ParametricDFT.mera_code(m, n)
        @test n_row == 2 * (m - 1)  # 2*(2-1) = 2
        @test n_col == 2 * (n - 1)  # 2*(2-1) = 2
        n_gates = n_row + n_col
        @test n_gates == 4
        @test length(tensors) == total + n_gates  # 4 Hadamards + 4 phase gates
        @test length(tensors) > 0

        # Test that the circuit can be applied
        state = rand(ComplexF64, fill(2, total)...)
        result = optcode(tensors..., state)
        @test size(result) == size(state)
    end

    @testset "mera_code with custom phases" begin
        m, n = 2, 2
        total = m + n
        n_gates = 2 * (m - 1) + 2 * (n - 1)
        phases = [0.1, 0.2, 0.3, 0.4]

        optcode, tensors, n_row, n_col = ParametricDFT.mera_code(m, n; phases=phases)
        @test n_row + n_col == n_gates
        @test length(tensors) == total + n_gates

        state = rand(ComplexF64, fill(2, total)...)
        result = optcode(tensors..., state)
        @test size(result) == size(state)
    end

    @testset "gate count verification" begin
        # For m qubits (power of 2): n_row_gates = 2*(m-1)
        # Total tensors = (m+n) Hadamards + n_row_gates + n_col_gates
        for (m, n) in [(2, 2), (4, 4), (2, 4), (4, 2)]
            total = m + n
            n_row_expected = 2 * (m - 1)
            n_col_expected = 2 * (n - 1)
            _, tensors, n_row, n_col = ParametricDFT.mera_code(m, n)
            @test n_row == n_row_expected
            @test n_col == n_col_expected
            @test length(tensors) == total + n_row + n_col
        end
    end

    @testset "forward and inverse transforms" begin
        Random.seed!(42)
        m, n = 2, 2
        total = m + n
        n_gates = 2 * (m - 1) + 2 * (n - 1)
        phases = rand(n_gates) * 2π

        optcode, tensors, _, _ = ParametricDFT.mera_code(m, n; phases=phases)
        optcode_inv, _, _, _ = ParametricDFT.mera_code(m, n; phases=phases, inverse=true)

        state = rand(ComplexF64, fill(2, total)...)

        # Forward transform
        result = optcode(tensors..., state)

        # Inverse transform (conjugate tensors for inverse)
        reconstructed = optcode_inv(conj.(tensors)..., result)

        # Should recover original
        @test isapprox(reconstructed, state, rtol=1e-10)
    end

    @testset "norm preservation (unitarity)" begin
        Random.seed!(42)
        m, n = 4, 2
        total = m + n
        n_gates = 2 * (m - 1) + 2 * (n - 1)
        phases = rand(n_gates) * 2π

        optcode, tensors, _, _ = ParametricDFT.mera_code(m, n; phases=phases)

        state = rand(ComplexF64, fill(2, total)...)
        result = optcode(tensors..., state)

        @test isapprox(norm(result), norm(state), rtol=1e-10)
    end

    @testset "get_mera_gate_indices" begin
        m, n = 2, 2
        n_gates = 2 * (m - 1) + 2 * (n - 1)
        phases = [0.1, 0.2, 0.3, 0.4]
        _, tensors, _, _ = ParametricDFT.mera_code(m, n; phases=phases)

        indices = ParametricDFT.get_mera_gate_indices(tensors, n_gates)
        @test length(indices) == n_gates
    end

    @testset "extract_mera_phases" begin
        m, n = 2, 2
        n_gates = 2 * (m - 1) + 2 * (n - 1)
        original_phases = [0.1, 0.2, 0.3, 0.4]
        _, tensors, _, _ = ParametricDFT.mera_code(m, n; phases=original_phases)

        indices = ParametricDFT.get_mera_gate_indices(tensors, n_gates)
        extracted = ParametricDFT.extract_mera_phases(tensors, indices)

        @test length(extracted) == n_gates
        @test isapprox(extracted, original_phases, rtol=1e-10)
    end

    @testset "error handling" begin
        # Wrong phase length
        @test_throws AssertionError ParametricDFT.mera_code(2, 2; phases=[0.1, 0.2])
        # Non-power-of-2 qubits
        @test_throws AssertionError ParametricDFT.mera_code(3, 3)
    end

    @testset "all tensors are 2×2" begin
        m, n = 4, 2
        _, tensors, _, _ = ParametricDFT.mera_code(m, n)
        @test all(t -> size(t) == (2, 2), tensors)
    end

    @testset "Hadamard count" begin
        for (m, n) in [(2, 2), (4, 4), (2, 4)]
            total = m + n
            _, tensors, _, _ = ParametricDFT.mera_code(m, n)
            n_hadamards = count(t -> t ≈ mat(H), tensors)
            @test n_hadamards == total
        end
    end

    @testset "different phases produce different results" begin
        Random.seed!(42)
        m, n = 2, 2
        n_gates = 2 * (m - 1) + 2 * (n - 1)

        phases1 = zeros(n_gates)
        phases2 = fill(π/4, n_gates)

        optcode1, tensors1, _, _ = ParametricDFT.mera_code(m, n; phases=phases1)
        optcode2, tensors2, _, _ = ParametricDFT.mera_code(m, n; phases=phases2)

        total = m + n
        state = rand(ComplexF64, fill(2, total)...)
        result1 = optcode1(tensors1..., state)
        result2 = optcode2(tensors2..., state)

        @test !isapprox(result1, result2, rtol=1e-5)
    end

    @testset "minimum size m=1, n=1" begin
        # Single qubit per dimension — no MERA layers, just Hadamards
        m, n = 1, 1
        total = m + n
        optcode, tensors, n_row, n_col = ParametricDFT.mera_code(m, n)
        @test n_row == 0
        @test n_col == 0
        @test length(tensors) == total  # Just 2 Hadamards

        state = rand(ComplexF64, fill(2, total)...)
        result = optcode(tensors..., state)
        @test size(result) == size(state)
    end

    @testset "larger circuit 4×4" begin
        Random.seed!(42)
        m, n = 4, 4
        total = m + n
        n_gates = 2 * (m - 1) + 2 * (n - 1)  # 6 + 6 = 12
        phases = rand(n_gates) * 2π

        optcode, tensors, n_row, n_col = ParametricDFT.mera_code(m, n; phases=phases)
        @test n_row == 6
        @test n_col == 6
        @test length(tensors) == total + n_gates

        state = rand(ComplexF64, fill(2, total)...)
        result = optcode(tensors..., state)
        @test isapprox(norm(result), norm(state), rtol=1e-10)

        # Roundtrip
        optcode_inv, _, _, _ = ParametricDFT.mera_code(m, n; phases=phases, inverse=true)
        reconstructed = optcode_inv(conj.(tensors)..., result)
        @test isapprox(reconstructed, state, rtol=1e-10)
    end
end
```

**Step 2: Add include to `test/runtests.jl`**

Add after the `include("tebd_tests.jl")` line (line 42):
```julia
include("mera_tests.jl")
```

**Step 3: Wire up `src/mera.jl` in `src/ParametricDFT.jl`**

Add `include("mera.jl")` after the TEBD include (after line 67):
```julia
# 3c. MERA circuit (standalone circuit code)
include("mera.jl")
```

Add exports after the TEBD exports (after line 28):
```julia
# MERA circuit exports
export mera_code
export get_mera_gate_indices, extract_mera_phases
```

**Step 4: Run tests**

```bash
cd /tmp/ParametricDFT-work
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30
```

Expected: All MERA circuit tests pass. Fix any issues.

**Step 5: Commit**

```bash
git add src/mera.jl src/ParametricDFT.jl test/mera_tests.jl test/runtests.jl
git commit -m "feat: add MERA circuit construction with tests"
git fsck --connectivity-only
```

---

### Task 3: Add `MERABasis` struct and interface methods to `src/basis.jl`

**Files:**
- Modify: `src/basis.jl` (append after TEBDBasis section)
- Modify: `src/ParametricDFT.jl` (add MERABasis exports)

**Step 1: Add MERABasis to `src/basis.jl`**

Append the following after the TEBDBasis `Base.:(==)` definition (after line 778):

```julia

# ============================================================================
# MERA Basis Implementation
# ============================================================================

"""
    MERABasis <: AbstractSparseBasis

Multi-scale Entanglement Renormalization Ansatz (MERA) basis.

This basis uses a hierarchical multi-scale circuit structure with:
- Hadamard layer on all qubits
- k layers of disentanglers and isometries per dimension (k = log2(n_qubits))
- Each layer doubles the stride, capturing correlations at increasing scales
- MERA applied independently to row and column qubits (separable 2D)

Requires m and n (number of qubits) to be powers of 2 (or 1 for trivial dimension).

# Fields
- `m::Int`: Number of row qubits (row dimension = 2^m)
- `n::Int`: Number of column qubits (col dimension = 2^n)
- `tensors::Vector`: Circuit parameters (Hadamard + phase gate tensors)
- `optcode::AbstractEinsum`: Optimized einsum code for forward transform
- `inverse_code::AbstractEinsum`: Optimized einsum code for inverse transform
- `n_row_gates::Int`: Number of row MERA phase gates (= 2*(m-1), or 0 if m=1)
- `n_col_gates::Int`: Number of column MERA phase gates (= 2*(n-1), or 0 if n=1)
- `phases::Vector{Float64}`: Phase parameters for all MERA gates

# Example
```julia
# Create default MERA basis for 16×16 images (m=4, n=4)
basis = MERABasis(4, 4)

# Create with custom initial phases
n_gates = 2*(4-1) + 2*(4-1)  # 12 gates
phases = rand(n_gates) * 2π
basis = MERABasis(4, 4; phases=phases)

# Transform an image
freq = forward_transform(basis, image)
reconstructed = inverse_transform(basis, freq)
```
"""
struct MERABasis <: AbstractSparseBasis
    m::Int
    n::Int
    tensors::Vector
    optcode::OMEinsum.AbstractEinsum
    inverse_code::OMEinsum.AbstractEinsum
    n_row_gates::Int
    n_col_gates::Int
    phases::Vector{Float64}
end

"""
    MERABasis(m::Int, n::Int; phases=nothing)

Construct a MERABasis with default or custom phases.

# Arguments
- `m::Int`: Number of row qubits (must be power of 2 or 1)
- `n::Int`: Number of column qubits (must be power of 2 or 1)
- `phases::Union{Nothing, Vector{<:Real}}`: Initial phases for MERA gates.
  If nothing, defaults to zeros.

# Returns
- `MERABasis`: Basis with MERA circuit parameters
"""
function MERABasis(m::Int, n::Int; phases::Union{Nothing, Vector{<:Real}}=nothing)
    n_row = m >= 2 ? 2 * (m - 1) : 0
    n_col = n >= 2 ? 2 * (n - 1) : 0
    n_gates = n_row + n_col
    if phases === nothing
        phases = zeros(n_gates)
    end

    optcode, tensors, _, _ = mera_code(m, n; phases=phases)
    inverse_code, _, _, _ = mera_code(m, n; phases=phases, inverse=true)

    return MERABasis(m, n, tensors, optcode, inverse_code, n_row, n_col, Float64.(phases))
end

"""
    MERABasis(m::Int, n::Int, tensors::Vector, n_row_gates::Int, n_col_gates::Int)

Construct a MERABasis with custom trained tensors.

# Arguments
- `m::Int`: Number of row qubits
- `n::Int`: Number of column qubits
- `tensors::Vector`: Pre-trained circuit parameters
- `n_row_gates::Int`: Number of row MERA gates
- `n_col_gates::Int`: Number of column MERA gates

# Returns
- `MERABasis`: Basis with custom parameters
"""
function MERABasis(m::Int, n::Int, tensors::Vector, n_row_gates::Int, n_col_gates::Int)
    optcode, _, _, _ = mera_code(m, n)
    inverse_code, _, _, _ = mera_code(m, n; inverse=true)

    # Extract phases from tensors
    n_gates = n_row_gates + n_col_gates
    gate_indices = get_mera_gate_indices(tensors, n_gates)
    phases = extract_mera_phases(tensors, gate_indices)

    return MERABasis(m, n, tensors, optcode, inverse_code, n_row_gates, n_col_gates, phases)
end

# ============================================================================
# Interface Implementation for MERABasis
# ============================================================================

"""
    forward_transform(basis::MERABasis, data::AbstractVector)

Apply forward MERA transform to a vector.
"""
function forward_transform(basis::MERABasis, data::AbstractVector)
    total = basis.m + basis.n
    expected_size = 2^total
    @assert length(data) == expected_size "Data length must be 2^(m+n) = $(expected_size), got $(length(data))"

    data_complex = Complex{Float64}.(data)
    return vec(basis.optcode(basis.tensors..., reshape(data_complex, fill(2, total)...)))
end

"""
    forward_transform(basis::MERABasis, image::AbstractMatrix)

Apply forward MERA transform to an image.
"""
function forward_transform(basis::MERABasis, image::AbstractMatrix)
    m, n = basis.m, basis.n
    expected_size = (2^m, 2^n)
    @assert size(image) == expected_size "Image must be $(expected_size), got $(size(image))"

    total = m + n
    img_complex = Complex{Float64}.(vec(image))
    result = vec(basis.optcode(basis.tensors..., reshape(img_complex, fill(2, total)...)))
    return reshape(result, size(image))
end

"""
    inverse_transform(basis::MERABasis, freq_domain::AbstractVector)

Apply inverse MERA transform to a vector.
"""
function inverse_transform(basis::MERABasis, freq_domain::AbstractVector)
    total = basis.m + basis.n
    expected_size = 2^total
    @assert length(freq_domain) == expected_size "Frequency domain length must be 2^(m+n) = $(expected_size), got $(length(freq_domain))"

    return vec(basis.inverse_code(conj.(basis.tensors)..., reshape(freq_domain, fill(2, total)...)))
end

"""
    inverse_transform(basis::MERABasis, freq_domain::AbstractMatrix)

Apply inverse MERA transform to a matrix.
"""
function inverse_transform(basis::MERABasis, freq_domain::AbstractMatrix)
    m, n = basis.m, basis.n
    expected_size = (2^m, 2^n)
    @assert size(freq_domain) == expected_size "Frequency domain must be $(expected_size), got $(size(freq_domain))"

    total = m + n
    freq_vec = Complex{Float64}.(vec(freq_domain))
    result = vec(basis.inverse_code(conj.(basis.tensors)..., reshape(freq_vec, fill(2, total)...)))
    return reshape(result, size(freq_domain))
end

function image_size(basis::MERABasis)
    return (2^basis.m, 2^basis.n)
end

function num_parameters(basis::MERABasis)
    total = 0
    for tensor in basis.tensors
        total += length(tensor)
    end
    return total
end

function num_gates(basis::MERABasis)
    return basis.n_row_gates + basis.n_col_gates
end

function get_phases(basis::MERABasis)
    return copy(basis.phases)
end

function basis_hash(basis::MERABasis)
    data = IOBuffer()
    write(data, "MERABasis:m=$(basis.m):n=$(basis.n):n_row=$(basis.n_row_gates):n_col=$(basis.n_col_gates):")
    for tensor in basis.tensors
        for val in tensor
            write(data, "$(real(val)),$(imag(val));")
        end
    end
    return bytes2hex(sha256(take!(data)))
end

function Base.show(io::IO, basis::MERABasis)
    h, w = image_size(basis)
    params = num_parameters(basis)
    n_g = num_gates(basis)
    print(io, "MERABasis($(basis.m)×$(basis.n) qubits, $(h)×$(w) images, $params parameters, $n_g gates)")
end

function Base.:(==)(a::MERABasis, b::MERABasis)
    return a.m == b.m && a.n == b.n &&
           a.n_row_gates == b.n_row_gates && a.n_col_gates == b.n_col_gates &&
           all(a.tensors .≈ b.tensors)
end
```

**Step 2: Add MERABasis exports to `src/ParametricDFT.jl`**

In the "Sparse basis exports" line (line 30), add `MERABasis`:
```julia
export AbstractSparseBasis, QFTBasis, EntangledQFTBasis, TEBDBasis, MERABasis
```

**Step 3: Add MERABasis tests to `test/mera_tests.jl`**

Append to `test/mera_tests.jl`:

```julia
@testset "MERABasis" begin

    @testset "construction" begin
        basis = MERABasis(2, 2)
        @test basis.m == 2
        @test basis.n == 2
        @test basis.n_row_gates == 2
        @test basis.n_col_gates == 2
        @test length(basis.phases) == 4
    end

    @testset "construction with custom phases" begin
        phases = [0.1, 0.2, 0.3, 0.4]
        basis = MERABasis(2, 2; phases=phases)
        @test isapprox(basis.phases, phases, atol=1e-10)
    end

    @testset "image_size" begin
        basis = MERABasis(2, 4)
        @test image_size(basis) == (4, 16)  # 2^2 × 2^4
    end

    @testset "num_gates" begin
        basis = MERABasis(4, 4)
        @test num_gates(basis) == 12  # 2*(4-1) + 2*(4-1) = 12
    end

    @testset "forward and inverse transform matrix" begin
        Random.seed!(42)
        basis = MERABasis(2, 2)
        image = rand(4, 4)  # 2^2 × 2^2

        freq = forward_transform(basis, image)
        @test size(freq) == size(image)

        recovered = inverse_transform(basis, freq)
        @test isapprox(real.(recovered), image, rtol=1e-10)
    end

    @testset "forward and inverse transform vector" begin
        Random.seed!(42)
        basis = MERABasis(2, 2)
        data = rand(16)  # 2^(2+2) = 16

        freq = forward_transform(basis, data)
        @test length(freq) == length(data)

        recovered = inverse_transform(basis, freq)
        @test isapprox(real.(recovered), data, rtol=1e-10)
    end

    @testset "basis_hash deterministic" begin
        basis1 = MERABasis(2, 2)
        basis2 = MERABasis(2, 2)
        @test basis_hash(basis1) == basis_hash(basis2)
    end

    @testset "show" begin
        basis = MERABasis(2, 2)
        str = sprint(show, basis)
        @test occursin("MERABasis", str)
        @test occursin("4×4", str)
    end

    @testset "equality" begin
        basis1 = MERABasis(2, 2)
        basis2 = MERABasis(2, 2)
        @test basis1 == basis2
    end

    @testset "minimum size m=1, n=1" begin
        basis = MERABasis(1, 1)
        @test basis.n_row_gates == 0
        @test basis.n_col_gates == 0
        @test image_size(basis) == (2, 2)

        image = rand(2, 2)
        freq = forward_transform(basis, image)
        recovered = inverse_transform(basis, freq)
        @test isapprox(real.(recovered), image, rtol=1e-10)
    end

    @testset "asymmetric m=2, n=4" begin
        basis = MERABasis(2, 4)
        @test basis.n_row_gates == 2  # 2*(2-1)
        @test basis.n_col_gates == 6  # 2*(4-1)

        image = rand(4, 16)  # 2^2 × 2^4
        freq = forward_transform(basis, image)
        recovered = inverse_transform(basis, freq)
        @test isapprox(real.(recovered), image, rtol=1e-10)
    end
end
```

**Step 4: Run tests**

```bash
cd /tmp/ParametricDFT-work
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30
```

**Step 5: Commit**

```bash
git add src/basis.jl src/ParametricDFT.jl test/mera_tests.jl
git commit -m "feat: add MERABasis struct and interface methods"
git fsck --connectivity-only
```

---

### Task 4: Add training dispatch for MERABasis

**Files:**
- Modify: `src/training.jl` (add `train_basis(::Type{MERABasis}, ...)`)

**Step 1: Add `train_basis` method for MERABasis**

Add after the TEBDBasis `train_basis` function (after line 408 in training.jl):

```julia
"""
    train_basis(::Type{MERABasis}, dataset; m, n, phases, ...)

Train a MERABasis on images. Same kwargs as TEBDBasis plus MERA-specific defaults.
Requires m and n to be powers of 2 (or 1).
"""
function train_basis(
    ::Type{MERABasis},
    dataset::Vector{<:AbstractMatrix};
    m::Int, n::Int,
    phases::Union{Nothing, Vector{<:Real}} = nothing,
    loss::AbstractLoss = MSELoss(round(Int, 2^(m+n) * 0.1)),
    epochs::Int = 3,
    steps_per_image::Int = 200,
    validation_split::Float64 = 0.2,
    shuffle::Bool = true,
    early_stopping_patience::Int = 2,
    save_loss_path::Union{Nothing, String} = nothing,
    optimizer::Union{Symbol, AbstractRiemannianOptimizer} = :gradient_descent,
    batch_size::Int = 1,
    device::Symbol = :cpu,
    checkpoint_interval::Int = 0,
    checkpoint_dir::Union{Nothing, String} = nothing
)
    @assert 0.0 <= validation_split < 1.0 "validation_split must be in [0, 1)"
    @assert length(dataset) > 0 "Dataset must not be empty"
    expected_size = (2^m, 2^n)
    for (i, img) in enumerate(dataset)
        @assert size(img) == expected_size "Image $i has size $(size(img)), expected $expected_size"
    end

    n_row_gates = m >= 2 ? 2 * (m - 1) : 0
    n_col_gates = n >= 2 ? 2 * (n - 1) : 0
    n_gates = n_row_gates + n_col_gates

    # Initialize phases: use small random values if not provided
    if phases === nothing
        phases = randn(n_gates) * 0.1
    end

    # Initialize circuit
    optcode, initial_tensors, _, _ = mera_code(m, n; phases=phases)
    inverse_code, _, _, _ = mera_code(m, n; phases=phases, inverse=true)

    # Checkpoint callback
    build_fn = tensors -> begin
        gidx = get_mera_gate_indices(tensors, n_gates)
        p = !isempty(gidx) ? extract_mera_phases(tensors, gidx) :
            (phases === nothing ? zeros(n_gates) : Float64.(phases))
        MERABasis(m, n, tensors, optcode, inverse_code, n_row_gates, n_col_gates, p)
    end

    final_tensors, _, train_losses, val_losses, step_train_losses = _train_basis_core(
        dataset, optcode, inverse_code, initial_tensors, m, n, loss,
        epochs, steps_per_image, validation_split, shuffle,
        early_stopping_patience, "MERABasis";
        save_loss_path=save_loss_path, optimizer=optimizer,
        batch_size=batch_size, device=device,
        checkpoint_interval=checkpoint_interval, checkpoint_dir=checkpoint_dir,
        build_basis_fn=build_fn
    )

    # Extract trained phases
    gate_indices = get_mera_gate_indices(final_tensors, n_gates)
    trained_phases = if !isempty(gate_indices)
        extract_mera_phases(final_tensors, gate_indices)
    else
        phases === nothing ? zeros(n_gates) : Float64.(phases)
    end

    trained_basis = MERABasis(m, n, final_tensors, optcode, inverse_code, n_row_gates, n_col_gates, trained_phases)
    history = (train_losses=train_losses, val_losses=val_losses, step_train_losses=step_train_losses, basis_name="MERA")

    return trained_basis, history
end
```

**Step 2: Add training smoke test to `test/mera_tests.jl`**

Append:

```julia
@testset "MERABasis Training" begin
    @testset "training smoke test" begin
        Random.seed!(42)
        m, n = 2, 2
        images = [rand(4, 4) for _ in 1:3]

        basis, history = train_basis(MERABasis, images; m=m, n=n, epochs=1,
                                     steps_per_image=2, validation_split=0.0)
        @test basis isa MERABasis
        @test basis.m == m
        @test basis.n == n
        @test length(history.train_losses) > 0
    end
end
```

**Step 3: Run tests**

```bash
cd /tmp/ParametricDFT-work
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30
```

**Step 4: Commit**

```bash
git add src/training.jl test/mera_tests.jl
git commit -m "feat: add train_basis dispatch for MERABasis"
git fsck --connectivity-only
```

---

### Task 5: Add serialization support for MERABasis

**Files:**
- Modify: `src/serialization.jl`
- Modify: `test/mera_tests.jl`

**Step 1: Add `MERABasisJSON` struct and serialization**

In `src/serialization.jl`, add after the `TEBDBasisJSON` struct definition (after line 56):

```julia
"""
    MERABasisJSON

Internal struct for JSON serialization of MERABasis.
"""
struct MERABasisJSON
    type::String
    version::String
    m::Int
    n::Int
    n_row_gates::Int
    n_col_gates::Int
    phases::Vector{Float64}
    tensors::Vector{Vector{Vector{Float64}}}
    hash::String
end
```

Add StructType registration after line 61:
```julia
StructTypes.StructType(::Type{MERABasisJSON}) = StructTypes.Struct()
```

Add `_basis_to_json` for MERABasis after the TEBDBasis version (after line 176):

```julia
"""
    _basis_to_json(basis::MERABasis)

Convert a MERABasis to JSON-serializable format.
"""
function _basis_to_json(basis::MERABasis)
    serialized_tensors = Vector{Vector{Vector{Float64}}}()
    for tensor in basis.tensors
        tensor_data = Vector{Vector{Float64}}()
        for val in tensor
            push!(tensor_data, [real(val), imag(val)])
        end
        push!(serialized_tensors, tensor_data)
    end

    return MERABasisJSON(
        "MERABasis",
        "1.0",
        basis.m,
        basis.n,
        basis.n_row_gates,
        basis.n_col_gates,
        basis.phases,
        serialized_tensors,
        basis_hash(basis)
    )
end
```

In `load_basis`, add MERABasis case after the TEBDBasis branch (after line 222):
```julia
    elseif basis_type == "MERABasis"
        json_data = JSON3.read(json_str, MERABasisJSON)
        return _json_to_mera_basis(json_data)
```

Add `_json_to_mera_basis` after `_json_to_tebd_basis` (after line 364):

```julia
"""
    _json_to_mera_basis(json_data::MERABasisJSON)

Convert JSON data back to a MERABasis object.
"""
function _json_to_mera_basis(json_data::MERABasisJSON)
    if json_data.type != "MERABasis"
        error("Unknown basis type: $(json_data.type)")
    end

    m = json_data.m
    n = json_data.n
    n_row_gates = json_data.n_row_gates
    n_col_gates = json_data.n_col_gates
    phases = json_data.phases

    # Reconstruct tensors
    optcode, template_tensors, _, _ = mera_code(m, n; phases=phases)
    inverse_code, _, _, _ = mera_code(m, n; phases=phases, inverse=true)

    tensors = Vector{Any}()
    for (i, tensor_data) in enumerate(json_data.tensors)
        template_shape = size(template_tensors[i])
        complex_vals = [Complex{Float64}(pair[1], pair[2]) for pair in tensor_data]
        tensor = reshape(complex_vals, template_shape)
        push!(tensors, tensor)
    end

    loaded_basis = MERABasis(m, n, tensors, optcode, inverse_code, n_row_gates, n_col_gates, Float64.(phases))
    loaded_hash = basis_hash(loaded_basis)

    if loaded_hash != json_data.hash
        @warn "Basis hash mismatch. File hash: $(json_data.hash), computed hash: $(loaded_hash). The basis may have been corrupted."
    end

    return loaded_basis
end
```

Add `basis_to_dict` for MERABasis after the TEBDBasis version (after line 434):

```julia
"""
    basis_to_dict(basis::MERABasis) -> Dict

Convert a MERABasis to a dictionary for custom serialization.
"""
function basis_to_dict(basis::MERABasis)
    json_data = _basis_to_json(basis)
    return Dict(
        "type" => json_data.type,
        "version" => json_data.version,
        "m" => json_data.m,
        "n" => json_data.n,
        "n_row_gates" => json_data.n_row_gates,
        "n_col_gates" => json_data.n_col_gates,
        "phases" => json_data.phases,
        "tensors" => json_data.tensors,
        "hash" => json_data.hash
    )
end
```

In `dict_to_basis`, add MERABasis case after the TEBDBasis branch (after line 475):

```julia
    elseif basis_type == "MERABasis"
        json_data = MERABasisJSON(
            d["type"],
            d["version"],
            d["m"],
            d["n"],
            d["n_row_gates"],
            d["n_col_gates"],
            d["phases"],
            d["tensors"],
            d["hash"]
        )
        return _json_to_mera_basis(json_data)
```

**Step 2: Add serialization test to `test/mera_tests.jl`**

Append:

```julia
@testset "MERABasis Serialization" begin
    @testset "save and load roundtrip" begin
        basis = MERABasis(2, 2; phases=[0.1, 0.2, 0.3, 0.4])
        path = tempname() * ".json"
        try
            save_basis(path, basis)
            loaded = load_basis(path)
            @test loaded isa MERABasis
            @test loaded == basis
            @test basis_hash(loaded) == basis_hash(basis)
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "basis_to_dict and dict_to_basis" begin
        basis = MERABasis(2, 2; phases=[0.1, 0.2, 0.3, 0.4])
        d = basis_to_dict(basis)
        @test d["type"] == "MERABasis"
        @test d["m"] == 2
        @test d["n"] == 2

        loaded = dict_to_basis(d)
        @test loaded isa MERABasis
        @test loaded == basis
    end
end
```

**Step 3: Run tests**

```bash
cd /tmp/ParametricDFT-work
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30
```

**Step 4: Commit**

```bash
git add src/serialization.jl test/mera_tests.jl
git commit -m "feat: add MERABasis serialization support"
git fsck --connectivity-only
```

---

### Task 6: Final verification and cleanup

**Step 1: Run full test suite**

```bash
cd /tmp/ParametricDFT-work
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1
```

Expected: All tests pass including existing QFT, EntangledQFT, TEBD tests (no regressions).

**Step 2: Verify exports work**

```bash
cd /tmp/ParametricDFT-work
julia --project=. -e '
using ParametricDFT
# Verify all exports are accessible
basis = MERABasis(2, 2)
println(basis)
println("image_size: ", image_size(basis))
println("num_parameters: ", num_parameters(basis))
println("num_gates: ", num_gates(basis))
println("phases: ", get_phases(basis))

# Verify transform works
image = rand(4, 4)
freq = forward_transform(basis, image)
recovered = inverse_transform(basis, freq)
println("Roundtrip error: ", maximum(abs.(real.(recovered) .- image)))

# Verify mera_code export
optcode, tensors, nr, nc = mera_code(2, 2)
println("mera_code: n_row=$nr, n_col=$nc")
'
```

**Step 3: Commit final state if any cleanup was needed**

```bash
cd /tmp/ParametricDFT-work
git status
# If clean, no commit needed
# If changes, commit with descriptive message
```
