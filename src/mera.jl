# ============================================================================
# MERA (Multi-scale Entanglement Renormalization Ansatz) Circuit Construction
# ============================================================================
# This file provides the MERA circuit construction with 2D separable topology.
# The circuit has two layers:
#   1. Hadamard layer on all m+n qubits (creates frequency/superposition basis)
#   2. Two separate MERA structures of controlled-phase gates:
#      - Row MERA: hierarchical disentangler + isometry layers on row qubits
#      - Column MERA: hierarchical disentangler + isometry layers on col qubits
# Each dimension requires power-of-2 qubits (or 1 for no MERA in that dimension).

"""
    _n_mera_gates(n_qubits::Int)

Calculate the number of phase gates for one dimension of a MERA circuit.

# Arguments
- `n_qubits::Int`: Number of qubits in this dimension (must be >= 2)

# Returns
- `Int`: Number of phase gates = 2 * (n_qubits - 1)
"""
_n_mera_gates(n_qubits::Int) = 2 * (n_qubits - 1)

"""
    _mera_single_dim(n_qubits::Int, qubit_offset::Int, total_qubits::Int, phases::Vector{<:Real})

Build MERA layers (disentanglers + isometries) for one dimension.

The MERA structure has k = log2(n_qubits) layers. Each layer l has stride s = 2^(l-1)
and n_pairs = n_qubits / (2*s) pairs of gates.

# Arguments
- `n_qubits::Int`: Number of qubits in this dimension (must be power of 2 and >= 2)
- `qubit_offset::Int`: Offset to add to qubit indices (0-based)
- `total_qubits::Int`: Total number of qubits in the full circuit
- `phases::Vector{<:Real}`: Phase parameters for all gates in this dimension

# Returns
- `ChainBlock`: Yao chain block containing all MERA gates for this dimension
"""
function _mera_single_dim(n_qubits::Int, qubit_offset::Int, total_qubits::Int, phases::Vector{<:Real})
    @assert ispow2(n_qubits) "n_qubits must be a power of 2, got $n_qubits"
    @assert n_qubits >= 2 "n_qubits must be >= 2, got $n_qubits"
    @assert length(phases) == _n_mera_gates(n_qubits) "phases must have length $(_n_mera_gates(n_qubits)), got $(length(phases))"

    k = Int(log2(n_qubits))
    qc = chain(total_qubits)
    phase_idx = 1

    for l in 1:k
        s = 2^(l - 1)
        n_pairs = n_qubits ÷ (2 * s)

        # Disentanglers
        for p in 0:(n_pairs - 1)
            q1 = 2 * p * s + 2
            q2 = mod1(2 * p * s + s + 2, n_qubits)
            push!(qc, control(total_qubits, q1 + qubit_offset, (q2 + qubit_offset) => shift(phases[phase_idx])))
            phase_idx += 1
        end

        # Isometries
        for p in 0:(n_pairs - 1)
            q1 = 2 * p * s + 1
            q2 = 2 * p * s + s + 1
            push!(qc, control(total_qubits, q1 + qubit_offset, (q2 + qubit_offset) => shift(phases[phase_idx])))
            phase_idx += 1
        end
    end

    return qc
end

"""
    mera_code(m::Int, n::Int; phases=nothing, inverse=false)

Generate an optimized tensor network representation of a 2D MERA circuit.

The MERA circuit consists of:
1. Hadamard layer: H gates on all m+n qubits (creates frequency basis)
2. Row MERA: hierarchical disentangler + isometry layers on row qubits 1:m
3. Column MERA: hierarchical disentangler + isometry layers on col qubits m+1:m+n

# Arguments
- `m::Int`: Number of row qubits (must be power of 2 if >= 2, or 1)
- `n::Int`: Number of column qubits (must be power of 2 if >= 2, or 1)
- `phases::Union{Nothing, Vector{<:Real}}`: Initial phases for MERA gates.
  If nothing, defaults to zeros. Length must equal n_row_gates + n_col_gates.
- `inverse::Bool`: If true, generate inverse transform code

# Returns
- `optcode::AbstractEinsum`: Optimized einsum contraction code
- `tensors::Vector`: Circuit parameters (Hadamard gates + MERA gate tensors)
- `n_row_gates::Int`: Number of row phase gates
- `n_col_gates::Int`: Number of column phase gates

# Example
```julia
# Create 2D MERA circuit for 4×4 (m=4 row qubits, n=4 col qubits)
optcode, tensors, n_row, n_col = mera_code(4, 4)

# Create with custom initial phases (6+6=12 gates total)
phases = rand(12) * 2π
optcode, tensors, n_row, n_col = mera_code(4, 4; phases=phases)
```
"""
function mera_code(m::Int, n::Int; phases::Union{Nothing, Vector{<:Real}}=nothing, inverse=false)
    total = m + n
    n_row_gates = m >= 2 ? _n_mera_gates(m) : 0
    n_col_gates = n >= 2 ? _n_mera_gates(n) : 0
    n_gates = n_row_gates + n_col_gates

    # Validate power-of-2 constraints
    if m >= 2
        @assert ispow2(m) "m must be a power of 2 when >= 2, got $m"
    end
    if n >= 2
        @assert ispow2(n) "n must be a power of 2 when >= 2, got $n"
    end

    # Default phases to zeros if not provided
    if phases === nothing
        phases = zeros(n_gates)
    end
    @assert length(phases) == n_gates "phases must have length $(n_gates) for $(m)×$(n) MERA ($(n_row_gates) row + $(n_col_gates) col gates)"

    # Build MERA circuit: Hadamard layer + MERA structures
    qc = chain(total)

    # Layer 1: Hadamard gates on all qubits (creates frequency basis)
    for i in 1:total
        push!(qc, put(total, i => H))
    end

    # Layer 2a: Row MERA on qubits 1:m
    if m >= 2
        row_phases = phases[1:n_row_gates]
        row_mera = _mera_single_dim(m, 0, total, row_phases)
        for gate in subblocks(row_mera)
            push!(qc, gate)
        end
    end

    # Layer 2b: Column MERA on qubits m+1:m+n
    if n >= 2
        col_phases = phases[n_row_gates+1:end]
        col_mera = _mera_single_dim(n, m, total, col_phases)
        for gate in subblocks(col_mera)
            push!(qc, gate)
        end
    end

    # Convert to tensor network
    tn = yao2einsum(qc; optimizer=nothing)

    # Reorder tensors: Hadamard gates first, then controlled-phase gates
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

Identify which tensors correspond to MERA gates.

# Arguments
- `tensors::Vector`: Circuit tensors
- `n_gates::Int`: Number of MERA gates

# Returns
- `Vector{Int}`: Indices of MERA gate tensors
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
- `gate_indices::Vector{Int}`: Indices of MERA gates

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
