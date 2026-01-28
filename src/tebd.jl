# ============================================================================
# TEBD (Time-Evolving Block Decimation) Circuit Construction
# ============================================================================
# This file provides the TEBD circuit construction with 2D ring topology.
# The circuit has two layers:
#   1. Hadamard layer on all m+n qubits (creates frequency/superposition basis)
#   2. Two separate rings of controlled-phase gates:
#      - Row ring: (x1,x2), (x2,x3), ..., (x_{m-1},x_m), (x_m,x1) for m gates
#      - Column ring: (y1,y2), (y2,y3), ..., (y_{n-1},y_n), (y_n,y1) for n gates
# This creates a 2D separable transform with periodic boundary conditions, suitable for image processing.

"""
    tebd_code(m::Int, n::Int; phases=nothing, inverse=false)

Generate an optimized tensor network representation of a 2D TEBD circuit with ring topology.

The TEBD circuit consists of:
1. Hadamard layer: H gates on all m+n qubits (creates frequency basis)
2. Two separate rings of controlled-phase gates:
   - Row ring: (1,2), (2,3), ..., (m-1,m), (m,1) for m gates on row qubits
   - Column ring: (m+1,m+2), (m+2,m+3), ..., (m+n-1,m+n), (m+n,m+1) for n gates on column qubits

This creates a 2D separable transform with periodic boundary conditions (ring topology).

# Arguments
- `m::Int`: Number of row qubits (row dimension = 2^m)
- `n::Int`: Number of column qubits (column dimension = 2^n)
- `phases::Union{Nothing, Vector{<:Real}}`: Initial phases for TEBD gates.
  If nothing, defaults to zeros. Length must equal m+n for ring topology.
- `inverse::Bool`: If true, generate inverse transform code

# Returns
- `optcode::AbstractEinsum`: Optimized einsum contraction code
- `tensors::Vector`: Circuit parameters (Hadamard gates + TEBD gate tensors)
- `n_row_gates::Int`: Number of row ring phase gates (= m)
- `n_col_gates::Int`: Number of column ring phase gates (= n)

# Example
```julia
# Create 2D TEBD circuit for 3×3 (m=3 row qubits, n=3 col qubits)
optcode, tensors, n_row, n_col = tebd_code(3, 3)

# Create with custom initial phases (6 gates total: 3 row ring + 3 col ring)
phases = rand(6) * 2π
optcode, tensors, n_row, n_col = tebd_code(3, 3; phases=phases)
```
"""
function tebd_code(m::Int, n::Int; phases::Union{Nothing, Vector{<:Real}}=nothing, inverse=false)
    total = m + n
    n_row_gates = m  # Row ring: (1,2), ..., (m-1,m), (m,1)
    n_col_gates = n  # Col ring: (m+1,m+2), ..., (m+n-1,m+n), (m+n,m+1)
    n_gates = n_row_gates + n_col_gates
    
    # Default phases to zeros if not provided
    if phases === nothing
        phases = zeros(n_gates)
    end
    @assert length(phases) == n_gates "phases must have length $(n_gates) for $(m)×$(n) ring topology ($(n_row_gates) row + $(n_col_gates) col gates)"
    
    # Build TEBD circuit: Hadamard layer + two rings of controlled-phase gates
    qc = chain(total)
    
    # Layer 1: Hadamard gates on all qubits (creates frequency basis)
    for i in 1:total
        push!(qc, put(total, i => H))
    end
    
    gate_idx = 1
    
    # Layer 2a: Row ring - controlled-phase gates on row qubits (x1 to xm)
    # Gates: (1,2), (2,3), ..., (m-1,m)
    for i in 1:(m-1)
        push!(qc, control(total, i, i+1 => shift(phases[gate_idx])))
        gate_idx += 1
    end
    # Wrap-around gate: (m,1) to close the row ring
    push!(qc, control(total, m, 1 => shift(phases[gate_idx])))
    gate_idx += 1
    
    # Layer 2b: Column ring - controlled-phase gates on column qubits (y1 to yn)
    # Gates: (m+1,m+2), (m+2,m+3), ..., (m+n-1,m+n)
    for i in 1:(n-1)
        push!(qc, control(total, m+i, m+i+1 => shift(phases[gate_idx])))
        gate_idx += 1
    end
    # Wrap-around gate: (m+n,m+1) to close the column ring
    push!(qc, control(total, m+n, m+1 => shift(phases[gate_idx])))
    gate_idx += 1
    
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
    get_tebd_gate_indices(tensors, n_gates::Int)

Identify which tensors correspond to TEBD gates.

# Arguments
- `tensors::Vector`: Circuit tensors
- `n_gates::Int`: Number of TEBD gates

# Returns
- `Vector{Int}`: Indices of TEBD gate tensors
"""
function get_tebd_gate_indices(tensors, n_gates::Int)
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
    extract_tebd_phases(tensors, gate_indices::Vector{Int})

Extract the phase parameters from TEBD gate tensors.

# Arguments
- `tensors::Vector`: Circuit tensors
- `gate_indices::Vector{Int}`: Indices of TEBD gates

# Returns
- `Vector{Float64}`: Phase parameters for each TEBD gate
"""
function extract_tebd_phases(tensors, gate_indices::Vector{Int})
    phases = Float64[]
    for idx in gate_indices
        push!(phases, angle(tensors[idx][2, 2]))
    end
    return phases
end

"""
    n_row_gates(m::Int)

Calculate number of row ring phase gates for m row qubits.
"""
n_row_gates(m::Int) = m

"""
    n_col_gates(n::Int)

Calculate number of column ring phase gates for n column qubits.
"""
n_col_gates(n::Int) = n

"""
    n_total_gates(m::Int, n::Int)

Calculate total number of phase gates for m×n TEBD circuit.
"""
n_total_gates(m::Int, n::Int) = n_row_gates(m) + n_col_gates(n)
