# ============================================================================
# TEBD (Time-Evolving Block Decimation) Circuit Construction
# ============================================================================
# This file provides the TEBD circuit construction with ring topology.
# Gates connect: (1,n), (1,2), (2,3), (3,4), ..., (n-1,n), (n,1)
# This creates a periodic boundary condition (ring) structure.

"""
    tebd_gate(theta::Real)

Create a parametric 2-qubit TEBD gate tensor.

The gate is a controlled-phase gate with the form:
    E = diag(1, 1, 1, e^(i*theta))

In tensor network form (2×2 matrix):
    E_tensor = [1 0; 0 e^(i*theta)]

# Arguments
- `theta::Real`: Phase parameter (in radians)

# Returns
- `Matrix{ComplexF64}`: 2×2 matrix in tensor network form
"""
function tebd_gate(theta::Real)
    return ComplexF64[1 0; 0 exp(im * theta)]
end

"""
    tebd_code(n::Int; phases=nothing, inverse=false)

Generate an optimized tensor network representation of a TEBD circuit with ring topology.

The TEBD circuit applies 2-qubit gates in a ring pattern:
- Gates 1 to n-1: connects (1,2), (2,3), ..., (n-1,n)
- Gate n: connects qubit n and qubit 1 (wrap-around to close the ring)

This creates a ring topology with periodic boundary conditions using exactly n gates.

# Arguments
- `n::Int`: Number of qubits
- `phases::Union{Nothing, Vector{<:Real}}`: Initial phases for TEBD gates.
  If nothing, defaults to zeros. Length must equal n (one for each edge in the ring).
- `inverse::Bool`: If true, generate inverse transform code

# Returns
- `optcode::AbstractEinsum`: Optimized einsum contraction code
- `tensors::Vector`: Circuit parameters (TEBD gate tensors)
- `n_gates::Int`: Number of TEBD gates

# Example
```julia
# Create TEBD circuit for 4 qubits with default (zero) phases
optcode, tensors, n_gates = tebd_code(4)

# Create with custom initial phases
phases = rand(4) * 2π  # 4 gates for 4 qubits (ring topology)
optcode, tensors, n_gates = tebd_code(4; phases=phases)
```
"""
function tebd_code(n::Int; phases::Union{Nothing, Vector{<:Real}}=nothing, inverse=false)
    # Number of gates in ring: n edges (1-2, 2-3, ..., n-1-n, n-1)
    n_gates = n
    
    # Default phases to zeros if not provided
    if phases === nothing
        phases = zeros(n_gates)
    end
    @assert length(phases) == n_gates "phases must have length n = $n_gates for ring topology"
    
    # Build TEBD circuit with ring topology
    qc = chain(n)
    
    # Gates 1 to n-1: Sequential nearest-neighbor connections (1-2, 2-3, ..., n-1-n)
    for i in 1:(n-1)
        push!(qc, control(n, i, i+1 => shift(phases[i])))
    end
    
    # Gate n: Last qubit to First qubit (n-1 wrap-around to close the ring)
    push!(qc, control(n, n, 1 => shift(phases[n])))
    
    # Convert to tensor network
    tn = yao2einsum(qc; optimizer=nothing)
    
    # Get tensors and indices
    ixs = tn.code.ixs
    tensors = tn.tensors
    
    if inverse
        code_reorder = DynamicEinCode([ixs..., tn.code.iy[n+1:end]], tn.code.iy[1:n])
    else
        code_reorder = DynamicEinCode([ixs..., tn.code.iy[1:n]], tn.code.iy[n+1:end])
    end
    optcode = optimize_code(code_reorder, uniformsize(tn.code, 2), TreeSA())
    
    return optcode, tensors, n_gates
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
    TEBDCircuitSpec

Specification for a TEBD circuit to be visualized.
"""
struct TEBDCircuitSpec
    n_qubits::Int
    phases::Vector{Float64}
    title::String
end

# Convenience constructor
function TEBDCircuitSpec(n::Int; phases=nothing, title="TEBD Circuit")
    if phases === nothing
        phases = zeros(n)
    end
    TEBDCircuitSpec(n, Float64.(phases), title)
end
