# ============================================================================
# Entangled QFT Circuit Construction
# ============================================================================
# This file implements the entangled QFT circuit that adds XY correlation gates
# between corresponding row and column qubits. The entanglement gates E_k have
# the same form as the M gates in standard QFT but with learnable phase parameters.

"""
    entanglement_gate(phi::Real)

Create the tensor network representation of a 2-qubit controlled-phase gate E
with learnable phase phi.

**Full gate form (4×4 matrix in computational basis):**
    E = diag(1, 1, 1, e^(i*phi))

This applies phase e^(i*phi) only when both qubits are in state |1⟩.

**Tensor network form (2×2 matrix):**
    E_tensor = [1  0; 0  e^(i*phi)]

In the einsum tensor network decomposition, controlled-phase gates are
represented as 2×2 matrices acting on the bond indices connecting the
control and target qubits. This function returns the tensor form, not
the full 4×4 gate matrix.

This gate has the same structure as the M gate in the QFT circuit, but with
a learnable phase parameter instead of the fixed QFT phases (2π/2^k).

# Arguments
- `phi::Real`: Phase parameter (in radians)

# Returns
- `Matrix{ComplexF64}`: 2×2 matrix in tensor network form
"""
function entanglement_gate(phi::Real)
    # Tensor network form of controlled-phase gate
    # Note: This is NOT the full 4×4 gate matrix, but the tensor decomposition
    # used in einsum contractions. The full gate diag(1,1,1,e^(iφ)) decomposes
    # to this 2×2 form when contracted over the qubit indices.
    return ComplexF64[1 0; 0 exp(im * phi)]
end

"""
    _build_manual_qft(n_qubits, qubit_offset, total_qubits)

Build a manual QFT gate chain (without using EasyBuild.qft_circuit) that returns
individual gate operations. This is needed for the `:middle` entangle_position mode
where entanglement gates are interleaved with QFT Hadamard gates.

The standard QFT for n qubits consists of:
- For qubit j = 1, ..., n:
  - H(j)
  - For target k = j+1, ..., n: ctrl(k → j, phase=2π/2^(k-j+1))

# Arguments
- `n_qubits::Int`: Number of qubits in this QFT block
- `qubit_offset::Int`: Offset to add to qubit indices (0 for row, m for col)
- `total_qubits::Int`: Total number of qubits in the full circuit

# Returns
- `Vector{AbstractBlock}`: Individual gate operations in order
"""
function _build_manual_qft(n_qubits::Int, qubit_offset::Int, total_qubits::Int)
    gates = AbstractBlock[]
    for j in 1:n_qubits
        q = qubit_offset + j
        # Hadamard on qubit j
        push!(gates, put(total_qubits, q => H))
        # Controlled phase gates
        for target in (j+1):n_qubits
            k = target - j + 1
            t = qubit_offset + target
            push!(gates, control(total_qubits, t, q => shift(2π / 2^k)))
        end
    end
    return gates
end

"""
    entangled_qft_code(m::Int, n::Int; entangle_phases=nothing, inverse=false, entangle_position=:back)

Generate an optimized tensor network representation of the entangled QFT circuit.

The entangled QFT extends the standard 2D QFT by adding entanglement gates E_k
between corresponding row and column qubits. Each entanglement gate
E_k is a controlled-phase gate with the same structure as the M gate in QFT:

**Full gate form (4×4 matrix):**
    E_k = diag(1, 1, 1, e^(i*phi_k))

**Tensor network form (2×2 matrix):**
    E_k_tensor = [1 0; 0 e^(i*phi_k)]

The gate acts on qubits (x_{n-k}, y_{n-k}), where phi_k is a learnable phase
parameter. In the tensor network, these are represented as 2×2 matrices.

For a square 2^n × 2^n image (m = n), we add exactly n entanglement gates,
one for each pair of corresponding row/column qubits.

# Arguments
- `m::Int`: Number of qubits for row indices (image height = 2^m)
- `n::Int`: Number of qubits for column indices (image width = 2^n)
- `entangle_phases::Union{Nothing, Vector{<:Real}}`: Initial phases for entanglement gates.
  If nothing, defaults to zeros (equivalent to standard QFT). Length must equal min(m, n).
- `inverse::Bool`: If true, generate inverse transform code
- `entangle_position::Symbol`: Where to place entanglement gates. One of:
  - `:back` (default): QFT_row ⊗ QFT_col → Entangle
  - `:front`: Entangle → QFT_row ⊗ QFT_col
  - `:middle`: Row and column QFT interleaved, with E_k placed after the row H(j)
    but BEFORE the column H(j). This produces a distinct result because E_k
    (a diagonal controlled-phase gate) does not commute with Hadamard gates.

# Returns
- `optcode::AbstractEinsum`: Optimized einsum contraction code
- `tensors::Vector`: Circuit parameters (unitary matrices + entanglement gates)
- `n_entangle::Int`: Number of entanglement gates added

# Example
```julia
# Create entangled QFT for 64×64 images with default (zero) phases
optcode, tensors, n_entangle = entangled_qft_code(6, 6)

# Create with custom initial phases
phases = rand(6) * 2π
optcode, tensors, n_entangle = entangled_qft_code(6, 6; entangle_phases=phases)

# Create with entanglement at front
optcode, tensors, n_entangle = entangled_qft_code(6, 6; entangle_position=:front)

# Create with entanglement interleaved in middle
optcode, tensors, n_entangle = entangled_qft_code(6, 6; entangle_position=:middle)
```
"""
function entangled_qft_code(m::Int, n::Int; entangle_phases::Union{Nothing, Vector{<:Real}}=nothing, inverse=false, entangle_position::Symbol=:back)
    @assert entangle_position in (:front, :middle, :back) "entangle_position must be :front, :middle, or :back, got :$entangle_position"

    # Number of entanglement gates = min(m, n) for one-to-one coupling
    n_entangle = min(m, n)

    # Default phases to zeros if not provided
    if entangle_phases === nothing
        entangle_phases = zeros(n_entangle)
    end
    @assert length(entangle_phases) == n_entangle "entangle_phases must have length min(m, n) = $n_entangle"

    total = m + n

    if entangle_position == :middle
        # Build circuit with row and column QFT interleaved together.
        # E_k is placed right after the row H gate but BEFORE the column H gate.
        # This produces a mathematically distinct result because E_k (a diagonal gate)
        # does not commute with the Hadamard gate on its column qubit.
        #
        # For each step j = 1 to max(m, n):
        #   1. Apply row H(j) (if j <= m)
        #   2. Apply E_k if its row qubit just got H (k = m-j+1)
        #   3. Apply row M gates for qubit j (if j <= m)
        #   4. Apply col H(m+j) (if j <= n)
        #   5. Apply col M gates for qubit j (if j <= n)
        #
        # E_k connects row qubit (m-k+1) with col qubit (m+n-k+1).
        # E_k is applied at step j = m-k+1, right after row H(j).

        qc = chain(total)

        for j in 1:max(m, n)
            # Step 1: Row H(j) (if j <= m)
            if j <= m
                push!(qc, put(total, j => H))
            end

            # Step 2: Apply E_k right after row H(j), before col H
            # E_k's row qubit is (m-k+1) = j, so k = m-j+1
            if j <= m
                k = m - j + 1
                if k >= 1 && k <= n_entangle
                    x_qubit = m - k + 1  # = j
                    y_qubit = m + n - k + 1
                    push!(qc, control(total, x_qubit, y_qubit => shift(entangle_phases[k])))
                end
            end

            # Step 3: Row M gates for qubit j (if j <= m)
            if j <= m
                for target in (j+1):m
                    k_phase = target - j + 1
                    push!(qc, control(total, target, j => shift(2π / 2^k_phase)))
                end
            end

            # Step 4: Col H(m+j) (if j <= n)
            if j <= n
                push!(qc, put(total, m + j => H))
            end

            # Step 5: Col M gates for qubit j (if j <= n)
            if j <= n
                for target in (j+1):n
                    k_phase = target - j + 1
                    push!(qc, control(total, m + target, m + j => shift(2π / 2^k_phase)))
                end
            end
        end
    else
        # :front or :back — use EasyBuild QFT and chain in appropriate order
        qc1 = Yao.EasyBuild.qft_circuit(m)  # QFT on row qubits (1:m)
        qc2 = Yao.EasyBuild.qft_circuit(n)  # QFT on column qubits (m+1:m+n)

        # Build entanglement layer
        entangle_gates = chain(total)
        for k in 1:n_entangle
            x_qubit = m - k + 1
            y_qubit = m + n - k + 1
            push!(entangle_gates, control(total, x_qubit, y_qubit => shift(entangle_phases[k])))
        end

        if entangle_position == :front
            # Entangle → QFT_row ⊗ QFT_col
            qc = chain(
                entangle_gates,
                subroutine(total, qc1, 1:m),
                subroutine(total, qc2, m+1:m+n)
            )
        else  # :back
            # QFT_row ⊗ QFT_col → Entangle (original default)
            qc = chain(
                subroutine(total, qc1, 1:m),
                subroutine(total, qc2, m+1:m+n),
                entangle_gates
            )
        end
    end

    # Convert to tensor network
    tn = yao2einsum(qc; optimizer=nothing)

    # Reorder tensors: Hadamard gates first, then M gates, then entanglement gates
    # Identify tensor types by their values
    is_hadamard(x) = x ≈ mat(H)

    # Sort: Hadamard first, then others (M gates and entanglement gates)
    perm_vec = sortperm(tn.tensors, by=x -> !is_hadamard(x))
    ixs = tn.code.ixs[perm_vec]
    tensors = tn.tensors[perm_vec]

    if inverse
        code_reorder = DynamicEinCode([ixs..., tn.code.iy[total+1:end]], tn.code.iy[1:total])
    else
        code_reorder = DynamicEinCode([ixs..., tn.code.iy[1:total]], tn.code.iy[total+1:end])
    end
    optcode = optimize_code_cached(code_reorder, uniformsize(tn.code, 2), TreeSA())

    return optcode, tensors, n_entangle
end

"""
    get_entangle_tensor_indices(tensors, n_entangle::Int)

Identify which tensors in the circuit correspond to entanglement gates.
Returns the indices of the entanglement gate tensors (the last n_entangle controlled-phase gates).

In the tensor network representation from yao2einsum, controlled-phase gates
have the form [1, 1; 1, e^(i*phi)] (not diagonal). The entanglement gates
are the last n_entangle such tensors after sorting (Hadamards first, then phase gates).

After training, tensors may drift slightly from the exact pattern, so we use
tolerance-based magnitude checks.

# Arguments
- `tensors::Vector`: Circuit tensors
- `n_entangle::Int`: Number of entanglement gates

# Returns
- `Vector{Int}`: Indices of entanglement gate tensors
"""
function get_entangle_tensor_indices(tensors, n_entangle::Int)
    # Controlled-phase gates in tensor network form have pattern [1, 1; 1, e^(i*phi)]
    # After training, complex values may drift slightly, so check magnitudes with tolerance.
    # We use a moderate tolerance (0.15) to account for optimization drift while still
    # distinguishing from Hadamard-like gates (which have elements ≈ 0.707).
    function is_ctrl_phase(x)
        size(x) != (2, 2) && return false
        tol = 0.15
        return isapprox(abs(x[1,1]), 1, atol=tol) &&
               isapprox(abs(x[1,2]), 1, atol=tol) &&
               isapprox(abs(x[2,1]), 1, atol=tol) &&
               isapprox(abs(x[2,2]), 1, atol=tol)
    end

    ctrl_phase_indices = findall(is_ctrl_phase, tensors)
    # The last n_entangle controlled-phase tensors are the entanglement gates
    if length(ctrl_phase_indices) >= n_entangle
        return ctrl_phase_indices[end-n_entangle+1:end]
    else
        return Int[]
    end
end

"""
    extract_entangle_phases(tensors, entangle_indices::Vector{Int})

Extract the phase parameters from entanglement gate tensors.

# Arguments
- `tensors::Vector`: Circuit tensors
- `entangle_indices::Vector{Int}`: Indices of entanglement gates

# Returns
- `Vector{Float64}`: Phase parameters phi_k for each entanglement gate
"""
function extract_entangle_phases(tensors, entangle_indices::Vector{Int})
    phases = Float64[]
    for idx in entangle_indices
        # Phase is arg(tensor[2,2])
        push!(phases, angle(tensors[idx][2, 2]))
    end
    return phases
end
