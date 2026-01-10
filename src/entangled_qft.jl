# ============================================================================
# Entangled QFT Circuit Construction
# ============================================================================
# This file implements the entangled QFT basis that adds XY correlation gates
# between corresponding row and column qubits. The entanglement gates E_k have
# the same form as the M gates in standard QFT but with learnable phase parameters.

"""
    entanglement_gate(phi::Real)

Create a 2-qubit entanglement gate matrix E with learnable phase phi.
The gate has the form:
    E = diag(1, 1, 1, e^(i*phi))

This is a controlled-phase gate that multiplies a phase e^(i*phi) when both
qubits are in state |1⟩. It has the same structure as the M gate in the QFT
circuit, but with a learnable phase parameter instead of the fixed QFT phases.

# Arguments
- `phi::Real`: Phase parameter (in radians)

# Returns
- `Matrix{ComplexF64}`: 2×2 diagonal matrix representing the gate in tensor form
"""
function entanglement_gate(phi::Real)
    # In tensor network form, this is a diagonal 2×2 matrix
    # representing the controlled-phase gate action on the |11⟩ component
    return ComplexF64[1 0; 0 exp(im * phi)]
end

"""
    entangled_qft_code(m::Int, n::Int; entangle_phases=nothing, inverse=false)

Generate an optimized tensor network representation of the entangled QFT circuit.

The entangled QFT extends the standard 2D QFT by adding entanglement gates E_k
between corresponding row and column qubits (x_k, y_k). Each entanglement gate
E_k has the same form as the M gate in the standard QFT:

    E_k = diag(1, 1, 1, e^(i*phi_k))

acting on qubits (x_{n-k}, y_{n-k}), where phi_k is a learnable phase parameter.

For a square 2^n × 2^n image (m = n), we add exactly n entanglement gates,
one for each pair of corresponding row/column qubits.

# Arguments
- `m::Int`: Number of qubits for row indices (image height = 2^m)
- `n::Int`: Number of qubits for column indices (image width = 2^n)
- `entangle_phases::Union{Nothing, Vector{<:Real}}`: Initial phases for entanglement gates.
  If nothing, defaults to zeros (equivalent to standard QFT). Length must equal min(m, n).
- `inverse::Bool`: If true, generate inverse transform code

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
```
"""
function entangled_qft_code(m::Int, n::Int; entangle_phases::Union{Nothing, Vector{<:Real}}=nothing, inverse=false)
    # Number of entanglement gates = min(m, n) for one-to-one coupling
    n_entangle = min(m, n)
    
    # Default phases to zeros if not provided
    if entangle_phases === nothing
        entangle_phases = zeros(n_entangle)
    end
    @assert length(entangle_phases) == n_entangle "entangle_phases must have length min(m, n) = $n_entangle"
    
    # Build standard QFT circuits for row and column qubits
    qc1 = Yao.EasyBuild.qft_circuit(m)  # QFT on row qubits (1:m)
    qc2 = Yao.EasyBuild.qft_circuit(n)  # QFT on column qubits (m+1:m+n)
    
    # Build entanglement layer: E_k connects x_{n-k} with y_{n-k}
    # In our qubit ordering: row qubits are 1:m, column qubits are m+1:m+n
    # E_k connects qubit (m - k + 1) with qubit (m + n - k + 1) for k = 1, ..., n_entangle
    entangle_gates = chain(m + n)
    for k in 1:n_entangle
        # x_{n-k} corresponds to row qubit index (m - k + 1)
        # y_{n-k} corresponds to col qubit index (m + n - k + 1)
        x_qubit = m - k + 1
        y_qubit = m + n - k + 1
        # Controlled-phase gate: control on x_qubit, shift(phi) on y_qubit
        push!(entangle_gates, control(m + n, x_qubit, y_qubit => shift(entangle_phases[k])))
    end
    
    # Full circuit: QFT_row ⊗ QFT_col followed by entanglement layer
    qc = chain(
        subroutine(m + n, qc1, 1:m),
        subroutine(m + n, qc2, m+1:m+n),
        entangle_gates
    )
    
    # Convert to tensor network
    tn = yao2einsum(qc; optimizer=nothing)
    
    # Reorder tensors: Hadamard gates first, then M gates, then entanglement gates
    # Identify tensor types by their values
    is_hadamard(x) = x ≈ mat(H)
    is_entangle(x) = size(x) == (2, 2) && x[1,1] ≈ 1 && x[2,1] ≈ 0 && x[1,2] ≈ 0 && abs(x[2,2]) ≈ 1
    
    # Sort: Hadamard first, then others (M gates and entanglement gates)
    perm_vec = sortperm(tn.tensors, by=x -> !is_hadamard(x))
    ixs = tn.code.ixs[perm_vec]
    tensors = tn.tensors[perm_vec]
    
    if inverse
        code_reorder = DynamicEinCode([ixs..., tn.code.iy[m+n+1:end]], tn.code.iy[1:m+n])
    else
        code_reorder = DynamicEinCode([ixs..., tn.code.iy[1:m+n]], tn.code.iy[m+n+1:end])
    end
    optcode = optimize_code(code_reorder, uniformsize(tn.code, 2), TreeSA())
    
    return optcode, tensors, n_entangle
end

"""
    get_entangle_tensor_indices(tensors, n_entangle::Int)

Identify which tensors in the circuit correspond to entanglement gates.
Returns the indices of the entanglement gate tensors (the last n_entangle controlled-phase gates).

In the tensor network representation from yao2einsum, controlled-phase gates
have the form [1, 1; 1, e^(i*phi)] (not diagonal). The entanglement gates
are the last n_entangle such tensors after sorting (Hadamards first, then phase gates).

# Arguments
- `tensors::Vector`: Circuit tensors
- `n_entangle::Int`: Number of entanglement gates

# Returns
- `Vector{Int}`: Indices of entanglement gate tensors
"""
function get_entangle_tensor_indices(tensors, n_entangle::Int)
    # Controlled-phase gates in tensor network form have pattern [1, 1; 1, e^(i*phi)]
    # This is different from the gate matrix form; it's the tensor decomposition
    is_ctrl_phase(x) = size(x) == (2, 2) && 
                       isapprox(x[1,1], 1) && 
                       isapprox(x[1,2], 1) && 
                       isapprox(x[2,1], 1) && 
                       abs(x[2,2]) ≈ 1
    
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

# ============================================================================
# Entangled QFT Basis Implementation
# ============================================================================

"""
    EntangledQFTBasis <: AbstractSparseBasis

Entangled Quantum Fourier Transform basis with XY correlation.

This basis extends the standard QFT by adding entanglement gates E_k between
corresponding row and column qubits. Each entanglement gate has the same form
as the M gate in QFT:

    E_k = diag(1, 1, 1, e^(i*phi_k))

acting on qubits (x_{n-k}, y_{n-k}), where phi_k is a learnable phase parameter.

# Fields
- `m::Int`: Number of qubits for row dimension (image height = 2^m)
- `n::Int`: Number of qubits for column dimension (image width = 2^n)
- `tensors::Vector`: Circuit parameters (unitary matrices + entanglement gates)
- `optcode::AbstractEinsum`: Optimized einsum code for forward transform
- `inverse_code::AbstractEinsum`: Optimized einsum code for inverse transform
- `n_entangle::Int`: Number of entanglement gates (= min(m, n))
- `entangle_phases::Vector{Float64}`: Phase parameters for entanglement gates

# Example
```julia
# Create default entangled QFT basis for 64×64 images
basis = EntangledQFTBasis(6, 6)

# Create with custom initial entanglement phases
phases = rand(6) * 2π
basis = EntangledQFTBasis(6, 6; entangle_phases=phases)

# Transform an image
freq = forward_transform(basis, image)

# Inverse transform
reconstructed = inverse_transform(basis, freq)
```
"""
struct EntangledQFTBasis <: AbstractSparseBasis
    m::Int
    n::Int
    tensors::Vector
    optcode::OMEinsum.AbstractEinsum
    inverse_code::OMEinsum.AbstractEinsum
    n_entangle::Int
    entangle_phases::Vector{Float64}
end

"""
    EntangledQFTBasis(m::Int, n::Int; entangle_phases=nothing)

Construct an EntangledQFTBasis with default or custom entanglement phases.

# Arguments
- `m::Int`: Number of qubits for rows (image height = 2^m)
- `n::Int`: Number of qubits for columns (image width = 2^n)
- `entangle_phases::Union{Nothing, Vector{<:Real}}`: Initial phases for entanglement gates.
  If nothing, defaults to zeros (equivalent to standard QFT initially).

# Returns
- `EntangledQFTBasis`: Basis with entangled QFT circuit parameters
"""
function EntangledQFTBasis(m::Int, n::Int; entangle_phases::Union{Nothing, Vector{<:Real}}=nothing)
    n_entangle = min(m, n)
    if entangle_phases === nothing
        entangle_phases = zeros(n_entangle)
    end
    
    optcode, tensors, _ = entangled_qft_code(m, n; entangle_phases=entangle_phases)
    inverse_code, _, _ = entangled_qft_code(m, n; entangle_phases=entangle_phases, inverse=true)
    
    return EntangledQFTBasis(m, n, tensors, optcode, inverse_code, n_entangle, Float64.(entangle_phases))
end

"""
    EntangledQFTBasis(m::Int, n::Int, tensors::Vector, n_entangle::Int)

Construct an EntangledQFTBasis with custom trained tensors.

# Arguments
- `m::Int`: Number of qubits for rows
- `n::Int`: Number of qubits for columns
- `tensors::Vector`: Pre-trained circuit parameters
- `n_entangle::Int`: Number of entanglement gates

# Returns
- `EntangledQFTBasis`: Basis with custom parameters
"""
function EntangledQFTBasis(m::Int, n::Int, tensors::Vector, n_entangle::Int)
    optcode, _, _ = entangled_qft_code(m, n)
    inverse_code, _, _ = entangled_qft_code(m, n; inverse=true)
    
    # Extract entanglement phases from tensors
    entangle_indices = get_entangle_tensor_indices(tensors, n_entangle)
    entangle_phases = extract_entangle_phases(tensors, entangle_indices)
    
    return EntangledQFTBasis(m, n, tensors, optcode, inverse_code, n_entangle, entangle_phases)
end

# ============================================================================
# Interface Implementation for EntangledQFTBasis
# ============================================================================

"""
    forward_transform(basis::EntangledQFTBasis, image::AbstractMatrix)

Apply forward entangled QFT transform to convert image to frequency domain.

# Arguments
- `basis::EntangledQFTBasis`: The basis to use for transformation
- `image::AbstractMatrix`: Input image (must be size 2^m × 2^n)

# Returns
- Frequency domain representation (Complex matrix of same size)
"""
function forward_transform(basis::EntangledQFTBasis, image::AbstractMatrix)
    m, n = basis.m, basis.n
    @assert size(image) == (2^m, 2^n) "Image size must be $(2^m)×$(2^n), got $(size(image))"
    
    img_complex = Complex{Float64}.(image)
    
    return reshape(
        basis.optcode(basis.tensors..., reshape(img_complex, fill(2, m+n)...)),
        2^m, 2^n
    )
end

"""
    inverse_transform(basis::EntangledQFTBasis, freq_domain::AbstractMatrix)

Apply inverse entangled QFT transform to convert frequency domain back to image.

# Arguments
- `basis::EntangledQFTBasis`: The basis to use for transformation
- `freq_domain::AbstractMatrix`: Frequency domain data (size 2^m × 2^n)

# Returns
- Reconstructed image (Complex matrix of same size)
"""
function inverse_transform(basis::EntangledQFTBasis, freq_domain::AbstractMatrix)
    m, n = basis.m, basis.n
    @assert size(freq_domain) == (2^m, 2^n) "Frequency domain size must be $(2^m)×$(2^n), got $(size(freq_domain))"
    
    return reshape(
        basis.inverse_code(conj.(basis.tensors)..., reshape(freq_domain, fill(2, m+n)...)),
        2^m, 2^n
    )
end

"""
    image_size(basis::EntangledQFTBasis)

Return the supported image dimensions for this basis.

# Returns
- `Tuple{Int,Int}`: (height, width) = (2^m, 2^n)
"""
function image_size(basis::EntangledQFTBasis)
    return (2^basis.m, 2^basis.n)
end

"""
    num_parameters(basis::EntangledQFTBasis)

Return the total number of learnable parameters in the basis.

For EntangledQFTBasis:
- Standard QFT parameters (Hadamard gates + M gates)
- Additional n entanglement gate phases (one per qubit pair)

# Returns
- `Int`: Total parameter count
"""
function num_parameters(basis::EntangledQFTBasis)
    total = 0
    for tensor in basis.tensors
        total += length(tensor)
    end
    return total
end

"""
    num_entangle_parameters(basis::EntangledQFTBasis)

Return the number of entanglement phase parameters.

# Returns
- `Int`: Number of entanglement gates (= min(m, n))
"""
function num_entangle_parameters(basis::EntangledQFTBasis)
    return basis.n_entangle
end

"""
    get_entangle_phases(basis::EntangledQFTBasis)

Get the current entanglement phase parameters.

# Returns
- `Vector{Float64}`: Phase parameters phi_k for each entanglement gate
"""
function get_entangle_phases(basis::EntangledQFTBasis)
    return copy(basis.entangle_phases)
end

"""
    basis_hash(basis::EntangledQFTBasis)

Compute a unique hash identifying this basis configuration and parameters.

# Returns
- `String`: SHA-256 hash of the basis parameters
"""
function basis_hash(basis::EntangledQFTBasis)
    data = IOBuffer()
    write(data, "EntangledQFTBasis:m=$(basis.m):n=$(basis.n):n_entangle=$(basis.n_entangle):")
    for tensor in basis.tensors
        for val in tensor
            write(data, "$(real(val)),$(imag(val));")
        end
    end
    return bytes2hex(sha256(take!(data)))
end

"""
    get_manifold(basis::EntangledQFTBasis)

Get the product manifold for Riemannian optimization of basis parameters.

# Returns
- `ProductManifold`: Manifold structure for the tensors
"""
function get_manifold(basis::EntangledQFTBasis)
    return generate_manifold(basis.tensors)
end

# ============================================================================
# Utility Functions for EntangledQFTBasis
# ============================================================================

"""
    Base.show(io::IO, basis::EntangledQFTBasis)

Pretty print the EntangledQFTBasis.
"""
function Base.show(io::IO, basis::EntangledQFTBasis)
    h, w = image_size(basis)
    params = num_parameters(basis)
    n_ent = num_entangle_parameters(basis)
    print(io, "EntangledQFTBasis($(basis.m)×$(basis.n) qubits, $(h)×$(w) images, $params parameters, $n_ent entanglement gates)")
end

"""
    Base.:(==)(a::EntangledQFTBasis, b::EntangledQFTBasis)

Check equality of two EntangledQFTBasis objects.
"""
function Base.:(==)(a::EntangledQFTBasis, b::EntangledQFTBasis)
    return a.m == b.m && a.n == b.n && a.n_entangle == b.n_entangle && all(a.tensors .≈ b.tensors)
end

