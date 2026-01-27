# ============================================================================
# Sparse Basis Abstraction
# ============================================================================
# This file defines an extensible abstraction for tensor network topologies
# used in sparse signal representation and compression.

"""
    AbstractSparseBasis

Abstract base type for sparse basis representations. All concrete basis types
should inherit from this and implement the required interface:

Required methods:
- `forward_transform(basis, image)` - transform image to frequency domain
- `inverse_transform(basis, freq_domain)` - inverse transform
- `image_size(basis)` - return supported image dimensions
- `num_parameters(basis)` - return total parameter count
- `basis_hash(basis)` - return unique hash for basis identification
"""
abstract type AbstractSparseBasis end

# ============================================================================
# QFT Basis Implementation
# ============================================================================

"""
    QFTBasis <: AbstractSparseBasis

Quantum Fourier Transform basis using tensor network representation.

# Fields
- `m::Int`: Number of qubits for row dimension (image height = 2^m)
- `n::Int`: Number of qubits for column dimension (image width = 2^n)
- `tensors::Vector`: Circuit parameters (unitary matrices)
- `optcode::AbstractEinsum`: Optimized einsum code for forward transform
- `inverse_code::AbstractEinsum`: Optimized einsum code for inverse transform

# Example
```julia
# Create default QFT basis for 64×64 images
basis = QFTBasis(6, 6)

# Transform an image
freq = forward_transform(basis, image)

# Inverse transform
reconstructed = inverse_transform(basis, freq)
```
"""
struct QFTBasis <: AbstractSparseBasis
    m::Int
    n::Int
    tensors::Vector
    optcode::OMEinsum.AbstractEinsum
    inverse_code::OMEinsum.AbstractEinsum
end

"""
    QFTBasis(m::Int, n::Int)

Construct a QFTBasis with default QFT circuit parameters.

# Arguments
- `m::Int`: Number of qubits for rows (image height = 2^m)
- `n::Int`: Number of qubits for columns (image width = 2^n)

# Returns
- `QFTBasis`: Basis with standard QFT circuit parameters
"""
function QFTBasis(m::Int, n::Int)
    optcode, tensors = qft_code(m, n)
    inverse_code, _ = qft_code(m, n; inverse=true)
    return QFTBasis(m, n, tensors, optcode, inverse_code)
end

"""
    QFTBasis(m::Int, n::Int, tensors::Vector)

Construct a QFTBasis with custom trained tensors.

# Arguments
- `m::Int`: Number of qubits for rows
- `n::Int`: Number of qubits for columns
- `tensors::Vector`: Pre-trained circuit parameters

# Returns
- `QFTBasis`: Basis with custom parameters
"""
function QFTBasis(m::Int, n::Int, tensors::Vector)
    optcode, _ = qft_code(m, n)
    inverse_code, _ = qft_code(m, n; inverse=true)
    return QFTBasis(m, n, tensors, optcode, inverse_code)
end

# ============================================================================
# Interface Implementation for QFTBasis
# ============================================================================

"""
    forward_transform(basis::QFTBasis, image::AbstractMatrix)

Apply forward transform to convert image to frequency domain.

# Arguments
- `basis::QFTBasis`: The basis to use for transformation
- `image::AbstractMatrix`: Input image (must be size 2^m × 2^n)

# Returns
- Frequency domain representation (Complex matrix of same size)
"""
function forward_transform(basis::QFTBasis, image::AbstractMatrix)
    m, n = basis.m, basis.n
    @assert size(image) == (2^m, 2^n) "Image size must be $(2^m)×$(2^n), got $(size(image))"
    
    # Convert to complex if needed
    img_complex = Complex{Float64}.(image)
    
    # Apply forward transform using tensor network
    return reshape(
        basis.optcode(basis.tensors..., reshape(img_complex, fill(2, m+n)...)),
        2^m, 2^n
    )
end

"""
    inverse_transform(basis::QFTBasis, freq_domain::AbstractMatrix)

Apply inverse transform to convert frequency domain back to image.

# Arguments
- `basis::QFTBasis`: The basis to use for transformation
- `freq_domain::AbstractMatrix`: Frequency domain data (size 2^m × 2^n)

# Returns
- Reconstructed image (Complex matrix of same size)
"""
function inverse_transform(basis::QFTBasis, freq_domain::AbstractMatrix)
    m, n = basis.m, basis.n
    @assert size(freq_domain) == (2^m, 2^n) "Frequency domain size must be $(2^m)×$(2^n), got $(size(freq_domain))"
    
    # Apply inverse transform using conjugated tensors
    return reshape(
        basis.inverse_code(conj.(basis.tensors)..., reshape(freq_domain, fill(2, m+n)...)),
        2^m, 2^n
    )
end

"""
    image_size(basis::QFTBasis)

Return the supported image dimensions for this basis.

# Returns
- `Tuple{Int,Int}`: (height, width) = (2^m, 2^n)
"""
function image_size(basis::QFTBasis)
    return (2^basis.m, 2^basis.n)
end

"""
    num_parameters(basis::QFTBasis)

Return the total number of learnable parameters in the basis.

For QFT basis:
- n Hadamard gates: 4n parameters each (2×2 unitary)
- n(n-1)/2 controlled-phase gates: 4 parameters each (diagonal 2×2)

# Returns
- `Int`: Total parameter count
"""
function num_parameters(basis::QFTBasis)
    total = 0
    for tensor in basis.tensors
        total += length(tensor)
    end
    return total
end

"""
    basis_hash(basis::QFTBasis)

Compute a unique hash identifying this basis configuration and parameters.

# Returns
- `String`: SHA-256 hash of the basis parameters
"""
function basis_hash(basis::QFTBasis)
    # Create a string representation of the basis for hashing
    data = IOBuffer()
    write(data, "QFTBasis:m=$(basis.m):n=$(basis.n):")
    for tensor in basis.tensors
        for val in tensor
            write(data, "$(real(val)),$(imag(val));")
        end
    end
    return bytes2hex(sha256(take!(data)))
end

"""
    get_manifold(basis::QFTBasis)

Get the product manifold for Riemannian optimization of basis parameters.

# Returns
- `ProductManifold`: Manifold structure for the tensors
"""
function get_manifold(basis::QFTBasis)
    return generate_manifold(basis.tensors)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    Base.show(io::IO, basis::QFTBasis)

Pretty print the QFTBasis.
"""
function Base.show(io::IO, basis::QFTBasis)
    h, w = image_size(basis)
    params = num_parameters(basis)
    print(io, "QFTBasis($(basis.m)×$(basis.n) qubits, $(h)×$(w) images, $params parameters)")
end

"""
    Base.:(==)(a::QFTBasis, b::QFTBasis)

Check equality of two QFTBasis objects.
"""
function Base.:(==)(a::QFTBasis, b::QFTBasis)
    return a.m == b.m && a.n == b.n && all(a.tensors .≈ b.tensors)
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


# ============================================================================
# TEBD Basis Implementation
# ============================================================================

"""
    TEBDBasis <: AbstractSparseBasis

Time-Evolving Block Decimation (TEBD) basis with 2D ring topology.

This basis uses m row qubits and n column qubits with two separate rings:
- Row ring: (x1,x2), (x2,x3), ..., (x_{m-1},x_m), (x_m,x1) for m gates
- Column ring: (y1,y2), (y2,y3), ..., (y_{n-1},y_n), (y_n,y1) for n gates

# Fields
- `m::Int`: Number of row qubits (row dimension = 2^m)
- `n::Int`: Number of column qubits (col dimension = 2^n)
- `tensors::Vector`: Circuit parameters (TEBD gate tensors)
- `optcode::AbstractEinsum`: Optimized einsum code for forward transform
- `inverse_code::AbstractEinsum`: Optimized einsum code for inverse transform
- `n_row_gates::Int`: Number of row ring phase gates (= m)
- `n_col_gates::Int`: Number of column ring phase gates (= n)
- `phases::Vector{Float64}`: Phase parameters for TEBD gates

# Example
```julia
# Create default TEBD basis for 8×8 images (m=3, n=3)
basis = TEBDBasis(3, 3)

# Create with custom initial phases (6 gates total: 3 row ring + 3 col ring)
phases = rand(6) * 2π
basis = TEBDBasis(3, 3; phases=phases)

# Transform an image
freq = forward_transform(basis, image)

# Inverse transform
reconstructed = inverse_transform(basis, freq)
```
"""
struct TEBDBasis <: AbstractSparseBasis
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
    TEBDBasis(m::Int, n::Int; phases=nothing)

Construct a TEBDBasis with default or custom phases.

# Arguments
- `m::Int`: Number of row qubits (row dimension = 2^m)
- `n::Int`: Number of column qubits (col dimension = 2^n)
- `phases::Union{Nothing, Vector{<:Real}}`: Initial phases for TEBD gates.
  If nothing, defaults to zeros. Length must be m+n for ring topology.

# Returns
- `TEBDBasis`: Basis with TEBD circuit parameters
"""
function TEBDBasis(m::Int, n::Int; phases::Union{Nothing, Vector{<:Real}}=nothing)
    n_row = m  # Row ring has m gates
    n_col = n  # Col ring has n gates
    n_gates = n_row + n_col
    if phases === nothing
        phases = zeros(n_gates)
    end
    
    optcode, tensors, _, _ = tebd_code(m, n; phases=phases)
    inverse_code, _, _, _ = tebd_code(m, n; phases=phases, inverse=true)
    
    return TEBDBasis(m, n, tensors, optcode, inverse_code, n_row, n_col, Float64.(phases))
end

"""
    TEBDBasis(m::Int, n::Int, tensors::Vector, n_row_gates::Int, n_col_gates::Int)

Construct a TEBDBasis with custom trained tensors.

# Arguments
- `m::Int`: Number of row qubits
- `n::Int`: Number of column qubits
- `tensors::Vector`: Pre-trained circuit parameters
- `n_row_gates::Int`: Number of row ring gates
- `n_col_gates::Int`: Number of column ring gates

# Returns
- `TEBDBasis`: Basis with custom parameters
"""
function TEBDBasis(m::Int, n::Int, tensors::Vector, n_row_gates::Int, n_col_gates::Int)
    optcode, _, _, _ = tebd_code(m, n)
    inverse_code, _, _, _ = tebd_code(m, n; inverse=true)
    
    # Extract phases from tensors
    n_gates = n_row_gates + n_col_gates
    gate_indices = get_tebd_gate_indices(tensors, n_gates)
    phases = extract_tebd_phases(tensors, gate_indices)
    
    return TEBDBasis(m, n, tensors, optcode, inverse_code, n_row_gates, n_col_gates, phases)
end

# ============================================================================
# Interface Implementation for TEBDBasis
# ============================================================================

"""
    forward_transform(basis::TEBDBasis, data::AbstractVector)

Apply forward TEBD transform to a vector.

# Arguments
- `basis::TEBDBasis`: The basis to use for transformation
- `data::AbstractVector`: Input vector (must have length 2^(m+n))

# Returns
- Transformed representation (Complex vector of same length)
"""
function forward_transform(basis::TEBDBasis, data::AbstractVector)
    total = basis.m + basis.n
    expected_size = 2^total
    @assert length(data) == expected_size "Data length must be 2^(m+n) = $(expected_size), got $(length(data))"
    
    data_complex = Complex{Float64}.(data)
    
    return vec(basis.optcode(basis.tensors..., reshape(data_complex, fill(2, total)...)))
end

"""
    forward_transform(basis::TEBDBasis, image::AbstractMatrix)

Apply forward TEBD transform to an image.

# Arguments
- `basis::TEBDBasis`: The basis to use for transformation
- `image::AbstractMatrix`: Input image (must be 2^m × 2^n)

# Returns
- Transformed representation as matrix (same shape as input)
"""
function forward_transform(basis::TEBDBasis, image::AbstractMatrix)
    m, n = basis.m, basis.n
    expected_size = (2^m, 2^n)
    @assert size(image) == expected_size "Image must be $(expected_size), got $(size(image))"
    
    total = m + n
    img_complex = Complex{Float64}.(vec(image))
    result = vec(basis.optcode(basis.tensors..., reshape(img_complex, fill(2, total)...)))
    
    return reshape(result, size(image))
end

"""
    inverse_transform(basis::TEBDBasis, freq_domain::AbstractVector)

Apply inverse TEBD transform to convert back to original domain.

# Arguments
- `basis::TEBDBasis`: The basis to use for transformation
- `freq_domain::AbstractVector`: Frequency domain data (length 2^(m+n))

# Returns
- Reconstructed data (Complex vector of same length)
"""
function inverse_transform(basis::TEBDBasis, freq_domain::AbstractVector)
    total = basis.m + basis.n
    expected_size = 2^total
    @assert length(freq_domain) == expected_size "Frequency domain length must be 2^(m+n) = $(expected_size), got $(length(freq_domain))"
    
    return vec(basis.inverse_code(conj.(basis.tensors)..., reshape(freq_domain, fill(2, total)...)))
end

"""
    inverse_transform(basis::TEBDBasis, freq_domain::AbstractMatrix)

Apply inverse TEBD transform to a matrix.

# Arguments
- `basis::TEBDBasis`: The basis to use for transformation
- `freq_domain::AbstractMatrix`: Frequency domain data (must be 2^m × 2^n)

# Returns
- Reconstructed data as matrix (same shape as input)
"""
function inverse_transform(basis::TEBDBasis, freq_domain::AbstractMatrix)
    m, n = basis.m, basis.n
    expected_size = (2^m, 2^n)
    @assert size(freq_domain) == expected_size "Frequency domain must be $(expected_size), got $(size(freq_domain))"
    
    total = m + n
    freq_vec = Complex{Float64}.(vec(freq_domain))
    result = vec(basis.inverse_code(conj.(basis.tensors)..., reshape(freq_vec, fill(2, total)...)))
    
    return reshape(result, size(freq_domain))
end

"""
    image_size(basis::TEBDBasis)

Return the supported image dimensions for this basis.

# Returns
- `Tuple{Int,Int}`: (height, width) = (2^m, 2^n)
"""
function image_size(basis::TEBDBasis)
    return (2^basis.m, 2^basis.n)
end

"""
    num_parameters(basis::TEBDBasis)

Return the total number of learnable parameters in the basis.

# Returns
- `Int`: Total parameter count
"""
function num_parameters(basis::TEBDBasis)
    total = 0
    for tensor in basis.tensors
        total += length(tensor)
    end
    return total
end

"""
    num_gates(basis::TEBDBasis)

Return the total number of TEBD gates.

# Returns
- `Int`: Number of gates (= m + n for ring topology)
"""
function num_gates(basis::TEBDBasis)
    return basis.n_row_gates + basis.n_col_gates
end

"""
    get_phases(basis::TEBDBasis)

Get the current phase parameters.

# Returns
- `Vector{Float64}`: Phase parameters for each TEBD gate
"""
function get_phases(basis::TEBDBasis)
    return copy(basis.phases)
end

"""
    basis_hash(basis::TEBDBasis)

Compute a unique hash identifying this basis configuration and parameters.

# Returns
- `String`: SHA-256 hash of the basis parameters
"""
function basis_hash(basis::TEBDBasis)
    data = IOBuffer()
    n_gates = basis.n_row_gates + basis.n_col_gates
    write(data, "TEBDBasis:m=$(basis.m):n=$(basis.n):n_row=$(basis.n_row_gates):n_col=$(basis.n_col_gates):")
    for tensor in basis.tensors
        for val in tensor
            write(data, "$(real(val)),$(imag(val));")
        end
    end
    return bytes2hex(sha256(take!(data)))
end

"""
    get_manifold(basis::TEBDBasis)

Get the product manifold for Riemannian optimization of basis parameters.

# Returns
- `ProductManifold`: Manifold structure for the tensors
"""
function get_manifold(basis::TEBDBasis)
    return generate_manifold(basis.tensors)
end

# ============================================================================
# Utility Functions for TEBDBasis
# ============================================================================

"""
    Base.show(io::IO, basis::TEBDBasis)

Pretty print the TEBDBasis.
"""
function Base.show(io::IO, basis::TEBDBasis)
    h, w = image_size(basis)
    params = num_parameters(basis)
    n_g = num_gates(basis)
    total_qubits = basis.m + basis.n
    print(io, "TEBDBasis($(basis.m)×$(basis.n) qubits, $(h)×$(w) images, $params parameters, $n_g gates)")
end

"""
    Base.:(==)(a::TEBDBasis, b::TEBDBasis)

Check equality of two TEBDBasis objects.
"""
function Base.:(==)(a::TEBDBasis, b::TEBDBasis)
    return a.m == b.m && a.n == b.n && 
           a.n_row_gates == b.n_row_gates && a.n_col_gates == b.n_col_gates && 
           all(a.tensors .≈ b.tensors)
end
