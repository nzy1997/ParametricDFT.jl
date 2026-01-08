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

