# ============================================================================
# Image Compression and Recovery
# ============================================================================
# This file provides functions for compressing images using a trained sparse
# basis and recovering images from compressed representations.

# ============================================================================
# Compressed Image Structure
# ============================================================================

"""
    CompressedImage

Sparse representation of an image in the frequency domain.

# Fields
- `indices::Vector{Int}`: Linear indices of non-zero coefficients
- `values_real::Vector{Float64}`: Real parts of coefficient values
- `values_imag::Vector{Float64}`: Imaginary parts of coefficient values
- `original_size::Tuple{Int,Int}`: Original image dimensions (height, width)
- `basis_hash::String`: Hash of the basis used for compression (for verification)

# Note
The compressed representation stores only the non-zero coefficients after
truncation, achieving compression by discarding small coefficients.
"""
struct CompressedImage
    indices::Vector{Int}
    values_real::Vector{Float64}
    values_imag::Vector{Float64}
    original_size::Tuple{Int,Int}
    basis_hash::String
end

# Define StructTypes for JSON3 serialization
StructTypes.StructType(::Type{CompressedImage}) = StructTypes.Struct()

# ============================================================================
# Compression Functions
# ============================================================================

"""
    compress(basis::AbstractSparseBasis, image::AbstractMatrix; ratio::Float64=0.9)

Compress an image using the given sparse basis.

# Arguments
- `basis::AbstractSparseBasis`: The trained basis to use for compression
- `image::AbstractMatrix`: Input image (must match basis dimensions)
- `ratio::Float64 = 0.9`: Compression ratio (0.9 means keep only 10% of coefficients)

# Returns
- `CompressedImage`: Sparse representation of the image

# Example
```julia
basis = load_basis("trained_basis.json")
image = load_grayscale_image("photo.png")
compressed = compress(basis, image; ratio=0.95)  # Keep top 5%
save_compressed("photo.cimg", compressed)
```
"""
function compress(
    basis::AbstractSparseBasis,
    image::AbstractMatrix;
    ratio::Float64 = 0.9
)
    @assert 0.0 <= ratio < 1.0 "Compression ratio must be in [0, 1)"
    
    expected_size = image_size(basis)
    @assert size(image) == expected_size "Image size $(size(image)) must match basis size $expected_size"
    
    # Forward transform to frequency domain
    freq_domain = forward_transform(basis, image)
    
    # Determine how many coefficients to keep
    total_coeffs = length(freq_domain)
    keep_count = max(1, round(Int, total_coeffs * (1.0 - ratio)))
    
    # Find top-k coefficients by magnitude with frequency weighting
    indices, values = _select_top_coefficients(freq_domain, keep_count)
    
    return CompressedImage(
        indices,
        real.(values),
        imag.(values),
        size(image),
        basis_hash(basis)
    )
end

"""
    compress_with_k(basis::AbstractSparseBasis, image::AbstractMatrix; k::Int)

Compress an image keeping exactly k coefficients.

# Arguments
- `basis::AbstractSparseBasis`: The trained basis to use
- `image::AbstractMatrix`: Input image
- `k::Int`: Exact number of coefficients to keep

# Returns
- `CompressedImage`: Sparse representation
"""
function compress_with_k(
    basis::AbstractSparseBasis,
    image::AbstractMatrix;
    k::Int
)
    expected_size = image_size(basis)
    @assert size(image) == expected_size "Image size $(size(image)) must match basis size $expected_size"
    @assert k > 0 "k must be positive"
    
    # Forward transform to frequency domain
    freq_domain = forward_transform(basis, image)
    
    # Keep exactly k coefficients
    keep_count = min(k, length(freq_domain))
    
    # Find top-k coefficients
    indices, values = _select_top_coefficients(freq_domain, keep_count)
    
    return CompressedImage(
        indices,
        real.(values),
        imag.(values),
        size(image),
        basis_hash(basis)
    )
end

"""
    _select_top_coefficients(freq_domain::AbstractMatrix, k::Int)

Select top k coefficients using frequency-weighted magnitude scoring.

Low-frequency components (near center) are prioritized as they typically
contain more important structural information.
"""
function _select_top_coefficients(freq_domain::AbstractMatrix, k::Int)
    m, n = size(freq_domain)
    k = min(k, length(freq_domain))
    
    # Calculate frequency distances from center (DC component)
    center_i, center_j = (m + 1) ÷ 2, (n + 1) ÷ 2
    max_dist = sqrt((m/2)^2 + (n/2)^2)
    
    # Create frequency-weighted scores
    scores = zeros(Float64, m, n)
    mags = abs.(freq_domain)
    
    @inbounds for j in 1:n, i in 1:m
        freq_dist = sqrt((i - center_i)^2 + (j - center_j)^2)
        freq_weight = 1.0 - (freq_dist / max_dist) * 0.5
        scores[i, j] = mags[i, j] * (1.0 + freq_weight)
    end
    
    # Select top k indices
    scores_flat = vec(scores)
    top_indices = partialsortperm(scores_flat, 1:k, rev=true)
    
    # Extract values at those indices
    freq_flat = vec(freq_domain)
    values = freq_flat[top_indices]
    
    return top_indices, values
end

# ============================================================================
# Recovery Functions
# ============================================================================

"""
    recover(basis::AbstractSparseBasis, compressed::CompressedImage; verify_hash::Bool=true)

Recover an image from its compressed representation.

# Arguments
- `basis::AbstractSparseBasis`: The basis used for compression
- `compressed::CompressedImage`: The compressed image data
- `verify_hash::Bool = true`: Whether to verify basis hash matches

# Returns
- `Matrix{Float64}`: Reconstructed image (real-valued)

# Example
```julia
basis = load_basis("trained_basis.json")
compressed = load_compressed("photo.cimg")
recovered = recover(basis, compressed)
```
"""
function recover(
    basis::AbstractSparseBasis,
    compressed::CompressedImage;
    verify_hash::Bool = true
)
    # Verify basis compatibility
    if verify_hash && basis_hash(basis) != compressed.basis_hash
        error("Basis hash mismatch. The compressed image was created with a different basis.\n" *
              "Expected: $(compressed.basis_hash)\n" *
              "Got: $(basis_hash(basis))")
    end
    
    expected_size = image_size(basis)
    @assert compressed.original_size == expected_size "Compressed image size $(compressed.original_size) doesn't match basis size $expected_size"
    
    # Reconstruct frequency domain from sparse representation
    freq_domain = _reconstruct_frequency_domain(compressed, expected_size)
    
    # Apply inverse transform
    reconstructed = inverse_transform(basis, freq_domain)
    
    # Return real-valued image
    return real.(reconstructed)
end

"""
    _reconstruct_frequency_domain(compressed::CompressedImage, size::Tuple{Int,Int})

Reconstruct full frequency domain matrix from sparse representation.
"""
function _reconstruct_frequency_domain(compressed::CompressedImage, size::Tuple{Int,Int})
    freq_domain = zeros(Complex{Float64}, size)
    
    for (idx, (re, im)) in zip(compressed.indices, zip(compressed.values_real, compressed.values_imag))
        freq_domain[idx] = Complex{Float64}(re, im)
    end
    
    return freq_domain
end

# ============================================================================
# Compressed Image I/O
# ============================================================================

"""
    CompressedImageJSON

Internal struct for JSON serialization of CompressedImage.
"""
struct CompressedImageJSON
    version::String
    indices::Vector{Int}
    values_real::Vector{Float64}
    values_imag::Vector{Float64}
    original_height::Int
    original_width::Int
    basis_hash::String
    num_coefficients::Int
    compression_ratio::Float64
end

StructTypes.StructType(::Type{CompressedImageJSON}) = StructTypes.Struct()

"""
    save_compressed(path::String, compressed::CompressedImage)

Save a compressed image to a JSON file.

# Arguments
- `path::String`: File path to save to
- `compressed::CompressedImage`: The compressed image to save

# Returns
- `String`: The path where the file was saved
"""
function save_compressed(path::String, compressed::CompressedImage)
    total_coeffs = compressed.original_size[1] * compressed.original_size[2]
    kept_coeffs = length(compressed.indices)
    ratio = 1.0 - (kept_coeffs / total_coeffs)
    
    json_data = CompressedImageJSON(
        "1.0",
        compressed.indices,
        compressed.values_real,
        compressed.values_imag,
        compressed.original_size[1],
        compressed.original_size[2],
        compressed.basis_hash,
        kept_coeffs,
        ratio
    )
    
    open(path, "w") do io
        JSON3.pretty(io, json_data)
    end
    
    return path
end

"""
    load_compressed(path::String) -> CompressedImage

Load a compressed image from a JSON file.

# Arguments
- `path::String`: Path to the compressed image file

# Returns
- `CompressedImage`: The loaded compressed image
"""
function load_compressed(path::String)
    json_str = read(path, String)
    json_data = JSON3.read(json_str, CompressedImageJSON)
    
    if json_data.version != "1.0"
        @warn "Compressed image version $(json_data.version) may not be fully compatible with current version 1.0"
    end
    
    return CompressedImage(
        json_data.indices,
        json_data.values_real,
        json_data.values_imag,
        (json_data.original_height, json_data.original_width),
        json_data.basis_hash
    )
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    compression_stats(compressed::CompressedImage) -> NamedTuple

Get statistics about the compression.

# Returns
A named tuple with:
- `original_size`: Original image dimensions
- `total_coefficients`: Total number of coefficients
- `kept_coefficients`: Number of non-zero coefficients kept
- `compression_ratio`: Ratio of discarded coefficients
- `storage_reduction`: Approximate storage reduction factor
"""
function compression_stats(compressed::CompressedImage)
    h, w = compressed.original_size
    total = h * w
    kept = length(compressed.indices)
    ratio = 1.0 - (kept / total)
    
    # Estimate storage: original = total * 8 bytes (Float64)
    # compressed = kept * (4 + 8 + 8) = kept * 20 bytes (index + 2 floats)
    original_bytes = total * 8
    compressed_bytes = kept * 20 + 100  # +100 for metadata overhead
    storage_factor = original_bytes / compressed_bytes
    
    return (
        original_size = compressed.original_size,
        total_coefficients = total,
        kept_coefficients = kept,
        compression_ratio = ratio,
        storage_reduction = storage_factor
    )
end

"""
    Base.show(io::IO, compressed::CompressedImage)

Pretty print the CompressedImage.
"""
function Base.show(io::IO, compressed::CompressedImage)
    stats = compression_stats(compressed)
    print(io, "CompressedImage($(stats.original_size[1])×$(stats.original_size[2]), " *
              "$(stats.kept_coefficients)/$(stats.total_coefficients) coeffs, " *
              "$(round(stats.compression_ratio * 100, digits=1))% compression)")
end

