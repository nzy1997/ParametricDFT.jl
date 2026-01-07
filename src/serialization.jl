# ============================================================================
# Sparse Basis Serialization
# ============================================================================
# This file provides JSON serialization for saving and loading trained bases.

# ============================================================================
# JSON Serialization Format
# ============================================================================

"""
    BasisJSON

Internal struct for JSON serialization of QFTBasis.
"""
struct BasisJSON
    type::String
    version::String
    m::Int
    n::Int
    tensors::Vector{Vector{Vector{Float64}}}  # Each tensor as [[real, imag], ...]
    hash::String
end

# Define StructTypes for JSON3 serialization
StructTypes.StructType(::Type{BasisJSON}) = StructTypes.Struct()

# ============================================================================
# Save Functions
# ============================================================================

"""
    save_basis(path::String, basis::AbstractSparseBasis)

Save a sparse basis to a JSON file.

# Arguments
- `path::String`: File path to save to (should end in .json)
- `basis::AbstractSparseBasis`: The basis to save

# Example
```julia
basis = train_basis(QFTBasis, images; m=6, n=6)
save_basis("trained_basis.json", basis)
```
"""
function save_basis(path::String, basis::AbstractSparseBasis)
    json_data = _basis_to_json(basis)
    open(path, "w") do io
        JSON3.pretty(io, json_data)
    end
    return path
end

"""
    _basis_to_json(basis::QFTBasis)

Convert a QFTBasis to JSON-serializable format.
"""
function _basis_to_json(basis::QFTBasis)
    # Convert tensors to serializable format
    # Each tensor is a complex matrix/array, we store as [[real, imag], ...]
    serialized_tensors = Vector{Vector{Vector{Float64}}}()
    
    for tensor in basis.tensors
        tensor_data = Vector{Vector{Float64}}()
        for val in tensor
            push!(tensor_data, [real(val), imag(val)])
        end
        push!(serialized_tensors, tensor_data)
    end
    
    return BasisJSON(
        "QFTBasis",
        "1.0",
        basis.m,
        basis.n,
        serialized_tensors,
        basis_hash(basis)
    )
end

# ============================================================================
# Load Functions
# ============================================================================

"""
    load_basis(path::String) -> AbstractSparseBasis

Load a sparse basis from a JSON file.

# Arguments
- `path::String`: Path to the JSON file

# Returns
- `AbstractSparseBasis`: The loaded basis (concrete type depends on file contents)

# Example
```julia
basis = load_basis("trained_basis.json")
freq = forward_transform(basis, image)
```
"""
function load_basis(path::String)
    json_str = read(path, String)
    json_data = JSON3.read(json_str, BasisJSON)
    return _json_to_basis(json_data)
end

"""
    _json_to_basis(json_data::BasisJSON)

Convert JSON data back to a basis object.
"""
function _json_to_basis(json_data::BasisJSON)
    if json_data.type != "QFTBasis"
        error("Unknown basis type: $(json_data.type)")
    end
    
    if json_data.version != "1.0"
        @warn "Basis version $(json_data.version) may not be fully compatible with current version 1.0"
    end
    
    m, n = json_data.m, json_data.n
    
    # Reconstruct tensors from serialized format
    # We need to know the original tensor shapes from the QFT code
    optcode, template_tensors = qft_code(m, n)
    inverse_code, _ = qft_code(m, n; inverse=true)
    
    tensors = Vector{Any}()
    for (i, tensor_data) in enumerate(json_data.tensors)
        # Get the shape from template
        template_shape = size(template_tensors[i])
        
        # Reconstruct complex values
        complex_vals = [Complex{Float64}(pair[1], pair[2]) for pair in tensor_data]
        
        # Reshape to original dimensions
        tensor = reshape(complex_vals, template_shape)
        push!(tensors, tensor)
    end
    
    # Verify hash matches
    loaded_basis = QFTBasis(m, n, tensors, optcode, inverse_code)
    loaded_hash = basis_hash(loaded_basis)
    
    if loaded_hash != json_data.hash
        @warn "Basis hash mismatch. File hash: $(json_data.hash), computed hash: $(loaded_hash). The basis may have been corrupted."
    end
    
    return loaded_basis
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    basis_to_dict(basis::AbstractSparseBasis) -> Dict

Convert a basis to a dictionary for custom serialization.

# Returns
- `Dict`: Dictionary representation of the basis
"""
function basis_to_dict(basis::QFTBasis)
    json_data = _basis_to_json(basis)
    return Dict(
        "type" => json_data.type,
        "version" => json_data.version,
        "m" => json_data.m,
        "n" => json_data.n,
        "tensors" => json_data.tensors,
        "hash" => json_data.hash
    )
end

"""
    dict_to_basis(d::Dict) -> AbstractSparseBasis

Convert a dictionary back to a basis.

# Arguments
- `d::Dict`: Dictionary with basis data

# Returns
- `AbstractSparseBasis`: The reconstructed basis
"""
function dict_to_basis(d::Dict)
    json_data = BasisJSON(
        d["type"],
        d["version"],
        d["m"],
        d["n"],
        d["tensors"],
        d["hash"]
    )
    return _json_to_basis(json_data)
end

