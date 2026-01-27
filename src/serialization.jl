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

"""
    EntangledBasisJSON

Internal struct for JSON serialization of EntangledQFTBasis.
"""
struct EntangledBasisJSON
    type::String
    version::String
    m::Int
    n::Int
    n_entangle::Int
    entangle_phases::Vector{Float64}
    tensors::Vector{Vector{Vector{Float64}}}  # Each tensor as [[real, imag], ...]
    hash::String
end

"""
    TEBDBasisJSON

Internal struct for JSON serialization of TEBDBasis.
"""
struct TEBDBasisJSON
    type::String
    version::String
    m::Int
    n::Int
    n_row_gates::Int
    n_col_gates::Int
    phases::Vector{Float64}
    tensors::Vector{Vector{Vector{Float64}}}  # Each tensor as [[real, imag], ...]
    hash::String
end

# Define StructTypes for JSON3 serialization
StructTypes.StructType(::Type{BasisJSON}) = StructTypes.Struct()
StructTypes.StructType(::Type{EntangledBasisJSON}) = StructTypes.Struct()
StructTypes.StructType(::Type{TEBDBasisJSON}) = StructTypes.Struct()

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

"""
    _basis_to_json(basis::EntangledQFTBasis)

Convert an EntangledQFTBasis to JSON-serializable format.
"""
function _basis_to_json(basis::EntangledQFTBasis)
    # Convert tensors to serializable format
    serialized_tensors = Vector{Vector{Vector{Float64}}}()
    
    for tensor in basis.tensors
        tensor_data = Vector{Vector{Float64}}()
        for val in tensor
            push!(tensor_data, [real(val), imag(val)])
        end
        push!(serialized_tensors, tensor_data)
    end
    
    return EntangledBasisJSON(
        "EntangledQFTBasis",
        "1.0",
        basis.m,
        basis.n,
        basis.n_entangle,
        basis.entangle_phases,
        serialized_tensors,
        basis_hash(basis)
    )
end

"""
    _basis_to_json(basis::TEBDBasis)

Convert a TEBDBasis to JSON-serializable format.
"""
function _basis_to_json(basis::TEBDBasis)
    # Convert tensors to serializable format
    serialized_tensors = Vector{Vector{Vector{Float64}}}()
    
    for tensor in basis.tensors
        tensor_data = Vector{Vector{Float64}}()
        for val in tensor
            push!(tensor_data, [real(val), imag(val)])
        end
        push!(serialized_tensors, tensor_data)
    end
    
    return TEBDBasisJSON(
        "TEBDBasis",
        "2.0",  # Version 2.0 for 2D topology
        basis.m,
        basis.n,
        basis.n_row_gates,
        basis.n_col_gates,
        basis.phases,
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
    
    # First, peek at the type to determine which struct to use
    json_obj = JSON3.read(json_str)
    basis_type = get(json_obj, :type, "QFTBasis")
    
    if basis_type == "EntangledQFTBasis"
        json_data = JSON3.read(json_str, EntangledBasisJSON)
        return _json_to_entangled_basis(json_data)
    elseif basis_type == "TEBDBasis"
        json_data = JSON3.read(json_str, TEBDBasisJSON)
        return _json_to_tebd_basis(json_data)
    else
        json_data = JSON3.read(json_str, BasisJSON)
        return _json_to_basis(json_data)
    end
end

"""
    _json_to_basis(json_data::BasisJSON)

Convert JSON data back to a QFTBasis object.
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

"""
    _json_to_entangled_basis(json_data::EntangledBasisJSON)

Convert JSON data back to an EntangledQFTBasis object.
"""
function _json_to_entangled_basis(json_data::EntangledBasisJSON)
    if json_data.type != "EntangledQFTBasis"
        error("Unknown basis type: $(json_data.type)")
    end
    
    if json_data.version != "1.0"
        @warn "Basis version $(json_data.version) may not be fully compatible with current version 1.0"
    end
    
    m, n = json_data.m, json_data.n
    n_entangle = json_data.n_entangle
    entangle_phases = json_data.entangle_phases
    
    # Reconstruct tensors from serialized format
    # We need to know the original tensor shapes from the entangled QFT code
    optcode, template_tensors, _ = entangled_qft_code(m, n; entangle_phases=entangle_phases)
    inverse_code, _, _ = entangled_qft_code(m, n; entangle_phases=entangle_phases, inverse=true)
    
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
    loaded_basis = EntangledQFTBasis(m, n, tensors, optcode, inverse_code, n_entangle, Float64.(entangle_phases))
    loaded_hash = basis_hash(loaded_basis)
    
    if loaded_hash != json_data.hash
        @warn "Basis hash mismatch. File hash: $(json_data.hash), computed hash: $(loaded_hash). The basis may have been corrupted."
    end
    
    return loaded_basis
end

"""
    _json_to_tebd_basis(json_data::TEBDBasisJSON)

Convert JSON data back to a TEBDBasis object.
"""
function _json_to_tebd_basis(json_data::TEBDBasisJSON)
    if json_data.type != "TEBDBasis"
        error("Unknown basis type: $(json_data.type)")
    end
    
    if json_data.version != "2.0"
        @warn "Basis version $(json_data.version) may not be fully compatible with current version 2.0"
    end
    
    m = json_data.m
    n = json_data.n
    n_row_gates = json_data.n_row_gates
    n_col_gates = json_data.n_col_gates
    phases = json_data.phases
    
    # Reconstruct tensors from serialized format
    optcode, template_tensors, _, _ = tebd_code(m, n; phases=phases)
    inverse_code, _, _, _ = tebd_code(m, n; phases=phases, inverse=true)
    
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
    loaded_basis = TEBDBasis(m, n, tensors, optcode, inverse_code, n_row_gates, n_col_gates, Float64.(phases))
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
    basis_to_dict(basis::EntangledQFTBasis) -> Dict

Convert an EntangledQFTBasis to a dictionary for custom serialization.

# Returns
- `Dict`: Dictionary representation of the basis
"""
function basis_to_dict(basis::EntangledQFTBasis)
    json_data = _basis_to_json(basis)
    return Dict(
        "type" => json_data.type,
        "version" => json_data.version,
        "m" => json_data.m,
        "n" => json_data.n,
        "n_entangle" => json_data.n_entangle,
        "entangle_phases" => json_data.entangle_phases,
        "tensors" => json_data.tensors,
        "hash" => json_data.hash
    )
end

"""
    basis_to_dict(basis::TEBDBasis) -> Dict

Convert a TEBDBasis to a dictionary for custom serialization.

# Returns
- `Dict`: Dictionary representation of the basis
"""
function basis_to_dict(basis::TEBDBasis)
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

"""
    dict_to_basis(d::Dict) -> AbstractSparseBasis

Convert a dictionary back to a basis.

# Arguments
- `d::Dict`: Dictionary with basis data

# Returns
- `AbstractSparseBasis`: The reconstructed basis
"""
function dict_to_basis(d::Dict)
    basis_type = get(d, "type", "QFTBasis")
    
    if basis_type == "EntangledQFTBasis"
        json_data = EntangledBasisJSON(
            d["type"],
            d["version"],
            d["m"],
            d["n"],
            d["n_entangle"],
            d["entangle_phases"],
            d["tensors"],
            d["hash"]
        )
        return _json_to_entangled_basis(json_data)
    elseif basis_type == "TEBDBasis"
        json_data = TEBDBasisJSON(
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
        return _json_to_tebd_basis(json_data)
    else
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
end

