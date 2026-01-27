# ============================================================================
# Sparse Basis Training
# ============================================================================
# This file provides training functionality to learn optimal basis parameters
# from a dataset of images.

# ============================================================================
# Generic Internal Training Core
# ============================================================================

"""
    _train_basis_core(dataset, optcode, inverse_code, initial_tensors, m, n, loss, 
                      epochs, steps_per_image, validation_split, shuffle, 
                      early_stopping_patience, verbose, basis_name; extra_info="")

Core training loop shared by all basis types. Returns (final_tensors, best_val_loss).
"""
function _train_basis_core(
    dataset::Vector{<:AbstractMatrix},
    optcode::OMEinsum.AbstractEinsum,
    inverse_code::OMEinsum.AbstractEinsum,
    initial_tensors::Vector,
    m::Int, n::Int,
    loss::AbstractLoss,
    epochs::Int,
    steps_per_image::Int,
    validation_split::Float64,
    shuffle::Bool,
    early_stopping_patience::Int,
    verbose::Bool,
    basis_name::String;
    extra_info::String = ""
)
    # Convert images to complex matrices
    complex_dataset = [Complex{Float64}.(img) for img in dataset]
    
    # Split into training and validation sets
    n_images = length(complex_dataset)
    n_validation = max(1, round(Int, n_images * validation_split))
    
    indices = shuffle ? Random.shuffle(1:n_images) : collect(1:n_images)
    validation_indices = indices[1:n_validation]
    training_indices = indices[n_validation+1:end]
    
    training_data = complex_dataset[training_indices]
    validation_data = complex_dataset[validation_indices]
    
    if verbose
        println("Training $basis_name:")
        println("  Image size: $(2^m)×$(2^n)")
        println("  Training images: $(length(training_data))")
        println("  Validation images: $(length(validation_data))")
        !isempty(extra_info) && println(extra_info)
        println("  Epochs: $epochs")
        println("  Steps per image: $steps_per_image")
    end
    
    # Initialize manifold
    M = generate_manifold(initial_tensors)
    current_theta = tensors2point(initial_tensors, M)
    
    # Track best parameters
    best_theta = current_theta
    best_val_loss = Inf
    patience_counter = 0
    
    # Training loop
    for epoch in 1:epochs
        verbose && println("\nEpoch $epoch/$epochs")
        
        # Shuffle training data each epoch
        shuffle && epoch > 1 && Random.shuffle!(training_data)
        
        epoch_losses = Float64[]
        
        # Train on each image
        for (idx, img_matrix) in enumerate(training_data)
            current_theta = _train_on_single_image(
                img_matrix, current_theta, M, optcode, inverse_code,
                m, n, loss, steps_per_image
            )
            
            tensors = point2tensors(current_theta, M)
            train_loss = loss_function(tensors, m, n, optcode, img_matrix, loss; inverse_code=inverse_code)
            push!(epoch_losses, train_loss)
            
            verbose && print("  [$idx/$(length(training_data))] loss: $(round(train_loss, digits=6))\r")
        end
        
        verbose && println()
        
        # Compute validation loss
        tensors = point2tensors(current_theta, M)
        val_loss = _compute_validation_loss(validation_data, tensors, optcode, inverse_code, m, n, loss)
        avg_train_loss = isempty(epoch_losses) ? Inf : sum(epoch_losses) / length(epoch_losses)
        
        if verbose
            println("  Avg train loss: $(round(avg_train_loss, digits=6))")
            println("  Validation loss: $(round(val_loss, digits=6))")
        end
        
        # Check for improvement
        if val_loss < best_val_loss
            improvement = best_val_loss == Inf ? 100.0 : (best_val_loss - val_loss) / best_val_loss * 100
            verbose && println("  ✓ Validation improved by $(round(improvement, digits=2))%")
            best_val_loss = val_loss
            best_theta = current_theta
            patience_counter = 0
        else
            patience_counter += 1
            verbose && println("  ✗ No improvement (patience: $patience_counter/$early_stopping_patience)")
            
            if patience_counter >= early_stopping_patience && epoch > 1
                verbose && println("\nEarly stopping: validation loss not improving")
                break
            end
        end
    end
    
    final_tensors = point2tensors(best_theta, M)
    verbose && println("\n✓ Training completed. Best validation loss: $(round(best_val_loss, digits=6))")
    
    return final_tensors, best_val_loss
end

# ============================================================================
# Public train_basis Methods
# ============================================================================

"""
    train_basis(::Type{QFTBasis}, dataset::Vector{<:AbstractMatrix}; kwargs...)

Train a QFTBasis on a dataset of images using Riemannian gradient descent.

# Arguments
- `::Type{QFTBasis}`: The basis type to train
- `dataset::Vector{<:AbstractMatrix}`: Training images (each must be 2^m × 2^n)

# Keyword Arguments
- `m::Int`: Number of qubits for rows (image height = 2^m)
- `n::Int`: Number of qubits for columns (image width = 2^n)
- `loss::AbstractLoss = MSELoss(k)`: Loss function for training
- `epochs::Int = 3`: Number of training epochs
- `steps_per_image::Int = 200`: Gradient descent steps per image
- `validation_split::Float64 = 0.2`: Fraction of data for validation
- `shuffle::Bool = true`: Whether to shuffle data each epoch
- `early_stopping_patience::Int = 2`: Epochs without improvement before stopping
- `verbose::Bool = true`: Whether to print training progress

# Returns
- `QFTBasis`: Trained basis with optimized parameters

# Example
```julia
images = [load_image(path) for path in image_paths]
k = round(Int, 64 * 64 * 0.1)  # Keep 10% of coefficients
basis = train_basis(QFTBasis, images; m=6, n=6, loss=MSELoss(k), epochs=5)
save_basis("trained_basis.json", basis)
```
"""
function train_basis(
    ::Type{QFTBasis},
    dataset::Vector{<:AbstractMatrix};
    m::Int, n::Int,
    loss::AbstractLoss = MSELoss(round(Int, 2^(m+n) * 0.1)),
    epochs::Int = 3,
    steps_per_image::Int = 200,
    validation_split::Float64 = 0.2,
    shuffle::Bool = true,
    early_stopping_patience::Int = 2,
    verbose::Bool = true
)
    @assert 0.0 <= validation_split < 1.0 "validation_split must be in [0, 1)"
    @assert length(dataset) > 0 "Dataset must not be empty"
    expected_size = (2^m, 2^n)
    for (i, img) in enumerate(dataset)
        @assert size(img) == expected_size "Image $i has size $(size(img)), expected $expected_size"
    end
    
    # Initialize circuit
    optcode, initial_tensors = qft_code(m, n)
    inverse_code, _ = qft_code(m, n; inverse=true)
    
    final_tensors, _ = _train_basis_core(
        dataset, optcode, inverse_code, initial_tensors, m, n, loss,
        epochs, steps_per_image, validation_split, shuffle,
        early_stopping_patience, verbose, "QFTBasis"
    )
    
    return QFTBasis(m, n, final_tensors, optcode, inverse_code)
end

"""
    train_basis(::Type{EntangledQFTBasis}, dataset::Vector{<:AbstractMatrix}; kwargs...)

Train an EntangledQFTBasis on a dataset of images using Riemannian gradient descent.

# Arguments
- `::Type{EntangledQFTBasis}`: The basis type to train
- `dataset::Vector{<:AbstractMatrix}`: Training images (each must be 2^m × 2^n)

# Keyword Arguments
- `m::Int`, `n::Int`: Number of qubits for rows/columns
- `entangle_phases::Union{Nothing, Vector{<:Real}}`: Initial phases (default: zeros)
- `loss`, `epochs`, `steps_per_image`, `validation_split`, `shuffle`, 
  `early_stopping_patience`, `verbose`: Same as QFTBasis

# Returns
- `EntangledQFTBasis`: Trained basis with optimized entanglement phases

# Example
```julia
k = round(Int, 64 * 64 * 0.1)
basis = train_basis(EntangledQFTBasis, images; m=6, n=6, loss=MSELoss(k), epochs=5)
phases = get_entangle_phases(basis)
```
"""
function train_basis(
    ::Type{EntangledQFTBasis},
    dataset::Vector{<:AbstractMatrix};
    m::Int, n::Int,
    entangle_phases::Union{Nothing, Vector{<:Real}} = nothing,
    loss::AbstractLoss = MSELoss(round(Int, 2^(m+n) * 0.1)),
    epochs::Int = 3,
    steps_per_image::Int = 200,
    validation_split::Float64 = 0.2,
    shuffle::Bool = true,
    early_stopping_patience::Int = 2,
    verbose::Bool = true
)
    @assert 0.0 <= validation_split < 1.0 "validation_split must be in [0, 1)"
    @assert length(dataset) > 0 "Dataset must not be empty"
    expected_size = (2^m, 2^n)
    for (i, img) in enumerate(dataset)
        @assert size(img) == expected_size "Image $i has size $(size(img)), expected $expected_size"
    end
    
    n_entangle = min(m, n)
    
    # Initialize circuit
    optcode, initial_tensors, _ = entangled_qft_code(m, n; entangle_phases=entangle_phases)
    inverse_code, _, _ = entangled_qft_code(m, n; entangle_phases=entangle_phases, inverse=true)
    
    final_tensors, _ = _train_basis_core(
        dataset, optcode, inverse_code, initial_tensors, m, n, loss,
        epochs, steps_per_image, validation_split, shuffle,
        early_stopping_patience, verbose, "EntangledQFTBasis";
        extra_info = "  Entanglement gates: $n_entangle"
    )
    
    # Extract trained phases
    entangle_indices = get_entangle_tensor_indices(final_tensors, n_entangle)
    trained_phases = if !isempty(entangle_indices)
        extract_entangle_phases(final_tensors, entangle_indices)
    else
        entangle_phases === nothing ? zeros(n_entangle) : Float64.(entangle_phases)
    end
    
    if verbose
        initial_str = entangle_phases === nothing ? "zeros" : "$(round.(entangle_phases, digits=4))"
        println("  Initial entanglement phases: $initial_str")
        println("  Trained entanglement phases: $(round.(trained_phases, digits=4))")
    end
    
    return EntangledQFTBasis(m, n, final_tensors, optcode, inverse_code, n_entangle, trained_phases)
end

"""
    train_basis(::Type{TEBDBasis}, dataset::Vector{<:AbstractMatrix}; kwargs...)

Train a TEBDBasis on a dataset of images using Riemannian gradient descent.

# Arguments
- `::Type{TEBDBasis}`: The basis type to train
- `dataset::Vector{<:AbstractMatrix}`: Training images (each must be 2^m × 2^n)

# Keyword Arguments
- `m::Int`: Number of row qubits (image height = 2^m)
- `n::Int`: Number of column qubits (image width = 2^n)
- `phases::Union{Nothing, Vector{<:Real}}`: Initial phases for TEBD gates (default: zeros)
  Length must be m+n for ring topology.
- `loss`, `epochs`, `steps_per_image`, `validation_split`, `shuffle`, 
  `early_stopping_patience`, `verbose`: Same as QFTBasis

# Returns
- `TEBDBasis`: Trained basis with optimized parameters

# Example
```julia
k = round(Int, 64 * 64 * 0.1)  # Keep 10% of coefficients
basis = train_basis(TEBDBasis, images; m=6, n=6, loss=MSELoss(k), epochs=5)
```
"""
function train_basis(
    ::Type{TEBDBasis},
    dataset::Vector{<:AbstractMatrix};
    m::Int, n::Int,
    phases::Union{Nothing, Vector{<:Real}} = nothing,
    loss::AbstractLoss = MSELoss(round(Int, 2^(m+n) * 0.1)),
    epochs::Int = 3,
    steps_per_image::Int = 200,
    validation_split::Float64 = 0.2,
    shuffle::Bool = true,
    early_stopping_patience::Int = 2,
    verbose::Bool = true
)
    @assert 0.0 <= validation_split < 1.0 "validation_split must be in [0, 1)"
    @assert length(dataset) > 0 "Dataset must not be empty"
    expected_size = (2^m, 2^n)
    for (i, img) in enumerate(dataset)
        @assert size(img) == expected_size "Image $i has size $(size(img)), expected $expected_size"
    end
    
    n_row_gates = m  # Row ring has m gates
    n_col_gates = n  # Col ring has n gates
    n_gates = n_row_gates + n_col_gates
    
    # Initialize circuit
    optcode, initial_tensors, _, _ = tebd_code(m, n; phases=phases)
    inverse_code, _, _, _ = tebd_code(m, n; phases=phases, inverse=true)
    
    final_tensors, _ = _train_basis_core(
        dataset, optcode, inverse_code, initial_tensors, m, n, loss,
        epochs, steps_per_image, validation_split, shuffle,
        early_stopping_patience, verbose, "TEBDBasis";
        extra_info = "  Row qubits: $m, Col qubits: $n\n  Row ring gates: $n_row_gates, Col ring gates: $n_col_gates"
    )
    
    # Extract trained phases
    gate_indices = get_tebd_gate_indices(final_tensors, n_gates)
    trained_phases = if !isempty(gate_indices)
        extract_tebd_phases(final_tensors, gate_indices)
    else
        phases === nothing ? zeros(n_gates) : Float64.(phases)
    end
    
    if verbose
        initial_str = phases === nothing ? "zeros" : "$(round.(phases, digits=4))"
        println("  Initial phases: $initial_str")
        println("  Trained phases: $(round.(trained_phases, digits=4))")
    end
    
    return TEBDBasis(m, n, final_tensors, optcode, inverse_code, n_row_gates, n_col_gates, trained_phases)
end

# ============================================================================
# Internal Helper Functions
# ============================================================================

"""
    _train_on_single_image(img_matrix, theta, M, optcode, inverse_code, m, n, loss, steps)

Train on a single image using gradient descent.
"""
function _train_on_single_image(
    img_matrix::AbstractMatrix,
    theta,
    M::ProductManifold,
    optcode::OMEinsum.AbstractEinsum,
    inverse_code::OMEinsum.AbstractEinsum,
    m::Int, n::Int,
    loss::AbstractLoss,
    steps::Int
)
    f(M, p) = loss_function(point2tensors(p, M), m, n, optcode, img_matrix, loss; inverse_code=inverse_code)
    grad_f(M, p) = ManifoldDiff.gradient(M, x -> f(M, x), p, RiemannianProjectionBackend(AutoZygote()))
    
    return gradient_descent(
        M, f, grad_f, theta;
        debug = [],
        stopping_criterion = StopAfterIteration(steps) | StopWhenGradientNormLess(1e-5)
    )
end

"""
    _compute_validation_loss(validation_data, tensors, optcode, inverse_code, m, n, loss)

Compute average loss over validation set.
"""
function _compute_validation_loss(
    validation_data::Vector{<:AbstractMatrix},
    tensors::Vector,
    optcode::OMEinsum.AbstractEinsum,
    inverse_code::OMEinsum.AbstractEinsum,
    m::Int, n::Int,
    loss::AbstractLoss
)
    isempty(validation_data) && return Inf
    total = sum(loss_function(tensors, m, n, optcode, img, loss; inverse_code=inverse_code) 
                for img in validation_data)
    return total / length(validation_data)
end

# ============================================================================
# Convenience Functions
# ============================================================================

"""
    train_basis_from_files(::Type{QFTBasis}, image_paths::Vector{String}; 
                          target_size::Int, kwargs...)

Train a QFTBasis from image files, automatically resizing to power-of-2 dimensions.

# Arguments
- `::Type{QFTBasis}`: The basis type to train
- `image_paths::Vector{String}`: Paths to image files

# Keyword Arguments
- `target_size::Int`: Target size (must be power of 2, e.g., 64, 128, 256, 512)
- `kwargs...`: Additional arguments passed to `train_basis`

# Returns
- `QFTBasis`: Trained basis

# Note
This function requires the Images.jl package to be loaded.
"""
function train_basis_from_files end  # Defined in extension or requires Images.jl
