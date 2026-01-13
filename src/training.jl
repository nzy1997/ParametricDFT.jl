# ============================================================================
# Sparse Basis Training
# ============================================================================
# This file provides training functionality to learn optimal basis parameters
# from a dataset of images.

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
# Load images as matrices
images = [load_image(path) for path in image_paths]

# Train basis for 64×64 images with 90% compression
k = round(Int, 64 * 64 * 0.1)  # Keep 10% of coefficients
basis = train_basis(QFTBasis, images; m=6, n=6, loss=MSELoss(k), epochs=5)

# Save the trained basis
save_basis("trained_basis.json", basis)
```
"""
function train_basis(
    ::Type{QFTBasis},
    dataset::Vector{<:AbstractMatrix};
    m::Int,
    n::Int,
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
    
    # Convert images to complex matrices
    complex_dataset = [Complex{Float64}.(img) for img in dataset]
    
    # Split into training and validation sets
    n_images = length(complex_dataset)
    n_validation = max(1, round(Int, n_images * validation_split))
    
    if shuffle
        indices = Random.shuffle(1:n_images)
    else
        indices = collect(1:n_images)
    end
    
    validation_indices = indices[1:n_validation]
    training_indices = indices[n_validation+1:end]
    
    training_data = complex_dataset[training_indices]
    validation_data = complex_dataset[validation_indices]
    
    if verbose
        println("Training QFTBasis:")
        println("  Image size: $(2^m)×$(2^n)")
        println("  Training images: $(length(training_data))")
        println("  Validation images: $(length(validation_data))")
        println("  Epochs: $epochs")
        println("  Steps per image: $steps_per_image")
    end
    
    # Initialize basis and get manifold
    optcode, initial_tensors = qft_code(m, n)
    inverse_code, _ = qft_code(m, n; inverse=true)
    M = generate_manifold(initial_tensors)
    current_theta = tensors2point(initial_tensors, M)
    
    # Track best parameters
    best_theta = current_theta
    best_val_loss = Inf
    patience_counter = 0
    
    # Training loop
    for epoch in 1:epochs
        if verbose
            println("\nEpoch $epoch/$epochs")
        end
        
        # Shuffle training data each epoch
        if shuffle && epoch > 1
            Random.shuffle!(training_data)
        end
        
        epoch_losses = Float64[]
        
        # Train on each image
        for (idx, img_matrix) in enumerate(training_data)
            # Train on this image
            current_theta = _train_on_single_image(
                img_matrix, current_theta, M, optcode, inverse_code,
                m, n, loss, steps_per_image
            )
            
            # Compute training loss
            tensors = point2tensors(current_theta, M)
            train_loss = _compute_basis_loss(img_matrix, tensors, optcode, inverse_code, m, n, loss)
            push!(epoch_losses, train_loss)
            
            if verbose
                print("  [$idx/$(length(training_data))] loss: $(round(train_loss, digits=6))\r")
            end
        end
        
        if verbose
            println()  # New line after progress
        end
        
        # Compute validation loss
        tensors = point2tensors(current_theta, M)
        val_loss = _compute_validation_loss(validation_data, tensors, optcode, inverse_code, m, n, loss)
        avg_train_loss = length(epoch_losses) > 0 ? sum(epoch_losses) / length(epoch_losses) : Inf
        
        if verbose
            println("  Avg train loss: $(round(avg_train_loss, digits=6))")
            println("  Validation loss: $(round(val_loss, digits=6))")
        end
        
        # Check for improvement
        if val_loss < best_val_loss
            improvement = best_val_loss == Inf ? 100.0 : (best_val_loss - val_loss) / best_val_loss * 100
            if verbose
                println("  ✓ Validation improved by $(round(improvement, digits=2))%")
            end
            best_val_loss = val_loss
            best_theta = current_theta
            patience_counter = 0
        else
            patience_counter += 1
            if verbose
                println("  ✗ No improvement (patience: $patience_counter/$early_stopping_patience)")
            end
            
            if patience_counter >= early_stopping_patience && epoch > 1
                if verbose
                    println("\nEarly stopping: validation loss not improving")
                end
                break
            end
        end
    end
    
    # Construct final basis with best parameters
    final_tensors = point2tensors(best_theta, M)
    
    if verbose
        println("\n✓ Training completed. Best validation loss: $(round(best_val_loss, digits=6))")
    end
    
    return QFTBasis(m, n, final_tensors, optcode, inverse_code)
end

"""
    train_basis(::Type{EntangledQFTBasis}, dataset::Vector{<:AbstractMatrix}; kwargs...)

Train an EntangledQFTBasis on a dataset of images using Riemannian gradient descent.

The EntangledQFTBasis extends the standard QFT by adding entanglement gates E_k
between corresponding row and column qubits. Each entanglement gate has a learnable
phase parameter phi_k.

# Arguments
- `::Type{EntangledQFTBasis}`: The basis type to train
- `dataset::Vector{<:AbstractMatrix}`: Training images (each must be 2^m × 2^n)

# Keyword Arguments
- `m::Int`: Number of qubits for rows (image height = 2^m)
- `n::Int`: Number of qubits for columns (image width = 2^n)
- `entangle_phases::Union{Nothing, Vector{<:Real}}`: Initial phases for entanglement gates.
  If nothing, defaults to zeros.
- `loss::AbstractLoss = MSELoss(k)`: Loss function for training
- `epochs::Int = 3`: Number of training epochs
- `steps_per_image::Int = 200`: Gradient descent steps per image
- `validation_split::Float64 = 0.2`: Fraction of data for validation
- `shuffle::Bool = true`: Whether to shuffle data each epoch
- `early_stopping_patience::Int = 2`: Epochs without improvement before stopping
- `verbose::Bool = true`: Whether to print training progress

# Returns
- `EntangledQFTBasis`: Trained basis with optimized parameters including entanglement phases

# Example
```julia
# Load images as matrices
images = [load_image(path) for path in image_paths]

# Train entangled basis for 64×64 images with 90% compression
k = round(Int, 64 * 64 * 0.1)  # Keep 10% of coefficients
basis = train_basis(EntangledQFTBasis, images; m=6, n=6, loss=MSELoss(k), epochs=5)

# The trained basis will have optimized entanglement phases
phases = get_entangle_phases(basis)

# Save the trained basis
save_basis("entangled_basis.json", basis)
```
"""
function train_basis(
    ::Type{EntangledQFTBasis},
    dataset::Vector{<:AbstractMatrix};
    m::Int,
    n::Int,
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
    
    # Convert images to complex matrices
    complex_dataset = [Complex{Float64}.(img) for img in dataset]
    
    # Split into training and validation sets
    n_images = length(complex_dataset)
    n_validation = max(1, round(Int, n_images * validation_split))
    
    if shuffle
        indices = Random.shuffle(1:n_images)
    else
        indices = collect(1:n_images)
    end
    
    validation_indices = indices[1:n_validation]
    training_indices = indices[n_validation+1:end]
    
    training_data = complex_dataset[training_indices]
    validation_data = complex_dataset[validation_indices]
    
    n_entangle = min(m, n)
    
    if verbose
        println("Training EntangledQFTBasis:")
        println("  Image size: $(2^m)×$(2^n)")
        println("  Training images: $(length(training_data))")
        println("  Validation images: $(length(validation_data))")
        println("  Entanglement gates: $n_entangle")
        println("  Epochs: $epochs")
        println("  Steps per image: $steps_per_image")
    end
    
    # Initialize basis and get manifold
    optcode, initial_tensors, _ = entangled_qft_code(m, n; entangle_phases=entangle_phases)
    inverse_code, _, _ = entangled_qft_code(m, n; entangle_phases=entangle_phases, inverse=true)
    M = generate_manifold(initial_tensors)
    current_theta = tensors2point(initial_tensors, M)
    
    # Track best parameters
    best_theta = current_theta
    best_val_loss = Inf
    patience_counter = 0
    
    # Training loop
    for epoch in 1:epochs
        if verbose
            println("\nEpoch $epoch/$epochs")
        end
        
        # Shuffle training data each epoch
        if shuffle && epoch > 1
            Random.shuffle!(training_data)
        end
        
        epoch_losses = Float64[]
        
        # Train on each image
        for (idx, img_matrix) in enumerate(training_data)
            # Train on this image
            current_theta = _train_on_single_image(
                img_matrix, current_theta, M, optcode, inverse_code,
                m, n, loss, steps_per_image
            )
            
            # Compute training loss
            tensors = point2tensors(current_theta, M)
            train_loss = _compute_basis_loss(img_matrix, tensors, optcode, inverse_code, m, n, loss)
            push!(epoch_losses, train_loss)
            
            if verbose
                print("  [$idx/$(length(training_data))] loss: $(round(train_loss, digits=6))\r")
            end
        end
        
        if verbose
            println()  # New line after progress
        end
        
        # Compute validation loss
        tensors = point2tensors(current_theta, M)
        val_loss = _compute_validation_loss(validation_data, tensors, optcode, inverse_code, m, n, loss)
        avg_train_loss = length(epoch_losses) > 0 ? sum(epoch_losses) / length(epoch_losses) : Inf
        
        if verbose
            println("  Avg train loss: $(round(avg_train_loss, digits=6))")
            println("  Validation loss: $(round(val_loss, digits=6))")
        end
        
        # Check for improvement
        if val_loss < best_val_loss
            improvement = best_val_loss == Inf ? 100.0 : (best_val_loss - val_loss) / best_val_loss * 100
            if verbose
                println("  ✓ Validation improved by $(round(improvement, digits=2))%")
            end
            best_val_loss = val_loss
            best_theta = current_theta
            patience_counter = 0
        else
            patience_counter += 1
            if verbose
                println("  ✗ No improvement (patience: $patience_counter/$early_stopping_patience)")
            end
            
            if patience_counter >= early_stopping_patience && epoch > 1
                if verbose
                    println("\nEarly stopping: validation loss not improving")
                end
                break
            end
        end
    end
    
    # Construct final basis with best parameters
    final_tensors = point2tensors(best_theta, M)
    
    # Extract the effective entanglement phases from the trained tensors.
    # During training, all tensor elements become free parameters, so we need to
    # extract the phase angles from the (2,2) elements of the entanglement gate tensors.
    entangle_indices = get_entangle_tensor_indices(final_tensors, n_entangle)
    trained_phases = if !isempty(entangle_indices)
        extract_entangle_phases(final_tensors, entangle_indices)
    else
        # Fallback to initial phases if extraction fails
        entangle_phases === nothing ? zeros(n_entangle) : Float64.(entangle_phases)
    end
    
    if verbose
        println("\n✓ Training completed. Best validation loss: $(round(best_val_loss, digits=6))")
        initial_phases_str = entangle_phases === nothing ? "zeros" : "$(round.(entangle_phases, digits=4))"
        println("  Initial entanglement phases: $initial_phases_str")
        println("  Trained entanglement phases: $(round.(trained_phases, digits=4))")
    end
    
    return EntangledQFTBasis(m, n, final_tensors, optcode, inverse_code, n_entangle, trained_phases)
end

# ============================================================================
# Internal Training Functions
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
    m::Int,
    n::Int,
    loss::AbstractLoss,
    steps::Int
)
    # Define loss function for this image
    f(M, p) = begin
        tensors = point2tensors(p, M)
        return loss_function(tensors, m, n, optcode, img_matrix, loss; inverse_code=inverse_code)
    end
    
    # Define gradient function
    grad_f(M, p) = ManifoldDiff.gradient(
        M, x -> f(M, x), p,
        RiemannianProjectionBackend(AutoZygote())
    )
    
    # Run gradient descent
    result = gradient_descent(
        M, f, grad_f, theta;
        debug = [],
        stopping_criterion = StopAfterIteration(steps) | StopWhenGradientNormLess(1e-5)
    )
    
    return result
end

"""
    _compute_basis_loss(img_matrix, tensors, optcode, inverse_code, m, n, loss)

Compute loss for a single image given current tensors.
"""
function _compute_basis_loss(
    img_matrix::AbstractMatrix,
    tensors::Vector,
    optcode::OMEinsum.AbstractEinsum,
    inverse_code::OMEinsum.AbstractEinsum,
    m::Int,
    n::Int,
    loss::AbstractLoss
)
    return loss_function(tensors, m, n, optcode, img_matrix, loss; inverse_code=inverse_code)
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
    m::Int,
    n::Int,
    loss::AbstractLoss
)
    if length(validation_data) == 0
        return Inf
    end
    
    total_loss = 0.0
    for img_matrix in validation_data
        total_loss += _compute_basis_loss(img_matrix, tensors, optcode, inverse_code, m, n, loss)
    end
    
    return total_loss / length(validation_data)
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

