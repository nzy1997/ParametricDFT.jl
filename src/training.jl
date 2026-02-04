# ============================================================================
# Sparse Basis Training
# ============================================================================
# This file provides training functionality to learn optimal basis parameters
# from a dataset of images.

# ============================================================================
# Device Abstraction
# ============================================================================

"""
    to_device(x, device::Symbol)

Move array `x` to the specified device. Returns `x` unchanged for `:cpu`.
For `:gpu`, requires CUDA.jl to be loaded (via the CUDAExt extension).
"""
to_device(x, ::Val{:cpu}) = x

function to_device(x, device::Symbol)
    if device === :cpu
        return to_device(x, Val(:cpu))
    elseif device === :gpu
        # Check if the CUDAExt has added the Val{:gpu} method
        if hasmethod(to_device, Tuple{typeof(x), Val{:gpu}})
            return to_device(x, Val(:gpu))
        else
            error("GPU support requires CUDA.jl. Install and load CUDA.jl first: `using CUDA`")
        end
    else
        error("Unknown device: $device. Supported: :cpu, :gpu")
    end
end

# ============================================================================
# Generic Internal Training Core
# ============================================================================

"""
    _train_basis_core(dataset, optcode, inverse_code, initial_tensors, m, n, loss,
                      epochs, steps_per_image, validation_split, shuffle,
                      early_stopping_patience, verbose, basis_name; extra_info="",
                      save_loss_path=nothing, optimizer=:gradient_descent,
                      batch_size=1, device=:cpu)

Core training loop shared by all basis types. Returns (final_tensors, best_val_loss, train_losses, val_losses, step_train_losses).

If `save_loss_path` is provided, saves per-step training losses to a JSON file with
`epoch`, `step`, and `loss` fields for each entry.

# Supported optimizers
- `:gradient_descent` (default): Riemannian gradient descent
- `:conjugate_gradient`: Riemannian conjugate gradient descent
- `:quasi_newton`: Riemannian L-BFGS quasi-Newton method

# Batch and Device Support
- `batch_size::Int = 1`: Number of images per batch. When > 1, the optimizer minimizes the
  average loss over the batch instead of a single image. batch_size=1 recovers original behavior.
- `device::Symbol = :cpu`: Device for computation. Use `:gpu` for GPU acceleration (requires CUDA.jl).
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
    extra_info::String = "",
    save_loss_path::Union{Nothing, String} = nothing,
    optimizer::Symbol = :gradient_descent,
    batch_size::Int = 1,
    device::Symbol = :cpu
)
    # Convert images to complex matrices and move to device
    complex_dataset = [to_device(Complex{Float64}.(img), device) for img in dataset]

    # Split into training and validation sets
    n_images = length(complex_dataset)
    n_validation = max(1, round(Int, n_images * validation_split))

    indices = shuffle ? Random.shuffle(1:n_images) : collect(1:n_images)
    validation_indices = indices[1:n_validation]
    training_indices = indices[n_validation+1:end]

    training_data = complex_dataset[training_indices]
    validation_data = complex_dataset[validation_indices]

    # Clamp batch_size to valid range
    batch_size = clamp(batch_size, 1, length(training_data))
    n_batches = ceil(Int, length(training_data) / batch_size)

    if verbose
        optimizer_names = Dict(:gradient_descent => "Riemannian Gradient Descent",
                               :conjugate_gradient => "Riemannian Conjugate Gradient",
                               :quasi_newton => "Riemannian L-BFGS")
        device_names = Dict(:cpu => "CPU", :gpu => "GPU (CUDA)")
        println("Training $basis_name:")
        println("  Image size: $(2^m)×$(2^n)")
        println("  Training images: $(length(training_data))")
        println("  Validation images: $(length(validation_data))")
        !isempty(extra_info) && println(extra_info)
        println("  Optimizer: $(get(optimizer_names, optimizer, string(optimizer)))")
        println("  Device: $(get(device_names, device, string(device)))")
        println("  Batch size: $batch_size ($(n_batches) batches per epoch)")
        println("  Epochs: $epochs")
        println("  Steps per batch: $steps_per_image")
    end

    # Move initial tensors to device for optimization
    device_tensors = [to_device(t, device) for t in initial_tensors]

    # Initialize manifold
    M = generate_manifold(device_tensors)
    current_theta = tensors2point(device_tensors, M)

    # Track best parameters
    best_theta = current_theta
    best_val_loss = Inf
    patience_counter = 0

    # Track training history
    train_losses = Float64[]
    val_losses = Float64[]
    step_train_losses = Float64[]

    # Per-step loss records for JSON export (epoch, step, loss)
    loss_records = Dict{String, Any}[]

    # Training loop
    for epoch in 1:epochs
        verbose && println("\nEpoch $epoch/$epochs")

        # Shuffle training data each epoch
        shuffle && epoch > 1 && Random.shuffle!(training_data)

        epoch_losses = Float64[]

        # Train on batches
        for batch_idx in 1:n_batches
            start_idx = (batch_idx - 1) * batch_size + 1
            end_idx = min(batch_idx * batch_size, length(training_data))
            batch = training_data[start_idx:end_idx]

            current_theta = _train_on_batch(
                batch, current_theta, M, optcode, inverse_code,
                m, n, loss, steps_per_image; optimizer=optimizer
            )

            # Compute average loss over batch for tracking
            tensors = point2tensors(current_theta, M)
            batch_loss = sum(
                loss_function(tensors, m, n, optcode, img, loss; inverse_code=inverse_code)
                for img in batch
            ) / length(batch)
            push!(epoch_losses, batch_loss)
            push!(step_train_losses, batch_loss)
            push!(loss_records, Dict{String, Any}("epoch" => epoch, "step" => batch_idx, "loss" => batch_loss))

            verbose && print("  [batch $batch_idx/$n_batches] loss: $(round(batch_loss, digits=6))\r")
        end

        verbose && println()

        # Compute validation loss
        tensors = point2tensors(current_theta, M)
        val_loss = _compute_validation_loss(validation_data, tensors, optcode, inverse_code, m, n, loss)
        avg_train_loss = isempty(epoch_losses) ? Inf : sum(epoch_losses) / length(epoch_losses)

        # Store losses for visualization
        push!(train_losses, avg_train_loss)
        push!(val_losses, val_loss)

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

    # Move final tensors back to CPU for serialization
    final_tensors_raw = point2tensors(best_theta, M)
    final_tensors = [Array(t) for t in final_tensors_raw]
    verbose && println("\n✓ Training completed. Best validation loss: $(round(best_val_loss, digits=6))")

    # Save loss history to JSON if path provided
    if save_loss_path !== nothing
        _save_loss_history(save_loss_path, basis_name, loss_records, train_losses, val_losses)
        verbose && println("  Loss history saved to: $save_loss_path")
    end

    return final_tensors, best_val_loss, train_losses, val_losses, step_train_losses
end

# ============================================================================
# Public train_basis Methods
# ============================================================================

"""
    train_basis(::Type{QFTBasis}, dataset::Vector{<:AbstractMatrix}; kwargs...)

Train a QFTBasis on a dataset of images using Riemannian optimization.

# Arguments
- `::Type{QFTBasis}`: The basis type to train
- `dataset::Vector{<:AbstractMatrix}`: Training images (each must be 2^m × 2^n)

# Keyword Arguments
- `m::Int`: Number of qubits for rows (image height = 2^m)
- `n::Int`: Number of qubits for columns (image width = 2^n)
- `loss::AbstractLoss = MSELoss(k)`: Loss function for training
- `epochs::Int = 3`: Number of training epochs
- `steps_per_image::Int = 200`: Optimization steps per batch
- `validation_split::Float64 = 0.2`: Fraction of data for validation
- `shuffle::Bool = true`: Whether to shuffle data each epoch
- `early_stopping_patience::Int = 2`: Epochs without improvement before stopping
- `verbose::Bool = true`: Whether to print training progress
- `save_loss_path::Union{Nothing, String} = nothing`: Path to save training loss history as JSON
- `optimizer::Symbol = :gradient_descent`: Riemannian optimizer to use.
  Options: `:gradient_descent`, `:conjugate_gradient`, `:quasi_newton`
- `batch_size::Int = 1`: Number of images per batch. The optimizer minimizes average loss over
  the batch. batch_size=1 is equivalent to single-image training.
- `device::Symbol = :cpu`: Computation device. Use `:gpu` for CUDA acceleration (requires CUDA.jl).

# Returns
- `Tuple{QFTBasis, NamedTuple}`: Trained basis and training history
  - `basis`: Trained QFTBasis with optimized parameters
  - `history`: NamedTuple with fields `train_losses`, `val_losses`, `step_train_losses`, `basis_name`

# Example
```julia
images = [load_image(path) for path in image_paths]
k = round(Int, 64 * 64 * 0.1)  # Keep 10% of coefficients
basis, history = train_basis(QFTBasis, images; m=6, n=6, loss=MSELoss(k), epochs=5,
                             optimizer=:conjugate_gradient, batch_size=8,
                             save_loss_path="qft_loss.json")
save_basis("trained_basis.json", basis)
println("Final training loss: ", history.train_losses[end])
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
    verbose::Bool = true,
    save_loss_path::Union{Nothing, String} = nothing,
    optimizer::Symbol = :gradient_descent,
    batch_size::Int = 1,
    device::Symbol = :cpu
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

    final_tensors, _, train_losses, val_losses, step_train_losses = _train_basis_core(
        dataset, optcode, inverse_code, initial_tensors, m, n, loss,
        epochs, steps_per_image, validation_split, shuffle,
        early_stopping_patience, verbose, "QFTBasis";
        save_loss_path=save_loss_path, optimizer=optimizer,
        batch_size=batch_size, device=device
    )

    trained_basis = QFTBasis(m, n, final_tensors, optcode, inverse_code)
    history = (train_losses=train_losses, val_losses=val_losses, step_train_losses=step_train_losses, basis_name="QFT")

    return trained_basis, history
end

"""
    train_basis(::Type{EntangledQFTBasis}, dataset::Vector{<:AbstractMatrix}; kwargs...)

Train an EntangledQFTBasis on a dataset of images using Riemannian optimization.

# Arguments
- `::Type{EntangledQFTBasis}`: The basis type to train
- `dataset::Vector{<:AbstractMatrix}`: Training images (each must be 2^m × 2^n)

# Keyword Arguments
- `m::Int`, `n::Int`: Number of qubits for rows/columns
- `entangle_phases::Union{Nothing, Vector{<:Real}}`: Initial phases (default: zeros)
- `loss`, `epochs`, `steps_per_image`, `validation_split`, `shuffle`,
  `early_stopping_patience`, `verbose`: Same as QFTBasis
- `save_loss_path::Union{Nothing, String} = nothing`: Path to save training loss history as JSON
- `optimizer::Symbol = :gradient_descent`: Riemannian optimizer to use.
  Options: `:gradient_descent`, `:conjugate_gradient`, `:quasi_newton`
- `batch_size::Int = 1`: Number of images per batch for training
- `device::Symbol = :cpu`: Computation device (`:cpu` or `:gpu`)

# Returns
- `Tuple{EntangledQFTBasis, NamedTuple}`: Trained basis and training history
  - `basis`: Trained EntangledQFTBasis with optimized entanglement phases
  - `history`: NamedTuple with fields `train_losses`, `val_losses`, `step_train_losses`, `basis_name`

# Example
```julia
k = round(Int, 64 * 64 * 0.1)
basis, history = train_basis(EntangledQFTBasis, images; m=6, n=6, loss=MSELoss(k), epochs=5,
                             optimizer=:conjugate_gradient, batch_size=8)
phases = get_entangle_phases(basis)
println("Training loss per epoch: ", history.train_losses)
```
"""
function train_basis(
    ::Type{EntangledQFTBasis},
    dataset::Vector{<:AbstractMatrix};
    m::Int, n::Int,
    entangle_phases::Union{Nothing, Vector{<:Real}} = nothing,
    entangle_position::Symbol = :back,
    loss::AbstractLoss = MSELoss(round(Int, 2^(m+n) * 0.1)),
    epochs::Int = 3,
    steps_per_image::Int = 200,
    validation_split::Float64 = 0.2,
    shuffle::Bool = true,
    early_stopping_patience::Int = 2,
    verbose::Bool = true,
    save_loss_path::Union{Nothing, String} = nothing,
    optimizer::Symbol = :gradient_descent,
    batch_size::Int = 1,
    device::Symbol = :cpu
)
    @assert 0.0 <= validation_split < 1.0 "validation_split must be in [0, 1)"
    @assert length(dataset) > 0 "Dataset must not be empty"
    expected_size = (2^m, 2^n)
    for (i, img) in enumerate(dataset)
        @assert size(img) == expected_size "Image $i has size $(size(img)), expected $expected_size"
    end
    
    n_entangle = min(m, n)
    
    # Initialize circuit
    optcode, initial_tensors, _ = entangled_qft_code(m, n; entangle_phases=entangle_phases, entangle_position=entangle_position)
    inverse_code, _, _ = entangled_qft_code(m, n; entangle_phases=entangle_phases, inverse=true, entangle_position=entangle_position)
    
    final_tensors, _, train_losses, val_losses, step_train_losses = _train_basis_core(
        dataset, optcode, inverse_code, initial_tensors, m, n, loss,
        epochs, steps_per_image, validation_split, shuffle,
        early_stopping_patience, verbose, "EntangledQFTBasis";
        extra_info = "  Entanglement gates: $n_entangle",
        save_loss_path=save_loss_path, optimizer=optimizer,
        batch_size=batch_size, device=device
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

    trained_basis = EntangledQFTBasis(m, n, final_tensors, optcode, inverse_code, n_entangle, trained_phases, entangle_position)
    history = (train_losses=train_losses, val_losses=val_losses, step_train_losses=step_train_losses, basis_name="Entangled QFT")

    return trained_basis, history
end

"""
    train_basis(::Type{TEBDBasis}, dataset::Vector{<:AbstractMatrix}; kwargs...)

Train a TEBDBasis on a dataset of images using Riemannian optimization.

# Arguments
- `::Type{TEBDBasis}`: The basis type to train
- `dataset::Vector{<:AbstractMatrix}`: Training images (each must be 2^m × 2^n)

# Keyword Arguments
- `m::Int`: Number of row qubits (image height = 2^m)
- `n::Int`: Number of column qubits (image width = 2^n)
- `phases::Union{Nothing, Vector{<:Real}}`: Initial phases for TEBD gates.
  If nothing, uses small random values (randn * 0.1) to break symmetry.
  Length must be m+n for ring topology.
- `loss`, `epochs`, `steps_per_image`, `validation_split`, `shuffle`,
  `early_stopping_patience`, `verbose`: Same as QFTBasis
- `save_loss_path::Union{Nothing, String} = nothing`: Path to save training loss history as JSON
- `optimizer::Symbol = :gradient_descent`: Riemannian optimizer to use.
  Options: `:gradient_descent`, `:conjugate_gradient`, `:quasi_newton`
- `batch_size::Int = 1`: Number of images per batch for training
- `device::Symbol = :cpu`: Computation device (`:cpu` or `:gpu`)

# Returns
- `Tuple{TEBDBasis, NamedTuple}`: Trained basis and training history
  - `basis`: Trained TEBDBasis with optimized parameters
  - `history`: NamedTuple with fields `train_losses`, `val_losses`, `step_train_losses`, `basis_name`

# Example
```julia
k = round(Int, 64 * 64 * 0.1)  # Keep 10% of coefficients
basis, history = train_basis(TEBDBasis, images; m=6, n=6, loss=MSELoss(k), epochs=5,
                             optimizer=:quasi_newton, batch_size=8, device=:gpu)
println("Validation loss per epoch: ", history.val_losses)
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
    verbose::Bool = true,
    save_loss_path::Union{Nothing, String} = nothing,
    optimizer::Symbol = :gradient_descent,
    batch_size::Int = 1,
    device::Symbol = :cpu
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
    
    # Initialize phases: use small random values if not provided
    # Zero phases create a symmetric point where gradients are zero,
    # preventing the optimizer from learning. Small random values break symmetry.
    if phases === nothing
        phases = randn(n_gates) * 0.1
    end
    
    # Initialize circuit
    optcode, initial_tensors, _, _ = tebd_code(m, n; phases=phases)
    inverse_code, _, _, _ = tebd_code(m, n; phases=phases, inverse=true)
    
    final_tensors, _, train_losses, val_losses, step_train_losses = _train_basis_core(
        dataset, optcode, inverse_code, initial_tensors, m, n, loss,
        epochs, steps_per_image, validation_split, shuffle,
        early_stopping_patience, verbose, "TEBDBasis";
        extra_info = "  Row qubits: $m, Col qubits: $n\n  Row ring gates: $n_row_gates, Col ring gates: $n_col_gates",
        save_loss_path=save_loss_path, optimizer=optimizer,
        batch_size=batch_size, device=device
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

    trained_basis = TEBDBasis(m, n, final_tensors, optcode, inverse_code, n_row_gates, n_col_gates, trained_phases)
    history = (train_losses=train_losses, val_losses=val_losses, step_train_losses=step_train_losses, basis_name="TEBD")

    return trained_basis, history
end

# ============================================================================
# Internal Helper Functions
# ============================================================================

"""
    _save_loss_history(path, basis_name, loss_records, train_losses, val_losses)

Internal function to save training loss history to a JSON file.
"""
function _save_loss_history(
    path::String,
    basis_name::String,
    loss_records::Vector{Dict{String, Any}},
    train_losses::Vector{Float64},
    val_losses::Vector{Float64}
)
    mkpath(dirname(path))
    json_data = Dict{String, Any}(
        "basis_name" => basis_name,
        "epoch_losses" => [
            Dict{String, Any}("epoch" => i, "train_loss" => train_losses[i], "val_loss" => val_losses[i])
            for i in eachindex(train_losses)
        ],
        "step_losses" => loss_records
    )
    open(path, "w") do io
        JSON3.pretty(io, json_data)
    end
end

"""
    save_loss_history(path::String, history::NamedTuple)

Save training loss history returned by `train_basis` to a JSON file.

The JSON contains:
- `basis_name`: Name of the trained basis
- `epoch_losses`: Array of `{epoch, train_loss, val_loss}` per epoch
- `step_losses`: Array of `{epoch, step, loss}` per training step

# Arguments
- `path::String`: Output file path (`.json`)
- `history::NamedTuple`: Training history returned by `train_basis`

# Example
```julia
basis, history = train_basis(QFTBasis, images; m=5, n=5, epochs=3)
save_loss_history("training_loss.json", history)
```
"""
function save_loss_history(path::String, history::NamedTuple)
    # Reconstruct step loss records with epoch/step info from step_train_losses
    # We need to infer epoch boundaries from the epoch-level train_losses
    n_epochs = length(history.train_losses)
    n_steps = length(history.step_train_losses)
    steps_per_epoch = n_epochs > 0 ? n_steps ÷ n_epochs : n_steps

    loss_records = Dict{String, Any}[]
    for (global_step, loss_val) in enumerate(history.step_train_losses)
        epoch = (global_step - 1) ÷ steps_per_epoch + 1
        step = (global_step - 1) % steps_per_epoch + 1
        push!(loss_records, Dict{String, Any}("epoch" => epoch, "step" => step, "loss" => loss_val))
    end

    _save_loss_history(path, history.basis_name, loss_records, history.train_losses, history.val_losses)
end

"""
    load_loss_history(path::String) -> TrainingHistory

Load training loss history from a JSON file (saved by `save_loss_history` or `save_loss_path`)
and return a `TrainingHistory` object that can be passed directly to plotting functions.

# Arguments
- `path::String`: Path to the JSON file

# Returns
- `TrainingHistory`: Object with `train_losses`, `val_losses`, `step_train_losses`, and `basis_name`

# Example
```julia
using ParametricDFT
history = load_loss_history("training_loss.json")
fig = plot_training_loss(history)
save("loss_curve.png", fig)
```
"""
function load_loss_history(path::String)
    json_str = read(path, String)
    json_data = JSON3.read(json_str)

    basis_name = String(json_data[:basis_name])

    train_losses = Float64[entry[:train_loss] for entry in json_data[:epoch_losses]]
    val_losses = Float64[entry[:val_loss] for entry in json_data[:epoch_losses]]
    step_train_losses = Float64[entry[:loss] for entry in json_data[:step_losses]]

    return TrainingHistory(train_losses, val_losses, step_train_losses, basis_name)
end

"""
    _train_on_batch(batch, theta, M, optcode, inverse_code, m, n, loss, steps; optimizer=:gradient_descent)

Train on a batch of images using the specified Riemannian optimizer.
The optimizer minimizes the average loss over all images in the batch.

When `batch` contains a single image, this is equivalent to single-image training.

# Supported optimizers
- `:gradient_descent` (default): Riemannian gradient descent
- `:conjugate_gradient`: Riemannian conjugate gradient descent (often faster convergence)
- `:quasi_newton`: Riemannian L-BFGS quasi-Newton method (quasi-Newton with limited memory)
"""
function _train_on_batch(
    batch::Vector{<:AbstractMatrix},
    theta,
    M::ProductManifold,
    optcode::OMEinsum.AbstractEinsum,
    inverse_code::OMEinsum.AbstractEinsum,
    m::Int, n::Int,
    loss::AbstractLoss,
    steps::Int;
    optimizer::Symbol = :gradient_descent
)
    n_imgs = length(batch)
    f(M, p) = begin
        ts = point2tensors(p, M)
        sum(loss_function(ts, m, n, optcode, img, loss; inverse_code=inverse_code) for img in batch) / n_imgs
    end
    grad_f(M, p) = ManifoldDiff.gradient(M, x -> f(M, x), p, RiemannianProjectionBackend(AutoZygote()))

    sc = StopAfterIteration(steps) | StopWhenGradientNormLess(1e-5)

    if optimizer == :gradient_descent
        return gradient_descent(M, f, grad_f, theta; debug=[], stopping_criterion=sc)
    elseif optimizer == :conjugate_gradient
        return conjugate_gradient_descent(M, f, grad_f, theta; stopping_criterion=sc)
    elseif optimizer == :quasi_newton
        return quasi_Newton(M, f, grad_f, theta; stopping_criterion=sc, memory_size=20)
    else
        error("Unknown optimizer: $optimizer. Supported: :gradient_descent, :conjugate_gradient, :quasi_newton")
    end
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
