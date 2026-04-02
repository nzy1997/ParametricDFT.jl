

"""Move array to device. `:gpu` requires CUDA.jl via CUDAExt."""
to_device(x, ::Val{:cpu}) = x

"""Move array to CPU."""
to_cpu(x::AbstractArray) = Array(x)
to_cpu(x) = x

function to_device(x, device::Symbol)
    device === :cpu && return to_device(x, Val(:cpu))
    device === :gpu && hasmethod(to_device, Tuple{typeof(x), Val{:gpu}}) && return to_device(x, Val(:gpu))
    device === :gpu && error("GPU support requires CUDA.jl. Install and load CUDA.jl first: `using CUDA`")
    error("Unknown device: $device. Supported: :cpu, :gpu")
end


"""Core training loop shared by all basis types.
Returns `(final_tensors, best_val_loss, train_losses, val_losses, step_train_losses)`.
Uses `optimize!` from optimizers.jl for all optimization (GPU and CPU).
Supports optimizers: `RiemannianGD()`, `RiemannianAdam()`, or symbols `:gradient_descent`, `:adam`."""
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
    basis_name::String;
    save_loss_path::Union{Nothing, String} = nothing,
    optimizer::Union{Symbol, AbstractRiemannianOptimizer} = :gradient_descent,
    batch_size::Int = 1,
    device::Symbol = :cpu,
    checkpoint_interval::Int = 0,
    checkpoint_dir::Union{Nothing, String} = nothing,
    build_basis_fn::Union{Nothing, Function} = nothing
)
    # Convert Symbol to optimizer type for backward compatibility
    opt = if optimizer isa AbstractRiemannianOptimizer
        optimizer
    elseif optimizer === :adam
        RiemannianAdam(lr=0.001)
    elseif optimizer === :gradient_descent
        RiemannianGD(lr=0.01)
    else
        error("Unknown optimizer: $optimizer. Use RiemannianGD(), RiemannianAdam(), or :gradient_descent/:adam")
    end

    # Convert images to complex matrices and move to device
    complex_dataset = [to_device(Complex{Float64}.(img), device) for img in dataset]

    # Split into training and validation sets
    n_images = length(complex_dataset)
    n_validation = clamp(round(Int, n_images * validation_split), 0, n_images - 1)

    indices = shuffle ? Random.shuffle(1:n_images) : collect(1:n_images)
    validation_indices = indices[1:n_validation]
    training_indices = indices[n_validation+1:end]

    training_data = complex_dataset[training_indices]
    validation_data = complex_dataset[validation_indices]

    # Clamp batch_size to valid range
    batch_size = clamp(batch_size, 1, length(training_data))
    n_batches = ceil(Int, length(training_data) / batch_size)

    # Move tensors to device and ensure Matrix{ComplexF64} for optimize!
    device_tensors = [to_device(Matrix{ComplexF64}(t), device) for t in initial_tensors]

    # Pre-compute batched einsum codes for batch_size > 1
    # This is done once and reused for all epochs/batches (TreeSA optimization is expensive)
    batched_optcode = nothing
    batched_inverse_code = nothing
    if batch_size > 1
        n_gates = length(initial_tensors)
        flat_batched, blabel = make_batched_code(optcode, n_gates)
        batched_optcode = optimize_batched_code(flat_batched, blabel, batch_size)
        # Batch the inverse code too (used by MSELoss)
        if inverse_code !== nothing && loss isa MSELoss
            flat_batched_inv, blabel_inv = make_batched_code(inverse_code, n_gates)
            batched_inverse_code = optimize_batched_code(flat_batched_inv, blabel_inv, batch_size)
        end
    end

    # Single code path: work directly with tensor list
    current_tensors = device_tensors
    best_tensors = copy.(current_tensors)

    # Track best parameters
    best_val_loss = Inf
    patience_counter = 0

    # Track training history
    train_losses = Float64[]
    val_losses = Float64[]
    step_train_losses = Float64[]

    # Per-step loss records for JSON export (epoch, step, loss)
    loss_records = Dict{String, Any}[]

    # Checkpointing setup
    do_checkpoint = checkpoint_interval > 0 && checkpoint_dir !== nothing
    if do_checkpoint
        mkpath(checkpoint_dir)
    end
    global_step = 0

    # Training loop
    for epoch in 1:epochs

        # Shuffle training data each epoch
        shuffle && epoch > 1 && Random.shuffle!(training_data)

        epoch_losses = Float64[]

        # Train on batches
        for batch_idx in 1:n_batches
            start_idx = (batch_idx - 1) * batch_size + 1
            end_idx = min(batch_idx * batch_size, length(training_data))
            batch = training_data[start_idx:end_idx]

            # Construct loss function for this batch
            batch_loss_fn = if batched_optcode !== nothing
                ts -> loss_function(ts, m, n, optcode, batch, loss;
                                    inverse_code=inverse_code, batched_optcode=batched_optcode,
                                    batched_inverse_code=batched_inverse_code)
            else
                ts -> begin
                    total = zero(real(eltype(ts[1])))
                    for img in batch
                        total += loss_function(ts, m, n, optcode, img, loss; inverse_code=inverse_code)
                    end
                    return total / length(batch)
                end
            end

            # Construct gradient function
            batch_grad_fn = ts -> begin
                _, back = Zygote.pullback(batch_loss_fn, ts)
                grads = back(one(real(eltype(ts[1]))))[1]
                return grads
            end

            # Run optimizer with per-iteration loss tracing
            # Scale max_iter by batch size so steps_per_image is honored per image,
            # not per batch. Without this, batch_size=16 would do 16x fewer total steps.
            batch_loss_trace = Float64[]
            batch_max_iter = steps_per_image * length(batch)
            current_tensors = optimize!(opt, current_tensors, batch_loss_fn, batch_grad_fn;
                                         max_iter=batch_max_iter, tol=1e-8,
                                         loss_trace=batch_loss_trace)

            # Collect per-iteration losses into step_train_losses
            append!(step_train_losses, batch_loss_trace)

            # Use last traced loss or recompute for epoch tracking
            batch_loss = isempty(batch_loss_trace) ? Float64(batch_loss_fn(current_tensors)) : last(batch_loss_trace)
            push!(epoch_losses, batch_loss)

            # Ensure at least one entry per batch (optimizer may converge immediately)
            if isempty(batch_loss_trace)
                push!(step_train_losses, batch_loss)
            end
            push!(loss_records, Dict{String, Any}("epoch" => epoch, "step" => batch_idx, "loss" => batch_loss))
            global_step += 1

            # Checkpoint: save basis and loss history periodically
            if do_checkpoint && global_step % checkpoint_interval == 0
                _save_checkpoint(
                    checkpoint_dir, global_step, basis_name,
                    current_tensors,
                    loss_records, train_losses, val_losses,
                    build_basis_fn
                )
            end
        end

        # Compute validation loss
        val_loss = _compute_validation_loss(validation_data, current_tensors, optcode, inverse_code, m, n, loss)
        avg_train_loss = isempty(epoch_losses) ? Inf : sum(epoch_losses) / length(epoch_losses)

        # Store losses for visualization
        push!(train_losses, avg_train_loss)
        push!(val_losses, val_loss)

        # Check for improvement
        if val_loss < best_val_loss
            best_val_loss = val_loss
            # Snapshot current tensors for early stopping
            best_tensors = copy.(current_tensors)
            patience_counter = 0
        else
            patience_counter += 1

            if patience_counter >= early_stopping_patience && epoch > 1
                break
            end
        end
    end

    # Move final tensors back to CPU for serialization
    # Ensure tensors are ComplexF64 for consistency with CPU operations
    final_tensors = [ComplexF64.(Array(t)) for t in best_tensors]

    # Save loss history to JSON if path provided
    if save_loss_path !== nothing
        _save_loss_history(save_loss_path, basis_name, loss_records, train_losses, val_losses)
    end

    return final_tensors, best_val_loss, train_losses, val_losses, step_train_losses
end


"""
    train_basis(::Type{QFTBasis}, dataset; m, n, loss, epochs, steps_per_image,
                optimizer, batch_size, device, ...)

Train a QFTBasis on images. Returns `(basis, history)`.
Key kwargs: `optimizer` (`RiemannianGD()`/`RiemannianAdam()`/`:gradient_descent`/`:adam`),
`batch_size`, `device` (`:cpu`/`:gpu`).
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
    save_loss_path::Union{Nothing, String} = nothing,
    optimizer::Union{Symbol, AbstractRiemannianOptimizer} = :gradient_descent,
    batch_size::Int = 1,
    device::Symbol = :cpu,
    checkpoint_interval::Int = 0,
    checkpoint_dir::Union{Nothing, String} = nothing
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

    # Checkpoint callback: construct QFTBasis from tensors
    build_fn = tensors -> QFTBasis(m, n, tensors, optcode, inverse_code)

    final_tensors, _, train_losses, val_losses, step_train_losses = _train_basis_core(
        dataset, optcode, inverse_code, initial_tensors, m, n, loss,
        epochs, steps_per_image, validation_split, shuffle,
        early_stopping_patience, "QFTBasis";
        save_loss_path=save_loss_path, optimizer=optimizer,
        batch_size=batch_size, device=device,
        checkpoint_interval=checkpoint_interval, checkpoint_dir=checkpoint_dir,
        build_basis_fn=build_fn
    )

    trained_basis = QFTBasis(m, n, final_tensors, optcode, inverse_code)
    history = (train_losses=train_losses, val_losses=val_losses, step_train_losses=step_train_losses, basis_name="QFT")

    return trained_basis, history
end

"""
    train_basis(::Type{EntangledQFTBasis}, dataset; m, n, entangle_phases, ...)

Train an EntangledQFTBasis on images. Same kwargs as QFTBasis plus `entangle_phases`.
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
    save_loss_path::Union{Nothing, String} = nothing,
    optimizer::Union{Symbol, AbstractRiemannianOptimizer} = :gradient_descent,
    batch_size::Int = 1,
    device::Symbol = :cpu,
    checkpoint_interval::Int = 0,
    checkpoint_dir::Union{Nothing, String} = nothing
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

    # Checkpoint callback: construct EntangledQFTBasis from tensors
    build_fn = tensors -> begin
        eidx = get_entangle_tensor_indices(tensors, n_entangle)
        phases = !isempty(eidx) ? extract_entangle_phases(tensors, eidx) :
                 (entangle_phases === nothing ? zeros(n_entangle) : Float64.(entangle_phases))
        EntangledQFTBasis(m, n, tensors, optcode, inverse_code, n_entangle, phases, entangle_position)
    end

    final_tensors, _, train_losses, val_losses, step_train_losses = _train_basis_core(
        dataset, optcode, inverse_code, initial_tensors, m, n, loss,
        epochs, steps_per_image, validation_split, shuffle,
        early_stopping_patience, "EntangledQFTBasis";
        save_loss_path=save_loss_path, optimizer=optimizer,
        batch_size=batch_size, device=device,
        checkpoint_interval=checkpoint_interval, checkpoint_dir=checkpoint_dir,
        build_basis_fn=build_fn
    )

    # Extract trained phases
    entangle_indices = get_entangle_tensor_indices(final_tensors, n_entangle)
    trained_phases = if !isempty(entangle_indices)
        extract_entangle_phases(final_tensors, entangle_indices)
    else
        entangle_phases === nothing ? zeros(n_entangle) : Float64.(entangle_phases)
    end

    trained_basis = EntangledQFTBasis(m, n, final_tensors, optcode, inverse_code, n_entangle, trained_phases, entangle_position)
    history = (train_losses=train_losses, val_losses=val_losses, step_train_losses=step_train_losses, basis_name="Entangled QFT")

    return trained_basis, history
end

"""
    train_basis(::Type{TEBDBasis}, dataset; m, n, phases, ...)

Train a TEBDBasis on images. Same kwargs as QFTBasis plus `phases` (initial TEBD gate phases).
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
    save_loss_path::Union{Nothing, String} = nothing,
    optimizer::Union{Symbol, AbstractRiemannianOptimizer} = :gradient_descent,
    batch_size::Int = 1,
    device::Symbol = :cpu,
    checkpoint_interval::Int = 0,
    checkpoint_dir::Union{Nothing, String} = nothing
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

    # Checkpoint callback: construct TEBDBasis from tensors
    build_fn = tensors -> begin
        gidx = get_tebd_gate_indices(tensors, n_gates)
        p = !isempty(gidx) ? extract_tebd_phases(tensors, gidx) :
            (phases === nothing ? zeros(n_gates) : Float64.(phases))
        TEBDBasis(m, n, tensors, optcode, inverse_code, n_row_gates, n_col_gates, p)
    end

    final_tensors, _, train_losses, val_losses, step_train_losses = _train_basis_core(
        dataset, optcode, inverse_code, initial_tensors, m, n, loss,
        epochs, steps_per_image, validation_split, shuffle,
        early_stopping_patience, "TEBDBasis";
        save_loss_path=save_loss_path, optimizer=optimizer,
        batch_size=batch_size, device=device,
        checkpoint_interval=checkpoint_interval, checkpoint_dir=checkpoint_dir,
        build_basis_fn=build_fn
    )

    # Extract trained phases
    gate_indices = get_tebd_gate_indices(final_tensors, n_gates)
    trained_phases = if !isempty(gate_indices)
        extract_tebd_phases(final_tensors, gate_indices)
    else
        phases === nothing ? zeros(n_gates) : Float64.(phases)
    end

    trained_basis = TEBDBasis(m, n, final_tensors, optcode, inverse_code, n_row_gates, n_col_gates, trained_phases)
    history = (train_losses=train_losses, val_losses=val_losses, step_train_losses=step_train_losses, basis_name="TEBD")

    return trained_basis, history
end

"""Train a MERABasis on images. Same kwargs as TEBDBasis."""
function train_basis(
    ::Type{MERABasis},
    dataset::Vector{<:AbstractMatrix};
    m::Int, n::Int,
    phases::Union{Nothing, Vector{<:Real}} = nothing,
    loss::AbstractLoss = MSELoss(round(Int, 2^(m+n) * 0.1)),
    epochs::Int = 3,
    steps_per_image::Int = 200,
    validation_split::Float64 = 0.2,
    shuffle::Bool = true,
    early_stopping_patience::Int = 2,
    save_loss_path::Union{Nothing, String} = nothing,
    optimizer::Union{Symbol, AbstractRiemannianOptimizer} = :gradient_descent,
    batch_size::Int = 1,
    device::Symbol = :cpu,
    checkpoint_interval::Int = 0,
    checkpoint_dir::Union{Nothing, String} = nothing
)
    @assert 0.0 <= validation_split < 1.0 "validation_split must be in [0, 1)"
    @assert length(dataset) > 0 "Dataset must not be empty"
    expected_size = (2^m, 2^n)
    for (i, img) in enumerate(dataset)
        @assert size(img) == expected_size "Image $i has size $(size(img)), expected $expected_size"
    end

    n_row_gates = m >= 2 ? 2 * (m - 1) : 0
    n_col_gates = n >= 2 ? 2 * (n - 1) : 0
    n_gates = n_row_gates + n_col_gates

    # Initialize phases: use small random values if not provided
    # Zero phases create a symmetric point where gradients are zero,
    # preventing the optimizer from learning. Small random values break symmetry.
    if phases === nothing
        phases = randn(n_gates) * 0.1
    end

    # Initialize circuit
    optcode, initial_tensors, _, _ = mera_code(m, n; phases=phases)
    inverse_code, _, _, _ = mera_code(m, n; phases=phases, inverse=true)

    # Checkpoint callback: construct MERABasis from tensors
    build_fn = tensors -> begin
        gidx = get_mera_gate_indices(tensors, n_gates)
        p = !isempty(gidx) ? extract_mera_phases(tensors, gidx) :
            (phases === nothing ? zeros(n_gates) : Float64.(phases))
        MERABasis(m, n, tensors, optcode, inverse_code, n_row_gates, n_col_gates, p)
    end

    final_tensors, _, train_losses, val_losses, step_train_losses = _train_basis_core(
        dataset, optcode, inverse_code, initial_tensors, m, n, loss,
        epochs, steps_per_image, validation_split, shuffle,
        early_stopping_patience, "MERABasis";
        save_loss_path=save_loss_path, optimizer=optimizer,
        batch_size=batch_size, device=device,
        checkpoint_interval=checkpoint_interval, checkpoint_dir=checkpoint_dir,
        build_basis_fn=build_fn
    )

    # Extract trained phases
    gate_indices = get_mera_gate_indices(final_tensors, n_gates)
    trained_phases = if !isempty(gate_indices)
        extract_mera_phases(final_tensors, gate_indices)
    else
        phases === nothing ? zeros(n_gates) : Float64.(phases)
    end

    trained_basis = MERABasis(m, n, final_tensors, optcode, inverse_code, n_row_gates, n_col_gates, trained_phases)
    history = (train_losses=train_losses, val_losses=val_losses, step_train_losses=step_train_losses, basis_name="MERA")

    return trained_basis, history
end


"""Save a training checkpoint (basis + loss history)."""
function _save_checkpoint(
    checkpoint_dir::String,
    step::Int,
    basis_name::String,
    tensors::Vector,
    loss_records::Vector{Dict{String, Any}},
    train_losses::Vector{Float64},
    val_losses::Vector{Float64},
    build_basis_fn::Union{Nothing, Function}
)
    tag = "step" * lpad(step, 4, '0')

    # Save loss history up to this point
    loss_path = joinpath(checkpoint_dir, "$(tag)_loss.json")
    _save_loss_history(loss_path, basis_name, loss_records, train_losses, val_losses)

    # Save basis if builder is available
    if build_basis_fn !== nothing
        cpu_tensors = [ComplexF64.(Array(t)) for t in tensors]
        basis = build_basis_fn(cpu_tensors)
        basis_path = joinpath(checkpoint_dir, "$(tag)_basis.json")
        save_basis(basis_path, basis)
    end
end

"""Save training loss history to JSON."""
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

"""Save training history (from `train_basis`) to a JSON file."""
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

"""Load training loss history from JSON. Returns a `TrainingHistory` object."""
function load_loss_history(path::String)
    json_str = read(path, String)
    json_data = JSON3.read(json_str)

    basis_name = String(json_data[:basis_name])

    train_losses = Float64[entry[:train_loss] for entry in json_data[:epoch_losses]]
    val_losses = Float64[entry[:val_loss] for entry in json_data[:epoch_losses]]
    step_train_losses = Float64[entry[:loss] for entry in json_data[:step_losses]]

    return TrainingHistory(train_losses, val_losses, step_train_losses, basis_name)
end

"""Compute average loss over validation set."""
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


"""Train QFTBasis from image files. Requires Images.jl."""
function train_basis_from_files end  # Defined in extension or requires Images.jl
