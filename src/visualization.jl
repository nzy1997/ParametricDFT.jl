# ============================================================================
# Training Loss Visualization
# ============================================================================
# This file provides visualization functionality for training loss curves

"""
    TrainingHistory

Stores the training history including training and validation losses per epoch.

# Fields
- `train_losses::Vector{Float64}`: Average training loss per epoch
- `val_losses::Vector{Float64}`: Validation loss per epoch
- `basis_name::String`: Name of the basis being trained
"""
struct TrainingHistory
    train_losses::Vector{Float64}
    val_losses::Vector{Float64}
    basis_name::String
end

"""
    plot_training_loss(history::TrainingHistory; kwargs...)

Plot training and validation loss curves for a single basis.

# Arguments
- `history::TrainingHistory`: Training history to plot

# Keyword Arguments
- `title::String`: Plot title (default: "Training Loss - \$(history.basis_name)")
- `xlabel::String`: X-axis label (default: "Epoch")
- `ylabel::String`: Y-axis label (default: "Loss")
- `yscale::Symbol`: Y-axis scale (default: :log10)
- `legend::Symbol`: Legend position (default: :topright)
- `size::Tuple{Int,Int}`: Plot size (default: (800, 500))

# Returns
- `Plots.Plot`: The generated plot

# Example
```julia
using ParametricDFT
basis, history = train_basis(QFTBasis, images; m=5, n=5, epochs=3)
hist = TrainingHistory(history.train_losses, history.val_losses, "QFT")
plot_training_loss(hist)
savefig("qft_training_loss.png")
```
"""
function plot_training_loss(history::TrainingHistory;
                           title::String = "Training Loss - $(history.basis_name)",
                           xlabel::String = "Epoch",
                           ylabel::String = "Loss",
                           yscale::Symbol = :log10,
                           legend::Symbol = :topright,
                           size::Tuple{Int,Int} = (800, 500))

    epochs = 1:length(history.train_losses)

    p = Plots.plot(epochs, history.train_losses,
             label="Training Loss",
             xlabel=xlabel,
             ylabel=ylabel,
             title=title,
             yscale=yscale,
             legend=legend,
             size=size,
             linewidth=2,
             marker=:circle,
             markersize=4,
             color=:blue)

    Plots.plot!(p, epochs, history.val_losses,
          label="Validation Loss",
          linewidth=2,
          marker=:square,
          markersize=4,
          color=:red)

    return p
end

"""
    plot_training_comparison(histories::Vector{TrainingHistory}; kwargs...)

Plot training loss curves for multiple bases on the same plot for comparison.

# Arguments
- `histories::Vector{TrainingHistory}`: Vector of training histories to compare

# Keyword Arguments
- `title::String`: Plot title (default: "Training Loss Comparison")
- `xlabel::String`: X-axis label (default: "Epoch")
- `ylabel::String`: Y-axis label (default: "Loss")
- `yscale::Symbol`: Y-axis scale (default: :log10)
- `legend::Symbol`: Legend position (default: :topright)
- `size::Tuple{Int,Int}`: Plot size (default: (1000, 600))
- `loss_type::Symbol`: Type of loss to plot (:train, :validation, or :both, default: :both)

# Returns
- `Plots.Plot`: The generated plot

# Example
```julia
using ParametricDFT
basis1, hist1 = train_basis(QFTBasis, images; m=5, n=5, epochs=3)
basis2, hist2 = train_basis(TEBDBasis, images; m=5, n=5, epochs=3)
histories = [
    TrainingHistory(hist1.train_losses, hist1.val_losses, "QFT"),
    TrainingHistory(hist2.train_losses, hist2.val_losses, "TEBD")
]
plot_training_comparison(histories)
savefig("training_comparison.png")
```
"""
function plot_training_comparison(histories::Vector{TrainingHistory};
                                 title::String = "Training Loss Comparison",
                                 xlabel::String = "Epoch",
                                 ylabel::String = "Loss",
                                 yscale::Symbol = :log10,
                                 legend::Symbol = :topright,
                                 size::Tuple{Int,Int} = (1000, 600),
                                 loss_type::Symbol = :both)

    @assert loss_type in [:train, :validation, :both] "loss_type must be :train, :validation, or :both"

    p = Plots.plot(xlabel=xlabel,
             ylabel=ylabel,
             title=title,
             yscale=yscale,
             legend=legend,
             size=size)

    colors = [:blue, :red, :green, :purple, :orange, :brown]
    markers_train = [:circle, :diamond, :utriangle, :dtriangle, :star5, :hexagon]
    markers_val = [:square, :pentagon, :cross, :xcross, :star4, :octagon]

    for (idx, history) in enumerate(histories)
        color = colors[mod1(idx, length(colors))]
        marker_train = markers_train[mod1(idx, length(markers_train))]
        marker_val = markers_val[mod1(idx, length(markers_val))]

        epochs = 1:length(history.train_losses)

        # Plot training loss
        if loss_type in [:train, :both]
            Plots.plot!(p, epochs, history.train_losses,
                  label="$(history.basis_name) (Train)",
                  linewidth=2,
                  marker=marker_train,
                  markersize=4,
                  color=color,
                  linestyle=:solid)
        end

        # Plot validation loss
        if loss_type in [:validation, :both]
            Plots.plot!(p, epochs, history.val_losses,
                  label="$(history.basis_name) (Val)",
                  linewidth=2,
                  marker=marker_val,
                  markersize=4,
                  color=color,
                  linestyle=:dash)
        end
    end

    return p
end

"""
    plot_training_grid(histories::Vector{TrainingHistory}; kwargs...)

Create a grid of individual training loss plots for multiple bases.

# Arguments
- `histories::Vector{TrainingHistory}`: Vector of training histories to plot

# Keyword Arguments
- `title::String`: Overall title (default: "Training Loss Comparison")
- `yscale::Symbol`: Y-axis scale (default: :log10)
- `size::Tuple{Int,Int}`: Total plot size (default: (1200, 800))
- `layout::Union{Nothing, Tuple{Int,Int}}`: Grid layout (default: auto)

# Returns
- `Plots.Plot`: The generated grid plot

# Example
```julia
using ParametricDFT
histories = [hist1, hist2, hist3]  # TrainingHistory objects
plot_training_grid(histories, layout=(2, 2))
savefig("training_grid.png")
```
"""
function plot_training_grid(histories::Vector{TrainingHistory};
                           title::String = "Training Loss Comparison",
                           yscale::Symbol = :log10,
                           size::Tuple{Int,Int} = (1200, 800),
                           layout::Union{Nothing, Tuple{Int,Int}} = nothing)

    n_plots = length(histories)

    # Auto-determine layout if not provided
    if layout === nothing
        n_cols = min(3, n_plots)
        n_rows = ceil(Int, n_plots / n_cols)
        layout = (n_rows, n_cols)
    end

    plots_array = []

    for history in histories
        epochs = 1:length(history.train_losses)

        p = Plots.plot(epochs, history.train_losses,
                label="Training",
                xlabel="Epoch",
                ylabel="Loss",
                title=history.basis_name,
                yscale=yscale,
                legend=:topright,
                linewidth=2,
                marker=:circle,
                markersize=3,
                color=:blue)

        Plots.plot!(p, epochs, history.val_losses,
              label="Validation",
              linewidth=2,
              marker=:square,
              markersize=3,
              color=:red)

        push!(plots_array, p)
    end

    return Plots.plot(plots_array...,
                layout=layout,
                size=size,
                plot_title=title,
                plot_titlefontsize=14)
end

"""
    save_training_plots(histories::Vector{TrainingHistory}, output_dir::String; prefix::String="")

Generate and save all training visualization plots to a directory.

# Arguments
- `histories::Vector{TrainingHistory}`: Vector of training histories
- `output_dir::String`: Directory to save plots

# Keyword Arguments
- `prefix::String`: Prefix for filenames (default: "")

# Returns
- `Vector{String}`: Paths to saved plot files

# Example
```julia
using ParametricDFT
histories = [
    TrainingHistory(hist1.train_losses, hist1.val_losses, "QFT"),
    TrainingHistory(hist2.train_losses, hist2.val_losses, "TEBD")
]
save_training_plots(histories, "output/")
```
"""
function save_training_plots(histories::Vector{TrainingHistory},
                            output_dir::String;
                            prefix::String = "")

    # Ensure output directory exists
    if !isdir(output_dir)
        mkpath(output_dir)
    end

    saved_files = String[]

    # Generate prefix
    file_prefix = isempty(prefix) ? "" : prefix * "_"

    # 1. Individual plots for each basis
    for history in histories
        safe_name = lowercase(replace(history.basis_name, " " => "_"))
        filename = joinpath(output_dir, "$(file_prefix)$(safe_name)_loss.png")

        p = plot_training_loss(history)
        Plots.savefig(p, filename)
        push!(saved_files, filename)
    end

    # 2. Comparison plot (all bases, both train and val)
    if length(histories) > 1
        filename = joinpath(output_dir, "$(file_prefix)comparison_all.png")
        p = plot_training_comparison(histories; loss_type=:both)
        Plots.savefig(p, filename)
        push!(saved_files, filename)

        # 3. Training loss only comparison
        filename = joinpath(output_dir, "$(file_prefix)comparison_train.png")
        p = plot_training_comparison(histories; loss_type=:train,
                                     title="Training Loss Comparison")
        Plots.savefig(p, filename)
        push!(saved_files, filename)

        # 4. Validation loss only comparison
        filename = joinpath(output_dir, "$(file_prefix)comparison_val.png")
        p = plot_training_comparison(histories; loss_type=:validation,
                                     title="Validation Loss Comparison")
        Plots.savefig(p, filename)
        push!(saved_files, filename)

        # 5. Grid plot
        filename = joinpath(output_dir, "$(file_prefix)grid.png")
        p = plot_training_grid(histories)
        Plots.savefig(p, filename)
        push!(saved_files, filename)
    end

    return saved_files
end
