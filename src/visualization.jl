# ============================================================================
# Training Loss Visualization
# ============================================================================
# This file provides visualization functionality for training loss curves

"""
    TrainingHistory

Stores the training history including losses per epoch and per step.

# Fields
- `train_losses::Vector{Float64}`: Average training loss per epoch
- `val_losses::Vector{Float64}`: Validation loss per epoch
- `step_train_losses::Vector{Float64}`: Training loss per step (per image processed)
- `basis_name::String`: Name of the basis being trained
"""
struct TrainingHistory
    train_losses::Vector{Float64}
    val_losses::Vector{Float64}
    step_train_losses::Vector{Float64}
    basis_name::String
end

"""
    ema_smooth(values::Vector{Float64}, alpha::Float64) -> Vector{Float64}

Compute exponential moving average for smoothing noisy loss curves.

# Arguments
- `values::Vector{Float64}`: Raw values to smooth
- `alpha::Float64`: Smoothing factor in (0, 1). Higher values = more smoothing.
  Common values: 0.6 (light), 0.9 (heavy), 0.95 (very heavy).

# Returns
- `Vector{Float64}`: Smoothed values (same length as input)
"""
function ema_smooth(values::Vector{Float64}, alpha::Float64)
    @assert 0.0 < alpha < 1.0 "alpha must be in (0, 1)"
    isempty(values) && return Float64[]
    smoothed = similar(values)
    smoothed[1] = values[1]
    for i in 2:length(values)
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * values[i]
    end
    return smoothed
end

"""
    plot_training_loss(history::TrainingHistory; kwargs...)

Plot training and validation loss curves for a single basis.

# Arguments
- `history::TrainingHistory`: Training history to plot

# Keyword Arguments
- `title::String`: Plot title (default: "Training Loss - \\\$(history.basis_name)")
- `xlabel::String`: X-axis label (default: "Epoch")
- `ylabel::String`: Y-axis label (default: "Loss")
- `yscale::Function`: Y-axis scale function (default: log10)
- `size::Tuple{Int,Int}`: Figure size (default: (800, 500))

# Returns
- `Makie.Figure`: The generated figure

# Example
```julia
using ParametricDFT
basis, history = train_basis(QFTBasis, images; m=5, n=5, epochs=3)
hist = TrainingHistory(history.train_losses, history.val_losses,
                       history.step_train_losses, "QFT")
fig = plot_training_loss(hist)
save("qft_training_loss.png", fig)
```
"""
function plot_training_loss(history::TrainingHistory;
                           title::String = "Training Loss - $(history.basis_name)",
                           xlabel::String = "Epoch",
                           ylabel::String = "Loss",
                           yscale::Function = log10,
                           size::Tuple{Int,Int} = (800, 500))

    epochs = 1:length(history.train_losses)

    fig = Figure(; size=size)
    ax = Axis(fig[1, 1];
              title=title,
              xlabel=xlabel,
              ylabel=ylabel,
              yscale=yscale)

    lines!(ax, epochs, history.train_losses;
           label="Training Loss",
           linewidth=2,
           color=:blue)
    scatter!(ax, epochs, history.train_losses;
             label=nothing,
             markersize=8,
             marker=:circle,
             color=:blue)

    lines!(ax, epochs, history.val_losses;
           label="Validation Loss",
           linewidth=2,
           color=:red)
    scatter!(ax, epochs, history.val_losses;
             label=nothing,
             markersize=8,
             marker=:rect,
             color=:red)

    axislegend(ax; position=:rt)

    return fig
end

"""
    plot_training_loss_steps(history::TrainingHistory; kwargs...)

Plot per-step training loss curve for a single basis.

# Arguments
- `history::TrainingHistory`: Training history to plot

# Keyword Arguments
- `title::String`: Plot title (default: "Step Training Loss - \\\$(history.basis_name)")
- `xlabel::String`: X-axis label (default: "Step")
- `ylabel::String`: Y-axis label (default: "Loss")
- `yscale::Function`: Y-axis scale function (default: log10)
- `size::Tuple{Int,Int}`: Figure size (default: (800, 500))
- `smoothing::Float64`: EMA smoothing factor in (0, 1), 0 disables smoothing (default: 0.0)

# Returns
- `Makie.Figure`: The generated figure

# Example
```julia
using ParametricDFT
basis, history = train_basis(QFTBasis, images; m=5, n=5, epochs=3)
hist = TrainingHistory(history.train_losses, history.val_losses, history.step_train_losses, "QFT")
fig = plot_training_loss_steps(hist; smoothing=0.8)
save("qft_step_loss.png", fig)
```
"""
function plot_training_loss_steps(history::TrainingHistory;
                                 title::String = "Step Training Loss - $(history.basis_name)",
                                 xlabel::String = "Step",
                                 ylabel::String = "Loss",
                                 yscale::Function = log10,
                                 size::Tuple{Int,Int} = (800, 500),
                                 smoothing::Float64 = 0.0)

    steps = 1:length(history.step_train_losses)

    fig = Figure(; size=size)
    ax = Axis(fig[1, 1];
              title=title,
              xlabel=xlabel,
              ylabel=ylabel,
              yscale=yscale)

    if smoothing > 0
        lines!(ax, steps, history.step_train_losses;
               label=nothing,
               linewidth=0.5,
               color=(:blue, 0.3))
        smoothed = ema_smooth(history.step_train_losses, smoothing)
        lines!(ax, steps, smoothed;
               label="Training Loss (smoothed)",
               linewidth=2,
               color=:blue)
    else
        lines!(ax, steps, history.step_train_losses;
               label="Training Loss",
               linewidth=1.5,
               color=:blue)
    end

    axislegend(ax; position=:rt)

    return fig
end

"""
    plot_training_comparison_steps(histories::Vector{TrainingHistory}; kwargs...)

Plot per-step training loss curves for multiple bases on the same plot.

# Arguments
- `histories::Vector{TrainingHistory}`: Vector of training histories to compare

# Keyword Arguments
- `title::String`: Plot title (default: "Step Training Loss Comparison")
- `xlabel::String`: X-axis label (default: "Step")
- `ylabel::String`: Y-axis label (default: "Loss")
- `yscale::Function`: Y-axis scale function (default: log10)
- `size::Tuple{Int,Int}`: Figure size (default: (1000, 600))
- `smoothing::Float64`: EMA smoothing factor in (0, 1), 0 disables smoothing (default: 0.0)

# Returns
- `Makie.Figure`: The generated figure

# Example
```julia
using ParametricDFT
histories = [hist_qft, hist_entangled, hist_tebd]
fig = plot_training_comparison_steps(histories; smoothing=0.8)
save("step_comparison.png", fig)
```
"""
function plot_training_comparison_steps(histories::Vector{TrainingHistory};
                                       title::String = "Step Training Loss Comparison",
                                       xlabel::String = "Step",
                                       ylabel::String = "Loss",
                                       yscale::Function = log10,
                                       size::Tuple{Int,Int} = (1000, 600),
                                       smoothing::Float64 = 0.0)

    fig = Figure(; size=size)
    ax = Axis(fig[1, 1];
              title=title,
              xlabel=xlabel,
              ylabel=ylabel,
              yscale=yscale)

    colors = [:blue, :red, :green, :purple, :orange, :brown]

    for (idx, history) in enumerate(histories)
        color = colors[mod1(idx, length(colors))]
        steps = 1:length(history.step_train_losses)

        if smoothing > 0
            lines!(ax, steps, history.step_train_losses;
                   label=nothing,
                   linewidth=0.5,
                   color=(color, 0.3))
            smoothed = ema_smooth(history.step_train_losses, smoothing)
            lines!(ax, steps, smoothed;
                   label=history.basis_name,
                   linewidth=2,
                   color=color)
        else
            lines!(ax, steps, history.step_train_losses;
                   label=history.basis_name,
                   linewidth=1.5,
                   color=color)
        end
    end

    axislegend(ax; position=:rt)

    return fig
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
- `yscale::Function`: Y-axis scale function (default: log10)
- `size::Tuple{Int,Int}`: Figure size (default: (1000, 600))
- `loss_type::Symbol`: Type of loss to plot (:train, :validation, or :both, default: :both)

# Returns
- `Makie.Figure`: The generated figure

# Example
```julia
using ParametricDFT
basis1, hist1 = train_basis(QFTBasis, images; m=5, n=5, epochs=3)
basis2, hist2 = train_basis(TEBDBasis, images; m=5, n=5, epochs=3)
histories = [
    TrainingHistory(hist1.train_losses, hist1.val_losses, hist1.step_train_losses, "QFT"),
    TrainingHistory(hist2.train_losses, hist2.val_losses, hist2.step_train_losses, "TEBD")
]
fig = plot_training_comparison(histories)
save("training_comparison.png", fig)
```
"""
function plot_training_comparison(histories::Vector{TrainingHistory};
                                 title::String = "Training Loss Comparison",
                                 xlabel::String = "Epoch",
                                 ylabel::String = "Loss",
                                 yscale::Function = log10,
                                 size::Tuple{Int,Int} = (1000, 600),
                                 loss_type::Symbol = :both)

    @assert loss_type in [:train, :validation, :both] "loss_type must be :train, :validation, or :both"

    fig = Figure(; size=size)
    ax = Axis(fig[1, 1];
              title=title,
              xlabel=xlabel,
              ylabel=ylabel,
              yscale=yscale)

    colors = [:blue, :red, :green, :purple, :orange, :brown]
    markers_train = [:circle, :diamond, :utriangle, :dtriangle, :star5, :hexagon]
    markers_val = [:rect, :pentagon, :cross, :xcross, :star4, :octagon]

    for (idx, history) in enumerate(histories)
        color = colors[mod1(idx, length(colors))]
        marker_train = markers_train[mod1(idx, length(markers_train))]
        marker_val = markers_val[mod1(idx, length(markers_val))]

        epochs = 1:length(history.train_losses)

        # Plot training loss
        if loss_type in [:train, :both]
            lines!(ax, epochs, history.train_losses;
                   label="$(history.basis_name) (Train)",
                   linewidth=2,
                   color=color,
                   linestyle=:solid)
            scatter!(ax, epochs, history.train_losses;
                     label=nothing,
                     markersize=8,
                     marker=marker_train,
                     color=color)
        end

        # Plot validation loss
        if loss_type in [:validation, :both]
            lines!(ax, epochs, history.val_losses;
                   label="$(history.basis_name) (Val)",
                   linewidth=2,
                   color=color,
                   linestyle=:dash)
            scatter!(ax, epochs, history.val_losses;
                     label=nothing,
                     markersize=8,
                     marker=marker_val,
                     color=color)
        end
    end

    axislegend(ax; position=:rt)

    return fig
end

"""
    plot_training_grid(histories::Vector{TrainingHistory}; kwargs...)

Create a grid of individual training loss plots for multiple bases.

# Arguments
- `histories::Vector{TrainingHistory}`: Vector of training histories to plot

# Keyword Arguments
- `title::String`: Overall title (default: "Training Loss Comparison")
- `yscale::Function`: Y-axis scale function (default: log10)
- `size::Tuple{Int,Int}`: Total figure size (default: (1200, 800))
- `layout::Union{Nothing, Tuple{Int,Int}}`: Grid layout (default: auto)

# Returns
- `Makie.Figure`: The generated grid figure

# Example
```julia
using ParametricDFT
histories = [hist1, hist2, hist3]  # TrainingHistory objects
fig = plot_training_grid(histories, layout=(2, 2))
save("training_grid.png", fig)
```
"""
function plot_training_grid(histories::Vector{TrainingHistory};
                           title::String = "Training Loss Comparison",
                           yscale::Function = log10,
                           size::Tuple{Int,Int} = (1200, 800),
                           layout::Union{Nothing, Tuple{Int,Int}} = nothing)

    n_plots = length(histories)

    # Auto-determine layout if not provided
    if layout === nothing
        n_cols = min(3, n_plots)
        n_rows = ceil(Int, n_plots / n_cols)
        layout = (n_rows, n_cols)
    end

    fig = Figure(; size=size)
    Label(fig[0, :], title; fontsize=20, font=:bold)

    for (i, history) in enumerate(histories)
        row = div(i - 1, layout[2]) + 1
        col = mod1(i, layout[2])

        epochs = 1:length(history.train_losses)

        ax = Axis(fig[row, col];
                  title=history.basis_name,
                  xlabel="Epoch",
                  ylabel="Loss",
                  yscale=yscale)

        lines!(ax, epochs, history.train_losses;
               label="Training",
               linewidth=2,
               color=:blue)
        scatter!(ax, epochs, history.train_losses;
                 label=nothing,
                 markersize=6,
                 marker=:circle,
                 color=:blue)

        lines!(ax, epochs, history.val_losses;
               label="Validation",
               linewidth=2,
               color=:red)
        scatter!(ax, epochs, history.val_losses;
                 label=nothing,
                 markersize=6,
                 marker=:rect,
                 color=:red)

        axislegend(ax; position=:rt)
    end

    return fig
end

"""
    save_training_plots(histories::Vector{TrainingHistory}, output_dir::String; prefix::String="", smoothing::Float64=0.6)

Generate and save all training visualization plots to a directory.

# Arguments
- `histories::Vector{TrainingHistory}`: Vector of training histories
- `output_dir::String`: Directory to save plots

# Keyword Arguments
- `prefix::String`: Prefix for filenames (default: "")
- `smoothing::Float64`: EMA smoothing factor for step-level plots (default: 0.6).
  Set to 0.0 to disable smoothed plots.

# Returns
- `Vector{String}`: Paths to saved plot files

# Example
```julia
using ParametricDFT
histories = [
    TrainingHistory(hist1.train_losses, hist1.val_losses, hist1.step_train_losses, "QFT"),
    TrainingHistory(hist2.train_losses, hist2.val_losses, hist2.step_train_losses, "TEBD")
]
save_training_plots(histories, "output/"; smoothing=0.8)
```
"""
function save_training_plots(histories::Vector{TrainingHistory},
                            output_dir::String;
                            prefix::String = "",
                            smoothing::Float64 = 0.6)

    saved_files = String[]

    # Generate prefix
    file_prefix = isempty(prefix) ? "" : prefix * "_"

    has_steps = any(h -> !isempty(h.step_train_losses), histories)

    # Generate plots in both log and linear scale
    for (scale_name, scale_fn) in [("log", log10), ("linear", identity)]
        scale_dir = joinpath(output_dir, scale_name)
        mkpath(scale_dir)

        # 1. Individual epoch-based plots for each basis
        for history in histories
            safe_name = lowercase(replace(history.basis_name, " " => "_"))
            filename = joinpath(scale_dir, "$(file_prefix)$(safe_name)_loss.png")

            fig = plot_training_loss(history; yscale=scale_fn)
            save(filename, fig)
            push!(saved_files, filename)
        end

        # 2. Individual step-based plots for each basis
        if has_steps
            for history in histories
                if !isempty(history.step_train_losses)
                    safe_name = lowercase(replace(history.basis_name, " " => "_"))
                    filename = joinpath(scale_dir, "$(file_prefix)$(safe_name)_step_loss.png")

                    fig = plot_training_loss_steps(history; yscale=scale_fn)
                    save(filename, fig)
                    push!(saved_files, filename)

                    # Smoothed variant
                    if smoothing > 0
                        filename_smooth = joinpath(scale_dir, "$(file_prefix)$(safe_name)_step_loss_smooth.png")
                        fig_smooth = plot_training_loss_steps(history; yscale=scale_fn, smoothing=smoothing)
                        save(filename_smooth, fig_smooth)
                        push!(saved_files, filename_smooth)
                    end
                end
            end
        end

        # 3. Comparison plot (all bases, both train and val)
        if length(histories) > 1
            filename = joinpath(scale_dir, "$(file_prefix)comparison_all.png")
            fig = plot_training_comparison(histories; loss_type=:both, yscale=scale_fn)
            save(filename, fig)
            push!(saved_files, filename)

            # 4. Training loss only comparison
            filename = joinpath(scale_dir, "$(file_prefix)comparison_train.png")
            fig = plot_training_comparison(histories; loss_type=:train,
                                         title="Training Loss Comparison", yscale=scale_fn)
            save(filename, fig)
            push!(saved_files, filename)

            # 5. Validation loss only comparison
            filename = joinpath(scale_dir, "$(file_prefix)comparison_val.png")
            fig = plot_training_comparison(histories; loss_type=:validation,
                                         title="Validation Loss Comparison", yscale=scale_fn)
            save(filename, fig)
            push!(saved_files, filename)

            # 6. Step-based comparison plot
            if has_steps
                filename = joinpath(scale_dir, "$(file_prefix)comparison_steps.png")
                fig = plot_training_comparison_steps(histories; yscale=scale_fn)
                save(filename, fig)
                push!(saved_files, filename)

                # Smoothed variant
                if smoothing > 0
                    filename_smooth = joinpath(scale_dir, "$(file_prefix)comparison_steps_smooth.png")
                    fig_smooth = plot_training_comparison_steps(histories; yscale=scale_fn, smoothing=smoothing)
                    save(filename_smooth, fig_smooth)
                    push!(saved_files, filename_smooth)
                end
            end

            # 7. Grid plot
            filename = joinpath(scale_dir, "$(file_prefix)grid.png")
            fig = plot_training_grid(histories; yscale=scale_fn)
            save(filename, fig)
            push!(saved_files, filename)
        end
    end

    return saved_files
end
