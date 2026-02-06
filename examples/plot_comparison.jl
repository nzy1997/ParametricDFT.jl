# ================================================================================
# Plot Comparison Results
# ================================================================================
# Reads loss history JSON files from FullComparison/loss_history/ and generates
# comparison plots across optimizers, devices, bases, and datasets.
#
# Run after optimizer_basis_comparison.jl:
#   julia --project=examples examples/plot_comparison.jl
#
# Input:  examples/FullComparison/loss_history/*.json
# Output: examples/FullComparison/plots/*.png
# ================================================================================

using ParametricDFT
using CairoMakie

# ================================================================================
# Configuration
# ================================================================================

const LOSS_DIR = joinpath(@__DIR__, "FullComparison", "loss_history")
const PLOTS_DIR = joinpath(@__DIR__, "FullComparison", "plots")

# Experiment dimensions
const DATASETS   = ["mnist", "quickdraw"]
const OPTIMIZERS = ["gradient_descent", "adam"]
const DEVICES    = ["cpu", "gpu"]
const BASES      = ["qft", "entangledqft", "tebd"]

# Display names
const DS_LABEL  = Dict("mnist" => "MNIST", "quickdraw" => "QuickDraw")
const OPT_LABEL = Dict("gradient_descent" => "GD", "adam" => "Adam")
const DEV_LABEL = Dict("cpu" => "CPU", "gpu" => "GPU")
const BAS_LABEL = Dict("qft" => "QFT", "entangledqft" => "Entangled QFT", "tebd" => "TEBD")

# Color scheme: basis -> color
const BAS_COLOR = Dict("qft" => :dodgerblue, "entangledqft" => :crimson, "tebd" => :forestgreen)

# Optimizer×Device styling
const OPTDEV_COLOR = Dict(
    ("cpu", "gradient_descent") => :royalblue,
    ("cpu", "adam")             => :darkorange,
    ("gpu", "gradient_descent") => :mediumseagreen,
    ("gpu", "adam")             => :mediumorchid,
)
const OPTDEV_STYLE = Dict(
    ("cpu", "gradient_descent") => :solid,
    ("cpu", "adam")             => :dash,
    ("gpu", "gradient_descent") => :solid,
    ("gpu", "adam")             => :dash,
)
const OPTDEV_WIDTH = Dict(
    ("cpu", "gradient_descent") => 1.5,
    ("cpu", "adam")             => 1.5,
    ("gpu", "gradient_descent") => 2.5,
    ("gpu", "adam")             => 2.5,
)

# ================================================================================
# Helpers
# ================================================================================

"""Build the loss history filename for a given config."""
function loss_path(ds, opt, dev, bas)
    joinpath(LOSS_DIR, "$(ds)_$(opt)_$(dev)_$(bas)_loss.json")
end

"""Load a TrainingHistory, return nothing if file missing."""
function try_load(ds, opt, dev, bas)
    p = loss_path(ds, opt, dev, bas)
    isfile(p) || return nothing
    h = load_loss_history(p)
    # Override basis_name with a descriptive label
    label = "$(DEV_LABEL[dev]) $(OPT_LABEL[opt]) - $(BAS_LABEL[bas])"
    return TrainingHistory(h.train_losses, h.val_losses, h.step_train_losses, label)
end

"""EMA smoothing."""
function smooth(vals::Vector{Float64}, alpha::Float64)
    isempty(vals) && return Float64[]
    s = similar(vals)
    s[1] = vals[1]
    for i in 2:length(vals)
        s[i] = alpha * s[i-1] + (1 - alpha) * vals[i]
    end
    return s
end

# ================================================================================
# Plot generators
# ================================================================================

"""
Plot 1: Per-basis comparison of optimizer×device.
One figure per (dataset, basis) showing 4 step-loss curves (CPU GD, CPU Adam, GPU GD, GPU Adam).
"""
function plot_per_basis(ds, bas; smoothing=0.7)
    fig = Figure(size=(900, 550))
    ax = Axis(fig[1, 1];
        title="$(DS_LABEL[ds]) — $(BAS_LABEL[bas]): Step Loss by Optimizer × Device",
        xlabel="Training Step (batch)",
        ylabel="Loss",
        yscale=log10)

    for dev in DEVICES, opt in OPTIMIZERS
        h = try_load(ds, opt, dev, bas)
        h === nothing && continue

        steps = 1:length(h.step_train_losses)
        key = (dev, opt)
        label = "$(DEV_LABEL[dev]) $(OPT_LABEL[opt])"

        # Raw (faint)
        lines!(ax, steps, h.step_train_losses;
            color=(OPTDEV_COLOR[key], 0.2), linewidth=0.5)
        # Smoothed
        sm = smooth(h.step_train_losses, smoothing)
        lines!(ax, steps, sm;
            label=label, color=OPTDEV_COLOR[key],
            linewidth=OPTDEV_WIDTH[key], linestyle=OPTDEV_STYLE[key])
    end

    axislegend(ax; position=:rt)
    return fig
end

"""
Plot 2: Per optimizer×device comparison of bases.
One figure per (dataset, device, optimizer) showing 3 step-loss curves (QFT, EntangledQFT, TEBD).
"""
function plot_per_optdev(ds, dev, opt; smoothing=0.7)
    fig = Figure(size=(900, 550))
    ax = Axis(fig[1, 1];
        title="$(DS_LABEL[ds]) — $(DEV_LABEL[dev]) $(OPT_LABEL[opt]): Step Loss by Basis",
        xlabel="Training Step (batch)",
        ylabel="Loss",
        yscale=log10)

    for bas in BASES
        h = try_load(ds, opt, dev, bas)
        h === nothing && continue

        steps = 1:length(h.step_train_losses)
        color = BAS_COLOR[bas]

        lines!(ax, steps, h.step_train_losses;
            color=(color, 0.2), linewidth=0.5)
        sm = smooth(h.step_train_losses, smoothing)
        lines!(ax, steps, sm;
            label=BAS_LABEL[bas], color=color, linewidth=2)
    end

    axislegend(ax; position=:rt)
    return fig
end

"""
Plot 3: Grand overview — all 12 curves on one plot per dataset.
"""
function plot_overview(ds; smoothing=0.8)
    fig = Figure(size=(1100, 650))
    ax = Axis(fig[1, 1];
        title="$(DS_LABEL[ds]) — All Configurations: Step Loss",
        xlabel="Training Step (batch)",
        ylabel="Loss",
        yscale=log10)

    palette = Makie.wong_colors()
    idx = 0
    for bas in BASES
        for dev in DEVICES, opt in OPTIMIZERS
            h = try_load(ds, opt, dev, bas)
            h === nothing && continue
            idx += 1

            steps = 1:length(h.step_train_losses)
            color = palette[mod1(idx, length(palette))]
            label = "$(BAS_LABEL[bas]) $(DEV_LABEL[dev]) $(OPT_LABEL[opt])"

            sm = smooth(h.step_train_losses, smoothing)
            lines!(ax, steps, sm;
                label=label, color=color, linewidth=1.8,
                linestyle=(dev == "gpu" ? :dash : :solid))
        end
    end

    axislegend(ax; position=:rt, labelsize=10, nbanks=2)
    return fig
end

"""
Plot 4: Epoch-level train + validation comparison per dataset.
Grid layout: rows = basis, columns = optimizer×device.
"""
function plot_epoch_grid(ds)
    devs_present = [d for d in DEVICES if any(isfile(loss_path(ds, o, d, b))
                     for o in OPTIMIZERS for b in BASES)]
    ncols = length(devs_present) * length(OPTIMIZERS)
    nrows = length(BASES)

    fig = Figure(size=(350 * ncols, 300 * nrows + 60))
    Label(fig[0, :], "$(DS_LABEL[ds]) — Epoch Loss (Train solid, Val dashed)";
        fontsize=18, font=:bold)

    col = 0
    for dev in devs_present, opt in OPTIMIZERS
        col += 1
        for (row, bas) in enumerate(BASES)
            h = try_load(ds, opt, dev, bas)
            h === nothing && continue

            epochs = 1:length(h.train_losses)
            title_str = row == 1 ? "$(DEV_LABEL[dev]) $(OPT_LABEL[opt])" : ""
            ylabel_str = col == 1 ? BAS_LABEL[bas] : ""

            ax = Axis(fig[row, col];
                title=title_str,
                xlabel=(row == nrows ? "Epoch" : ""),
                ylabel=ylabel_str,
                yscale=log10)

            lines!(ax, epochs, h.train_losses;
                label="Train", color=:steelblue, linewidth=2)
            scatter!(ax, epochs, h.train_losses;
                color=:steelblue, markersize=6)
            lines!(ax, epochs, h.val_losses;
                label="Val", color=:tomato, linewidth=2, linestyle=:dash)
            scatter!(ax, epochs, h.val_losses;
                color=:tomato, markersize=6, marker=:rect)

            if row == 1 && col == 1
                axislegend(ax; position=:rt, labelsize=9)
            end
        end
    end

    return fig
end

"""
Plot 5: Cross-dataset comparison per (basis, optimizer×device).
Side-by-side step loss for MNIST vs QuickDraw.
"""
function plot_cross_dataset(bas, dev, opt; smoothing=0.7)
    available_ds = [ds for ds in DATASETS if isfile(loss_path(ds, opt, dev, bas))]
    length(available_ds) < 2 && return nothing

    fig = Figure(size=(900, 550))
    ax = Axis(fig[1, 1];
        title="$(BAS_LABEL[bas]) $(DEV_LABEL[dev]) $(OPT_LABEL[opt]): MNIST vs QuickDraw",
        xlabel="Training Step (batch)",
        ylabel="Loss",
        yscale=log10)

    ds_colors = Dict("mnist" => :steelblue, "quickdraw" => :darkorange)

    for ds in available_ds
        h = try_load(ds, opt, dev, bas)
        h === nothing && continue

        steps = 1:length(h.step_train_losses)
        color = ds_colors[ds]

        lines!(ax, steps, h.step_train_losses;
            color=(color, 0.2), linewidth=0.5)
        sm = smooth(h.step_train_losses, smoothing)
        lines!(ax, steps, sm;
            label=DS_LABEL[ds], color=color, linewidth=2)
    end

    axislegend(ax; position=:rt)
    return fig
end

"""
Plot 6: Adam vs GD head-to-head per (dataset, device).
Each subplot is a basis; two curves (Adam, GD) per subplot.
"""
function plot_adam_vs_gd(ds, dev; smoothing=0.7)
    fig = Figure(size=(1100, 400))
    Label(fig[0, :], "$(DS_LABEL[ds]) $(DEV_LABEL[dev]): Adam vs GD";
        fontsize=16, font=:bold)

    for (col, bas) in enumerate(BASES)
        ax = Axis(fig[1, col];
            title=BAS_LABEL[bas],
            xlabel="Step",
            ylabel=(col == 1 ? "Loss" : ""),
            yscale=log10)

        for opt in OPTIMIZERS
            h = try_load(ds, opt, dev, bas)
            h === nothing && continue

            steps = 1:length(h.step_train_losses)
            color = opt == "adam" ? :darkorange : :steelblue

            lines!(ax, steps, h.step_train_losses;
                color=(color, 0.2), linewidth=0.5)
            sm = smooth(h.step_train_losses, smoothing)
            lines!(ax, steps, sm;
                label=OPT_LABEL[opt], color=color, linewidth=2)
        end

        axislegend(ax; position=:rt, labelsize=10)
    end

    return fig
end

"""
Plot 7: CPU vs GPU head-to-head per (dataset, optimizer).
Each subplot is a basis; two curves (CPU, GPU) per subplot.
"""
function plot_cpu_vs_gpu(ds, opt; smoothing=0.7)
    fig = Figure(size=(1100, 400))
    Label(fig[0, :], "$(DS_LABEL[ds]) $(OPT_LABEL[opt]): CPU vs GPU";
        fontsize=16, font=:bold)

    for (col, bas) in enumerate(BASES)
        ax = Axis(fig[1, col];
            title=BAS_LABEL[bas],
            xlabel="Step",
            ylabel=(col == 1 ? "Loss" : ""),
            yscale=log10)

        for dev in DEVICES
            h = try_load(ds, opt, dev, bas)
            h === nothing && continue

            steps = 1:length(h.step_train_losses)
            color = dev == "gpu" ? :mediumseagreen : :royalblue

            lines!(ax, steps, h.step_train_losses;
                color=(color, 0.2), linewidth=0.5)
            sm = smooth(h.step_train_losses, smoothing)
            lines!(ax, steps, sm;
                label=DEV_LABEL[dev], color=color, linewidth=2)
        end

        axislegend(ax; position=:rt, labelsize=10)
    end

    return fig
end

# ================================================================================
# Main
# ================================================================================

function main()
    if !isdir(LOSS_DIR)
        error("Loss history directory not found: $LOSS_DIR\n" *
              "Run optimizer_basis_comparison.jl first.")
    end

    # Detect which datasets/devices actually have data
    available_ds = [ds for ds in DATASETS
                    if any(isfile(loss_path(ds, o, d, b))
                           for o in OPTIMIZERS for d in DEVICES for b in BASES)]
    available_devs = [dev for dev in DEVICES
                      if any(isfile(loss_path(ds, o, dev, b))
                             for ds in available_ds for o in OPTIMIZERS for b in BASES)]

    n_files = length(filter(f -> endswith(f, ".json"), readdir(LOSS_DIR)))
    println("=" ^ 80)
    println("  Plot Comparison Results")
    println("  Found $n_files loss history files")
    println("  Datasets: $(join([DS_LABEL[d] for d in available_ds], ", "))")
    println("  Devices:  $(join([DEV_LABEL[d] for d in available_devs], ", "))")
    println("=" ^ 80)

    mkpath(PLOTS_DIR)
    saved = String[]

    # --- Plot 1: Per-basis (optimizer×device comparison) ---
    dir1 = joinpath(PLOTS_DIR, "by_basis")
    mkpath(dir1)
    for ds in available_ds, bas in BASES
        fig = plot_per_basis(ds, bas)
        path = joinpath(dir1, "$(ds)_$(bas).png")
        save(path, fig)
        push!(saved, path)
    end
    println("  [1/7] Per-basis plots: $(length(available_ds) * length(BASES)) saved")

    # --- Plot 2: Per optimizer×device (basis comparison) ---
    dir2 = joinpath(PLOTS_DIR, "by_optdev")
    mkpath(dir2)
    for ds in available_ds, dev in available_devs, opt in OPTIMIZERS
        fig = plot_per_optdev(ds, dev, opt)
        path = joinpath(dir2, "$(ds)_$(dev)_$(opt).png")
        save(path, fig)
        push!(saved, path)
    end
    println("  [2/7] Per-opt/device plots: $(length(available_ds) * length(available_devs) * length(OPTIMIZERS)) saved")

    # --- Plot 3: Grand overview per dataset ---
    dir3 = joinpath(PLOTS_DIR, "overview")
    mkpath(dir3)
    for ds in available_ds
        fig = plot_overview(ds)
        path = joinpath(dir3, "$(ds)_overview.png")
        save(path, fig)
        push!(saved, path)
    end
    println("  [3/7] Overview plots: $(length(available_ds)) saved")

    # --- Plot 4: Epoch grid per dataset ---
    dir4 = joinpath(PLOTS_DIR, "epoch_grid")
    mkpath(dir4)
    for ds in available_ds
        fig = plot_epoch_grid(ds)
        path = joinpath(dir4, "$(ds)_epoch_grid.png")
        save(path, fig)
        push!(saved, path)
    end
    println("  [4/7] Epoch grid plots: $(length(available_ds)) saved")

    # --- Plot 5: Cross-dataset comparison ---
    if length(available_ds) > 1
        dir5 = joinpath(PLOTS_DIR, "cross_dataset")
        mkpath(dir5)
        count5 = 0
        for bas in BASES, dev in available_devs, opt in OPTIMIZERS
            fig = plot_cross_dataset(bas, dev, opt)
            fig === nothing && continue
            path = joinpath(dir5, "$(bas)_$(dev)_$(opt).png")
            save(path, fig)
            push!(saved, path)
            count5 += 1
        end
        println("  [5/7] Cross-dataset plots: $count5 saved")
    else
        println("  [5/7] Cross-dataset plots: skipped (only 1 dataset)")
    end

    # --- Plot 6: Adam vs GD head-to-head ---
    dir6 = joinpath(PLOTS_DIR, "adam_vs_gd")
    mkpath(dir6)
    for ds in available_ds, dev in available_devs
        fig = plot_adam_vs_gd(ds, dev)
        path = joinpath(dir6, "$(ds)_$(dev).png")
        save(path, fig)
        push!(saved, path)
    end
    println("  [6/7] Adam vs GD plots: $(length(available_ds) * length(available_devs)) saved")

    # --- Plot 7: CPU vs GPU head-to-head ---
    if length(available_devs) > 1
        dir7 = joinpath(PLOTS_DIR, "cpu_vs_gpu")
        mkpath(dir7)
        for ds in available_ds, opt in OPTIMIZERS
            fig = plot_cpu_vs_gpu(ds, opt)
            path = joinpath(dir7, "$(ds)_$(opt).png")
            save(path, fig)
            push!(saved, path)
        end
        println("  [7/7] CPU vs GPU plots: $(length(available_ds) * length(OPTIMIZERS)) saved")
    else
        println("  [7/7] CPU vs GPU plots: skipped (only 1 device)")
    end

    println("\n" * "=" ^ 80)
    println("  Total plots saved: $(length(saved))")
    println("  Output directory:  $PLOTS_DIR")
    println("=" ^ 80)

    # List all subdirectories and their contents
    println("\nPlots directory structure:")
    for subdir in sort(readdir(PLOTS_DIR))
        subpath = joinpath(PLOTS_DIR, subdir)
        isdir(subpath) || continue
        files = sort(filter(f -> endswith(f, ".png"), readdir(subpath)))
        println("  $subdir/ ($(length(files)) files)")
        for f in files
            println("    $f")
        end
    end

    return saved
end

main()
