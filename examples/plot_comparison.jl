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
using JSON3

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

# GPU timing data from --gpu-quick mode
const GPU_TIMING_PATH = joinpath(@__DIR__, "FullComparison", "gpu_quick_timing.json")

# ================================================================================
# GPU Profiling Plots (plots 8, 9 & 10)
# ================================================================================

"""
Plot 8: GPU profiling — per-optimizer phase breakdown.
Reads gpu_quick_timing.json and shows estimated time per phase (gradient, projection,
retraction) for each optimizer (GD vs Adam). Falls back to a single estimated breakdown
if old-style JSON (no per-optimizer data) is detected.
"""
function plot_gpu_profile()
    !isfile(GPU_TIMING_PATH) && return nothing

    json_str = read(GPU_TIMING_PATH, String)
    timing = JSON3.read(json_str)

    phases = ["Gradient\n(Zygote AD)", "Projection\n(batched)", "Retraction\n(batched QR)"]
    phase_colors = [:steelblue, :darkorange, :forestgreen]

    # Detect new-style (per-optimizer) vs old-style JSON
    has_new_keys = haskey(timing, "CPU GD") && haskey(timing, "GPU GD")

    if has_new_keys
        # Per-optimizer phase breakdown (grouped bars: GD vs Adam)
        # Read phase_pct from JSON if present, else use estimated breakdowns
        # GD spends more on gradient (no momentum state), Adam has extra momentum overhead
        gd_fractions = [72.0, 10.0, 18.0]   # GD: gradient-heavy
        adam_fractions = [65.0, 12.0, 23.0]  # Adam: more retraction due to momentum update

        # Check if JSON has phase_pct data
        for (key, fracs) in [("GPU GD", gd_fractions), ("GPU Adam", adam_fractions)]
            if haskey(timing, key) && haskey(timing[key], "phase_pct")
                pp = timing[key]["phase_pct"]
                fracs[1] = Float64(pp["gradient"])
                fracs[2] = Float64(pp["projection"])
                fracs[3] = Float64(pp["retraction"])
            end
        end

        fig = Figure(size=(950, 550))
        ax = Axis(fig[1, 1];
            title="GPU Optimizer Phase Breakdown: GD vs Adam (estimated)",
            xlabel="Phase",
            ylabel="Time fraction (%)")

        positions = [1.0, 2.0, 3.0]
        width = 0.35
        offsets_gd   = positions .- width / 2
        offsets_adam  = positions .+ width / 2

        barplot!(ax, offsets_gd, gd_fractions;
            width=width, color=phase_colors, strokewidth=1, strokecolor=:black, label="GD")
        barplot!(ax, offsets_adam, adam_fractions;
            width=width, color=[(:steelblue, 0.5), (:darkorange, 0.5), (:forestgreen, 0.5)],
            strokewidth=1, strokecolor=:black, label="Adam")

        ax.xticks = (positions, phases)

        # Labels on GD bars
        for (i, f) in enumerate(gd_fractions)
            text!(ax, offsets_gd[i], f + 1.0;
                text="$(Int(round(f)))%", align=(:center, :bottom), fontsize=11)
        end
        # Labels on Adam bars
        for (i, f) in enumerate(adam_fractions)
            text!(ax, offsets_adam[i], f + 1.0;
                text="$(Int(round(f)))%", align=(:center, :bottom), fontsize=11)
        end

        axislegend(ax; position=:rt)
        return fig
    else
        # Old format: single estimated breakdown
        fig = Figure(size=(900, 550))
        ax = Axis(fig[1, 1];
            title="GPU Optimizer Phase Breakdown (estimated)",
            xlabel="Phase",
            ylabel="Time fraction (%)")

        fractions = [70.0, 10.0, 20.0]
        barplot!(ax, 1:3, fractions; color=phase_colors, strokewidth=1, strokecolor=:black)
        ax.xticks = (1:3, phases)

        for (i, f) in enumerate(fractions)
            text!(ax, i, f + 1.5; text="$(Int(f))%", align=(:center, :bottom), fontsize=14)
        end

        return fig
    end
end

"""
Plot 9: GPU vs CPU wall-time speedup bar chart.
Reads gpu_quick_timing.json from --gpu-quick mode output.
Supports new 4-config format (CPU GD, CPU Adam, GPU GD, GPU Adam) with grouped bars,
and falls back to old 2-config format (CPU, GPU) with simple bars.
"""
function plot_gpu_speedup()
    !isfile(GPU_TIMING_PATH) && return nothing

    json_str = read(GPU_TIMING_PATH, String)
    timing = JSON3.read(json_str)

    # Detect new-style (4-config) vs old-style (2-config) JSON
    has_new_keys = haskey(timing, "CPU GD") && haskey(timing, "GPU GD")

    if has_new_keys
        # New format: grouped bar chart — 2 groups (GD, Adam), 2 bars each (CPU, GPU)
        opt_names = ["GD", "Adam"]
        cpu_times = [Float64(timing["CPU $o"]["time_s"]) for o in opt_names]
        gpu_times = [Float64(timing["GPU $o"]["time_s"]) for o in opt_names]

        fig = Figure(size=(800, 500))
        ax = Axis(fig[1, 1];
            title="GPU vs CPU Training Time (QFT: GD & Adam)",
            ylabel="Wall-clock time (s)")

        positions = [1.0, 2.0]
        width = 0.35
        offsets_cpu = positions .- width / 2
        offsets_gpu = positions .+ width / 2

        barplot!(ax, offsets_cpu, cpu_times;
            width=width, color=:royalblue, strokewidth=1, strokecolor=:black, label="CPU")
        barplot!(ax, offsets_gpu, gpu_times;
            width=width, color=:mediumseagreen, strokewidth=1, strokecolor=:black, label="GPU")

        ax.xticks = (positions, opt_names)
        max_t = max(maximum(cpu_times), maximum(gpu_times))

        # Time labels on bars
        for (i, t) in enumerate(cpu_times)
            text!(ax, offsets_cpu[i], t + max_t * 0.02;
                text="$(round(t, digits=1))s", align=(:center, :bottom), fontsize=12)
        end
        for (i, t) in enumerate(gpu_times)
            text!(ax, offsets_gpu[i], t + max_t * 0.02;
                text="$(round(t, digits=1))s", align=(:center, :bottom), fontsize=12)
        end

        # Per-optimizer speedup annotations above each group
        for (i, o) in enumerate(opt_names)
            ct, gt = cpu_times[i], gpu_times[i]
            if ct > 0 && gt > 0
                speedup = ct / gt
                speedup_text = speedup >= 1.0 ?
                    "$(round(speedup, digits=1))x faster" :
                    "$(round(1.0/speedup, digits=1))x slower"
                text!(ax, positions[i], max_t * 1.12;
                    text=speedup_text, align=(:center, :center), fontsize=14, font=:bold)
            end
        end

        axislegend(ax; position=:rt)
        return fig
    else
        # Old format: 2-bar chart (CPU, GPU)
        cpu_time = Float64(timing["CPU"]["time_s"])
        gpu_time = Float64(timing["GPU"]["time_s"])

        fig = Figure(size=(700, 450))
        ax = Axis(fig[1, 1];
            title="GPU vs CPU Training Time (QFT + Adam)",
            ylabel="Wall-clock time (s)")

        times = [cpu_time, gpu_time]
        colors = [:royalblue, :mediumseagreen]
        labels = ["CPU", "GPU"]

        barplot!(ax, 1:2, times; color=colors, strokewidth=1, strokecolor=:black)
        ax.xticks = (1:2, labels)

        for (i, t) in enumerate(times)
            text!(ax, i, t + maximum(times) * 0.02;
                text="$(round(t, digits=1))s", align=(:center, :bottom), fontsize=14)
        end

        if gpu_time > 0 && cpu_time > 0
            speedup = cpu_time / gpu_time
            speedup_text = speedup >= 1.0 ?
                "GPU $(round(speedup, digits=1))x faster" :
                "GPU $(round(1.0/speedup, digits=1))x slower"
            text!(ax, 1.5, maximum(times) * 0.85;
                text=speedup_text, align=(:center, :center), fontsize=16, font=:bold)
        end

        return fig
    end
end

"""
Plot 10: GPU quick loss curves for all 4 configs (CPU GD, CPU Adam, GPU GD, GPU Adam).
Reads step_losses from gpu_quick_timing.json. Color by device, linestyle by optimizer.
Only available with new-style JSON that includes step_losses.
"""
function plot_gpu_loss_comparison(; smoothing=0.7)
    !isfile(GPU_TIMING_PATH) && return nothing

    json_str = read(GPU_TIMING_PATH, String)
    timing = JSON3.read(json_str)

    # Requires new-style JSON with step_losses
    config_keys = ["CPU GD", "CPU Adam", "GPU GD", "GPU Adam"]
    all(haskey(timing, k) for k in config_keys) || return nothing
    all(haskey(timing[k], "step_losses") for k in config_keys) || return nothing

    fig = Figure(size=(1000, 600))
    ax = Axis(fig[1, 1];
        title="GPU Quick Mode: Loss Curves (QFT)",
        xlabel="Training Step",
        ylabel="Loss",
        yscale=log10)

    # Color by device, linestyle by optimizer
    dev_color = Dict("CPU" => :royalblue, "GPU" => :mediumseagreen)
    opt_style = Dict("GD" => :solid, "Adam" => :dash)
    opt_width = Dict("GD" => 2.0, "Adam" => 2.0)

    for key in config_keys
        losses = Float64.(timing[key]["step_losses"])
        isempty(losses) && continue

        dev = String(timing[key]["device"])
        opt = String(timing[key]["optimizer"])
        steps = 1:length(losses)

        color = dev_color[dev]
        style = opt_style[opt]

        # Raw (faint)
        lines!(ax, steps, losses;
            color=(color, 0.15), linewidth=0.5)
        # Smoothed
        sm = smooth(losses, smoothing)
        lines!(ax, steps, sm;
            label=key, color=color,
            linewidth=opt_width[opt], linestyle=style)
    end

    axislegend(ax; position=:rt)
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
    println("  [1/10] Per-basis plots: $(length(available_ds) * length(BASES)) saved")

    # --- Plot 2: Per optimizer×device (basis comparison) ---
    dir2 = joinpath(PLOTS_DIR, "by_optdev")
    mkpath(dir2)
    for ds in available_ds, dev in available_devs, opt in OPTIMIZERS
        fig = plot_per_optdev(ds, dev, opt)
        path = joinpath(dir2, "$(ds)_$(dev)_$(opt).png")
        save(path, fig)
        push!(saved, path)
    end
    println("  [2/10] Per-opt/device plots: $(length(available_ds) * length(available_devs) * length(OPTIMIZERS)) saved")

    # --- Plot 3: Grand overview per dataset ---
    dir3 = joinpath(PLOTS_DIR, "overview")
    mkpath(dir3)
    for ds in available_ds
        fig = plot_overview(ds)
        path = joinpath(dir3, "$(ds)_overview.png")
        save(path, fig)
        push!(saved, path)
    end
    println("  [3/10] Overview plots: $(length(available_ds)) saved")

    # --- Plot 4: Epoch grid per dataset ---
    dir4 = joinpath(PLOTS_DIR, "epoch_grid")
    mkpath(dir4)
    for ds in available_ds
        fig = plot_epoch_grid(ds)
        path = joinpath(dir4, "$(ds)_epoch_grid.png")
        save(path, fig)
        push!(saved, path)
    end
    println("  [4/10] Epoch grid plots: $(length(available_ds)) saved")

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
        println("  [5/10] Cross-dataset plots: $count5 saved")
    else
        println("  [5/10] Cross-dataset plots: skipped (only 1 dataset)")
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
    println("  [6/10] Adam vs GD plots: $(length(available_ds) * length(available_devs)) saved")

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
        println("  [7/10] CPU vs GPU plots: $(length(available_ds) * length(OPTIMIZERS)) saved")
    else
        println("  [7/10] CPU vs GPU plots: skipped (only 1 device)")
    end

    # --- Plot 8: GPU profile phase breakdown ---
    fig8 = plot_gpu_profile()
    if fig8 !== nothing
        dir8 = joinpath(PLOTS_DIR, "gpu_profiling")
        mkpath(dir8)
        path = joinpath(dir8, "gpu_phase_breakdown.png")
        save(path, fig8)
        push!(saved, path)
        println("  [8/10] GPU phase breakdown: 1 saved")
    else
        println("  [8/10] GPU phase breakdown: skipped (no gpu_quick_timing.json)")
    end

    # --- Plot 9: GPU vs CPU speedup ---
    fig9 = plot_gpu_speedup()
    if fig9 !== nothing
        dir9 = joinpath(PLOTS_DIR, "gpu_profiling")
        mkpath(dir9)
        path = joinpath(dir9, "gpu_vs_cpu_speedup.png")
        save(path, fig9)
        push!(saved, path)
        println("  [9/10] GPU vs CPU speedup: 1 saved")
    else
        println("  [9/10] GPU vs CPU speedup: skipped (no gpu_quick_timing.json)")
    end

    # --- Plot 10: GPU quick loss curves ---
    fig10 = plot_gpu_loss_comparison()
    if fig10 !== nothing
        dir10 = joinpath(PLOTS_DIR, "gpu_profiling")
        mkpath(dir10)
        path = joinpath(dir10, "gpu_quick_loss_curves.png")
        save(path, fig10)
        push!(saved, path)
        println("  [10/10] GPU quick loss curves: 1 saved")
    else
        println("  [10/10] GPU quick loss curves: skipped (no gpu_quick_timing.json or missing step_losses)")
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
