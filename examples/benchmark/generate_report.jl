# ============================================================================
# Benchmark Report Generator
# ============================================================================
# Loads results from all dataset benchmarks and produces:
# 1. Rate-distortion tables (CSV)
# 2. Training loss curves (PNG)
# 3. Visual comparison grids (PNG)
# 4. Cross-dataset summary table (CSV)
# 5. Timing table (CSV)
#
# Usage:
#   julia --project=examples/benchmark examples/benchmark/generate_report.jl
# ============================================================================

include("config.jl")
include("data_loading.jl")
include("evaluation.jl")

using CairoMakie
using ParametricDFT

const DATASET_NAMES = [:quickdraw, :div2k, :atd12k]
const DISPLAY_NAMES = Dict(
    :quickdraw => "Quick Draw",
    :div2k => "DIV2K",
    :atd12k => "ATD-12K",
)
const BASIS_DISPLAY_NAMES = Dict(
    "qft" => "QFT",
    "entangled_qft" => "Entangled QFT",
    "tebd" => "TEBD",
    "mera" => "MERA",
    "fft" => "Classical FFT",
)
const BASIS_COLORS = Dict(
    "qft" => :blue,
    "entangled_qft" => :red,
    "tebd" => :green,
    "mera" => :purple,
    "fft" => :black,
)

# ============================================================================
# Load Results
# ============================================================================

function load_all_results()
    all_results = Dict{Symbol,Any}()
    for dataset_name in DATASET_NAMES
        metrics_path = joinpath(RESULTS_DIR, string(dataset_name), "metrics.json")
        if isfile(metrics_path)
            all_results[dataset_name] = load_benchmark_results(metrics_path)
            @info "Loaded results for $(DISPLAY_NAMES[dataset_name])"
        else
            @warn "No results found for $(DISPLAY_NAMES[dataset_name]) at $metrics_path"
        end
    end
    return all_results
end

# ============================================================================
# 1. Rate-Distortion Tables (CSV)
# ============================================================================

function generate_rate_distortion_csv(all_results)
    for (dataset_name, results) in all_results
        output_dir = joinpath(RESULTS_DIR, string(dataset_name))
        mkpath(output_dir)

        for metric_name in ["psnr", "ssim", "mse"]
            csv_path = joinpath(output_dir, "rate_distortion_$(metric_name).csv")
            open(csv_path, "w") do io
                # Header
                print(io, "Basis")
                for ratio in KEEP_RATIOS
                    print(io, ",$(round(Int, ratio * 100))%_kept")
                end
                println(io)

                # Rows
                for basis_name in ["qft", "entangled_qft", "tebd", "mera", "fft"]
                    if haskey(results, basis_name)
                        print(io, BASIS_DISPLAY_NAMES[basis_name])
                        basis_data = results[basis_name]
                        metrics = basis_data["metrics"]
                        for ratio in KEEP_RATIOS
                            ratio_key = string(ratio)
                            if haskey(metrics, ratio_key)
                                val = metrics[ratio_key]["mean_$(metric_name)"]
                                print(io, ",$(val)")
                            else
                                print(io, ",N/A")
                            end
                        end
                        println(io)
                    end
                end
            end
            @info "Saved $csv_path"
        end
    end
end

# ============================================================================
# 2. Training Loss Curves
# ============================================================================

function generate_training_curves(all_results)
    for (dataset_name, results) in all_results
        plots_dir = joinpath(RESULTS_DIR, string(dataset_name), "plots")
        mkpath(plots_dir)

        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1];
            xlabel = "Epoch",
            ylabel = "Validation Loss",
            title = "Training Convergence — $(DISPLAY_NAMES[dataset_name])",
            yscale = log10,
        )

        for basis_name in ["qft", "entangled_qft", "tebd", "mera"]
            if haskey(results, basis_name) && haskey(results[basis_name], "history")
                history = results[basis_name]["history"]
                val_losses = Float64.(history["val_losses"])
                if !isempty(val_losses)
                    lines!(ax, 1:length(val_losses), val_losses;
                        label = BASIS_DISPLAY_NAMES[basis_name],
                        color = BASIS_COLORS[basis_name],
                    )
                end
            end
        end

        axislegend(ax; position = :rt)
        save(joinpath(plots_dir, "training_curves.png"), fig; px_per_unit = 2)
        @info "Saved training curves for $(DISPLAY_NAMES[dataset_name])"
    end
end

# ============================================================================
# 3. Visual Comparison Grids
# ============================================================================

function generate_reconstruction_grids(all_results)
    for (dataset_name, results) in all_results
        plots_dir = joinpath(RESULTS_DIR, string(dataset_name), "plots")
        mkpath(plots_dir)
        output_dir = joinpath(RESULTS_DIR, string(dataset_name))

        # Load first test image using the appropriate loader
        dataset_config = DATASET_CONFIGS[dataset_name]
        # We need a test image — load just 1
        test_images = try
            if dataset_name == :quickdraw
                _, test, _ = load_quickdraw_dataset(; n_train = 1, n_test = 1)
                test
            elseif dataset_name == :div2k
                _, test, _ = load_div2k_dataset(; n_train = 1, n_test = 1)
                test
            else
                _, test, _ = load_atd12k_dataset(; n_train = 1, n_test = 1)
                test
            end
        catch e
            @warn "Could not load test image for $dataset_name: $e"
            continue
        end

        sample_img = test_images[1]
        basis_order = ["qft", "entangled_qft", "tebd", "mera", "fft"]

        # Load trained bases
        trained_bases = Dict{String,Any}()
        for basis_name in ["qft", "entangled_qft", "tebd", "mera"]
            basis_path = joinpath(output_dir, "trained_$(basis_name).json")
            if isfile(basis_path)
                trained_bases[basis_name] = load_basis(basis_path)
            end
        end

        n_rows = 1 + length(basis_order)  # original + each basis
        n_cols = length(KEEP_RATIOS)

        fig = Figure(size = (250 * n_cols, 200 * n_rows))

        # Column headers
        for (j, ratio) in enumerate(KEEP_RATIOS)
            Label(fig[0, j], "$(round(Int, ratio * 100))% kept"; fontsize = 14)
        end

        # Original row
        Label(fig[1, 0], "Original"; fontsize = 12, rotation = pi / 2)
        for j in 1:n_cols
            ax = Axis(fig[1, j]; aspect = DataAspect())
            hidedecorations!(ax)
            heatmap!(ax, rotr90(sample_img); colormap = :grays)
        end

        # Basis rows
        for (i, basis_name) in enumerate(basis_order)
            row = i + 1
            Label(fig[row, 0], get(BASIS_DISPLAY_NAMES, basis_name, basis_name);
                fontsize = 12, rotation = pi / 2)

            for (j, keep_ratio) in enumerate(KEEP_RATIOS)
                ax = Axis(fig[row, j]; aspect = DataAspect())
                hidedecorations!(ax)

                recovered = if basis_name == "fft"
                    fft_compress_recover(sample_img, keep_ratio)
                elseif haskey(trained_bases, basis_name)
                    basis = trained_bases[basis_name]
                    compressed = compress(basis, sample_img; ratio = 1.0 - keep_ratio)
                    real.(recover(basis, compressed))
                else
                    zeros(size(sample_img))
                end

                heatmap!(ax, rotr90(clamp.(recovered, 0.0, 1.0)); colormap = :grays)
            end
        end

        save(joinpath(plots_dir, "reconstruction_grid.png"), fig; px_per_unit = 2)
        @info "Saved reconstruction grid for $(DISPLAY_NAMES[dataset_name])"
    end
end

# ============================================================================
# 4. Cross-Dataset Summary Table
# ============================================================================

function generate_cross_dataset_summary(all_results)
    csv_path = joinpath(RESULTS_DIR, "cross_dataset_summary.csv")

    open(csv_path, "w") do io
        # Header
        print(io, "Basis")
        for dataset_name in DATASET_NAMES
            if haskey(all_results, dataset_name)
                print(io, ",$(DISPLAY_NAMES[dataset_name]) PSNR@10%")
            end
        end
        println(io, ",Avg Rank")

        basis_order = ["qft", "entangled_qft", "tebd", "mera", "fft"]

        # Compute ranks per dataset
        ranks = Dict{String,Vector{Float64}}()
        for basis_name in basis_order
            ranks[basis_name] = Float64[]
        end

        for dataset_name in DATASET_NAMES
            haskey(all_results, dataset_name) || continue
            results = all_results[dataset_name]

            # Get PSNR@10% for each basis
            psnr_values = Dict{String,Float64}()
            for basis_name in basis_order
                if haskey(results, basis_name)
                    metrics = results[basis_name]["metrics"]
                    if haskey(metrics, "0.1")
                        psnr_values[basis_name] = Float64(metrics["0.1"]["mean_psnr"])
                    end
                end
            end

            # Rank by PSNR (higher = better = lower rank)
            sorted = sort(collect(psnr_values); by = x -> -x[2])
            for (rank, (name, _)) in enumerate(sorted)
                push!(ranks[name], Float64(rank))
            end
        end

        # Write rows
        for basis_name in basis_order
            print(io, BASIS_DISPLAY_NAMES[basis_name])
            for dataset_name in DATASET_NAMES
                if haskey(all_results, dataset_name) && haskey(all_results[dataset_name], basis_name)
                    metrics = all_results[dataset_name][basis_name]["metrics"]
                    if haskey(metrics, "0.1")
                        print(io, ",$(Float64(metrics["0.1"]["mean_psnr"]))")
                    else
                        print(io, ",N/A")
                    end
                else
                    print(io, ",N/A")
                end
            end
            avg_rank = isempty(ranks[basis_name]) ? NaN : mean(ranks[basis_name])
            println(io, ",$avg_rank")
        end
    end

    @info "Saved cross-dataset summary to $csv_path"

    # Also print to console
    println("\n" * "=" ^ 80)
    println("CROSS-DATASET SUMMARY (PSNR @ 10% kept)")
    println("=" ^ 80)
    println(read(csv_path, String))
end

# ============================================================================
# 5. Timing Table
# ============================================================================

function generate_timing_table(all_results)
    csv_path = joinpath(RESULTS_DIR, "timing_summary.csv")

    open(csv_path, "w") do io
        print(io, "Basis")
        for dataset_name in DATASET_NAMES
            if haskey(all_results, dataset_name)
                print(io, ",$(DISPLAY_NAMES[dataset_name]) Time(s)")
            end
        end
        println(io)

        for basis_name in ["qft", "entangled_qft", "tebd", "mera", "fft"]
            print(io, BASIS_DISPLAY_NAMES[basis_name])
            for dataset_name in DATASET_NAMES
                if haskey(all_results, dataset_name) && haskey(all_results[dataset_name], basis_name)
                    t = all_results[dataset_name][basis_name]["time"]
                    @printf(io, ",%.1f", Float64(t))
                else
                    print(io, ",N/A")
                end
            end
            println(io)
        end
    end

    @info "Saved timing summary to $csv_path"
    println("\n" * "=" ^ 80)
    println("TIMING SUMMARY")
    println("=" ^ 80)
    println(read(csv_path, String))
end

# ============================================================================
# Cross-Dataset Plots
# ============================================================================

function generate_cross_dataset_plots(all_results)
    plots_dir = joinpath(RESULTS_DIR, "plots")
    mkpath(plots_dir)

    basis_order = ["qft", "entangled_qft", "tebd", "mera", "fft"]
    available_datasets = [d for d in DATASET_NAMES if haskey(all_results, d)]

    for (metric_name, ylabel, higher_better) in [
        ("psnr", "PSNR (dB)", true),
        ("ssim", "SSIM", true),
    ]
        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1];
            xlabel = "Dataset",
            ylabel = ylabel,
            title = "Cross-Dataset Comparison — $(uppercase(metric_name)) @ 10% kept",
            xticks = (1:length(available_datasets), [DISPLAY_NAMES[d] for d in available_datasets]),
        )

        n_bases = length(basis_order)
        bar_width = 0.15

        for (bi, basis_name) in enumerate(basis_order)
            values = Float64[]
            positions = Float64[]
            for (di, dataset_name) in enumerate(available_datasets)
                if haskey(all_results[dataset_name], basis_name)
                    metrics = all_results[dataset_name][basis_name]["metrics"]
                    if haskey(metrics, "0.1")
                        push!(values, Float64(metrics["0.1"]["mean_$(metric_name)"]))
                        push!(positions, di + (bi - (n_bases + 1) / 2) * bar_width)
                    end
                end
            end
            if !isempty(values)
                barplot!(ax, positions, values;
                    width = bar_width,
                    color = BASIS_COLORS[basis_name],
                    label = BASIS_DISPLAY_NAMES[basis_name],
                )
            end
        end

        axislegend(ax; position = :rt)
        save(joinpath(plots_dir, "cross_dataset_$(metric_name).png"), fig; px_per_unit = 2)
        @info "Saved cross-dataset $(metric_name) plot"
    end
end

# ============================================================================
# Main
# ============================================================================

function main()
    println("=" ^ 80)
    println("Generating Benchmark Report")
    println("=" ^ 80)

    all_results = load_all_results()

    if isempty(all_results)
        error("No results found. Run the benchmark scripts first.")
    end

    generate_rate_distortion_csv(all_results)
    generate_training_curves(all_results)
    generate_reconstruction_grids(all_results)
    generate_cross_dataset_summary(all_results)
    generate_cross_dataset_plots(all_results)
    generate_timing_table(all_results)

    println("\n" * "=" ^ 80)
    println("Report generation complete!")
    println("Results in: $RESULTS_DIR")
    println("=" ^ 80)
end

main()
