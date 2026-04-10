# ============================================================================
# Optimizer Benchmark: PDFT vs Manopt.jl
# ============================================================================
# Validates ParametricDFT's customized Riemannian GD optimizer against
# Manopt.jl's gradient_descent on QuickDraw images.
#
# Run:
#   julia --project=examples examples/optimizer_benchmark.jl
# ============================================================================

using ParametricDFT
using CUDA
using CairoMakie
using JSON3
using Printf
using Random
using Statistics
using Dates
using Downloads
using NPZ
using ImageQualityIndexes: assess_psnr, assess_ssim

# Manopt stack
using Manopt, Manifolds, ManifoldDiff
using RecursiveArrayTools: ArrayPartition
using ADTypes: AutoZygote
import Zygote

# ============================================================================
# Configuration
# ============================================================================

const M_PARAM = 5
const N_PARAM = 5
const IMAGE_SIZE = 32

const N_TRAIN = 20
const N_TEST = 5
const SEED = 42
const GPU_DEVICE = 1

const SMOKE_STEPS = 5
const FULL_STEPS = 50

const LOSS_K = round(Int, 2^(M_PARAM + N_PARAM) * 0.1)
const COMPRESSION_RATIOS = [0.8, 0.9, 0.95]

const DATA_DIR = joinpath(@__DIR__, "benchmark", "data", "quickdraw")
const OUTPUT_DIR = joinpath(@__DIR__, "OptimizerBenchmark", "quickdraw")
const QUICKDRAW_CATEGORIES = ["cat", "dog", "airplane", "apple", "bicycle"]
const QUICKDRAW_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

const OPTIMIZER_CONFIGS = [
    ("Manopt-GD",      :manopt, :gradient_descent, :cpu),
    ("PDFT-GD (cpu)",  :pdft,   :gradient_descent, :cpu),
    ("PDFT-GD (gpu)",  :pdft,   :gradient_descent, :gpu),
]

const PLOT_STYLES = Dict(
    "Manopt-GD"       => (color=:black,     linestyle=:solid),
    "PDFT-GD (cpu)"   => (color=:blue,      linestyle=:dash),
    "PDFT-GD (gpu)"   => (color=:blue,      linestyle=:solid),
)

# ============================================================================
# Data Loading
# ============================================================================

"""Center-pad a 28x28 QuickDraw image to the benchmark power-of-two size."""
function pad_to_power_of_two(img::AbstractMatrix, target_size::Int)
    h, w = size(img)
    padded = zeros(Float64, target_size, target_size)
    y_offset = (target_size - h) ÷ 2 + 1
    x_offset = (target_size - w) ÷ 2 + 1
    padded[y_offset:y_offset+h-1, x_offset:x_offset+w-1] = Float64.(img)
    return padded
end

"""Load QuickDraw bitmap images, auto-downloading missing category files."""
function load_quickdraw()
    mkpath(DATA_DIR)
    for category in QUICKDRAW_CATEGORIES
        filepath = joinpath(DATA_DIR, "$(category).npy")
        if !isfile(filepath)
            url = "$(QUICKDRAW_BASE_URL)/$(category).npy"
            @info "Downloading QuickDraw category" category url
            Downloads.download(url, filepath)
        end
    end

    images = Matrix{Float64}[]
    labels = String[]

    per_category = ceil(Int, (N_TRAIN + N_TEST) / length(QUICKDRAW_CATEGORIES))
    for category in QUICKDRAW_CATEGORIES
        data = npzread(joinpath(DATA_DIR, "$(category).npy"))
        n_to_load = min(size(data, 1), per_category)
        for i in 1:n_to_load
            img = reshape(Float64.(data[i, :]), 28, 28) ./ 255.0
            push!(images, pad_to_power_of_two(img, IMAGE_SIZE))
            push!(labels, category)
        end
    end
    @assert length(images) >= N_TRAIN + N_TEST "Need $(N_TRAIN + N_TEST) QuickDraw images, found $(length(images))"

    Random.seed!(SEED)
    selected = randperm(length(images))[1:(N_TRAIN + N_TEST)]
    images = images[selected]
    labels = labels[selected]

    train_images = images[1:N_TRAIN]
    test_images = images[N_TRAIN+1:end]
    test_labels = labels[N_TRAIN+1:end]

    println("  Train: $(length(train_images)) images ($(IMAGE_SIZE)×$(IMAGE_SIZE))")
    println("  Test:  $(length(test_images)) images")
    return train_images, test_images, test_labels
end

# ============================================================================
# Manopt Runner
# ============================================================================

"""Build ProductManifold of Stiefel(2,2,ℂ) for QFT circuit tensors."""
function _manopt_manifold(tensors)
    S = Stiefel(2, 2, ℂ)
    return ProductManifold(ntuple(_ -> S, length(tensors))...)
end

"""Convert Vector{Matrix} → ArrayPartition for Manopt."""
_tensors2point(tensors) = ArrayPartition(tensors...)

"""Convert ArrayPartition → Vector{Matrix} from Manopt."""
_point2tensors(p) = collect(p.x)

"""
Run Manopt gradient_descent on QFT circuit parameters.
Returns (loss_trace, final_tensors, elapsed).
"""
function run_manopt_gd(train_images, steps)
    # Build QFT circuit
    basis = QFTBasis(M_PARAM, N_PARAM)
    tensors = basis.tensors
    optcode = basis.optcode
    inverse_code = basis.inverse_code
    loss = ParametricDFT.MSELoss(LOSS_K)

    # Convert to Complex{Float64}
    images = [Complex{Float64}.(img) for img in train_images]

    # Manopt setup
    M = _manopt_manifold(tensors)
    p0 = _tensors2point(tensors)

    # Cost function (averaged over all images)
    f = (M_arg, p) -> begin
        ts = _point2tensors(p)
        total = sum(ParametricDFT.loss_function(ts, M_PARAM, N_PARAM, optcode, img, loss;
                    inverse_code=inverse_code) for img in images)
        return Float64(total / length(images))
    end

    # Riemannian gradient
    grad_f = (M_arg, p) -> ManifoldDiff.gradient(
        M_arg, x -> f(M_arg, x), p,
        ManifoldDiff.RiemannianProjectionBackend(AutoZygote())
    )

    elapsed = @elapsed begin
        result = Manopt.gradient_descent(
            M, f, grad_f, p0;
            stopping_criterion=Manopt.StopAfterIteration(steps),
            record=[:Cost],
            return_state=true
        )
    end

    loss_trace = Float64.(get_record(result))
    final_point = get_solver_result(result)
    final_tensors = _point2tensors(final_point)

    return loss_trace, final_tensors, elapsed
end

# ============================================================================
# PDFT Runner
# ============================================================================

"""
Run ParametricDFT's customized Riemannian GD on the same full-batch QFT objective
as Manopt. This avoids `train_basis` epoch/batch accounting so the requested
`steps` means the same number of optimizer iterations for both frameworks.
Returns (loss_trace, final_tensors, elapsed).
"""
function run_pdft_gd(train_images, steps, device::Symbol)
    basis = QFTBasis(M_PARAM, N_PARAM)
    optcode = basis.optcode
    inverse_code = basis.inverse_code
    loss = ParametricDFT.MSELoss(LOSS_K)

    tensors = [ParametricDFT.to_device(Matrix{ComplexF64}(t), device) for t in basis.tensors]
    images = [ParametricDFT.to_device(ComplexF64.(img), device) for img in train_images]

    flat_batched, blabel = ParametricDFT.make_batched_code(optcode, length(tensors))
    batched_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, length(images))
    flat_batched_inv, blabel_inv = ParametricDFT.make_batched_code(inverse_code, length(tensors))
    batched_inverse_code = ParametricDFT.optimize_batched_code(flat_batched_inv, blabel_inv, length(images))
    stacked_images = ParametricDFT.stack_image_batch(images, M_PARAM, N_PARAM)

    loss_fn = ts -> ParametricDFT.loss_function(
        ts, M_PARAM, N_PARAM, optcode, stacked_images, loss;
        inverse_code=inverse_code,
        batched_optcode=batched_optcode,
        batched_inverse_code=batched_inverse_code,
    )
    grad_fn = ts -> begin
        _, back = Zygote.pullback(loss_fn, ts)
        return back(one(real(eltype(ts[1]))))[1]
    end

    loss_trace = Float64[]
    elapsed = @elapsed begin
        tensors = ParametricDFT.optimize!(
            ParametricDFT.RiemannianGD(lr=0.01),
            tensors,
            loss_fn,
            grad_fn;
            max_iter=steps,
            tol=1e-8,
            loss_trace=loss_trace,
        )
        device == :gpu && CUDA.synchronize()
    end

    final_tensors = [ComplexF64.(Array(t)) for t in tensors]

    return loss_trace, final_tensors, elapsed
end

# ============================================================================
# JSON Serialization
# ============================================================================

"""Serialize complex tensor to JSON-safe format: Vector of [real, imag] pairs."""
function _tensors_to_json(tensors)
    return [[[real(v), imag(v)] for v in vec(t)] for t in tensors]
end

"""Deserialize tensors from JSON format back to Vector{Matrix{ComplexF64}}."""
function _tensors_from_json(json_tensors, rows::Int=2, cols::Int=2)
    return [reshape([Complex(v[1], v[2]) for v in t], rows, cols) for t in json_tensors]
end

"""Save a single config result to JSON, appending to existing file."""
function save_result!(path::String, metadata::Dict, config_result::Dict)
    data = if isfile(path)
        JSON3.read(read(path, String), Dict{String, Any})
    else
        Dict{String, Any}("metadata" => metadata, "configs" => Dict{String, Any}[])
    end

    push!(data["configs"], config_result)

    open(path, "w") do io
        JSON3.pretty(io, data)
    end
end

"""Load results from JSON."""
function load_results(path::String)
    isfile(path) || return nothing
    return JSON3.read(read(path, String), Dict{String, Any})
end

# ============================================================================
# Benchmark Result
# ============================================================================

struct BenchmarkResult
    label::String
    framework::Symbol
    optimizer::Symbol
    device::Symbol
    steps::Int
    elapsed::Float64
    loss_trace::Vector{Float64}
    final_loss::Float64
    tensors::Vector{Matrix{ComplexF64}}
    success::Bool
    error_msg::String
end

# ============================================================================
# Benchmark Runner
# ============================================================================

"""Run a single optimizer config. Returns BenchmarkResult."""
function run_config(label, framework, optimizer, device, train_images, steps)
    @printf("  %-20s ... ", label)
    flush(stdout)

    try
        if framework == :manopt
            loss_trace, tensors, elapsed = run_manopt_gd(train_images, steps)
        else
            optimizer == :gradient_descent || error("optimizer_benchmark currently compares Riemannian GD only")
            loss_trace, tensors, elapsed = run_pdft_gd(train_images, steps, device)
        end

        final_loss = isempty(loss_trace) ? NaN : last(loss_trace)
        @printf("%.1fs  loss=%.1f  (%d steps recorded)\n", elapsed, final_loss, length(loss_trace))

        return BenchmarkResult(label, framework, optimizer, device, steps,
                               elapsed, loss_trace, final_loss,
                               [Matrix{ComplexF64}(t) for t in tensors], true, "")
    catch e
        msg = sprint(showerror, e)
        @printf("FAILED: %s\n", first(msg, 80))
        return BenchmarkResult(label, framework, optimizer, device, steps,
                               0.0, Float64[], NaN,
                               Matrix{ComplexF64}[], false, msg)
    end
end

"""Run all 5 configs at given step count. Save JSON after each."""
function run_benchmark_phase(phase_name, steps, train_images, json_path, metadata)
    println("\n" * "=" ^ 70)
    println("  $phase_name ($steps steps, $(length(train_images)) images)")
    println("=" ^ 70)

    results = BenchmarkResult[]

    for (label, framework, optimizer, device) in OPTIMIZER_CONFIGS
        result = run_config(label, framework, optimizer, device, train_images, steps)
        push!(results, result)

        # Save to JSON after each config
        if result.success
            config_data = Dict{String, Any}(
                "label"           => result.label,
                "framework"       => string(result.framework),
                "optimizer"       => string(result.optimizer),
                "device"          => string(result.device),
                "steps"           => result.steps,
                "elapsed_seconds" => result.elapsed,
                "final_loss"      => result.final_loss,
                "loss_trace"      => result.loss_trace,
                "basis_tensors"   => _tensors_to_json(result.tensors),
            )
            save_result!(json_path, metadata, config_data)
            println("    -> Saved to $json_path")
        end
    end

    return results
end

# ============================================================================
# Compression Evaluation
# ============================================================================

struct CompressionResult
    label::String
    ratio::Float64
    psnr::Float64
    ssim::Float64
    kept_pct::Float64
end

"""Evaluate compression quality for all successful configs."""
function evaluate_all_compression(results, test_images, test_filenames)
    println("\n" * "=" ^ 70)
    println("  Compression Evaluation")
    println("=" ^ 70)

    all_comp = CompressionResult[]

    for r in filter(r -> r.success, results)
        # Reconstruct QFTBasis from trained tensors
        basis = QFTBasis(M_PARAM, N_PARAM, r.tensors)

        for ratio in COMPRESSION_RATIOS
            psnr_vals = Float64[]
            ssim_vals = Float64[]

            for img in test_images
                compressed = compress(basis, img; ratio=ratio)
                recovered = recover(basis, compressed; verify_hash=false)
                recovered_clamped = clamp.(recovered, 0.0, 1.0)

                push!(psnr_vals, assess_psnr(img, recovered_clamped))
                push!(ssim_vals, assess_ssim(img, recovered_clamped))
            end

            kept_pct = (1.0 - ratio) * 100
            push!(all_comp, CompressionResult(r.label, ratio, mean(psnr_vals), mean(ssim_vals), kept_pct))
            @printf("  %-20s  keep=%4.0f%%  PSNR=%.2f dB  SSIM=%.4f\n",
                    r.label, kept_pct, mean(psnr_vals), mean(ssim_vals))
        end
    end

    # Save compression results
    comp_path = joinpath(OUTPUT_DIR, "compression_results.json")
    comp_data = [Dict("label" => c.label, "ratio" => c.ratio,
                       "psnr" => c.psnr, "ssim" => c.ssim, "kept_pct" => c.kept_pct)
                 for c in all_comp]
    open(comp_path, "w") do io
        JSON3.pretty(io, comp_data)
    end
    println("  -> Saved to $comp_path")

    return all_comp
end

# ============================================================================
# Visualization
# ============================================================================

"""Plot 1: Per-step loss curves for all optimizers on one axis (normalized per pixel)."""
function plot_loss_curves(results)
    with_trace = filter(r -> r.success && !isempty(r.loss_trace), results)
    isempty(with_trace) && return nothing

    n_pixels = 2^(M_PARAM + N_PARAM)

    fig = Figure(size=(900, 600))
    ax = Axis(fig[1, 1];
              title="QuickDraw Training Loss Convergence ($(IMAGE_SIZE)x$(IMAGE_SIZE), QFT)",
              xlabel="Optimization Step",
              ylabel="MSE per pixel",
              yscale=log10)

    for r in with_trace
        style = get(PLOT_STYLES, r.label, (color=:gray, linestyle=:solid))
        steps = 1:length(r.loss_trace)
        normalized = r.loss_trace ./ n_pixels
        lines!(ax, steps, normalized;
               label=r.label, color=style.color, linestyle=style.linestyle, linewidth=2)
    end

    axislegend(ax; position=:rt)
    return fig
end

"""Plot 2: Training time bar chart."""
function plot_timing(results)
    successful = filter(r -> r.success, results)
    isempty(successful) && return nothing

    labels = [r.label for r in successful]
    times = [r.elapsed for r in successful]
    colors = [r.device == :gpu ? :steelblue : (r.framework == :manopt ? :gray60 : :salmon)
              for r in successful]

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1];
              title="QuickDraw Training Time ($(FULL_STEPS) full-batch GD steps)",
              ylabel="Time (seconds)",
              xticklabelrotation=π/6)

    barplot!(ax, 1:length(times), times; color=colors)
    ax.xticks = (1:length(labels), labels)

    return fig
end

"""Plot 3: Speedup against Manopt and CPU/GPU speedup for customized GD."""
function plot_optimizer_speedup(results)
    successful = filter(r -> r.success, results)

    speedups = Tuple{String, Float64}[]

    manopt = findfirst(r -> r.label == "Manopt-GD", successful)
    gd_cpu = findfirst(r -> r.label == "PDFT-GD (cpu)", successful)
    gd_gpu = findfirst(r -> r.label == "PDFT-GD (gpu)", successful)
    if manopt !== nothing && gd_cpu !== nothing
        push!(speedups, ("PDFT CPU / Manopt", successful[manopt].elapsed / successful[gd_cpu].elapsed))
    end
    if manopt !== nothing && gd_gpu !== nothing
        push!(speedups, ("PDFT GPU / Manopt", successful[manopt].elapsed / successful[gd_gpu].elapsed))
    end
    if gd_cpu !== nothing && gd_gpu !== nothing
        push!(speedups, ("GPU / CPU", successful[gd_cpu].elapsed / successful[gd_gpu].elapsed))
    end

    isempty(speedups) && return nothing

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1];
              title="Customized GD Speedup",
              ylabel="Speedup factor")

    labels = [s[1] for s in speedups]
    values = [s[2] for s in speedups]
    colors = [v >= 1.0 ? :forestgreen : :salmon for v in values]

    barplot!(ax, 1:length(values), values; color=colors)
    ax.xticks = (1:length(labels), labels)
    hlines!(ax, [1.0]; color=:black, linestyle=:dash, linewidth=1)

    return fig
end

"""Plot 4-5: Compression quality (PSNR and SSIM vs kept %)."""
function plot_compression_quality(comp_results)
    isempty(comp_results) && return nothing, nothing

    fig_psnr = Figure(size=(800, 500))
    ax_psnr = Axis(fig_psnr[1, 1];
                    title="QuickDraw Compression Quality — PSNR ($(IMAGE_SIZE)x$(IMAGE_SIZE), QFT)",
                    xlabel="Coefficients Kept (%)",
                    ylabel="PSNR (dB)")

    fig_ssim = Figure(size=(800, 500))
    ax_ssim = Axis(fig_ssim[1, 1];
                    title="QuickDraw Compression Quality — SSIM ($(IMAGE_SIZE)x$(IMAGE_SIZE), QFT)",
                    xlabel="Coefficients Kept (%)",
                    ylabel="SSIM")

    optimizer_labels = unique(c.label for c in comp_results)

    for opt_label in optimizer_labels
        style = get(PLOT_STYLES, opt_label, (color=:gray, linestyle=:solid))
        subset = filter(c -> c.label == opt_label, comp_results)
        sort!(subset; by=c -> c.kept_pct)

        xs = [c.kept_pct for c in subset]
        psnrs = [c.psnr for c in subset]
        ssims = [c.ssim for c in subset]

        lines!(ax_psnr, xs, psnrs; label=opt_label, color=style.color,
               linestyle=style.linestyle, linewidth=2)
        scatter!(ax_psnr, xs, psnrs; color=style.color, markersize=8)

        lines!(ax_ssim, xs, ssims; label=opt_label, color=style.color,
               linestyle=style.linestyle, linewidth=2)
        scatter!(ax_ssim, xs, ssims; color=style.color, markersize=8)
    end

    axislegend(ax_psnr; position=:rb)
    axislegend(ax_ssim; position=:rb)

    return fig_psnr, fig_ssim
end

"""Plot 6: Reconstruction grid — original + each optimizer at middle compression ratio."""
function plot_reconstruction_grid(results, test_images, test_filenames)
    successful = filter(r -> r.success && !isempty(r.tensors), results)
    isempty(successful) && return nothing

    n_show = min(length(test_images), 4)
    ratio = COMPRESSION_RATIOS[2]  # middle ratio (0.9 = keep 10%)
    kept_pct = round(Int, (1.0 - ratio) * 100)

    n_rows = 1 + length(successful)  # original + each optimizer
    fig = Figure(size=(250 * n_show, 220 * n_rows))

    # Row 1: Originals
    for (col, idx) in enumerate(1:n_show)
        ax = Axis(fig[1, col]; title=test_filenames[idx][1:min(end,12)],
                  aspect=DataAspect())
        hidedecorations!(ax)
        heatmap!(ax, rotr90(test_images[idx]); colormap=:grays)
    end
    Label(fig[1, 0], "Original"; rotation=π/2, fontsize=12)

    # Rows 2+: Reconstructions
    for (row_idx, r) in enumerate(successful)
        basis = QFTBasis(M_PARAM, N_PARAM, r.tensors)

        for (col, idx) in enumerate(1:n_show)
            img = test_images[idx]
            compressed = compress(basis, img; ratio=ratio)
            recovered = clamp.(recover(basis, compressed; verify_hash=false), 0.0, 1.0)
            psnr_val = assess_psnr(img, recovered)

            ax = Axis(fig[row_idx + 1, col];
                      title=@sprintf("%.1f dB", psnr_val),
                      titlesize=10, aspect=DataAspect())
            hidedecorations!(ax)
            heatmap!(ax, rotr90(recovered); colormap=:grays)
        end

        Label(fig[row_idx + 1, 0], r.label; rotation=π/2, fontsize=10)
    end

    Label(fig[0, 1:n_show], "Reconstruction Grid (keep $(kept_pct)%)"; fontsize=14)

    return fig
end

"""Plot 7: Final loss comparison bar chart."""
function plot_final_loss(results)
    successful = filter(r -> r.success, results)
    isempty(successful) && return nothing

    labels = [r.label for r in successful]
    losses = [r.final_loss for r in successful]
    colors = [r.device == :gpu ? :steelblue : (r.framework == :manopt ? :gray60 : :salmon)
              for r in successful]

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1];
              title="Final Training Loss ($(FULL_STEPS) steps)",
              ylabel="Loss",
              xticklabelrotation=π/6)

    barplot!(ax, 1:length(losses), losses; color=colors)
    ax.xticks = (1:length(labels), labels)

    return fig
end

"""Generate and save all 7 plots."""
function generate_all_plots(results, comp_results, test_images, test_filenames)
    plots_dir = joinpath(OUTPUT_DIR, "plots")
    mkpath(plots_dir)

    println("\nGenerating plots...")

    plots = [
        ("loss_curves.png",      plot_loss_curves(results)),
        ("timing.png",           plot_timing(results)),
        ("optimizer_speedup.png", plot_optimizer_speedup(results)),
        ("final_loss.png",       plot_final_loss(results)),
        ("reconstruction.png",   plot_reconstruction_grid(results, test_images, test_filenames)),
    ]

    fig_psnr, fig_ssim = plot_compression_quality(comp_results)
    push!(plots, ("psnr.png", fig_psnr))
    push!(plots, ("ssim.png", fig_ssim))

    for (fname, fig) in plots
        if fig !== nothing
            save(joinpath(plots_dir, fname), fig)
            println("  $fname")
        end
    end

    println("\nAll plots saved to: $plots_dir")
end

# ============================================================================
# Main
# ============================================================================

function main()
    println("=" ^ 70)
    println("  Optimizer Benchmark: PDFT vs Manopt.jl")
    println("=" ^ 70)
    println("  Image:    $(IMAGE_SIZE)x$(IMAGE_SIZE) (m=$(M_PARAM), n=$(N_PARAM))")
    println("  Data:     QuickDraw ($N_TRAIN train, $N_TEST test)")
    println("  Loss:     MSELoss(k=$(LOSS_K))")
    println("  Smoke:    $(SMOKE_STEPS) full-batch GD steps")
    println("  Full:     $(FULL_STEPS) full-batch GD steps")
    println("  Configs:  $(length(OPTIMIZER_CONFIGS))")
    println("  GPU:      CUDA device $GPU_DEVICE")
    println("=" ^ 70)

    CUDA.device!(GPU_DEVICE)
    @assert CUDA.functional() "GPU $GPU_DEVICE required"
    println("  CUDA:     $(CUDA.name(CUDA.device()))")

    mkpath(OUTPUT_DIR)

    # Load data
    println("\nLoading QuickDraw images...")
    train_images, test_images, test_filenames = load_quickdraw()

    metadata = Dict{String, Any}(
        "date"       => Dates.format(now(), "yyyy-mm-dd HH:MM"),
        "image_size" => "$(IMAGE_SIZE)x$(IMAGE_SIZE)",
        "dataset" => "quickdraw",
        "m" => M_PARAM, "n" => N_PARAM,
        "n_train" => N_TRAIN, "n_test" => N_TEST,
        "basis_type" => "QFT",
        "seed" => SEED,
        "loss_k" => LOSS_K,
        "step_semantics" => "full_batch_optimizer_iterations",
    )

    # Phase 1: Smoke test
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    smoke_path = joinpath(OUTPUT_DIR, "smoke_results_$(timestamp).json")
    smoke_results = run_benchmark_phase("Smoke Test", SMOKE_STEPS, train_images, smoke_path, metadata)

    # Check smoke failures
    failed = filter(r -> !r.success, smoke_results)
    if !isempty(failed)
        println("\nSmoke failures:")
        for r in failed
            println("  $(r.label): $(first(r.error_msg, 100))")
        end
    end
    failed_labels = Set(r.label for r in failed)

    # Phase 2: Full run (skip smoke failures)
    full_path = joinpath(OUTPUT_DIR, "full_results_$(timestamp).json")

    println("\n" * "=" ^ 70)
    println("  Full Benchmark ($(FULL_STEPS) steps)")
    println("=" ^ 70)

    full_results = BenchmarkResult[]
    for (label, framework, optimizer, device) in OPTIMIZER_CONFIGS
        if label in failed_labels
            @printf("  %-20s SKIP (failed in smoke)\n", label)
            continue
        end
        result = run_config(label, framework, optimizer, device, train_images, FULL_STEPS)
        push!(full_results, result)

        if result.success
            config_data = Dict{String, Any}(
                "label"           => result.label,
                "framework"       => string(result.framework),
                "optimizer"       => string(result.optimizer),
                "device"          => string(result.device),
                "steps"           => result.steps,
                "elapsed_seconds" => result.elapsed,
                "final_loss"      => result.final_loss,
                "loss_trace"      => result.loss_trace,
                "basis_tensors"   => _tensors_to_json(result.tensors),
            )
            save_result!(full_path, metadata, config_data)
            println("    -> Saved to $full_path")
        end
    end

    # Phase 3: Compression evaluation
    compression_results = evaluate_all_compression(full_results, test_images, test_filenames)

    # Phase 4: Plots
    generate_all_plots(full_results, compression_results, test_images, test_filenames)

    manopt = findfirst(r -> r.label == "Manopt-GD" && r.success, full_results)
    pdft_cpu = findfirst(r -> r.label == "PDFT-GD (cpu)" && r.success, full_results)
    pdft_gpu = findfirst(r -> r.label == "PDFT-GD (gpu)" && r.success, full_results)
    if manopt !== nothing && pdft_cpu !== nothing
        @printf("\nPDFT-GD CPU speedup over Manopt-GD: %.2fx\n",
                full_results[manopt].elapsed / full_results[pdft_cpu].elapsed)
    end
    if manopt !== nothing && pdft_gpu !== nothing
        @printf("PDFT-GD GPU speedup over Manopt-GD: %.2fx\n",
                full_results[manopt].elapsed / full_results[pdft_gpu].elapsed)
    end

    println("\n" * "=" ^ 70)
    println("  Benchmark complete! Results in: $OUTPUT_DIR")
    println("=" ^ 70)

    return full_results, compression_results
end

# ============================================================================
# Run
# ============================================================================

main()
