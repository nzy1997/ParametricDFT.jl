# ============================================================================
# Optimizer Benchmark: PDFT vs Manopt.jl
# ============================================================================
# Single public benchmark entry point with two built-in profiles:
#   1. QuickDraw fairness check against Manopt.jl on a small problem
#   2. DIV2K speed check for PDFT GD/Adam on a heavier workload
#
# Run:
#   julia --project=examples examples/optimizer_benchmark.jl
#   julia --project=examples examples/optimizer_benchmark.jl quickdraw_fairness
#   julia --project=examples examples/optimizer_benchmark.jl div2k_speed
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
using Images: load, Gray, channelview
using ImageQualityIndexes: assess_psnr, assess_ssim

# Manopt stack
using Manopt, Manifolds, ManifoldDiff
using RecursiveArrayTools: ArrayPartition
using ADTypes: AutoZygote
import Zygote

# ============================================================================
# Configuration
# ============================================================================

const SEED = 42
const GPU_DEVICE = 1
const QUICKDRAW_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
const QUICKDRAW_CATEGORIES = ["cat", "dog", "airplane", "apple", "bicycle"]
const QUICKDRAW_DIR = joinpath(@__DIR__, "benchmark", "data", "quickdraw")
const DIV2K_DIR = joinpath(@__DIR__, "benchmark", "data", "DIV2K_sample")
const BASE_OUTPUT_DIR = joinpath(@__DIR__, "OptimizerBenchmark")
const DEFAULT_PROFILE = :default

struct BenchmarkProfile
    name::String
    slug::String
    dataset::Symbol
    m::Int
    n::Int
    n_train::Int
    n_test::Int
    smoke_steps::Int
    full_steps::Int
    loss_keep_ratio::Float64
    compression_ratios::Vector{Float64}
    configs::Vector{NTuple{4, Any}}
    compare_to_manopt::Bool
    evaluate_compression::Bool
end

function quickdraw_profile()
    return BenchmarkProfile(
        "QuickDraw Fairness",
        "quickdraw",
        :quickdraw,
        5,
        5,
        20,
        5,
        5,
        30,
        0.1,
        [0.8, 0.9, 0.95],
        [
            ("Manopt-GD", :manopt, :gradient_descent, :cpu),
            ("PDFT-GD (cpu)", :pdft, :gradient_descent, :cpu),
            ("PDFT-GD (gpu)", :pdft, :gradient_descent, :gpu),
        ],
        true,
        true,
    )
end

function div2k_profile()
    return BenchmarkProfile(
        "DIV2K Speed",
        "div2k",
        :div2k,
        9,
        9,
        4,
        2,
        2,
        10,
        0.1,
        [0.9, 0.95],
        [
            ("PDFT-GD (cpu)", :pdft, :gradient_descent, :cpu),
            ("PDFT-GD (gpu)", :pdft, :gradient_descent, :gpu),
            ("PDFT-Adam (cpu)", :pdft, :adam, :cpu),
            ("PDFT-Adam (gpu)", :pdft, :adam, :gpu),
        ],
        false,
        true,
    )
end

function resolve_profiles(arg::String)
    arg == "default" && return [quickdraw_profile(), div2k_profile()]
    arg == "quickdraw_fairness" && return [quickdraw_profile()]
    arg == "div2k_speed" && return [div2k_profile()]
    error("Unknown benchmark profile '$arg'. Use default, quickdraw_fairness, or div2k_speed")
end

image_size(profile::BenchmarkProfile) = 2^profile.m
loss_k(profile::BenchmarkProfile) = round(Int, 2^(profile.m + profile.n) * profile.loss_keep_ratio)
output_dir(profile::BenchmarkProfile) = joinpath(BASE_OUTPUT_DIR, profile.slug)

# ============================================================================
# Data Loading
# ============================================================================

"""Center-pad a small image to the benchmark power-of-two size."""
function pad_to_power_of_two(img::AbstractMatrix, target_size::Int)
    h, w = size(img)
    padded = zeros(Float64, target_size, target_size)
    y_offset = (target_size - h) ÷ 2 + 1
    x_offset = (target_size - w) ÷ 2 + 1
    padded[y_offset:y_offset+h-1, x_offset:x_offset+w-1] = Float64.(img)
    return padded
end

"""Load QuickDraw bitmap images, auto-downloading missing category files."""
function load_quickdraw(profile::BenchmarkProfile)
    mkpath(QUICKDRAW_DIR)
    for category in QUICKDRAW_CATEGORIES
        filepath = joinpath(QUICKDRAW_DIR, "$(category).npy")
        if !isfile(filepath)
            url = "$(QUICKDRAW_BASE_URL)/$(category).npy"
            @info "Downloading QuickDraw category" category url
            Downloads.download(url, filepath)
        end
    end

    images = Matrix{Float64}[]
    labels = String[]
    per_category = ceil(Int, (profile.n_train + profile.n_test) / length(QUICKDRAW_CATEGORIES))

    for category in QUICKDRAW_CATEGORIES
        data = npzread(joinpath(QUICKDRAW_DIR, "$(category).npy"))
        n_to_load = min(size(data, 1), per_category)
        for i in 1:n_to_load
            img = reshape(Float64.(data[i, :]), 28, 28) ./ 255.0
            push!(images, pad_to_power_of_two(img, image_size(profile)))
            push!(labels, category)
        end
    end

    n_required = profile.n_train + profile.n_test
    @assert length(images) >= n_required "Need $n_required QuickDraw images, found $(length(images))"

    Random.seed!(SEED)
    selected = randperm(length(images))[1:n_required]
    images = images[selected]
    labels = labels[selected]

    train_images = images[1:profile.n_train]
    test_images = images[profile.n_train+1:end]
    test_labels = labels[profile.n_train+1:end]

    println("  Train: $(length(train_images)) images ($(image_size(profile))x$(image_size(profile)))")
    println("  Test:  $(length(test_images)) images")
    return train_images, test_images, test_labels
end

"""Load DIV2K images, center-crop to the profile image size."""
function load_div2k(profile::BenchmarkProfile)
    all_files = sort(filter(f -> endswith(lowercase(f), ".png"), readdir(DIV2K_DIR; join=true)))
    n_required = profile.n_train + profile.n_test
    @assert length(all_files) >= n_required "Need $n_required DIV2K images, found $(length(all_files))"

    Random.seed!(SEED)
    selected = all_files[randperm(length(all_files))[1:n_required]]

    images = Matrix{Float64}[]
    labels = String[]
    side = image_size(profile)

    for path in selected
        img = load(path)
        gray = Gray.(img)
        h, w = size(gray)
        y0 = (h - side) ÷ 2 + 1
        x0 = (w - side) ÷ 2 + 1
        cropped = gray[y0:y0+side-1, x0:x0+side-1]
        push!(images, Float64.(channelview(cropped)))
        push!(labels, basename(path))
    end

    train_images = images[1:profile.n_train]
    test_images = images[profile.n_train+1:end]
    test_labels = labels[profile.n_train+1:end]

    println("  Train: $(length(train_images)) images ($(side)x$(side))")
    println("  Test:  $(length(test_images)) images")
    return train_images, test_images, test_labels
end

function load_dataset(profile::BenchmarkProfile)
    profile.dataset == :quickdraw && return load_quickdraw(profile)
    profile.dataset == :div2k && return load_div2k(profile)
    error("Unsupported dataset $(profile.dataset)")
end

# ============================================================================
# Shared Manopt Helpers
# ============================================================================

"""Build ProductManifold of Stiefel(2,2,ℂ) for QFT circuit tensors."""
function _manopt_manifold(tensors)
    S = Stiefel(2, 2, ℂ)
    return ProductManifold(ntuple(_ -> S, length(tensors))...)
end

_tensors2point(tensors) = ArrayPartition(tensors...)
_point2tensors(p) = collect(p.x)

# ============================================================================
# Benchmark Runners
# ============================================================================

"""Run Manopt gradient_descent on QFT circuit parameters."""
function run_manopt_gd(profile::BenchmarkProfile, train_images, steps)
    basis = QFTBasis(profile.m, profile.n)
    tensors = basis.tensors
    optcode = basis.optcode
    inverse_code = basis.inverse_code
    loss = ParametricDFT.MSELoss(loss_k(profile))
    images = [ComplexF64.(img) for img in train_images]

    M = _manopt_manifold(tensors)
    p0 = _tensors2point(tensors)

    f = (M_arg, p) -> begin
        ts = _point2tensors(p)
        total = sum(ParametricDFT.loss_function(ts, profile.m, profile.n, optcode, img, loss;
                    inverse_code=inverse_code) for img in images)
        return Float64(total / length(images))
    end

    grad_f = (M_arg, p) -> ManifoldDiff.gradient(
        M_arg, x -> f(M_arg, x), p,
        ManifoldDiff.RiemannianProjectionBackend(AutoZygote())
    )

    elapsed = @elapsed begin
        result = Manopt.gradient_descent(
            M, f, grad_f, p0;
            stopping_criterion=Manopt.StopAfterIteration(steps),
            record=[:Cost],
            return_state=true,
        )
    end

    loss_trace = Float64.(get_record(result))
    final_point = get_solver_result(result)
    final_tensors = _point2tensors(final_point)
    return loss_trace, [Matrix{ComplexF64}(t) for t in final_tensors], elapsed
end

"""Run ParametricDFT optimize! directly for fair full-batch step comparison."""
function run_pdft(profile::BenchmarkProfile, train_images, steps, optimizer::Symbol, device::Symbol)
    basis = QFTBasis(profile.m, profile.n)
    optcode = basis.optcode
    inverse_code = basis.inverse_code
    loss = ParametricDFT.MSELoss(loss_k(profile))

    tensors = [ParametricDFT.to_device(Matrix{ComplexF64}(t), device) for t in basis.tensors]
    images = [ParametricDFT.to_device(ComplexF64.(img), device) for img in train_images]

    flat_batched, blabel = ParametricDFT.make_batched_code(optcode, length(tensors))
    batched_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, length(images))
    flat_batched_inv, blabel_inv = ParametricDFT.make_batched_code(inverse_code, length(tensors))
    batched_inverse_code = ParametricDFT.optimize_batched_code(flat_batched_inv, blabel_inv, length(images))
    stacked_images = ParametricDFT.stack_image_batch(images, profile.m, profile.n)

    loss_fn = ts -> ParametricDFT.loss_function(
        ts, profile.m, profile.n, optcode, stacked_images, loss;
        inverse_code=inverse_code,
        batched_optcode=batched_optcode,
        batched_inverse_code=batched_inverse_code,
    )
    grad_fn = ts -> begin
        _, back = Zygote.pullback(loss_fn, ts)
        return back(one(real(eltype(ts[1]))))[1]
    end

    opt = optimizer == :adam ? ParametricDFT.RiemannianAdam(lr=0.001) : ParametricDFT.RiemannianGD(lr=0.01)

    loss_trace = Float64[]
    elapsed = @elapsed begin
        tensors = ParametricDFT.optimize!(opt, tensors, loss_fn, grad_fn;
                                          max_iter=steps, tol=1e-8, loss_trace=loss_trace)
        device == :gpu && CUDA.synchronize()
    end

    return loss_trace, [ComplexF64.(Array(t)) for t in tensors], elapsed
end

# ============================================================================
# JSON Serialization
# ============================================================================

"""Serialize complex tensor to JSON-safe format: Vector of [real, imag] pairs."""
function _tensors_to_json(tensors)
    return [[[real(v), imag(v)] for v in vec(t)] for t in tensors]
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

# ============================================================================
# Benchmark Result Types
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

struct CompressionResult
    label::String
    ratio::Float64
    psnr::Float64
    ssim::Float64
    kept_pct::Float64
end

# ============================================================================
# Benchmark Execution
# ============================================================================

function run_config(profile::BenchmarkProfile, label, framework, optimizer, device, train_images, steps)
    @printf("  %-20s ... ", label)
    flush(stdout)

    try
        loss_trace, tensors, elapsed = if framework == :manopt
            run_manopt_gd(profile, train_images, steps)
        else
            run_pdft(profile, train_images, steps, optimizer, device)
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

function run_phase(profile::BenchmarkProfile, phase_name::String, steps::Int, train_images, json_path::String, metadata::Dict)
    println("\n" * "=" ^ 70)
    println("  $(profile.name): $phase_name ($steps steps, $(length(train_images)) images)")
    println("=" ^ 70)

    results = BenchmarkResult[]
    for (label, framework, optimizer, device) in profile.configs
        result = run_config(profile, label, framework, optimizer, device, train_images, steps)
        push!(results, result)

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

function evaluate_all_compression(profile::BenchmarkProfile, results, test_images)
    profile.evaluate_compression || return CompressionResult[]

    println("\n" * "=" ^ 70)
    println("  $(profile.name): Compression Evaluation")
    println("=" ^ 70)

    all_comp = CompressionResult[]

    for r in filter(r -> r.success, results)
        basis = QFTBasis(profile.m, profile.n, r.tensors)

        for ratio in profile.compression_ratios
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

    comp_path = joinpath(output_dir(profile), "compression_results.json")
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

function plot_styles(profile::BenchmarkProfile)
    if profile.slug == "div2k"
        return Dict(
            "PDFT-GD (cpu)" => (color=:blue, linestyle=:dash),
            "PDFT-GD (gpu)" => (color=:blue, linestyle=:solid),
            "PDFT-Adam (cpu)" => (color=:red, linestyle=:dash),
            "PDFT-Adam (gpu)" => (color=:red, linestyle=:solid),
        )
    end
    return Dict(
        "Manopt-GD" => (color=:black, linestyle=:solid),
        "PDFT-GD (cpu)" => (color=:blue, linestyle=:dash),
        "PDFT-GD (gpu)" => (color=:blue, linestyle=:solid),
    )
end

function plot_loss_curves(profile::BenchmarkProfile, results)
    with_trace = filter(r -> r.success && !isempty(r.loss_trace), results)
    isempty(with_trace) && return nothing

    n_pixels = 2^(profile.m + profile.n)
    styles = plot_styles(profile)

    fig = Figure(size=(900, 600))
    ax = Axis(fig[1, 1];
              title="$(profile.name) Loss Convergence ($(image_size(profile))x$(image_size(profile)), QFT)",
              xlabel="Optimization Step",
              ylabel="MSE per pixel",
              yscale=log10)

    for r in with_trace
        style = get(styles, r.label, (color=:gray, linestyle=:solid))
        steps = 1:length(r.loss_trace)
        normalized = r.loss_trace ./ n_pixels
        lines!(ax, steps, normalized; label=r.label, color=style.color,
               linestyle=style.linestyle, linewidth=2)
    end

    axislegend(ax; position=:rt)
    return fig
end

function plot_timing(profile::BenchmarkProfile, results)
    successful = filter(r -> r.success, results)
    isempty(successful) && return nothing

    labels = [r.label for r in successful]
    times = [r.elapsed for r in successful]
    colors = [r.device == :gpu ? :steelblue : (r.framework == :manopt ? :gray60 : :salmon)
              for r in successful]

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1];
              title="$(profile.name) Training Time ($(successful[1].steps) steps)",
              ylabel="Time (seconds)",
              xticklabelrotation=π/6)

    barplot!(ax, 1:length(times), times; color=colors)
    ax.xticks = (1:length(labels), labels)
    return fig
end

function plot_speedup(profile::BenchmarkProfile, results)
    successful = filter(r -> r.success, results)
    isempty(successful) && return nothing

    speedups = Tuple{String, Float64}[]

    if profile.compare_to_manopt
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
    else
        gd_cpu = findfirst(r -> r.label == "PDFT-GD (cpu)", successful)
        gd_gpu = findfirst(r -> r.label == "PDFT-GD (gpu)", successful)
        adam_cpu = findfirst(r -> r.label == "PDFT-Adam (cpu)", successful)
        adam_gpu = findfirst(r -> r.label == "PDFT-Adam (gpu)", successful)
        if gd_cpu !== nothing && gd_gpu !== nothing
            push!(speedups, ("GD GPU / CPU", successful[gd_cpu].elapsed / successful[gd_gpu].elapsed))
        end
        if adam_cpu !== nothing && adam_gpu !== nothing
            push!(speedups, ("Adam GPU / CPU", successful[adam_cpu].elapsed / successful[adam_gpu].elapsed))
        end
    end

    isempty(speedups) && return nothing

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1]; title="$(profile.name) Speedup", ylabel="Speedup factor")

    labels = [s[1] for s in speedups]
    values = [s[2] for s in speedups]
    colors = [v >= 1.0 ? :forestgreen : :salmon for v in values]

    barplot!(ax, 1:length(values), values; color=colors)
    ax.xticks = (1:length(labels), labels)
    hlines!(ax, [1.0]; color=:black, linestyle=:dash, linewidth=1)
    return fig
end

function plot_compression_quality(profile::BenchmarkProfile, comp_results)
    isempty(comp_results) && return nothing, nothing

    styles = plot_styles(profile)
    fig_psnr = Figure(size=(800, 500))
    ax_psnr = Axis(fig_psnr[1, 1];
                   title="$(profile.name) Compression Quality — PSNR",
                   xlabel="Coefficients Kept (%)",
                   ylabel="PSNR (dB)")

    fig_ssim = Figure(size=(800, 500))
    ax_ssim = Axis(fig_ssim[1, 1];
                   title="$(profile.name) Compression Quality — SSIM",
                   xlabel="Coefficients Kept (%)",
                   ylabel="SSIM")

    optimizer_labels = unique(c.label for c in comp_results)
    for opt_label in optimizer_labels
        style = get(styles, opt_label, (color=:gray, linestyle=:solid))
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

function plot_reconstruction_grid(profile::BenchmarkProfile, results, test_images, test_labels)
    successful = filter(r -> r.success && !isempty(r.tensors), results)
    isempty(successful) && return nothing

    n_show = min(length(test_images), 4)
    ratio = profile.compression_ratios[min(2, length(profile.compression_ratios))]
    kept_pct = round(Int, (1.0 - ratio) * 100)

    n_rows = 1 + length(successful)
    fig = Figure(size=(250 * n_show, 220 * n_rows))

    for (col, idx) in enumerate(1:n_show)
        ax = Axis(fig[1, col]; title=test_labels[idx][1:min(end, 12)], aspect=DataAspect())
        hidedecorations!(ax)
        heatmap!(ax, rotr90(test_images[idx]); colormap=:grays)
    end
    Label(fig[1, 0], "Original"; rotation=π/2, fontsize=12)

    for (row_idx, r) in enumerate(successful)
        basis = QFTBasis(profile.m, profile.n, r.tensors)
        for (col, idx) in enumerate(1:n_show)
            img = test_images[idx]
            compressed = compress(basis, img; ratio=ratio)
            recovered = clamp.(recover(basis, compressed; verify_hash=false), 0.0, 1.0)
            psnr_val = assess_psnr(img, recovered)

            ax = Axis(fig[row_idx + 1, col]; title=@sprintf("%.1f dB", psnr_val),
                      titlesize=10, aspect=DataAspect())
            hidedecorations!(ax)
            heatmap!(ax, rotr90(recovered); colormap=:grays)
        end
        Label(fig[row_idx + 1, 0], r.label; rotation=π/2, fontsize=10)
    end

    Label(fig[0, 1:n_show], "Reconstruction Grid (keep $(kept_pct)%)"; fontsize=14)
    return fig
end

function plot_final_loss(profile::BenchmarkProfile, results)
    successful = filter(r -> r.success, results)
    isempty(successful) && return nothing

    labels = [r.label for r in successful]
    losses = [r.final_loss for r in successful]
    colors = [r.device == :gpu ? :steelblue : (r.framework == :manopt ? :gray60 : :salmon)
              for r in successful]

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1]; title="$(profile.name) Final Training Loss ($(successful[1].steps) steps)",
              ylabel="Loss", xticklabelrotation=π/6)
    barplot!(ax, 1:length(losses), losses; color=colors)
    ax.xticks = (1:length(labels), labels)
    return fig
end

function generate_all_plots(profile::BenchmarkProfile, results, comp_results, test_images, test_labels)
    plots_dir = joinpath(output_dir(profile), "plots")
    mkpath(plots_dir)

    println("\nGenerating plots for $(profile.name)...")
    plots = [
        ("loss_curves.png", plot_loss_curves(profile, results)),
        ("timing.png", plot_timing(profile, results)),
        ("speedup.png", plot_speedup(profile, results)),
        ("final_loss.png", plot_final_loss(profile, results)),
        ("reconstruction.png", plot_reconstruction_grid(profile, results, test_images, test_labels)),
    ]

    fig_psnr, fig_ssim = plot_compression_quality(profile, comp_results)
    push!(plots, ("psnr.png", fig_psnr))
    push!(plots, ("ssim.png", fig_ssim))

    for (fname, fig) in plots
        if fig !== nothing
            save(joinpath(plots_dir, fname), fig)
            println("  $fname")
        end
    end
end

# ============================================================================
# Reporting
# ============================================================================

function print_summary(profile::BenchmarkProfile, results)
    successful = filter(r -> r.success, results)
    isempty(successful) && return

    println("\n" * "=" ^ 70)
    println("  $(profile.name) Summary")
    println("=" ^ 70)
    for r in successful
        @printf("  %-20s %7.2fs  loss=%.2f\n", r.label, r.elapsed, r.final_loss)
    end

    if profile.compare_to_manopt
        manopt = findfirst(r -> r.label == "Manopt-GD", successful)
        pdft_cpu = findfirst(r -> r.label == "PDFT-GD (cpu)", successful)
        pdft_gpu = findfirst(r -> r.label == "PDFT-GD (gpu)", successful)
        if manopt !== nothing && pdft_cpu !== nothing
            @printf("\n  PDFT-GD CPU speedup over Manopt-GD: %.2fx\n",
                    successful[manopt].elapsed / successful[pdft_cpu].elapsed)
        end
        if manopt !== nothing && pdft_gpu !== nothing
            @printf("  PDFT-GD GPU speedup over Manopt-GD: %.2fx\n",
                    successful[manopt].elapsed / successful[pdft_gpu].elapsed)
        end
    else
        gd_cpu = findfirst(r -> r.label == "PDFT-GD (cpu)", successful)
        gd_gpu = findfirst(r -> r.label == "PDFT-GD (gpu)", successful)
        adam_cpu = findfirst(r -> r.label == "PDFT-Adam (cpu)", successful)
        adam_gpu = findfirst(r -> r.label == "PDFT-Adam (gpu)", successful)
        if gd_cpu !== nothing && gd_gpu !== nothing
            @printf("\n  GD GPU/CPU speedup: %.2fx\n",
                    successful[gd_cpu].elapsed / successful[gd_gpu].elapsed)
        end
        if adam_cpu !== nothing && adam_gpu !== nothing
            @printf("  Adam GPU/CPU speedup: %.2fx\n",
                    successful[adam_cpu].elapsed / successful[adam_gpu].elapsed)
        end
    end
end

# ============================================================================
# Main
# ============================================================================

function run_profile(profile::BenchmarkProfile)
    println("=" ^ 70)
    println("  Optimizer Benchmark: $(profile.name)")
    println("=" ^ 70)
    println("  Dataset:   $(profile.dataset)")
    println("  Image:     $(image_size(profile))x$(image_size(profile)) (m=$(profile.m), n=$(profile.n))")
    println("  Train:     $(profile.n_train) images")
    println("  Test:      $(profile.n_test) images")
    println("  Loss:      MSELoss(k=$(loss_k(profile)))")
    println("  Smoke:     $(profile.smoke_steps) full-batch steps")
    println("  Full:      $(profile.full_steps) full-batch steps")
    println("  Configs:   $(length(profile.configs))")
    println("  GPU:       CUDA device $GPU_DEVICE")
    println("=" ^ 70)

    CUDA.device!(GPU_DEVICE)
    @assert CUDA.functional() "GPU $GPU_DEVICE required"
    println("  CUDA:      $(CUDA.name(CUDA.device()))")

    mkpath(output_dir(profile))

    println("\nLoading dataset...")
    train_images, test_images, test_labels = load_dataset(profile)

    metadata = Dict{String, Any}(
        "date" => Dates.format(now(), "yyyy-mm-dd HH:MM"),
        "profile" => profile.slug,
        "dataset" => String(profile.dataset),
        "image_size" => "$(image_size(profile))x$(image_size(profile))",
        "m" => profile.m,
        "n" => profile.n,
        "n_train" => profile.n_train,
        "n_test" => profile.n_test,
        "basis_type" => "QFT",
        "seed" => SEED,
        "loss_k" => loss_k(profile),
        "step_semantics" => "full_batch_optimizer_iterations",
    )

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    smoke_path = joinpath(output_dir(profile), "smoke_results_$(timestamp).json")
    smoke_results = run_phase(profile, "Smoke Test", profile.smoke_steps, train_images, smoke_path, metadata)

    failed = filter(r -> !r.success, smoke_results)
    if !isempty(failed)
        println("\nSmoke failures:")
        for r in failed
            println("  $(r.label): $(first(r.error_msg, 100))")
        end
    end
    failed_labels = Set(r.label for r in failed)

    full_path = joinpath(output_dir(profile), "full_results_$(timestamp).json")
    println("\n" * "=" ^ 70)
    println("  $(profile.name): Full Benchmark ($(profile.full_steps) steps)")
    println("=" ^ 70)

    full_results = BenchmarkResult[]
    for (label, framework, optimizer, device) in profile.configs
        if label in failed_labels
            @printf("  %-20s SKIP (failed in smoke)\n", label)
            continue
        end
        result = run_config(profile, label, framework, optimizer, device, train_images, profile.full_steps)
        push!(full_results, result)

        if result.success
            config_data = Dict{String, Any}(
                "label" => result.label,
                "framework" => string(result.framework),
                "optimizer" => string(result.optimizer),
                "device" => string(result.device),
                "steps" => result.steps,
                "elapsed_seconds" => result.elapsed,
                "final_loss" => result.final_loss,
                "loss_trace" => result.loss_trace,
                "basis_tensors" => _tensors_to_json(result.tensors),
            )
            save_result!(full_path, metadata, config_data)
            println("    -> Saved to $full_path")
        end
    end

    compression_results = evaluate_all_compression(profile, full_results, test_images)
    generate_all_plots(profile, full_results, compression_results, test_images, test_labels)
    print_summary(profile, full_results)

    println("\nResults saved to: $(output_dir(profile))")
    return full_results, compression_results
end

function main()
    arg = isempty(ARGS) ? String(DEFAULT_PROFILE) : ARGS[1]
    profiles = resolve_profiles(arg)
    results = []
    for profile in profiles
        push!(results, run_profile(profile))
        println()
    end
    return results
end

main()
