# ============================================================================
# Benchmark Evaluation Utilities
# ============================================================================
# Provides metrics computation, training wrapper with timing, FFT baseline,
# and result serialization. Include after config.jl.
# ============================================================================

using ImageQualityIndexes: assess_psnr, assess_ssim
using FFTW
using JSON3
using Statistics
using Printf

# ============================================================================
# Metrics
# ============================================================================

"""
    compute_metrics(original::AbstractMatrix, recovered::AbstractMatrix)

Compute MSE, PSNR, and SSIM between original and recovered images.
Returns a NamedTuple `(mse, psnr, ssim)`.
"""
function compute_metrics(original::AbstractMatrix, recovered::AbstractMatrix)
    recovered_clamped = clamp.(real.(recovered), 0.0, 1.0)
    mse = mean((original .- recovered_clamped) .^ 2)
    psnr = mse > 0 ? 10 * log10(1.0 / mse) : Inf
    ssim = assess_ssim(Gray.(original), Gray.(recovered_clamped))
    return (mse = mse, psnr = psnr, ssim = ssim)
end

# ============================================================================
# Basis Evaluation
# ============================================================================

"""
    evaluate_basis(basis, test_images, keep_ratios)

Evaluate a trained basis at multiple compression ratios on a test set.

Returns `Dict(keep_ratio => (mean_mse, std_mse, mean_psnr, std_psnr, mean_ssim, std_ssim))`.

Note: `keep_ratio` is the fraction of coefficients *kept*.
The `compress()` function's `ratio` parameter is the fraction *discarded*,
so we pass `ratio = 1.0 - keep_ratio`.
"""
function evaluate_basis(basis, test_images::Vector{<:AbstractMatrix}, keep_ratios::Vector{Float64})
    results = Dict{Float64,NamedTuple}()

    for keep_ratio in keep_ratios
        discard_ratio = 1.0 - keep_ratio
        mse_vals, psnr_vals, ssim_vals = Float64[], Float64[], Float64[]

        for img in test_images
            compressed = compress(basis, img; ratio = discard_ratio)
            recovered = recover(basis, compressed)
            metrics = compute_metrics(img, recovered)
            push!(mse_vals, metrics.mse)
            push!(psnr_vals, metrics.psnr)
            push!(ssim_vals, metrics.ssim)
        end

        results[keep_ratio] = (
            mean_mse = mean(mse_vals), std_mse = std(mse_vals),
            mean_psnr = mean(psnr_vals), std_psnr = std(psnr_vals),
            mean_ssim = mean(ssim_vals), std_ssim = std(ssim_vals),
        )
    end

    return results
end

# ============================================================================
# FFT Baseline
# ============================================================================

"""
    fft_compress_recover(img::AbstractMatrix, keep_ratio::Float64)

Compress and recover an image using classical FFT.
"""
function fft_compress_recover(img::AbstractMatrix, keep_ratio::Float64)
    freq = fftshift(fft(img))
    total = length(freq)
    keep = max(1, round(Int, total * keep_ratio))

    flat = vec(freq)
    idx = partialsortperm(abs.(flat), 1:keep, rev = true)
    compressed = zeros(ComplexF64, size(freq))
    compressed[idx] = freq[idx]

    return real.(ifft(ifftshift(compressed)))
end

"""
    evaluate_fft_baseline_timed(test_images, keep_ratios)

Evaluate classical FFT baseline at multiple ratios. Returns `(metrics_dict, elapsed_seconds)`.
"""
function evaluate_fft_baseline_timed(test_images::Vector{<:AbstractMatrix}, keep_ratios::Vector{Float64})
    elapsed = @elapsed begin
        results = Dict{Float64,NamedTuple}()
        for keep_ratio in keep_ratios
            mse_vals, psnr_vals, ssim_vals = Float64[], Float64[], Float64[]
            for img in test_images
                recovered = fft_compress_recover(img, keep_ratio)
                metrics = compute_metrics(img, recovered)
                push!(mse_vals, metrics.mse)
                push!(psnr_vals, metrics.psnr)
                push!(ssim_vals, metrics.ssim)
            end
            results[keep_ratio] = (
                mean_mse = mean(mse_vals), std_mse = std(mse_vals),
                mean_psnr = mean(psnr_vals), std_psnr = std(psnr_vals),
                mean_ssim = mean(ssim_vals), std_ssim = std(ssim_vals),
            )
        end
    end
    return results, elapsed
end

# ============================================================================
# Training Wrapper
# ============================================================================

"""
    train_and_time(BasisType, dataset, dataset_config, preset)

Train a basis with timing. Returns `(trained_basis, history, elapsed_seconds)`.

Sets `Random.seed!(42)` before training for reproducibility.
Computes `k = round(Int, 0.10 * img_size^2)` for MSELoss.
"""
function train_and_time(
    BasisType::Type{<:AbstractSparseBasis},
    dataset::Vector{<:AbstractMatrix},
    dataset_config::NamedTuple,
    preset::NamedTuple;
    save_loss_path::Union{Nothing,String} = nothing,
)
    m, n = dataset_config.m, dataset_config.n
    k = round(Int, 0.10 * dataset_config.img_size^2)

    Random.seed!(42)
    elapsed = @elapsed begin
        basis, history = train_basis(
            BasisType, dataset;
            m = m, n = n,
            loss = MSELoss(k),
            epochs = preset.epochs,
            steps_per_image = preset.steps_per_image,
            validation_split = preset.validation_split,
            early_stopping_patience = preset.patience,
            optimizer = preset.optimizer,
            device = preset.device,
            save_loss_path = save_loss_path,
        )
    end

    return basis, history, elapsed
end

# ============================================================================
# Result I/O
# ============================================================================

"""
    save_benchmark_results(path, results_dict)

Save benchmark results (metrics + timing) to JSON.
"""
function save_benchmark_results(path::String, results_dict::Dict)
    mkpath(dirname(path))
    # Convert to serializable format
    serializable = Dict{String,Any}()
    for (name, data) in results_dict
        entry = Dict{String,Any}()
        if haskey(data, :metrics)
            # Convert metric keys from Float64 to String for JSON
            metrics_serializable = Dict{String,Any}()
            for (ratio, vals) in data[:metrics]
                metrics_serializable[string(ratio)] = Dict(pairs(vals))
            end
            entry["metrics"] = metrics_serializable
        end
        if haskey(data, :time)
            entry["time"] = data[:time]
        end
        if haskey(data, :history)
            h = data[:history]
            entry["history"] = Dict(
                "train_losses" => h.train_losses,
                "val_losses" => h.val_losses,
                "step_train_losses" => h.step_train_losses,
                "basis_name" => h.basis_name,
            )
        end
        serializable[name] = entry
    end

    open(path, "w") do io
        JSON3.pretty(io, serializable)
    end
    @info "Results saved to $path"
end

"""
    load_benchmark_results(path)

Load benchmark results from JSON.
"""
function load_benchmark_results(path::String)
    return JSON3.read(read(path, String))
end

# ============================================================================
# Summary Printing
# ============================================================================

"""
    print_dataset_summary(results, keep_ratios)

Print a formatted comparison table for a single dataset.
"""
function print_dataset_summary(results::Dict, keep_ratios::Vector{Float64})
    println("\n" * "=" ^ 100)
    println("COMPRESSION QUALITY COMPARISON")
    println("=" ^ 100)

    # Collect basis names
    basis_names = sort(collect(keys(results)))

    # PSNR table
    println("\nPSNR (dB) — higher is better:")
    println("-" ^ 100)
    @printf("%-25s", "Basis")
    for ratio in keep_ratios
        @printf(" | %10s", "$(round(Int, ratio * 100))%% kept")
    end
    @printf(" | %12s\n", "Train time")
    println("-" ^ 100)

    for name in basis_names
        data = results[name]
        @printf("%-25s", name)
        for ratio in keep_ratios
            if haskey(data[:metrics], ratio)
                @printf(" | %10.2f", data[:metrics][ratio].mean_psnr)
            else
                @printf(" | %10s", "N/A")
            end
        end
        @printf(" | %10.1fs\n", data[:time])
    end

    # SSIM table
    println("\nSSIM — higher is better:")
    println("-" ^ 100)
    @printf("%-25s", "Basis")
    for ratio in keep_ratios
        @printf(" | %10s", "$(round(Int, ratio * 100))%% kept")
    end
    println()
    println("-" ^ 100)

    for name in basis_names
        data = results[name]
        @printf("%-25s", name)
        for ratio in keep_ratios
            if haskey(data[:metrics], ratio)
                @printf(" | %10.4f", data[:metrics][ratio].mean_ssim)
            else
                @printf(" | %10s", "N/A")
            end
        end
        println()
    end
    println("=" ^ 100)
end
