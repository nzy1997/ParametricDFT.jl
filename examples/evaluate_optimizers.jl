# ================================================================================
# Quick Evaluation Script for Optimizer Comparison Results
# ================================================================================
# This script loads the trained QFT bases from optimizer_comparison.jl and
# evaluates their compression quality metrics (PSNR, SSIM, MSE) on the MNIST test set.
#
# Run with:
#   julia --project=examples examples/evaluate_optimizers.jl
# ================================================================================

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using ParametricDFT
using MLDatasets
using Images
using ImageQualityIndexes
using Random
using Statistics
using Printf
using FFTW

# ================================================================================
# Configuration
# ================================================================================

const M_QUBITS = 5
const N_QUBITS = 5
const IMG_SIZE = 32
const NUM_TEST_IMAGES = 10
const COMPRESSION_RATIOS = [0.95, 0.90, 0.85, 0.80]

const INPUT_DIR = joinpath(@__DIR__, "OptimizerComparison")
const OUTPUT_PATH = joinpath(INPUT_DIR, "evaluation_metrics.md")

# ================================================================================
# Utility Functions
# ================================================================================

"""Pad 28x28 image to 32x32 (center-padded)."""
function pad_image(raw_img::AbstractMatrix)
    padded = zeros(Float64, IMG_SIZE, IMG_SIZE)
    padded[3:30, 3:30] = Float64.(raw_img)
    return padded
end

"""Compute quality metrics between original and recovered images."""
function compute_metrics(original::AbstractMatrix, recovered::AbstractMatrix)
    recovered_clamped = clamp.(real.(recovered), 0.0, 1.0)
    mse = mean((original .- recovered_clamped).^2)
    psnr = mse > 0 ? 10 * log10(1.0 / mse) : Inf
    ssim = assess_ssim(Gray.(original), Gray.(recovered_clamped))
    return (mse=mse, psnr=psnr, ssim=ssim)
end

"""Classical FFT compression for comparison."""
function fft_compress(img::AbstractMatrix, ratio::Float64)
    freq = fftshift(fft(img))
    total = length(freq)
    keep = max(1, round(Int, total * (1 - ratio)))

    flat = vec(freq)
    idx = partialsortperm(abs.(flat), 1:keep, rev=true)
    compressed = zeros(ComplexF64, size(freq))
    compressed[idx] = freq[idx]

    return real.(ifft(ifftshift(compressed)))
end

# ================================================================================
# Main Evaluation
# ================================================================================

function main()
    println("="^80)
    println("       Optimizer Comparison - Evaluation Metrics")
    println("="^80)

    # Load test images
    println("\nLoading MNIST test set...")
    mnist_test = MNIST(split=:test)

    Random.seed!(42)
    test_indices = randperm(size(mnist_test.features, 3))[1:NUM_TEST_IMAGES]
    test_images = [pad_image(mnist_test.features[:, :, i]) for i in test_indices]
    println("  Loaded $NUM_TEST_IMAGES test images")

    # Load trained bases
    println("\nLoading trained bases...")

    optimizers = [
        (:gradient_descent, "Gradient Descent"),
        (:conjugate_gradient, "Conjugate Gradient"),
        (:quasi_newton, "L-BFGS"),
    ]

    trained_bases = Dict{String, Any}()
    for (opt_sym, opt_name) in optimizers
        path = joinpath(INPUT_DIR, "trained_qft_$(opt_sym).json")
        if isfile(path)
            trained_bases["Trained QFT ($opt_name)"] = load_basis(path)
            println("  ✓ Loaded: trained_qft_$(opt_sym).json")
        else
            println("  ✗ Not found: trained_qft_$(opt_sym).json")
        end
    end

    # Add untrained baseline
    standard_qft = QFTBasis(M_QUBITS, N_QUBITS)
    trained_bases["Standard QFT (untrained)"] = standard_qft

    basis_names = ["Standard QFT (untrained)"]
    for (_, opt_name) in optimizers
        push!(basis_names, "Trained QFT ($opt_name)")
    end
    push!(basis_names, "Classical FFT")

    # Evaluate
    println("\nEvaluating compression quality...")
    results = Dict{Tuple{String, Float64}, NamedTuple}()

    for ratio in COMPRESSION_RATIOS
        kept_pct = round(Int, (1 - ratio) * 100)
        println("\n--- $(kept_pct)% kept ---")

        for (basis_name, basis) in trained_bases
            mse_vals, psnr_vals, ssim_vals = Float64[], Float64[], Float64[]

            for test_img in test_images
                compressed = compress(basis, test_img; ratio=ratio)
                recovered = recover(basis, compressed)
                metrics = compute_metrics(test_img, recovered)
                push!(mse_vals, metrics.mse)
                push!(psnr_vals, metrics.psnr)
                push!(ssim_vals, metrics.ssim)
            end

            results[(basis_name, ratio)] = (
                mse=mean(mse_vals), psnr=mean(psnr_vals), ssim=mean(ssim_vals),
                mse_std=std(mse_vals), psnr_std=std(psnr_vals), ssim_std=std(ssim_vals)
            )
            @printf("  %-35s PSNR: %.2f dB  SSIM: %.4f  MSE: %.6f\n",
                    basis_name, mean(psnr_vals), mean(ssim_vals), mean(mse_vals))
        end

        # Classical FFT
        mse_vals, psnr_vals, ssim_vals = Float64[], Float64[], Float64[]
        for test_img in test_images
            recovered = fft_compress(test_img, ratio)
            metrics = compute_metrics(test_img, recovered)
            push!(mse_vals, metrics.mse)
            push!(psnr_vals, metrics.psnr)
            push!(ssim_vals, metrics.ssim)
        end
        results[("Classical FFT", ratio)] = (
            mse=mean(mse_vals), psnr=mean(psnr_vals), ssim=mean(ssim_vals),
            mse_std=std(mse_vals), psnr_std=std(psnr_vals), ssim_std=std(ssim_vals)
        )
        @printf("  %-35s PSNR: %.2f dB  SSIM: %.4f  MSE: %.6f\n",
                "Classical FFT", mean(psnr_vals), mean(ssim_vals), mean(mse_vals))
    end

    # Generate markdown summary
    println("\n" * "="^80)
    println("Generating summary...")

    md_content = """
# Optimizer Comparison - Evaluation Metrics

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST |
| Test images | $NUM_TEST_IMAGES |
| Image size | $(IMG_SIZE)×$(IMG_SIZE) |
| Compression ratios | $(join(["$(round(Int, (1-r)*100))%" for r in COMPRESSION_RATIOS], ", ")) |

## Compression Quality Comparison

### PSNR (dB) - higher is better

| Basis / Optimizer | $(join(["$(round(Int, (1-r)*100))% kept" for r in COMPRESSION_RATIOS], " | ")) |
|-------------------|$(join(["------" for _ in COMPRESSION_RATIOS], "|"))|
$(join([
    "| $name | " * join([@sprintf("%.2f", haskey(results, (name, r)) ? results[(name, r)].psnr : NaN) for r in COMPRESSION_RATIOS], " | ") * " |"
    for name in basis_names
], "\n"))

### SSIM - higher is better

| Basis / Optimizer | $(join(["$(round(Int, (1-r)*100))% kept" for r in COMPRESSION_RATIOS], " | ")) |
|-------------------|$(join(["------" for _ in COMPRESSION_RATIOS], "|"))|
$(join([
    "| $name | " * join([@sprintf("%.4f", haskey(results, (name, r)) ? results[(name, r)].ssim : NaN) for r in COMPRESSION_RATIOS], " | ") * " |"
    for name in basis_names
], "\n"))

### MSE - lower is better

| Basis / Optimizer | $(join(["$(round(Int, (1-r)*100))% kept" for r in COMPRESSION_RATIOS], " | ")) |
|-------------------|$(join(["------" for _ in COMPRESSION_RATIOS], "|"))|
$(join([
    "| $name | " * join([@sprintf("%.6f", haskey(results, (name, r)) ? results[(name, r)].mse : NaN) for r in COMPRESSION_RATIOS], " | ") * " |"
    for name in basis_names
], "\n"))

## Summary

"""

    # Find best optimizer at 10% kept
    ratio_10 = 0.90
    best_name, best_psnr = "", -Inf
    for name in basis_names
        if haskey(results, (name, ratio_10)) && results[(name, ratio_10)].psnr > best_psnr
            best_psnr = results[(name, ratio_10)].psnr
            best_name = name
        end
    end

    md_content *= """
**Best at 10% kept:** $best_name (PSNR: $(round(best_psnr, digits=2)) dB)

### Comparison at 10% coefficient retention:

| Optimizer | PSNR (dB) | SSIM | MSE |
|-----------|-----------|------|-----|
"""

    for name in basis_names
        if haskey(results, (name, ratio_10))
            r = results[(name, ratio_10)]
            md_content *= "| $name | $(round(r.psnr, digits=2)) | $(round(r.ssim, digits=4)) | $(round(r.mse, digits=6)) |\n"
        end
    end

    # Write to file
    open(OUTPUT_PATH, "w") do io
        write(io, md_content)
    end

    println("  ✓ Saved: $OUTPUT_PATH")

    # Print summary to console
    println("\n" * "="^80)
    println("                              RESULTS SUMMARY")
    println("="^80)
    println("\nBest at 10% kept: $best_name (PSNR: $(round(best_psnr, digits=2)) dB)")
    println("\nPSNR at 10% kept:")
    for name in basis_names
        if haskey(results, (name, ratio_10))
            @printf("  %-35s %.2f dB\n", name, results[(name, ratio_10)].psnr)
        end
    end

    println("\n" * "="^80)
    println("Evaluation completed!")
    println("="^80)

    return results
end

main()
