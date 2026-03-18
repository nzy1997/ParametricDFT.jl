#!/usr/bin/env julia
# Quick evaluation of a trained basis vs FFT on CLIC test images
# Usage: julia --project=examples/benchmark examples/benchmark/quick_eval.jl

include("config.jl")
include("data_loading.jl")
include("evaluation.jl")

using ParametricDFT

dataset = :clic
dc = DATASET_CONFIGS[dataset]

# Load a few test images
println("Loading CLIC test images...")
_, test_images, _ = load_clic_dataset(; n_train=1, n_test=5, img_size=dc.img_size)
println("Loaded $(length(test_images)) test images ($(dc.img_size)×$(dc.img_size))")

# Load trained QFT basis from moderate run
trained_path = joinpath(RESULTS_DIR, "clic", "trained_qft.json")
println("\nLoading trained QFT from: $trained_path")
trained_qft = load_basis(trained_path)

# Also create an untrained QFT for comparison
untrained_qft = QFTBasis(dc.m, dc.n)

keep_ratios = [0.05, 0.10, 0.15, 0.20]

println("\n=== Evaluating Trained QFT (moderate, 1000 steps) ===")
trained_results = evaluate_basis(trained_qft, test_images, keep_ratios)

println("\n=== Evaluating Untrained QFT (baseline) ===")
untrained_results = evaluate_basis(untrained_qft, test_images, keep_ratios)

println("\n=== Evaluating FFT Baseline ===")
fft_results = Dict{Float64,NamedTuple}()
for kr in keep_ratios
    mse_v, psnr_v, ssim_v = Float64[], Float64[], Float64[]
    for img in test_images
        rec = fft_compress_recover(img, kr)
        m = compute_metrics(img, rec)
        push!(mse_v, m.mse); push!(psnr_v, m.psnr); push!(ssim_v, m.ssim)
    end
    fft_results[kr] = (
        mean_mse=mean(mse_v), std_mse=std(mse_v),
        mean_psnr=mean(psnr_v), std_psnr=std(psnr_v),
        mean_ssim=mean(ssim_v), std_ssim=std(ssim_v),
    )
end

# Print comparison table
println("\n" * "="^75)
println("CLIC Compression Quality — PSNR (dB)")
println("="^75)
@printf("%-20s %10s %10s %10s %10s\n", "Basis", "5%", "10%", "15%", "20%")
println("-"^75)
for (name, res) in [("Trained QFT", trained_results),
                     ("Untrained QFT", untrained_results),
                     ("Classical FFT", fft_results)]
    @printf("%-20s %10.2f %10.2f %10.2f %10.2f\n",
            name, res[0.05].mean_psnr, res[0.10].mean_psnr,
            res[0.15].mean_psnr, res[0.20].mean_psnr)
end

println("\n" * "="^75)
println("CLIC Compression Quality — SSIM")
println("="^75)
@printf("%-20s %10s %10s %10s %10s\n", "Basis", "5%", "10%", "15%", "20%")
println("-"^75)
for (name, res) in [("Trained QFT", trained_results),
                     ("Untrained QFT", untrained_results),
                     ("Classical FFT", fft_results)]
    @printf("%-20s %10.4f %10.4f %10.4f %10.4f\n",
            name, res[0.05].mean_ssim, res[0.10].mean_ssim,
            res[0.15].mean_ssim, res[0.20].mean_ssim)
end

# Print deltas
println("\n" * "="^75)
println("Improvement over FFT — PSNR (dB)")
println("="^75)
@printf("%-20s %10s %10s %10s %10s\n", "Basis", "5%", "10%", "15%", "20%")
println("-"^75)
for (name, res) in [("Trained QFT", trained_results),
                     ("Untrained QFT", untrained_results)]
    @printf("%-20s %+10.3f %+10.3f %+10.3f %+10.3f\n",
            name,
            res[0.05].mean_psnr - fft_results[0.05].mean_psnr,
            res[0.10].mean_psnr - fft_results[0.10].mean_psnr,
            res[0.15].mean_psnr - fft_results[0.15].mean_psnr,
            res[0.20].mean_psnr - fft_results[0.20].mean_psnr)
end
println("="^75)
