# ================================================================================
# MERA Basis Demo: Training and Image Compression
# ================================================================================
# This example demonstrates the MERA (Multi-scale Entanglement Renormalization
# Ansatz) basis for image compression:
#   1. Creates synthetic training images
#   2. Trains a MERABasis
#   3. Compresses and recovers images at various ratios
#   4. Compares MERA against QFT and Classical FFT baselines
#
# Run with:
#   julia --project=examples examples/mera_demo.jl
#
# No external datasets required — uses synthetic gradient + circle images.
# ================================================================================

using ParametricDFT
using Random
using Statistics
using Printf
using FFTW
using CairoMakie
using LinearAlgebra

# ================================================================================
# Configuration
# ================================================================================

const M_QUBITS = 4              # 2^4 = 16 rows (must be power of 2 for MERA)
const N_QUBITS = 4              # 2^4 = 16 columns
const IMG_SIZE = 2^M_QUBITS     # 16×16 images

const NUM_TRAINING_IMAGES = 15
const TRAINING_EPOCHS = 2
const STEPS_PER_IMAGE = 30
const NUM_TEST_IMAGES = 5

const COMPRESSION_RATIOS = [0.95, 0.90, 0.80, 0.70]  # Keep 5%, 10%, 20%, 30%

const OUTPUT_DIR = joinpath(@__DIR__, "MERADemo")

# ================================================================================
# Synthetic Image Generation
# ================================================================================

"""Generate a synthetic image with gradients, circles, and noise."""
function generate_image(seed::Int)
    Random.seed!(seed)
    img = zeros(Float64, IMG_SIZE, IMG_SIZE)

    # Random gradient background
    angle = rand() * 2π
    for i in 1:IMG_SIZE, j in 1:IMG_SIZE
        img[i, j] = 0.3 * (cos(angle) * (i - 1) + sin(angle) * (j - 1)) / IMG_SIZE
    end

    # Add circles with random intensity
    n_circles = rand(2:5)
    for _ in 1:n_circles
        cx, cy = rand(1:IMG_SIZE), rand(1:IMG_SIZE)
        r = rand(2:IMG_SIZE÷3)
        intensity = rand() * 0.6 + 0.2
        for i in 1:IMG_SIZE, j in 1:IMG_SIZE
            if (i - cx)^2 + (j - cy)^2 < r^2
                img[i, j] += intensity
            end
        end
    end

    # Add sinusoidal texture
    freq = rand(1:3)
    img .+= 0.1 * sin.(2π * freq .* (1:IMG_SIZE) ./ IMG_SIZE) * ones(IMG_SIZE)'

    # Add small noise and clamp
    img .+= randn(IMG_SIZE, IMG_SIZE) * 0.02
    return clamp.(img, 0.0, 1.0)
end

# ================================================================================
# Classical FFT Compression Baseline
# ================================================================================

"""Classical FFT compression: keep top-k coefficients by magnitude."""
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
# Metrics
# ================================================================================

"""Compute MSE and PSNR between original and recovered images."""
function compute_metrics(original::AbstractMatrix, recovered::AbstractMatrix)
    recovered_clamped = clamp.(real.(recovered), 0.0, 1.0)
    mse = mean((original .- recovered_clamped) .^ 2)
    psnr = mse > 0 ? 10 * log10(1.0 / mse) : Inf
    return (mse=mse, psnr=psnr)
end

# ================================================================================
# Visualization
# ================================================================================

"""Plot original vs recovered images for each basis at a given compression ratio."""
function plot_compression_comparison(
    test_img::AbstractMatrix, bases::Vector, basis_names::Vector{String},
    ratio::Float64, output_path::String
)
    n_bases = length(bases)
    fig = Figure(size=(300 * (n_bases + 2), 300))

    # Original
    ax = Axis(fig[1, 1]; title="Original", aspect=DataAspect())
    heatmap!(ax, rotr90(test_img); colormap=:grays, colorrange=(0, 1))
    hidedecorations!(ax)

    # Each basis
    for (i, (basis, name)) in enumerate(zip(bases, basis_names))
        compressed = compress(basis, test_img; ratio=ratio)
        recovered = clamp.(real.(recover(basis, compressed)), 0.0, 1.0)
        metrics = compute_metrics(test_img, recovered)

        ax = Axis(fig[1, i + 1]; title="$name\nPSNR: $(@sprintf("%.1f", metrics.psnr)) dB",
                  aspect=DataAspect())
        heatmap!(ax, rotr90(recovered); colormap=:grays, colorrange=(0, 1))
        hidedecorations!(ax)
    end

    # Classical FFT
    recovered_fft = clamp.(fft_compress(test_img, ratio), 0.0, 1.0)
    metrics_fft = compute_metrics(test_img, recovered_fft)
    ax = Axis(fig[1, n_bases + 2]; title="FFT\nPSNR: $(@sprintf("%.1f", metrics_fft.psnr)) dB",
              aspect=DataAspect())
    heatmap!(ax, rotr90(recovered_fft); colormap=:grays, colorrange=(0, 1))
    hidedecorations!(ax)

    kept_pct = round(Int, (1 - ratio) * 100)
    Label(fig[0, :], "Compression Comparison ($kept_pct% coefficients kept)";
          fontsize=16, font=:bold)

    save(output_path, fig; px_per_unit=2)
    return fig
end

"""Plot PSNR vs compression ratio for all bases."""
function plot_psnr_curves(results::Dict, ratios::Vector{Float64},
                          basis_names::Vector{String}, output_path::String)
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1]; xlabel="Coefficients Kept (%)",
              ylabel="PSNR (dB)", title="Compression Quality")

    colors = [:blue, :red, :green, :orange, :purple, :brown]
    for (i, name) in enumerate(basis_names)
        kept_pcts = [round(Int, (1 - r) * 100) for r in ratios]
        psnrs = [results[(name, r)].psnr for r in ratios]
        lines!(ax, kept_pcts, psnrs; label=name, color=colors[mod1(i, length(colors))],
               linewidth=2)
        scatter!(ax, kept_pcts, psnrs; color=colors[mod1(i, length(colors))], markersize=8)
    end

    axislegend(ax; position=:rb)
    save(output_path, fig; px_per_unit=2)
    return fig
end

# ================================================================================
# Main Demo
# ================================================================================

function main()
    println("=" ^ 80)
    println("  MERA Basis Demo: Training and Image Compression")
    println("  Image size: $(IMG_SIZE)×$(IMG_SIZE) | Qubits: $(M_QUBITS)×$(N_QUBITS)")
    println("=" ^ 80)

    mkpath(OUTPUT_DIR)

    # --------------------------------------------------------------------------
    # Step 1: Generate synthetic dataset
    # --------------------------------------------------------------------------
    println("\nStep 1: Generating synthetic dataset...")

    Random.seed!(42)
    training_images = [generate_image(i) for i in 1:NUM_TRAINING_IMAGES]
    test_images = [generate_image(i + 1000) for i in 1:NUM_TEST_IMAGES]

    println("  Training: $(length(training_images)) images")
    println("  Test:     $(length(test_images)) images")

    # --------------------------------------------------------------------------
    # Step 2: Create untrained bases
    # --------------------------------------------------------------------------
    println("\nStep 2: Creating untrained bases...")

    qft_basis = QFTBasis(M_QUBITS, N_QUBITS)
    mera_default = MERABasis(M_QUBITS, N_QUBITS)

    println("  QFT:  $(num_parameters(qft_basis)) parameters")
    println("  MERA: $(num_parameters(mera_default)) parameters ($(num_gates(mera_default)) gates)")

    # --------------------------------------------------------------------------
    # Step 3: Train MERA basis
    # --------------------------------------------------------------------------
    println("\nStep 3: Training bases...")

    total_coefficients = IMG_SIZE * IMG_SIZE
    k = round(Int, total_coefficients * 0.1)

    println("  Loss: MSELoss($k) — reconstruct from top $k/$total_coefficients coefficients")
    println("  Epochs: $TRAINING_EPOCHS | Steps/image: $STEPS_PER_IMAGE")

    loss_dir = joinpath(OUTPUT_DIR, "loss_history")
    mkpath(loss_dir)

    println("\n  Training QFT...")
    trained_qft, qft_history = @time train_basis(
        QFTBasis, training_images;
        m=M_QUBITS, n=N_QUBITS,
        loss=ParametricDFT.MSELoss(k),
        epochs=TRAINING_EPOCHS,
        steps_per_image=STEPS_PER_IMAGE,
        validation_split=0.2,
        save_loss_path=joinpath(loss_dir, "qft_loss.json")
    )

    println("\n  Training MERA...")
    trained_mera, mera_history = @time train_basis(
        MERABasis, training_images;
        m=M_QUBITS, n=N_QUBITS,
        loss=ParametricDFT.MSELoss(k),
        epochs=TRAINING_EPOCHS,
        steps_per_image=STEPS_PER_IMAGE,
        validation_split=0.2,
        save_loss_path=joinpath(loss_dir, "mera_loss.json")
    )

    println("\n  Training complete!")
    println("  Trained QFT:  $(num_parameters(trained_qft)) parameters")
    println("  Trained MERA: $(num_parameters(trained_mera)) parameters")
    println("  MERA phases:  $(round.(get_phases(trained_mera), digits=4))")

    # --------------------------------------------------------------------------
    # Step 4: Visualize training loss
    # --------------------------------------------------------------------------
    println("\nStep 4: Visualizing training loss...")

    histories = [
        TrainingHistory(qft_history.train_losses, qft_history.val_losses,
                        qft_history.step_train_losses, "QFT"),
        TrainingHistory(mera_history.train_losses, mera_history.val_losses,
                        mera_history.step_train_losses, "MERA")
    ]

    plots_dir = joinpath(OUTPUT_DIR, "plots")
    saved_plots = save_training_plots(histories, plots_dir)
    println("  Saved $(length(saved_plots)) training plots to $plots_dir")

    # --------------------------------------------------------------------------
    # Step 5: Save trained bases
    # --------------------------------------------------------------------------
    println("\nStep 5: Saving trained bases...")

    qft_path = joinpath(OUTPUT_DIR, "trained_qft.json")
    mera_path = joinpath(OUTPUT_DIR, "trained_mera.json")
    save_basis(qft_path, trained_qft)
    save_basis(mera_path, trained_mera)
    println("  Saved: trained_qft.json ($(round(filesize(qft_path)/1024, digits=1)) KB)")
    println("  Saved: trained_mera.json ($(round(filesize(mera_path)/1024, digits=1)) KB)")

    # --------------------------------------------------------------------------
    # Step 6: Evaluate compression quality
    # --------------------------------------------------------------------------
    println("\nStep 6: Evaluating compression quality on test set...")

    all_bases = [qft_basis, trained_qft, mera_default, trained_mera]
    all_names = ["Standard QFT", "Trained QFT", "Standard MERA", "Trained MERA"]

    results = Dict{Tuple{String,Float64}, NamedTuple}()

    for ratio in COMPRESSION_RATIOS
        kept_pct = round(Int, (1 - ratio) * 100)
        println("\n  $kept_pct% kept:")

        for (basis, name) in zip(all_bases, all_names)
            mse_vals, psnr_vals = Float64[], Float64[]
            for img in test_images
                compressed = compress(basis, img; ratio=ratio)
                recovered = recover(basis, compressed)
                m = compute_metrics(img, recovered)
                push!(mse_vals, m.mse)
                push!(psnr_vals, m.psnr)
            end
            results[(name, ratio)] = (mse=mean(mse_vals), psnr=mean(psnr_vals))
            @printf("    %-20s MSE: %.6f  PSNR: %.2f dB\n", name, mean(mse_vals), mean(psnr_vals))
        end

        # Classical FFT baseline
        mse_vals, psnr_vals = Float64[], Float64[]
        for img in test_images
            recovered = fft_compress(img, ratio)
            m = compute_metrics(img, recovered)
            push!(mse_vals, m.mse)
            push!(psnr_vals, m.psnr)
        end
        results[("Classical FFT", ratio)] = (mse=mean(mse_vals), psnr=mean(psnr_vals))
        @printf("    %-20s MSE: %.6f  PSNR: %.2f dB\n", "Classical FFT", mean(mse_vals), mean(psnr_vals))
    end

    # --------------------------------------------------------------------------
    # Step 7: Generate comparison plots
    # --------------------------------------------------------------------------
    println("\nStep 7: Generating comparison plots...")

    # Visual compression comparison (first test image, 10% kept)
    comparison_path = joinpath(OUTPUT_DIR, "compression_comparison.png")
    plot_compression_comparison(
        test_images[1], all_bases, all_names,
        0.90, comparison_path
    )
    println("  Saved: compression_comparison.png")

    # PSNR curves
    curve_names = vcat(all_names, ["Classical FFT"])
    psnr_path = joinpath(OUTPUT_DIR, "psnr_curves.png")
    plot_psnr_curves(results, COMPRESSION_RATIOS, curve_names, psnr_path)
    println("  Saved: psnr_curves.png")

    # --------------------------------------------------------------------------
    # Step 8: Print summary
    # --------------------------------------------------------------------------
    println("\n" * "=" ^ 80)
    println("  SUMMARY")
    println("=" ^ 80)

    println("\nCompression Quality (PSNR dB) — higher is better:")
    @printf("\n  %-20s", "Basis")
    for ratio in COMPRESSION_RATIOS
        @printf(" | %8s", "$(round(Int, (1-ratio)*100))% kept")
    end
    println()
    println("  " * "-" ^ 65)

    for name in curve_names
        @printf("  %-20s", name)
        for ratio in COMPRESSION_RATIOS
            @printf(" | %8.2f", results[(name, ratio)].psnr)
        end
        println()
    end

    # Best at 10% kept
    ratio_10 = 0.90
    best_name, best_psnr = "", -Inf
    for name in curve_names
        if results[(name, ratio_10)].psnr > best_psnr
            best_psnr = results[(name, ratio_10)].psnr
            best_name = name
        end
    end

    println("\n  Best at 10% kept: $best_name (PSNR: $(round(best_psnr, digits=2)) dB)")
    println("\n  Output directory: $OUTPUT_DIR")
    println("=" ^ 80)
end

main()
