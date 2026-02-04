# ================================================================================
# Optimizer Comparison Demo
# ================================================================================
# This example compares three Riemannian optimizers on the QFTBasis training task:
#   1. Riemannian Gradient Descent (baseline)
#   2. Riemannian Conjugate Gradient (Manopt.jl conjugate_gradient_descent)
#   3. Riemannian L-BFGS (Manopt.jl quasi_Newton)
#
# Each optimizer trains a QFTBasis on the same MNIST dataset with identical
# initial conditions, then training loss curves are compared.
#
# Run with:
#   julia --project=examples examples/optimizer_comparison.jl
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
using CairoMakie

# ================================================================================
# Configuration
# ================================================================================

const M_QUBITS = 5           # 2^5 = 32 rows
const N_QUBITS = 5           # 2^5 = 32 columns
const IMG_SIZE = 32

# Larger dataset and more training steps for meaningful optimizer comparison
const NUM_TRAINING_IMAGES = 80
const TRAINING_EPOCHS = 20
const STEPS_PER_IMAGE = 50
const NUM_TEST_IMAGES = 10
const EARLY_STOPPING_PATIENCE = 20  # Effectively disabled to see full curves

# Compression ratios to test
const COMPRESSION_RATIOS = [0.95, 0.90, 0.85, 0.80]

# Output directory
const OUTPUT_DIR = joinpath(@__DIR__, "OptimizerComparison")

# Optimizers to compare
const OPTIMIZERS = [
    (:gradient_descent, "Gradient Descent"),
    (:conjugate_gradient, "Conjugate Gradient"),
    (:quasi_newton, "L-BFGS"),
]

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
# Main Demo
# ================================================================================

function main()
    println("="^100)
    println("       Optimizer Comparison: Gradient Descent vs Conjugate Gradient vs L-BFGS")
    println("="^100)

    mkpath(OUTPUT_DIR)

    # ============================================================================
    # Step 1: Load MNIST Dataset
    # ============================================================================

    println("\n" * "="^80)
    println("Step 1: Loading MNIST Dataset")
    println("="^80)

    mnist_train = MNIST(split=:train)
    mnist_test = MNIST(split=:test)

    println("  Training pool: $(size(mnist_train.features, 3)) images")
    println("  Test pool: $(size(mnist_test.features, 3)) images")

    Random.seed!(42)
    train_indices = randperm(size(mnist_train.features, 3))[1:NUM_TRAINING_IMAGES]
    test_indices = randperm(size(mnist_test.features, 3))[1:NUM_TEST_IMAGES]

    training_images = [pad_image(mnist_train.features[:, :, i]) for i in train_indices]
    test_images = [pad_image(mnist_test.features[:, :, i]) for i in test_indices]

    println("  Prepared $(length(training_images)) training images")
    println("  Prepared $(length(test_images)) test images")
    println("  Image size: $(IMG_SIZE)x$(IMG_SIZE) (padded from 28x28)")

    # ============================================================================
    # Step 2: Train QFTBasis with Each Optimizer
    # ============================================================================

    println("\n" * "="^80)
    println("Step 2: Training QFTBasis with Each Optimizer")
    println("="^80)

    total_coefficients = IMG_SIZE * IMG_SIZE
    k = round(Int, total_coefficients * 0.1)  # Keep 10% for MSE loss

    # Expected tracked steps per optimizer:
    n_train = round(Int, NUM_TRAINING_IMAGES * 0.8)
    expected_steps = n_train * TRAINING_EPOCHS
    println("\nTraining configuration:")
    println("  Images: $NUM_TRAINING_IMAGES ($n_train training, $(NUM_TRAINING_IMAGES - n_train) validation)")
    println("  Epochs: $TRAINING_EPOCHS | Steps/image: $STEPS_PER_IMAGE")
    println("  Loss: MSELoss($k) - reconstruct from top $k/$total_coefficients coefficients")
    println("  Expected tracked steps per optimizer: $expected_steps")
    println("  Total gradient evaluations per optimizer: $(expected_steps * STEPS_PER_IMAGE)")

    loss_dir = joinpath(OUTPUT_DIR, "loss_history")
    mkpath(loss_dir)

    trained_bases = Dict{Symbol, Any}()
    all_histories = Dict{Symbol, Any}()

    for (opt_sym, opt_name) in OPTIMIZERS
        println("\n" * "-"^80)
        println("Training with: $opt_name (optimizer = :$opt_sym)")
        println("-"^80)

        # Use same random seed for fair comparison of data splits
        Random.seed!(123)

        basis, history = @time train_basis(
            QFTBasis, training_images;
            m=M_QUBITS, n=N_QUBITS,
            loss=ParametricDFT.MSELoss(k),
            epochs=TRAINING_EPOCHS,
            steps_per_image=STEPS_PER_IMAGE,
            validation_split=0.2,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            verbose=true,
            optimizer=opt_sym,
            save_loss_path=joinpath(loss_dir, "qft_$(opt_sym)_loss.json")
        )

        trained_bases[opt_sym] = basis
        all_histories[opt_sym] = history
        println("  Final train loss: $(round(history.train_losses[end], digits=6))")
        println("  Final val loss:   $(round(history.val_losses[end], digits=6))")
        println("  Tracked steps:    $(length(history.step_train_losses))")
    end

    # ============================================================================
    # Step 3: Visualize Training Loss Comparison
    # ============================================================================

    println("\n" * "="^80)
    println("Step 3: Visualizing Training Loss Comparison")
    println("="^80)

    # Build TrainingHistory objects with optimizer name included
    histories = TrainingHistory[]
    for (opt_sym, opt_name) in OPTIMIZERS
        h = all_histories[opt_sym]
        push!(histories, TrainingHistory(
            h.train_losses, h.val_losses, h.step_train_losses,
            "QFT ($opt_name)"
        ))
    end

    # Generate all standard comparison plots
    plots_dir = joinpath(OUTPUT_DIR, "plots")
    println("\nGenerating training loss plots...")
    saved_plots = save_training_plots(histories, plots_dir; smoothing=0.8)

    println("\nSaved $(length(saved_plots)) plots to: $plots_dir")
    for plot_path in saved_plots
        relpath_str = relpath(plot_path, plots_dir)
        println("  $relpath_str")
    end

    # ============================================================================
    # Step 4: Evaluate on Test Set
    # ============================================================================

    println("\n" * "="^80)
    println("Step 4: Evaluating on Test Set ($NUM_TEST_IMAGES images)")
    println("="^80)

    # Include untrained QFT and classical FFT as baselines
    standard_qft = QFTBasis(M_QUBITS, N_QUBITS)

    eval_bases = Dict{String, Any}(
        "Standard QFT (untrained)" => standard_qft,
    )
    for (opt_sym, opt_name) in OPTIMIZERS
        eval_bases["Trained QFT ($opt_name)"] = trained_bases[opt_sym]
    end

    basis_names = ["Standard QFT (untrained)"]
    for (_, opt_name) in OPTIMIZERS
        push!(basis_names, "Trained QFT ($opt_name)")
    end
    push!(basis_names, "Classical FFT")

    results = Dict{Tuple{String, Float64}, NamedTuple}()

    for ratio in COMPRESSION_RATIOS
        kept_pct = round(Int, (1 - ratio) * 100)
        println("\n--- Compression ratio: $(kept_pct)% kept ---")

        for (basis_name, basis) in eval_bases
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
            @printf("  %-40s MSE: %.6f  PSNR: %.2f dB  SSIM: %.4f\n",
                    basis_name, mean(mse_vals), mean(psnr_vals), mean(ssim_vals))
        end

        # Classical FFT baseline
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
        @printf("  %-40s MSE: %.6f  PSNR: %.2f dB  SSIM: %.4f\n",
                "Classical FFT", mean(mse_vals), mean(psnr_vals), mean(ssim_vals))
    end

    # ============================================================================
    # Step 5: Print Comparison Table
    # ============================================================================

    println("\n" * "="^100)
    println("                   OPTIMIZER COMPARISON - COMPRESSION QUALITY")
    println("="^100)

    # PSNR table
    println("\nPSNR (dB) - higher is better:")
    println("-"^100)
    @printf("%-42s", "Basis / Optimizer")
    for ratio in COMPRESSION_RATIOS
        @printf(" | %12s", "$(round(Int, (1-ratio)*100))% kept")
    end
    println()
    println("-"^100)

    for name in basis_names
        @printf("%-42s", name)
        for ratio in COMPRESSION_RATIOS
            if haskey(results, (name, ratio))
                @printf(" | %12.2f", results[(name, ratio)].psnr)
            else
                @printf(" | %12s", "N/A")
            end
        end
        println()
    end

    # SSIM table
    println("\nSSIM - higher is better:")
    println("-"^100)
    @printf("%-42s", "Basis / Optimizer")
    for ratio in COMPRESSION_RATIOS
        @printf(" | %12s", "$(round(Int, (1-ratio)*100))% kept")
    end
    println()
    println("-"^100)

    for name in basis_names
        @printf("%-42s", name)
        for ratio in COMPRESSION_RATIOS
            if haskey(results, (name, ratio))
                @printf(" | %12.4f", results[(name, ratio)].ssim)
            else
                @printf(" | %12s", "N/A")
            end
        end
        println()
    end
    println("="^100)

    # ============================================================================
    # Step 6: Optimizer Summary
    # ============================================================================

    println("\n" * "="^100)
    println("                              OPTIMIZER SUMMARY")
    println("="^100)

    ratio_10 = 0.90  # 10% kept
    println("\nFinal training loss comparison:")
    println("-"^60)
    for (opt_sym, opt_name) in OPTIMIZERS
        h = all_histories[opt_sym]
        best_val = minimum(h.val_losses)
        final_train = h.train_losses[end]
        n_steps = length(h.step_train_losses)
        @printf("  %-25s Train: %.6f  Best Val: %.6f  Steps: %d\n",
                opt_name, final_train, best_val, n_steps)
    end

    println("\nCompression quality at 10% kept (PSNR dB):")
    println("-"^60)
    best_name, best_psnr = "", -Inf
    for (_, opt_name) in OPTIMIZERS
        name = "Trained QFT ($opt_name)"
        if haskey(results, (name, ratio_10))
            psnr = results[(name, ratio_10)].psnr
            @printf("  %-25s PSNR: %.2f dB\n", opt_name, psnr)
            if psnr > best_psnr
                best_psnr = psnr
                best_name = opt_name
            end
        end
    end

    if haskey(results, ("Standard QFT (untrained)", ratio_10))
        @printf("  %-25s PSNR: %.2f dB (baseline)\n",
                "Untrained", results[("Standard QFT (untrained)", ratio_10)].psnr)
    end
    if haskey(results, ("Classical FFT", ratio_10))
        @printf("  %-25s PSNR: %.2f dB (reference)\n",
                "Classical FFT", results[("Classical FFT", ratio_10)].psnr)
    end

    println("\nBest optimizer at 10% kept: $best_name (PSNR: $(round(best_psnr, digits=2)) dB)")

    # ============================================================================
    # Step 7: Save Trained Bases
    # ============================================================================

    println("\n" * "="^80)
    println("Step 7: Saving Trained Bases")
    println("="^80)

    for (opt_sym, opt_name) in OPTIMIZERS
        path = joinpath(OUTPUT_DIR, "trained_qft_$(opt_sym).json")
        save_basis(path, trained_bases[opt_sym])
        println("  Saved: trained_qft_$(opt_sym).json ($(round(filesize(path)/1024, digits=2)) KB)")
    end

    # ============================================================================
    # Summary
    # ============================================================================

    println("\n" * "="^100)
    println("                                    SUMMARY")
    println("="^100)

    println("""

    Configuration:
    |- Dataset:             MNIST
    |- Training images:     $NUM_TRAINING_IMAGES
    |- Test images:         $NUM_TEST_IMAGES
    |- Image size:          $(IMG_SIZE)x$(IMG_SIZE)
    |- Epochs:              $TRAINING_EPOCHS
    |- Steps per image:     $STEPS_PER_IMAGE
    |- Loss:                MSELoss($k)

    Optimizers compared:
    |- Riemannian Gradient Descent (baseline)
    |- Riemannian Conjugate Gradient
    |- Riemannian L-BFGS (quasi-Newton)

    Best optimizer: $best_name (PSNR: $(round(best_psnr, digits=2)) dB at 10% kept)

    Output files saved to: $OUTPUT_DIR
    |- trained_qft_gradient_descent.json
    |- trained_qft_conjugate_gradient.json
    |- trained_qft_quasi_newton.json
    |- loss_history/              : Training loss data (JSON)
    |- plots/                     : Training loss visualizations
       |- log/                    : Log-scale plots
       |  |- comparison_all.png   : All optimizers (train + val)
       |  |- comparison_steps.png : Step-level comparison
       |  |- grid.png             : Grid view
       |- linear/                 : Linear-scale plots
    """)

    println("="^100)
    println("Demo completed successfully!")
    println("="^100)

    return results, all_histories
end

# Run the demo
main()
