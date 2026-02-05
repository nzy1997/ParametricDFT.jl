# ================================================================================
# GPU vs CPU Training Comparison
# ================================================================================
# This example compares training results between:
#   - CPU training using Manifolds.jl/Manopt.jl
#   - GPU training using custom Riemannian optimizer
#
# Both should produce similar training loss curves and compression quality,
# validating that the custom GPU optimizer is working correctly.
#
# Run with:
#   julia --project=examples examples/gpu_cpu_comparison.jl
#
# On a machine with NVIDIA GPU, this will run both CPU and GPU training.
# Without GPU, it will only run CPU training and skip GPU comparison.
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
using LinearAlgebra

# ================================================================================
# Check GPU Availability
# ================================================================================

const HAS_GPU = try
    using CUDA
    CUDA.functional()
catch
    false
end

if HAS_GPU
    using CUDA
    println("GPU detected: $(CUDA.name(CUDA.device()))")
else
    println("No GPU detected. Will run CPU-only comparison with different optimizers.")
end

# ================================================================================
# Configuration
# ================================================================================

const M_QUBITS = 4           # 2^4 = 16 rows
const N_QUBITS = 4           # 2^4 = 16 columns
const IMG_SIZE = 16          # Smaller size for faster comparison

const NUM_TRAINING_IMAGES = 30
const TRAINING_EPOCHS = 5
const STEPS_PER_BATCH = 50
const BATCH_SIZE = 5
const NUM_TEST_IMAGES = 10

const COMPRESSION_RATIO = 0.90  # Keep 10% of coefficients

const OUTPUT_DIR = joinpath(@__DIR__, "GPU_CPU_Comparison")

# ================================================================================
# Utility Functions
# ================================================================================

"""Resize 28×28 MNIST image to target size."""
function resize_image(raw_img::AbstractMatrix, target_size::Int)
    # Simple center crop/pad to target size
    h, w = size(raw_img)
    result = zeros(Float64, target_size, target_size)

    # Calculate offsets for centering
    src_start_h = max(1, (h - target_size) ÷ 2 + 1)
    src_end_h = min(h, src_start_h + target_size - 1)
    dst_start_h = max(1, (target_size - h) ÷ 2 + 1)

    src_start_w = max(1, (w - target_size) ÷ 2 + 1)
    src_end_w = min(w, src_start_w + target_size - 1)
    dst_start_w = max(1, (target_size - w) ÷ 2 + 1)

    src_h = src_end_h - src_start_h + 1
    src_w = src_end_w - src_start_w + 1

    result[dst_start_h:dst_start_h+src_h-1, dst_start_w:dst_start_w+src_w-1] =
        Float64.(raw_img[src_start_h:src_end_h, src_start_w:src_end_w])

    return result
end

"""Compute quality metrics between original and recovered images."""
function compute_metrics(original::AbstractMatrix, recovered::AbstractMatrix)
    recovered_clamped = clamp.(real.(recovered), 0.0, 1.0)
    mse = mean((original .- recovered_clamped).^2)
    psnr = mse > 0 ? 10 * log10(1.0 / mse) : Inf
    ssim = assess_ssim(Gray.(original), Gray.(recovered_clamped))
    return (mse=mse, psnr=psnr, ssim=ssim)
end

"""Evaluate a trained basis on test set."""
function evaluate_basis(basis, test_images, ratio)
    psnr_vals = Float64[]
    ssim_vals = Float64[]

    for img in test_images
        compressed = compress(basis, img; ratio=ratio)
        recovered = recover(basis, compressed)
        metrics = compute_metrics(img, recovered)
        push!(psnr_vals, metrics.psnr)
        push!(ssim_vals, metrics.ssim)
    end

    return (psnr=mean(psnr_vals), ssim=mean(ssim_vals),
            psnr_std=std(psnr_vals), ssim_std=std(ssim_vals))
end

"""Compare two training histories."""
function compare_histories(hist1, hist2, name1, name2)
    println("\n--- Training Loss Comparison ---")
    println(@sprintf("%-20s  %12s  %12s", "Epoch", name1, name2))
    println("-"^50)

    n_epochs = min(length(hist1.train_losses), length(hist2.train_losses))
    for i in 1:n_epochs
        @printf("%-20s  %12.4f  %12.4f\n",
                "Epoch $i train", hist1.train_losses[i], hist2.train_losses[i])
    end

    println()
    for i in 1:n_epochs
        @printf("%-20s  %12.4f  %12.4f\n",
                "Epoch $i val", hist1.val_losses[i], hist2.val_losses[i])
    end

    # Compute relative difference in final loss
    final_diff = abs(hist1.val_losses[end] - hist2.val_losses[end]) /
                 max(hist1.val_losses[end], hist2.val_losses[end]) * 100
    println(@sprintf("\nFinal validation loss difference: %.2f%%", final_diff))

    return final_diff
end

"""Compare compression quality between two bases."""
function compare_quality(metrics1, metrics2, name1, name2)
    println("\n--- Compression Quality Comparison ---")
    println(@sprintf("%-20s  %12s  %12s", "Metric", name1, name2))
    println("-"^50)
    @printf("%-20s  %12.2f  %12.2f\n", "PSNR (dB)", metrics1.psnr, metrics2.psnr)
    @printf("%-20s  %12.4f  %12.4f\n", "SSIM", metrics1.ssim, metrics2.ssim)

    psnr_diff = abs(metrics1.psnr - metrics2.psnr)
    ssim_diff = abs(metrics1.ssim - metrics2.ssim)

    println(@sprintf("\nPSNR difference: %.2f dB", psnr_diff))
    println(@sprintf("SSIM difference: %.4f", ssim_diff))

    return (psnr_diff=psnr_diff, ssim_diff=ssim_diff)
end

# ================================================================================
# Main Comparison
# ================================================================================

function main()
    println("="^70)
    println("         GPU vs CPU Training Comparison")
    println("="^70)

    mkpath(OUTPUT_DIR)

    # =========================================================================
    # Step 1: Load Dataset
    # =========================================================================

    println("\n[Step 1] Loading MNIST dataset...")

    mnist_train = MNIST(split=:train)
    mnist_test = MNIST(split=:test)

    Random.seed!(42)
    train_indices = randperm(size(mnist_train.features, 3))[1:NUM_TRAINING_IMAGES]
    test_indices = randperm(size(mnist_test.features, 3))[1:NUM_TEST_IMAGES]

    training_images = [resize_image(mnist_train.features[:, :, i], IMG_SIZE) for i in train_indices]
    test_images = [resize_image(mnist_test.features[:, :, i], IMG_SIZE) for i in test_indices]

    println("  Training images: $(length(training_images))")
    println("  Test images: $(length(test_images))")
    println("  Image size: $(IMG_SIZE)×$(IMG_SIZE)")

    # =========================================================================
    # Step 2: Training Configuration
    # =========================================================================

    total_coefficients = IMG_SIZE * IMG_SIZE
    k = round(Int, total_coefficients * 0.1)  # 10% retention

    println("\n[Step 2] Training configuration:")
    println("  Epochs: $TRAINING_EPOCHS")
    println("  Steps per batch: $STEPS_PER_BATCH")
    println("  Batch size: $BATCH_SIZE")
    println("  Loss: MSELoss($k) - top $k/$total_coefficients coefficients")
    println("  Compression ratio: $(round(Int, (1-COMPRESSION_RATIO)*100))% retained")

    # =========================================================================
    # Step 2.5: Diagnostic - Compare initial loss on CPU vs GPU
    # =========================================================================

    if HAS_GPU
        println("\n[Diagnostic] Comparing initial loss on CPU vs GPU...")

        # Initialize same tensors
        optcode, init_tensors = qft_code(M_QUBITS, N_QUBITS)
        inverse_code, _ = qft_code(M_QUBITS, N_QUBITS; inverse=true)

        test_img = Complex{Float64}.(training_images[1])

        # CPU loss
        cpu_init_loss = ParametricDFT.loss_function(init_tensors, M_QUBITS, N_QUBITS,
            optcode, test_img, ParametricDFT.MSELoss(k); inverse_code=inverse_code)

        # GPU loss
        gpu_tensors = [CuArray{ComplexF64}(t) for t in init_tensors]
        gpu_img = CuArray{ComplexF64}(test_img)

        gpu_init_loss = ParametricDFT.loss_function(gpu_tensors, M_QUBITS, N_QUBITS,
            optcode, gpu_img, ParametricDFT.MSELoss(k); inverse_code=inverse_code)

        println("  CPU initial loss: ", cpu_init_loss)
        println("  GPU initial loss: ", gpu_init_loss)
        println("  Ratio (GPU/CPU): ", gpu_init_loss / cpu_init_loss)

        # Check tensor types
        println("  CPU tensor type: ", typeof(init_tensors[1]))
        println("  GPU tensor type: ", typeof(gpu_tensors[1]))
        println("  GPU tensor eltype: ", eltype(gpu_tensors[1]))

        if abs(gpu_init_loss / cpu_init_loss - 1.0) > 0.1
            println("  WARNING: Large discrepancy in initial loss!")
        end
    end

    # =========================================================================
    # Step 3: Train on CPU
    # =========================================================================

    println("\n[Step 3] Training on CPU (Manopt.jl)...")
    println("-"^50)

    cpu_time = @elapsed begin
        cpu_basis, cpu_history = train_basis(
            QFTBasis, training_images;
            m=M_QUBITS, n=N_QUBITS,
            loss=ParametricDFT.MSELoss(k),
            epochs=TRAINING_EPOCHS,
            steps_per_image=STEPS_PER_BATCH,
            validation_split=0.2,
            verbose=true,
            optimizer=:gradient_descent,
            batch_size=BATCH_SIZE,
            device=:cpu
        )
    end
    println(@sprintf("\nCPU training time: %.2f seconds", cpu_time))

    # Evaluate CPU basis
    cpu_metrics = evaluate_basis(cpu_basis, test_images, COMPRESSION_RATIO)
    println(@sprintf("CPU test PSNR: %.2f dB (±%.2f)", cpu_metrics.psnr, cpu_metrics.psnr_std))
    println(@sprintf("CPU test SSIM: %.4f (±%.4f)", cpu_metrics.ssim, cpu_metrics.ssim_std))

    # =========================================================================
    # Step 4: Train on GPU (or CPU with GPU optimizer for comparison)
    # =========================================================================

    if HAS_GPU
        println("\n[Step 4] Training on GPU (Custom Riemannian optimizer)...")
        println("-"^50)

        gpu_time = @elapsed begin
            gpu_basis, gpu_history = train_basis(
                QFTBasis, training_images;
                m=M_QUBITS, n=N_QUBITS,
                loss=ParametricDFT.MSELoss(k),
                epochs=TRAINING_EPOCHS,
                steps_per_image=STEPS_PER_BATCH,
                validation_split=0.2,
                verbose=true,
                optimizer=:gradient_descent,
                batch_size=BATCH_SIZE,
                device=:gpu
            )
        end
        println(@sprintf("\nGPU training time: %.2f seconds", gpu_time))
        println(@sprintf("Speedup: %.2fx", cpu_time / gpu_time))

        # Evaluate GPU basis
        gpu_metrics = evaluate_basis(gpu_basis, test_images, COMPRESSION_RATIO)
        println(@sprintf("GPU test PSNR: %.2f dB (±%.2f)", gpu_metrics.psnr, gpu_metrics.psnr_std))
        println(@sprintf("GPU test SSIM: %.4f (±%.4f)", gpu_metrics.ssim, gpu_metrics.ssim_std))

        comparison_name = "GPU"
        comparison_history = gpu_history
        comparison_metrics = gpu_metrics
        comparison_basis = gpu_basis
    else
        # Without GPU, compare different CPU optimizers instead
        println("\n[Step 4] Training with Conjugate Gradient (CPU fallback comparison)...")
        println("-"^50)

        cg_time = @elapsed begin
            cg_basis, cg_history = train_basis(
                QFTBasis, training_images;
                m=M_QUBITS, n=N_QUBITS,
                loss=ParametricDFT.MSELoss(k),
                epochs=TRAINING_EPOCHS,
                steps_per_image=STEPS_PER_BATCH,
                validation_split=0.2,
                verbose=true,
                optimizer=:conjugate_gradient,
                batch_size=BATCH_SIZE,
                device=:cpu
            )
        end
        println(@sprintf("\nCG training time: %.2f seconds", cg_time))

        # Evaluate CG basis
        cg_metrics = evaluate_basis(cg_basis, test_images, COMPRESSION_RATIO)
        println(@sprintf("CG test PSNR: %.2f dB (±%.2f)", cg_metrics.psnr, cg_metrics.psnr_std))
        println(@sprintf("CG test SSIM: %.4f (±%.4f)", cg_metrics.ssim, cg_metrics.ssim_std))

        comparison_name = "CG"
        comparison_history = cg_history
        comparison_metrics = cg_metrics
        comparison_basis = cg_basis
    end

    # =========================================================================
    # Step 5: Compare Results
    # =========================================================================

    println("\n" * "="^70)
    println("                    COMPARISON RESULTS")
    println("="^70)

    loss_diff = compare_histories(cpu_history, comparison_history, "CPU (GD)", comparison_name)
    quality_diff = compare_quality(cpu_metrics, comparison_metrics, "CPU (GD)", comparison_name)

    # =========================================================================
    # Step 6: Validation Criteria
    # =========================================================================

    println("\n" * "="^70)
    println("                    VALIDATION")
    println("="^70)

    # Define acceptable thresholds
    LOSS_DIFF_THRESHOLD = 50.0    # Allow up to 50% difference in loss (optimizers may find different local minima)
    PSNR_DIFF_THRESHOLD = 3.0     # Allow up to 3 dB difference in PSNR
    SSIM_DIFF_THRESHOLD = 0.05    # Allow up to 0.05 difference in SSIM

    loss_ok = loss_diff < LOSS_DIFF_THRESHOLD
    psnr_ok = quality_diff.psnr_diff < PSNR_DIFF_THRESHOLD
    ssim_ok = quality_diff.ssim_diff < SSIM_DIFF_THRESHOLD

    println("\nValidation thresholds:")
    println(@sprintf("  Loss difference < %.0f%%: %s (%.1f%%)",
                    LOSS_DIFF_THRESHOLD, loss_ok ? "PASS" : "FAIL", loss_diff))
    println(@sprintf("  PSNR difference < %.1f dB: %s (%.2f dB)",
                    PSNR_DIFF_THRESHOLD, psnr_ok ? "PASS" : "FAIL", quality_diff.psnr_diff))
    println(@sprintf("  SSIM difference < %.2f: %s (%.4f)",
                    SSIM_DIFF_THRESHOLD, ssim_ok ? "PASS" : "FAIL", quality_diff.ssim_diff))

    all_pass = loss_ok && psnr_ok && ssim_ok

    println("\n" * "="^70)
    if all_pass
        if HAS_GPU
            println("  SUCCESS: GPU and CPU training produce comparable results!")
        else
            println("  SUCCESS: Different optimizers produce comparable results!")
        end
    else
        println("  WARNING: Results differ more than expected.")
        println("  This may indicate an issue with the GPU optimizer.")
    end
    println("="^70)

    # =========================================================================
    # Step 7: Save Results
    # =========================================================================

    # Save training histories
    save_loss_history(joinpath(OUTPUT_DIR, "cpu_gd_history.json"), cpu_history)
    save_loss_history(joinpath(OUTPUT_DIR, "$(lowercase(comparison_name))_history.json"), comparison_history)

    # Save bases
    save_basis(joinpath(OUTPUT_DIR, "cpu_gd_basis.json"), cpu_basis)
    save_basis(joinpath(OUTPUT_DIR, "$(lowercase(comparison_name))_basis.json"), comparison_basis)

    # Generate comparison plots
    histories = [
        TrainingHistory(cpu_history.train_losses, cpu_history.val_losses,
                       cpu_history.step_train_losses, "CPU (GD)"),
        TrainingHistory(comparison_history.train_losses, comparison_history.val_losses,
                       comparison_history.step_train_losses, comparison_name)
    ]

    plots_dir = joinpath(OUTPUT_DIR, "plots")
    mkpath(plots_dir)
    save_training_plots(histories, plots_dir)

    println("\nResults saved to: $OUTPUT_DIR")

    # =========================================================================
    # Step 8: Summary Report
    # =========================================================================

    summary = """
    # GPU vs CPU Training Comparison Report

    ## Configuration
    - Image size: $(IMG_SIZE)×$(IMG_SIZE)
    - Training images: $NUM_TRAINING_IMAGES
    - Test images: $NUM_TEST_IMAGES
    - Epochs: $TRAINING_EPOCHS
    - Steps per batch: $STEPS_PER_BATCH
    - Batch size: $BATCH_SIZE
    - Compression: $(round(Int, (1-COMPRESSION_RATIO)*100))% coefficients retained

    ## Training Time
    | Method | Time (s) |
    |--------|----------|
    | CPU (GD) | $(round(cpu_time, digits=2)) |
    | $comparison_name | $(round(HAS_GPU ? gpu_time : cg_time, digits=2)) |

    ## Final Validation Loss
    | Method | Loss |
    |--------|------|
    | CPU (GD) | $(round(cpu_history.val_losses[end], digits=4)) |
    | $comparison_name | $(round(comparison_history.val_losses[end], digits=4)) |

    ## Compression Quality (Test Set)
    | Method | PSNR (dB) | SSIM |
    |--------|-----------|------|
    | CPU (GD) | $(round(cpu_metrics.psnr, digits=2)) ± $(round(cpu_metrics.psnr_std, digits=2)) | $(round(cpu_metrics.ssim, digits=4)) ± $(round(cpu_metrics.ssim_std, digits=4)) |
    | $comparison_name | $(round(comparison_metrics.psnr, digits=2)) ± $(round(comparison_metrics.psnr_std, digits=2)) | $(round(comparison_metrics.ssim, digits=4)) ± $(round(comparison_metrics.ssim_std, digits=4)) |

    ## Validation
    - Loss difference: $(round(loss_diff, digits=1))% (threshold: $(LOSS_DIFF_THRESHOLD)%) - $(loss_ok ? "PASS" : "FAIL")
    - PSNR difference: $(round(quality_diff.psnr_diff, digits=2)) dB (threshold: $(PSNR_DIFF_THRESHOLD) dB) - $(psnr_ok ? "PASS" : "FAIL")
    - SSIM difference: $(round(quality_diff.ssim_diff, digits=4)) (threshold: $(SSIM_DIFF_THRESHOLD)) - $(ssim_ok ? "PASS" : "FAIL")

    **Overall: $(all_pass ? "PASS" : "FAIL")**
    """

    open(joinpath(OUTPUT_DIR, "report.md"), "w") do io
        write(io, summary)
    end

    println("\nReport saved to: $(joinpath(OUTPUT_DIR, "report.md"))")

    return all_pass
end

# Run the comparison
success = main()
exit(success ? 0 : 1)
