# ================================================================================
# Entangle Position Comparison Demo
# ================================================================================
# This example compares the three entanglement gate placements (:front, :middle,
# :back) for the EntangledQFTBasis, training each variant on the same dataset
# and measuring compression quality across several ratios.
#
# The standard (untrained) QFT and classical FFT serve as baselines.
#
# Run with:
#   julia --project=examples examples/entangle_position_demo.jl
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
using CairoMakie  # Used for training loss visualization

# ================================================================================
# Configuration
# ================================================================================

const M_QUBITS = 5           # 2^5 = 32 rows
const N_QUBITS = 5           # 2^5 = 32 columns
const IMG_SIZE = 32

const NUM_TRAINING_IMAGES = 20
const TRAINING_EPOCHS = 2
const STEPS_PER_IMAGE = 50
const NUM_TEST_IMAGES = 5

const COMPRESSION_RATIOS = [0.95, 0.90, 0.85, 0.80]  # Keep 5%, 10%, 15%, 20%

const POSITIONS = [:front, :middle, :back]

const OUTPUT_DIR = joinpath(@__DIR__, "EntanglePositionDemo")

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

"""Classical FFT compression for comparison baseline."""
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
    println("="^90)
    println("  Entangle Position Comparison: :front vs :middle vs :back")
    println("="^90)

    mkpath(OUTPUT_DIR)

    # =========================================================================
    # Step 1: Load MNIST
    # =========================================================================

    println("\n--- Step 1: Loading MNIST ---")
    mnist_train = MNIST(split=:train)
    mnist_test  = MNIST(split=:test)

    Random.seed!(42)
    train_idx = randperm(size(mnist_train.features, 3))[1:NUM_TRAINING_IMAGES]
    test_idx  = randperm(size(mnist_test.features, 3))[1:NUM_TEST_IMAGES]

    training_images = [pad_image(mnist_train.features[:, :, i]) for i in train_idx]
    test_images     = [pad_image(mnist_test.features[:, :, i])  for i in test_idx]
    test_labels     = [string(mnist_test.targets[i])             for i in test_idx]

    println("  Training images: $(length(training_images))")
    println("  Test images:     $(length(test_images))")

    # =========================================================================
    # Step 2: Create baselines
    # =========================================================================

    println("\n--- Step 2: Creating baselines ---")
    standard_qft = QFTBasis(M_QUBITS, N_QUBITS)
    println("  Standard QFT: $(num_parameters(standard_qft)) parameters")

    # =========================================================================
    # Step 3: Train one EntangledQFTBasis per position
    # =========================================================================

    total_coefficients = IMG_SIZE * IMG_SIZE
    k = round(Int, total_coefficients * 0.1)

    println("\n--- Step 3: Training EntangledQFTBasis for each position ---")
    println("  Loss: MSELoss($k) | Epochs: $TRAINING_EPOCHS | Steps/image: $STEPS_PER_IMAGE")

    trained = Dict{Symbol, EntangledQFTBasis}()
    train_histories = Dict{Symbol, Any}()

    for pos in POSITIONS
        println("\n  [$pos] Training...")
        @time trained[pos], train_histories[pos] = train_basis(
            EntangledQFTBasis, training_images;
            m=M_QUBITS, n=N_QUBITS,
            entangle_position=pos,
            loss=ParametricDFT.MSELoss(k),
            epochs=TRAINING_EPOCHS,
            steps_per_image=STEPS_PER_IMAGE,
            validation_split=0.2,
            verbose=true
        )
        println("  [$pos] Parameters: $(num_parameters(trained[pos]))")
        println("  [$pos] Phases:     $(round.(get_entangle_phases(trained[pos]), digits=4))")
    end

    # =========================================================================
    # Step 4: Save trained bases
    # =========================================================================

    println("\n--- Step 4: Saving trained bases ---")
    for pos in POSITIONS
        path = joinpath(OUTPUT_DIR, "trained_entangled_$(pos).json")
        save_basis(path, trained[pos])
        println("  Saved: trained_entangled_$(pos).json ($(round(filesize(path)/1024, digits=2)) KB)")
    end

    # =========================================================================
    # Step 4.5: Visualize Training Losses
    # =========================================================================

    println("\n--- Step 4.5: Visualizing Training Losses ---")

    histories = [
        TrainingHistory(
            train_histories[pos].train_losses,
            train_histories[pos].val_losses,
            train_histories[pos].step_train_losses,
            "Entangled :$pos"
        )
        for pos in POSITIONS
    ]

    plots_dir = joinpath(OUTPUT_DIR, "plots")
    saved_plots = save_training_plots(histories, plots_dir)

    println("  Training loss plots saved to: $plots_dir")
    for plot_path in saved_plots
        println("    ✓ $(relpath(plot_path, plots_dir))")
    end

    # =========================================================================
    # Step 5: Evaluate on test set
    # =========================================================================

    println("\n--- Step 5: Evaluating on test set ---")

    basis_map = Dict{String, AbstractSparseBasis}(
        "Standard QFT"     => standard_qft,
        "Entangled :front"  => trained[:front],
        "Entangled :middle" => trained[:middle],
        "Entangled :back"   => trained[:back],
    )

    basis_names = [
        "Standard QFT",
        "Entangled :front",
        "Entangled :middle",
        "Entangled :back",
        "Classical FFT",
    ]

    results = Dict{Tuple{String, Float64}, NamedTuple}()

    for ratio in COMPRESSION_RATIOS
        kept_pct = round(Int, (1 - ratio) * 100)
        println("\n  Compression: $kept_pct% kept")

        for (name, basis) in basis_map
            mse_v, psnr_v, ssim_v = Float64[], Float64[], Float64[]
            for img in test_images
                compressed = compress(basis, img; ratio=ratio)
                recovered  = recover(basis, compressed)
                m_val = compute_metrics(img, recovered)
                push!(mse_v, m_val.mse); push!(psnr_v, m_val.psnr); push!(ssim_v, m_val.ssim)
            end
            results[(name, ratio)] = (mse=mean(mse_v), psnr=mean(psnr_v), ssim=mean(ssim_v))
            @printf("    %-22s  MSE: %.6f  PSNR: %6.2f dB  SSIM: %.4f\n",
                    name, mean(mse_v), mean(psnr_v), mean(ssim_v))
        end

        # Classical FFT baseline
        mse_v, psnr_v, ssim_v = Float64[], Float64[], Float64[]
        for img in test_images
            recovered = fft_compress(img, ratio)
            m_val = compute_metrics(img, recovered)
            push!(mse_v, m_val.mse); push!(psnr_v, m_val.psnr); push!(ssim_v, m_val.ssim)
        end
        results[("Classical FFT", ratio)] = (mse=mean(mse_v), psnr=mean(psnr_v), ssim=mean(ssim_v))
        @printf("    %-22s  MSE: %.6f  PSNR: %6.2f dB  SSIM: %.4f\n",
                "Classical FFT", mean(mse_v), mean(psnr_v), mean(ssim_v))
    end

    # =========================================================================
    # Step 6: Save sample recovered images
    # =========================================================================

    println("\n--- Step 6: Saving sample images ---")
    sample_img   = test_images[1]
    sample_label = test_labels[1]
    sample_ratio = 0.90

    Images.save(joinpath(OUTPUT_DIR, "original_$(sample_label).png"), Gray.(sample_img))
    println("  original_$(sample_label).png")

    for (name, basis) in basis_map
        compressed = compress(basis, sample_img; ratio=sample_ratio)
        recovered  = recover(basis, compressed)
        safe = lowercase(replace(name, " " => "_", ":" => ""))
        path = joinpath(OUTPUT_DIR, "recovered_$(safe).png")
        Images.save(path, Gray.(clamp.(real.(recovered), 0.0, 1.0)))
        println("  recovered_$(safe).png")
    end

    recovered_fft = fft_compress(sample_img, sample_ratio)
    Images.save(joinpath(OUTPUT_DIR, "recovered_classical_fft.png"),
                Gray.(clamp.(recovered_fft, 0.0, 1.0)))
    println("  recovered_classical_fft.png")

    # =========================================================================
    # Step 7: Print comparison tables
    # =========================================================================

    println("\n" * "="^90)
    println("                     PSNR COMPARISON (dB) - higher is better")
    println("="^90)
    @printf("%-22s", "Basis")
    for r in COMPRESSION_RATIOS
        @printf(" | %10s", "$(round(Int,(1-r)*100))%% kept")
    end
    println()
    println("-"^90)
    for name in basis_names
        @printf("%-22s", name)
        for r in COMPRESSION_RATIOS
            @printf(" | %10.2f", results[(name, r)].psnr)
        end
        println()
    end
    println("="^90)

    println("\n" * "="^90)
    println("                     SSIM COMPARISON - higher is better")
    println("="^90)
    @printf("%-22s", "Basis")
    for r in COMPRESSION_RATIOS
        @printf(" | %10s", "$(round(Int,(1-r)*100))%% kept")
    end
    println()
    println("-"^90)
    for name in basis_names
        @printf("%-22s", name)
        for r in COMPRESSION_RATIOS
            @printf(" | %10.4f", results[(name, r)].ssim)
        end
        println()
    end
    println("="^90)

    # =========================================================================
    # Step 8: Position-vs-position improvement analysis
    # =========================================================================

    println("\n" * "="^90)
    println("           POSITION-vs-POSITION IMPROVEMENT (vs Standard QFT)")
    println("="^90)
    @printf("%-22s", "Position")
    for r in COMPRESSION_RATIOS
        @printf(" | %10s", "$(round(Int,(1-r)*100))%% kept")
    end
    println()
    println("-"^90)

    for pos in POSITIONS
        name = "Entangled :$pos"
        @printf("%-22s", name)
        for r in COMPRESSION_RATIOS
            base_psnr = results[("Standard QFT", r)].psnr
            pos_psnr  = results[(name, r)].psnr
            delta     = pos_psnr - base_psnr
            @printf(" | %+9.2f dB", delta)
        end
        println()
    end
    println("="^90)

    # =========================================================================
    # Step 9: Summary
    # =========================================================================

    ref_ratio = 0.90
    best_name, best_psnr = "", -Inf
    for name in basis_names
        p = results[(name, ref_ratio)].psnr
        if p > best_psnr
            best_psnr = p; best_name = name
        end
    end

    println("\n" * "="^90)
    println("                              SUMMARY")
    println("="^90)
    println("""
    Best at 10%% kept: $best_name (PSNR $(round(best_psnr, digits=2)) dB)

    Learned entanglement phases (10 qubits each):
      :front  $(round.(get_entangle_phases(trained[:front]),  digits=4))
      :middle $(round.(get_entangle_phases(trained[:middle]), digits=4))
      :back   $(round.(get_entangle_phases(trained[:back]),   digits=4))

    Output files: $OUTPUT_DIR
    ├─ trained_entangled_*.json   : Trained bases
    ├─ original_*.png / recovered_*.png : Sample images
    └─ plots/                     : Training loss visualizations
       ├─ log/                    : Log-scale (y-axis) plots
       └─ linear/                 : Linear-scale (y-axis) plots
    """)
    println("="^90)

    return results
end

main()
