# ================================================================================
# QFT vs Entangled QFT: Comprehensive Comparison on MNIST
# ================================================================================
# This script provides a rigorous comparison between standard QFT basis and
# entangled QFT basis, training both on the same dataset and evaluating on
# multiple test images to demonstrate statistical significance.
#
# Run with: julia --project=examples examples/qft_vs_entangled_comparison.jl
# ================================================================================

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using ParametricDFT
using MLDatasets
using Images
using ImageQualityIndexes
using Random
using Statistics
using Printf

# ================================================================================
# Configuration
# ================================================================================

const M_QUBITS = 5  # 2^5 = 32 rows
const N_QUBITS = 5  # 2^5 = 32 columns  
const IMG_SIZE = 32

# Training configuration (balanced for fair comparison)
const NUM_TRAINING_IMAGES = 25      # Training images
const TRAINING_EPOCHS = 4           # More epochs for better convergence
const STEPS_PER_IMAGE = 60          # Gradient steps per image
const NUM_TEST_IMAGES = 10          # Number of test images for evaluation

# Compression ratios to test
const COMPRESSION_RATIOS = [0.95, 0.90, 0.85, 0.80]  # Keep 5%, 10%, 15%, 20%

# Output directory
const OUTPUT_DIR = joinpath(@__DIR__, "QFTComparison")

# ================================================================================
# Utility Functions
# ================================================================================

function pad_mnist_image(raw_img::AbstractMatrix)
    padded = zeros(Float64, IMG_SIZE, IMG_SIZE)
    padded[3:30, 3:30] = Float64.(raw_img)
    return padded
end

function compute_metrics(original::AbstractMatrix, recovered::AbstractMatrix)
    recovered_clamped = clamp.(real.(recovered), 0.0, 1.0)
    mse = mean((original .- recovered_clamped).^2)
    psnr = mse > 0 ? 10 * log10(1.0 / mse) : Inf
    ssim = assess_ssim(Gray.(original), Gray.(recovered_clamped))
    return (mse=mse, psnr=psnr, ssim=ssim)
end

function print_metrics_table(results::Dict, ratios::Vector{Float64})
    println("\n" * "="^90)
    println("                     COMPRESSION QUALITY COMPARISON")
    println("="^90)
    
    # Header
    @printf("%-30s", "Basis Type")
    for ratio in ratios
        @printf(" | %10s", "$(round(Int, (1-ratio)*100))% kept")
    end
    println()
    println("-"^90)
    
    # MSE
    println("\nMean Squared Error (MSE) - lower is better:")
    println("-"^90)
    for basis_name in ["Standard QFT", "Trained QFT", "Entangled QFT (untrained)", "Entangled QFT (trained)"]
        @printf("%-30s", basis_name)
        for ratio in ratios
            if haskey(results, (basis_name, ratio))
                mse = results[(basis_name, ratio)].mse
                @printf(" | %10.6f", mse)
            else
                @printf(" | %10s", "N/A")
            end
        end
        println()
    end
    
    # PSNR
    println("\nPeak Signal-to-Noise Ratio (PSNR) - higher is better:")
    println("-"^90)
    for basis_name in ["Standard QFT", "Trained QFT", "Entangled QFT (untrained)", "Entangled QFT (trained)"]
        @printf("%-30s", basis_name)
        for ratio in ratios
            if haskey(results, (basis_name, ratio))
                psnr = results[(basis_name, ratio)].psnr
                @printf(" | %10.2f", psnr)
            else
                @printf(" | %10s", "N/A")
            end
        end
        println()
    end
    
    # SSIM
    println("\nStructural Similarity (SSIM) - higher is better:")
    println("-"^90)
    for basis_name in ["Standard QFT", "Trained QFT", "Entangled QFT (untrained)", "Entangled QFT (trained)"]
        @printf("%-30s", basis_name)
        for ratio in ratios
            if haskey(results, (basis_name, ratio))
                ssim = results[(basis_name, ratio)].ssim
                @printf(" | %10.4f", ssim)
            else
                @printf(" | %10s", "N/A")
            end
        end
        println()
    end
    println("="^90)
end

# ================================================================================
# Main Comparison
# ================================================================================

function main()
    println("="^90)
    println("       QFT vs Entangled QFT: Comprehensive Comparison on MNIST")
    println("="^90)
    
    # Create output directory
    if !isdir(OUTPUT_DIR)
        mkpath(OUTPUT_DIR)
    end
    
    # ============================================================================
    # Step 1: Load Data
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 1: Loading MNIST Dataset")
    println("="^70)
    
    mnist_train = MNIST(split=:train)
    mnist_test = MNIST(split=:test)
    
    println("Dataset loaded:")
    println("  Training pool: $(size(mnist_train.features, 3)) images")
    println("  Test pool: $(size(mnist_test.features, 3)) images")
    
    # Prepare training and test sets
    Random.seed!(42)
    train_indices = randperm(size(mnist_train.features, 3))[1:NUM_TRAINING_IMAGES]
    test_indices = randperm(size(mnist_test.features, 3))[1:NUM_TEST_IMAGES]
    
    training_images = [pad_mnist_image(mnist_train.features[:, :, i]) for i in train_indices]
    test_images = [pad_mnist_image(mnist_test.features[:, :, i]) for i in test_indices]
    test_labels = [mnist_test.targets[i] for i in test_indices]
    
    println("\nPrepared:")
    println("  Training images: $(length(training_images))")
    println("  Test images: $(length(test_images))")
    
    # ============================================================================
    # Step 2: Create Untrained Bases
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 2: Creating Untrained Bases")
    println("="^70)
    
    standard_basis = QFTBasis(M_QUBITS, N_QUBITS)
    entangled_untrained = EntangledQFTBasis(M_QUBITS, N_QUBITS)
    
    println("  Standard QFT: $(num_parameters(standard_basis)) parameters")
    println("  Entangled QFT: $(num_parameters(entangled_untrained)) parameters")
    println("  Extra entanglement gates: $(entangled_untrained.n_entangle)")
    
    # ============================================================================
    # Step 3: Train Both Bases
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 3: Training Bases")
    println("="^70)
    
    total_coefficients = IMG_SIZE * IMG_SIZE
    k = round(Int, total_coefficients * 0.1)  # Keep 10% for training loss
    
    println("\nTraining configuration:")
    println("  Training images: $NUM_TRAINING_IMAGES")
    println("  Epochs: $TRAINING_EPOCHS")
    println("  Steps per image: $STEPS_PER_IMAGE")
    println("  Loss: MSELoss($k) - reconstruct from top $k coefficients")
    
    # Train standard QFT
    println("\n--- Training Standard QFT Basis ---")
    @time trained_qft = train_basis(
        QFTBasis, training_images;
        m=M_QUBITS, n=N_QUBITS,
        loss=ParametricDFT.MSELoss(k),
        epochs=TRAINING_EPOCHS,
        steps_per_image=STEPS_PER_IMAGE,
        validation_split=0.2,
        verbose=true
    )
    
    # Train entangled QFT
    println("\n--- Training Entangled QFT Basis ---")
    @time trained_entangled = train_basis(
        EntangledQFTBasis, training_images;
        m=M_QUBITS, n=N_QUBITS,
        loss=ParametricDFT.MSELoss(k),
        epochs=TRAINING_EPOCHS,
        steps_per_image=STEPS_PER_IMAGE,
        validation_split=0.2,
        verbose=true
    )
    
    println("\n✓ Training completed!")
    println("  Trained QFT parameters: $(num_parameters(trained_qft))")
    println("  Trained Entangled parameters: $(num_parameters(trained_entangled))")
    println("  Learned entanglement phases: $(round.(get_entangle_phases(trained_entangled), digits=4))")
    
    # ============================================================================
    # Step 4: Evaluate on Test Set
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 4: Evaluating on Test Set ($NUM_TEST_IMAGES images)")
    println("="^70)
    
    bases = Dict(
        "Standard QFT" => standard_basis,
        "Trained QFT" => trained_qft,
        "Entangled QFT (untrained)" => entangled_untrained,
        "Entangled QFT (trained)" => trained_entangled
    )
    
    # Collect results: (basis_name, compression_ratio) => (avg_mse, avg_psnr, avg_ssim)
    results = Dict{Tuple{String, Float64}, NamedTuple}()
    
    for ratio in COMPRESSION_RATIOS
        println("\n--- Compression ratio: $(round(Int, (1-ratio)*100))% kept ---")
        
        for (basis_name, basis) in bases
            mse_values = Float64[]
            psnr_values = Float64[]
            ssim_values = Float64[]
            
            for (i, test_img) in enumerate(test_images)
                compressed = compress(basis, test_img; ratio=ratio)
                recovered = recover(basis, compressed)
                metrics = compute_metrics(test_img, recovered)
                
                push!(mse_values, metrics.mse)
                push!(psnr_values, metrics.psnr)
                push!(ssim_values, metrics.ssim)
            end
            
            avg_metrics = (
                mse = mean(mse_values),
                psnr = mean(psnr_values),
                ssim = mean(ssim_values),
                mse_std = std(mse_values),
                psnr_std = std(psnr_values),
                ssim_std = std(ssim_values)
            )
            
            results[(basis_name, ratio)] = avg_metrics
            @printf("  %-30s MSE: %.6f ± %.6f  PSNR: %.2f ± %.2f  SSIM: %.4f ± %.4f\n",
                    basis_name, avg_metrics.mse, avg_metrics.mse_std,
                    avg_metrics.psnr, avg_metrics.psnr_std,
                    avg_metrics.ssim, avg_metrics.ssim_std)
        end
    end
    
    # ============================================================================
    # Step 5: Print Summary Table
    # ============================================================================
    
    print_metrics_table(results, COMPRESSION_RATIOS)
    
    # ============================================================================
    # Step 6: Analyze Improvement
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 5: Improvement Analysis")
    println("="^70)
    
    println("\nImprovement of Trained Entangled QFT over Standard QFT:")
    println("-"^70)
    
    for ratio in COMPRESSION_RATIOS
        std_metrics = results[("Standard QFT", ratio)]
        ent_metrics = results[("Entangled QFT (trained)", ratio)]
        
        mse_improvement = (std_metrics.mse - ent_metrics.mse) / std_metrics.mse * 100
        psnr_improvement = ent_metrics.psnr - std_metrics.psnr
        ssim_improvement = (ent_metrics.ssim - std_metrics.ssim) / std_metrics.ssim * 100
        
        kept_pct = round(Int, (1-ratio)*100)
        @printf("  %2d%% kept: MSE: %+.2f%%  PSNR: %+.2f dB  SSIM: %+.2f%%\n",
                kept_pct, mse_improvement, psnr_improvement, ssim_improvement)
    end
    
    println("\nImprovement of Trained Entangled QFT over Trained QFT:")
    println("-"^70)
    
    for ratio in COMPRESSION_RATIOS
        qft_metrics = results[("Trained QFT", ratio)]
        ent_metrics = results[("Entangled QFT (trained)", ratio)]
        
        mse_improvement = (qft_metrics.mse - ent_metrics.mse) / qft_metrics.mse * 100
        psnr_improvement = ent_metrics.psnr - qft_metrics.psnr
        ssim_improvement = (ent_metrics.ssim - qft_metrics.ssim) / qft_metrics.ssim * 100
        
        kept_pct = round(Int, (1-ratio)*100)
        @printf("  %2d%% kept: MSE: %+.2f%%  PSNR: %+.2f dB  SSIM: %+.2f%%\n",
                kept_pct, mse_improvement, psnr_improvement, ssim_improvement)
    end
    
    # ============================================================================
    # Step 7: Save Results and Best Basis
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 6: Saving Results")
    println("="^70)
    
    # Save the trained bases
    qft_path = joinpath(OUTPUT_DIR, "trained_qft.json")
    entangled_path = joinpath(OUTPUT_DIR, "trained_entangled.json")
    
    save_basis(qft_path, trained_qft)
    save_basis(entangled_path, trained_entangled)
    
    println("  Saved trained QFT: $qft_path")
    println("  Saved trained entangled: $entangled_path")
    
    # Save example images at 10% compression
    test_img = test_images[1]
    test_label = test_labels[1]
    ratio = 0.90
    
    original_path = joinpath(OUTPUT_DIR, "original_digit_$(test_label).png")
    Images.save(original_path, Gray.(test_img))
    
    for (basis_name, basis) in bases
        compressed = compress(basis, test_img; ratio=ratio)
        recovered = recover(basis, compressed)
        
        safe_name = replace(basis_name, " " => "_", "(" => "", ")" => "")
        img_path = joinpath(OUTPUT_DIR, "recovered_$(safe_name).png")
        Images.save(img_path, Gray.(clamp.(real.(recovered), 0.0, 1.0)))
    end
    
    println("  Saved comparison images to: $OUTPUT_DIR")
    
    # ============================================================================
    # Summary
    # ============================================================================
    
    println("\n" * "="^90)
    println("                              SUMMARY")
    println("="^90)
    
    # Find best performing basis at 10% kept
    ratio_10 = 0.90
    best_basis = ""
    best_psnr = -Inf
    for (basis_name, basis) in bases
        if haskey(results, (basis_name, ratio_10))
            if results[(basis_name, ratio_10)].psnr > best_psnr
                best_psnr = results[(basis_name, ratio_10)].psnr
                best_basis = basis_name
            end
        end
    end
    
    println("""
    
    Comparison completed on MNIST dataset:
    - Training images: $NUM_TRAINING_IMAGES
    - Test images: $NUM_TEST_IMAGES
    - Compression ratios tested: $(join(["$(round(Int, (1-r)*100))%" for r in COMPRESSION_RATIOS], ", "))
    
    Key findings at 10% coefficient retention:
    - Best performing basis: $best_basis (PSNR: $(round(best_psnr, digits=2)) dB)
    - Standard QFT PSNR: $(round(results[("Standard QFT", ratio_10)].psnr, digits=2)) dB
    - Trained QFT PSNR: $(round(results[("Trained QFT", ratio_10)].psnr, digits=2)) dB  
    - Trained Entangled QFT PSNR: $(round(results[("Entangled QFT (trained)", ratio_10)].psnr, digits=2)) dB
    
    Learned entanglement phases (radians):
    $(round.(get_entangle_phases(trained_entangled), digits=4))
    
    Output files saved to: $OUTPUT_DIR
    """)
    
    println("="^90)
    println("Comparison completed successfully!")
    println("="^90)
    
    return results, trained_qft, trained_entangled
end

# Run the comparison
results, trained_qft, trained_entangled = main()
