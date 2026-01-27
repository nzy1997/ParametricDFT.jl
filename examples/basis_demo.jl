# ================================================================================
# Unified Basis Demo: QFT, Entangled QFT, and TEBD Comparison
# ================================================================================
# This example demonstrates all basis types in ParametricDFT:
#   1. Standard QFT Basis (untrained and trained)
#   2. Entangled QFT Basis (untrained and trained)
#   3. TEBD Basis (untrained and trained)
#
# The demo trains all bases on MNIST, compares compression quality, and outputs
# a comprehensive comparison table.
#
# Run with: julia --project=examples examples/basis_demo.jl
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

const M_QUBITS = 5           # 2^5 = 32 rows
const N_QUBITS = 5           # 2^5 = 32 columns
const IMG_SIZE = 32

# Training configuration
const NUM_TRAINING_IMAGES = 20
const TRAINING_EPOCHS = 2
const STEPS_PER_IMAGE = 50
const NUM_TEST_IMAGES = 5

# Compression ratios to test
const COMPRESSION_RATIOS = [0.95, 0.90, 0.85, 0.80]  # Keep 5%, 10%, 15%, 20%

# Output directory
const OUTPUT_DIR = joinpath(@__DIR__, "BasisDemo")

# ================================================================================
# Utility Functions
# ================================================================================

"""Pad 28×28 MNIST image to 32×32."""
function pad_mnist_image(raw_img::AbstractMatrix)
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

"""Print a formatted comparison table."""
function print_comparison_table(results::Dict, ratios::Vector{Float64}, basis_names::Vector{String})
    println("\n" * "="^100)
    println("                           COMPRESSION QUALITY COMPARISON")
    println("="^100)
    
    # Header
    @printf("%-35s", "Basis Type")
    for ratio in ratios
        @printf(" | %12s", "$(round(Int, (1-ratio)*100))% kept")
    end
    println()
    println("-"^100)
    
    # MSE Section
    println("\n📊 Mean Squared Error (MSE) - lower is better:")
    println("-"^100)
    for basis_name in basis_names
        @printf("%-35s", basis_name)
        for ratio in ratios
            if haskey(results, (basis_name, ratio))
                mse = results[(basis_name, ratio)].mse
                @printf(" | %12.6f", mse)
            else
                @printf(" | %12s", "N/A")
            end
        end
        println()
    end
    
    # PSNR Section
    println("\n📊 Peak Signal-to-Noise Ratio (PSNR dB) - higher is better:")
    println("-"^100)
    for basis_name in basis_names
        @printf("%-35s", basis_name)
        for ratio in ratios
            if haskey(results, (basis_name, ratio))
                psnr = results[(basis_name, ratio)].psnr
                @printf(" | %12.2f", psnr)
            else
                @printf(" | %12s", "N/A")
            end
        end
        println()
    end
    
    # SSIM Section
    println("\n📊 Structural Similarity (SSIM) - higher is better:")
    println("-"^100)
    for basis_name in basis_names
        @printf("%-35s", basis_name)
        for ratio in ratios
            if haskey(results, (basis_name, ratio))
                ssim = results[(basis_name, ratio)].ssim
                @printf(" | %12.4f", ssim)
            else
                @printf(" | %12s", "N/A")
            end
        end
        println()
    end
    println("="^100)
end

"""Print improvement analysis."""
function print_improvement_analysis(results::Dict, ratios::Vector{Float64})
    println("\n" * "="^100)
    println("                           IMPROVEMENT ANALYSIS")
    println("="^100)
    
    comparisons = [
        ("Trained QFT", "Standard QFT"),
        ("Trained Entangled QFT", "Standard QFT"),
        ("Trained TEBD", "Standard QFT"),
        ("Trained Entangled QFT", "Trained QFT"),
        ("Trained TEBD", "Trained QFT"),
    ]
    
    for (better, baseline) in comparisons
        println("\n📈 $better vs $baseline:")
        println("-"^80)
        @printf("%-15s | %15s | %15s | %15s\n", "Kept %", "MSE Δ%", "PSNR Δ (dB)", "SSIM Δ%")
        println("-"^80)
        
        for ratio in ratios
            if haskey(results, (better, ratio)) && haskey(results, (baseline, ratio))
                b_metrics = results[(better, ratio)]
                base_metrics = results[(baseline, ratio)]
                
                mse_imp = (base_metrics.mse - b_metrics.mse) / base_metrics.mse * 100
                psnr_imp = b_metrics.psnr - base_metrics.psnr
                ssim_imp = (b_metrics.ssim - base_metrics.ssim) / base_metrics.ssim * 100
                
                kept_pct = round(Int, (1-ratio)*100)
                @printf("%12d%% | %+14.2f%% | %+14.2f | %+14.2f%%\n",
                        kept_pct, mse_imp, psnr_imp, ssim_imp)
            end
        end
    end
    println("="^100)
end

# ================================================================================
# Main Demo
# ================================================================================

function main()
    println("="^100)
    println("       Unified Basis Demo: QFT, Entangled QFT, and TEBD Comparison")
    println("="^100)
    
    # Create output directory
    if !isdir(OUTPUT_DIR)
        mkpath(OUTPUT_DIR)
        println("\n📁 Created output directory: $OUTPUT_DIR")
    end
    
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
    
    # Prepare datasets
    Random.seed!(42)
    train_indices = randperm(size(mnist_train.features, 3))[1:NUM_TRAINING_IMAGES]
    test_indices = randperm(size(mnist_test.features, 3))[1:NUM_TEST_IMAGES]
    
    training_images = [pad_mnist_image(mnist_train.features[:, :, i]) for i in train_indices]
    test_images = [pad_mnist_image(mnist_test.features[:, :, i]) for i in test_indices]
    test_labels = [mnist_test.targets[i] for i in test_indices]
    
    println("  Prepared $(length(training_images)) training images")
    println("  Prepared $(length(test_images)) test images")
    println("  Image size: $(IMG_SIZE)×$(IMG_SIZE) (padded from 28×28)")
    
    # ============================================================================
    # Step 2: Create Untrained Bases
    # ============================================================================
    
    println("\n" * "="^80)
    println("Step 2: Creating Untrained Bases")
    println("="^80)
    
    standard_qft = QFTBasis(M_QUBITS, N_QUBITS)
    entangled_qft = EntangledQFTBasis(M_QUBITS, N_QUBITS)
    tebd_default = TEBDBasis(M_QUBITS, N_QUBITS)
    
    println("  Standard QFT:     $(num_parameters(standard_qft)) parameters")
    println("  Entangled QFT:    $(num_parameters(entangled_qft)) parameters ($(entangled_qft.n_entangle) entangle gates)")
    println("  TEBD:             $(num_parameters(tebd_default)) parameters ($(num_gates(tebd_default)) ring gates)")
    
    # ============================================================================
    # Step 3: Train All Bases
    # ============================================================================
    
    println("\n" * "="^80)
    println("Step 3: Training All Bases")
    println("="^80)
    
    total_coefficients = IMG_SIZE * IMG_SIZE
    k = round(Int, total_coefficients * 0.1)  # Keep 10% for MSE loss
    
    println("\nTraining configuration:")
    println("  Images: $NUM_TRAINING_IMAGES | Epochs: $TRAINING_EPOCHS | Steps/image: $STEPS_PER_IMAGE")
    println("  Loss: MSELoss($k) - reconstruct from top $k/$total_coefficients coefficients")
    
    # Train QFT
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
    
    # Train Entangled QFT
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
    
    # Train TEBD
    println("\n--- Training TEBD Basis ---")
    @time trained_tebd = train_basis(
        TEBDBasis, training_images;
        m=M_QUBITS, n=N_QUBITS,
        loss=ParametricDFT.MSELoss(k),
        epochs=TRAINING_EPOCHS,
        steps_per_image=STEPS_PER_IMAGE,
        validation_split=0.2,
        verbose=true
    )
    
    println("\n✓ All training completed!")
    println("  Trained QFT:          $(num_parameters(trained_qft)) parameters")
    println("  Trained Entangled:    $(num_parameters(trained_entangled)) parameters")
    println("  Trained TEBD:         $(num_parameters(trained_tebd)) parameters")
    println("  Entanglement phases:  $(round.(get_entangle_phases(trained_entangled), digits=4))")
    
    # ============================================================================
    # Step 4: Save Trained Bases
    # ============================================================================
    
    println("\n" * "="^80)
    println("Step 4: Saving Trained Bases")
    println("="^80)
    
    qft_path = joinpath(OUTPUT_DIR, "trained_qft.json")
    entangled_path = joinpath(OUTPUT_DIR, "trained_entangled_qft.json")
    tebd_path = joinpath(OUTPUT_DIR, "trained_tebd.json")
    
    save_basis(qft_path, trained_qft)
    save_basis(entangled_path, trained_entangled)
    save_basis(tebd_path, trained_tebd)
    
    println("  ✓ Saved: trained_qft.json ($(round(filesize(qft_path)/1024, digits=2)) KB)")
    println("  ✓ Saved: trained_entangled_qft.json ($(round(filesize(entangled_path)/1024, digits=2)) KB)")
    println("  ✓ Saved: trained_tebd.json ($(round(filesize(tebd_path)/1024, digits=2)) KB)")
    
    # ============================================================================
    # Step 5: Evaluate All Bases on Test Set
    # ============================================================================
    
    println("\n" * "="^80)
    println("Step 5: Evaluating on Test Set ($NUM_TEST_IMAGES images)")
    println("="^80)
    
    bases = Dict(
        "Standard QFT" => standard_qft,
        "Trained QFT" => trained_qft,
        "Standard Entangled QFT" => entangled_qft,
        "Trained Entangled QFT" => trained_entangled,
        "Standard TEBD" => tebd_default,
        "Trained TEBD" => trained_tebd,
    )
    
    basis_names = [
        "Standard QFT",
        "Trained QFT", 
        "Standard Entangled QFT",
        "Trained Entangled QFT",
        "Standard TEBD",
        "Trained TEBD",
        "Classical FFT"
    ]
    
    results = Dict{Tuple{String, Float64}, NamedTuple}()
    
    for ratio in COMPRESSION_RATIOS
        println("\n--- Compression ratio: $(round(Int, (1-ratio)*100))% kept ---")
        
        # Evaluate each basis
        for (basis_name, basis) in bases
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
            @printf("  %-25s MSE: %.6f  PSNR: %.2f dB  SSIM: %.4f\n",
                    basis_name, mean(mse_vals), mean(psnr_vals), mean(ssim_vals))
        end
        
        # Evaluate classical FFT
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
        @printf("  %-25s MSE: %.6f  PSNR: %.2f dB  SSIM: %.4f\n",
                "Classical FFT", mean(mse_vals), mean(psnr_vals), mean(ssim_vals))
    end
    
    # ============================================================================
    # Step 6: Save Sample Images
    # ============================================================================
    
    println("\n" * "="^80)
    println("Step 6: Saving Sample Images")
    println("="^80)
    
    # Use first test image as sample
    sample_img = test_images[1]
    sample_label = test_labels[1]
    sample_ratio = 0.90  # 10% kept
    
    # Save original
    original_path = joinpath(OUTPUT_DIR, "original_digit_$(sample_label).png")
    Images.save(original_path, Gray.(sample_img))
    println("  ✓ original_digit_$(sample_label).png")
    
    # Save recovered images for each basis
    for (basis_name, basis) in bases
        compressed = compress(basis, sample_img; ratio=sample_ratio)
        recovered = recover(basis, compressed)
        
        safe_name = lowercase(replace(basis_name, " " => "_"))
        img_path = joinpath(OUTPUT_DIR, "recovered_$(safe_name).png")
        Images.save(img_path, Gray.(clamp.(real.(recovered), 0.0, 1.0)))
        println("  ✓ recovered_$(safe_name).png")
    end
    
    # Save FFT recovered
    recovered_fft = fft_compress(sample_img, sample_ratio)
    fft_path = joinpath(OUTPUT_DIR, "recovered_classical_fft.png")
    Images.save(fft_path, Gray.(clamp.(recovered_fft, 0.0, 1.0)))
    println("  ✓ recovered_classical_fft.png")
    
    # ============================================================================
    # Step 7: Print Comparison Tables
    # ============================================================================
    
    print_comparison_table(results, COMPRESSION_RATIOS, basis_names)
    print_improvement_analysis(results, COMPRESSION_RATIOS)
    
    # ============================================================================
    # Summary
    # ============================================================================
    
    println("\n" * "="^100)
    println("                                    SUMMARY")
    println("="^100)
    
    # Find best basis at 10% kept
    ratio_10 = 0.90
    best_basis, best_psnr = "", -Inf
    for name in basis_names
        if haskey(results, (name, ratio_10)) && results[(name, ratio_10)].psnr > best_psnr
            best_psnr = results[(name, ratio_10)].psnr
            best_basis = name
        end
    end
    
    summary_text = """
    
    ✅ Comparison completed on MNIST dataset
    
    Configuration:
    ├─ Training images:     $NUM_TRAINING_IMAGES
    ├─ Test images:         $NUM_TEST_IMAGES  
    ├─ Image size:          $(IMG_SIZE)×$(IMG_SIZE)
    ├─ Compression ratios:  $(join(["$(round(Int, (1-r)*100))%" for r in COMPRESSION_RATIOS], ", "))
    └─ Training epochs:     $TRAINING_EPOCHS
    
    Basis Architectures:
    ├─ QFT:          $(M_QUBITS) + $(N_QUBITS) qubits (2D separable)
    ├─ Entangled:    $(M_QUBITS) + $(N_QUBITS) qubits + $(min(M_QUBITS, N_QUBITS)) entangle gates
    └─ TEBD:         $(M_QUBITS) + $(N_QUBITS) qubits (2D ring: $(M_QUBITS) row + $(N_QUBITS) col gates)
    
    🏆 Best at 10% kept: $best_basis (PSNR: $(round(best_psnr, digits=2)) dB)
    
    Results at 10% coefficient retention:
    ├─ Standard QFT:           PSNR $(round(results[("Standard QFT", ratio_10)].psnr, digits=2)) dB
    ├─ Trained QFT:            PSNR $(round(results[("Trained QFT", ratio_10)].psnr, digits=2)) dB
    ├─ Standard Entangled QFT: PSNR $(round(results[("Standard Entangled QFT", ratio_10)].psnr, digits=2)) dB
    ├─ Trained Entangled QFT:  PSNR $(round(results[("Trained Entangled QFT", ratio_10)].psnr, digits=2)) dB
    ├─ Standard TEBD:          PSNR $(round(results[("Standard TEBD", ratio_10)].psnr, digits=2)) dB
    ├─ Trained TEBD:           PSNR $(round(results[("Trained TEBD", ratio_10)].psnr, digits=2)) dB
    └─ Classical FFT:          PSNR $(round(results[("Classical FFT", ratio_10)].psnr, digits=2)) dB
    
    Learned Parameters:
    ├─ Entanglement phases: $(round.(get_entangle_phases(trained_entangled), digits=4))
    └─ TEBD phases:         $(round.(trained_tebd.phases, digits=4))
    
    Output files saved to: $OUTPUT_DIR
    ├─ trained_qft.json              : Trained QFT basis
    ├─ trained_entangled_qft.json    : Trained Entangled QFT basis  
    ├─ trained_tebd.json             : Trained TEBD basis
    ├─ original_digit_$(sample_label).png          : Original test image
    └─ recovered_*.png               : Recovered images for each basis
    """
    
    println(summary_text)
    
    # ============================================================================
    # Step 10: Write Summary to Markdown File
    # ============================================================================
    
    summary_path = joinpath(OUTPUT_DIR, "summary.md")
    
    # Build markdown content
    md_content = """
# Basis Comparison Summary

## Configuration

| Parameter | Value |
|-----------|-------|
| Training images | $NUM_TRAINING_IMAGES |
| Test images | $NUM_TEST_IMAGES |
| Image size | $(IMG_SIZE)×$(IMG_SIZE) |
| Compression ratios | $(join(["$(round(Int, (1-r)*100))%" for r in COMPRESSION_RATIOS], ", ")) |
| Training epochs | $TRAINING_EPOCHS |

## Basis Architectures

| Basis | Architecture |
|-------|-------------|
| QFT | $(M_QUBITS) + $(N_QUBITS) qubits (2D separable) |
| Entangled QFT | $(M_QUBITS) + $(N_QUBITS) qubits + $(min(M_QUBITS, N_QUBITS)) entangle gates |
| TEBD | $(M_QUBITS) + $(N_QUBITS) qubits (2D ring: $(M_QUBITS) row + $(N_QUBITS) col gates) |

## Results

### 🏆 Best at 10% kept: **$best_basis** (PSNR: $(round(best_psnr, digits=2)) dB)

### Compression Quality Comparison (PSNR in dB)

| Basis | $(join(["$(round(Int, (1-r)*100))% kept" for r in COMPRESSION_RATIOS], " | ")) |
|-------|$(join(["------" for _ in COMPRESSION_RATIOS], "|"))|
$(join([
    "| $name | " * join([@sprintf("%.2f", haskey(results, (name, r)) ? results[(name, r)].psnr : NaN) for r in COMPRESSION_RATIOS], " | ") * " |"
    for name in basis_names
], "\n"))

### Compression Quality Comparison (SSIM)

| Basis | $(join(["$(round(Int, (1-r)*100))% kept" for r in COMPRESSION_RATIOS], " | ")) |
|-------|$(join(["------" for _ in COMPRESSION_RATIOS], "|"))|
$(join([
    "| $name | " * join([@sprintf("%.4f", haskey(results, (name, r)) ? results[(name, r)].ssim : NaN) for r in COMPRESSION_RATIOS], " | ") * " |"
    for name in basis_names
], "\n"))

### Compression Quality Comparison (MSE)

| Basis | $(join(["$(round(Int, (1-r)*100))% kept" for r in COMPRESSION_RATIOS], " | ")) |
|-------|$(join(["------" for _ in COMPRESSION_RATIOS], "|"))|
$(join([
    "| $name | " * join([@sprintf("%.6f", haskey(results, (name, r)) ? results[(name, r)].mse : NaN) for r in COMPRESSION_RATIOS], " | ") * " |"
    for name in basis_names
], "\n"))

## Learned Parameters

### Entanglement Phases
```
$(round.(get_entangle_phases(trained_entangled), digits=4))
```

### TEBD Phases
```
$(round.(trained_tebd.phases, digits=4))
```

## Output Files

- `trained_qft.json` - Trained QFT basis
- `trained_entangled_qft.json` - Trained Entangled QFT basis
- `trained_tebd.json` - Trained TEBD basis
- `original_digit_$(sample_label).png` - Original test image
- `recovered_*.png` - Recovered images for each basis
"""
    
    open(summary_path, "w") do io
        write(io, md_content)
    end
    println("  ✓ Summary written to: $summary_path")
    
    println("="^100)
    println("Demo completed successfully!")
    println("="^100)
    
    return results
end

# Run the demo
main()
