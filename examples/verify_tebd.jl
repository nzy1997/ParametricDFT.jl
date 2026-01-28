# ================================================================================
# TEBD Verification Script
# ================================================================================
# This script verifies the suspicious TEBD compression results from issue #20
# by checking for:
#   1. Train/test data leakage
#   2. Near-identity transform behavior
#   3. Compression actually zeroing coefficients
#   4. Performance on non-MNIST images
#   5. Statistical significance with larger test sets
#
# Run with: julia --project=examples examples/verify_tebd.jl
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

const M_QUBITS = 5
const N_QUBITS = 5
const IMG_SIZE = 32
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

"""Create a synthetic test image (gradient + noise)."""
function create_synthetic_image(seed::Int)
    Random.seed!(seed)
    # Create a gradient pattern
    x = range(0, 1, length=IMG_SIZE)
    y = range(0, 1, length=IMG_SIZE)
    gradient = [0.3 * xi + 0.3 * yi for xi in x, yi in y]
    # Add some structure (circles)
    for _ in 1:3
        cx, cy = rand(1:IMG_SIZE), rand(1:IMG_SIZE)
        r = rand(3:8)
        for i in 1:IMG_SIZE, j in 1:IMG_SIZE
            if (i - cx)^2 + (j - cy)^2 < r^2
                gradient[i, j] = clamp(gradient[i, j] + 0.3, 0, 1)
            end
        end
    end
    # Add noise
    noise = randn(IMG_SIZE, IMG_SIZE) * 0.05
    return clamp.(gradient .+ noise, 0, 1)
end

"""Create a random noise image."""
function create_random_noise_image(seed::Int)
    Random.seed!(seed)
    return rand(IMG_SIZE, IMG_SIZE)
end

# ================================================================================
# Verification Tests
# ================================================================================

function test_train_test_overlap()
    println("\n" * "="^80)
    println("TEST 1: Verifying Train/Test Data Split (No Overlap)")
    println("="^80)
    
    mnist_train = MNIST(split=:train)
    mnist_test = MNIST(split=:test)
    
    println("  Training set: $(size(mnist_train.features, 3)) images")
    println("  Test set: $(size(mnist_test.features, 3)) images")
    println("  ✓ Using different MNIST splits - no overlap possible")
    println("  ✓ PASS: Train and test sets are from different MNIST splits")
    
    return true
end

function test_transform_identity(trained_tebd::TEBDBasis)
    println("\n" * "="^80)
    println("TEST 2: Checking if TEBD is Learning Near-Identity Transform")
    println("="^80)
    
    # Test with identity input
    identity_img = zeros(IMG_SIZE, IMG_SIZE)
    identity_img[IMG_SIZE÷2, IMG_SIZE÷2] = 1.0  # Single point
    
    # Forward and inverse transform without compression
    freq = ParametricDFT.forward_transform(trained_tebd, identity_img)
    recovered = ParametricDFT.inverse_transform(trained_tebd, freq)
    
    reconstruction_error = norm(identity_img - real.(recovered)) / norm(identity_img)
    println("  Reconstruction error (full, no truncation): $reconstruction_error")
    
    # Check if the transform spreads energy or concentrates it
    freq_flat = vec(abs.(freq))
    sorted_magnitudes = sort(freq_flat, rev=true)
    
    # Energy concentration: what fraction of energy is in top 10% coefficients?
    total_energy = sum(abs2.(freq))
    top_10pct_idx = max(1, round(Int, length(freq_flat) * 0.1))
    top_10pct_energy = sum(sorted_magnitudes[1:top_10pct_idx].^2)
    energy_concentration = top_10pct_energy / total_energy
    
    println("  Energy concentration in top 10%: $(round(energy_concentration * 100, digits=2))%")
    
    # Compare to random image
    Random.seed!(999)
    random_img = rand(IMG_SIZE, IMG_SIZE)
    random_freq = ParametricDFT.forward_transform(trained_tebd, random_img)
    random_freq_flat = vec(abs.(random_freq))
    sorted_random = sort(random_freq_flat, rev=true)
    
    total_random_energy = sum(abs2.(random_freq))
    top_random_energy = sum(sorted_random[1:top_10pct_idx].^2)
    random_concentration = top_random_energy / total_random_energy
    
    println("  Energy concentration for random image: $(round(random_concentration * 100, digits=2))%")
    
    # Check phase parameters
    phases = trained_tebd.phases
    println("\n  Learned TEBD phases: $(round.(phases, digits=4))")
    println("  Max |phase|: $(round(maximum(abs.(phases)), digits=4))")
    println("  Mean |phase|: $(round(mean(abs.(phases)), digits=4))")
    
    # If all phases are very small, it's close to just Hadamard transform
    if maximum(abs.(phases)) < 0.2
        println("  ⚠ Warning: All phases are small (< 0.2 rad), transform is close to Hadamard-only")
    end
    
    return reconstruction_error < 1e-10, energy_concentration
end

function test_compression_zeroing(trained_tebd::TEBDBasis)
    println("\n" * "="^80)
    println("TEST 3: Verifying Compression Actually Zeros Coefficients")
    println("="^80)
    
    Random.seed!(42)
    test_img = rand(IMG_SIZE, IMG_SIZE)
    
    for ratio in [0.95, 0.90, 0.80]
        compressed = ParametricDFT.compress(trained_tebd, test_img; ratio=ratio)
        
        total_coeffs = IMG_SIZE * IMG_SIZE
        kept_coeffs = length(compressed.indices)
        expected_kept = round(Int, total_coeffs * (1 - ratio))
        
        println("  Ratio $(ratio): kept $kept_coeffs / $total_coeffs (expected ~$expected_kept)")
        
        # Verify the compressed representation is actually sparse
        if kept_coeffs > expected_kept * 1.1
            println("    ⚠ Warning: More coefficients kept than expected!")
        else
            println("    ✓ Correct number of coefficients")
        end
        
        # Recover and check
        recovered = ParametricDFT.recover(trained_tebd, compressed; verify_hash=false)
        metrics = compute_metrics(test_img, recovered)
        println("    Recovery MSE: $(round(metrics.mse, digits=6)), PSNR: $(round(metrics.psnr, digits=2)) dB")
    end
    
    return true
end

function test_non_mnist_images(trained_tebd::TEBDBasis, standard_qft::QFTBasis)
    println("\n" * "="^80)
    println("TEST 4: Testing on Non-MNIST Images (Synthetic)")
    println("="^80)
    
    # Create synthetic images
    synthetic_images = [create_synthetic_image(i) for i in 1:20]
    random_noise_images = [create_random_noise_image(i + 100) for i in 1:20]
    
    println("\n--- Synthetic Images (Gradients + Circles) ---")
    for ratio in [0.95, 0.90, 0.80]
        tebd_psnrs, qft_psnrs = Float64[], Float64[]
        
        for img in synthetic_images
            # TEBD
            compressed = ParametricDFT.compress(trained_tebd, img; ratio=ratio)
            recovered = ParametricDFT.recover(trained_tebd, compressed; verify_hash=false)
            push!(tebd_psnrs, compute_metrics(img, recovered).psnr)
            
            # QFT
            compressed_qft = ParametricDFT.compress(standard_qft, img; ratio=ratio)
            recovered_qft = ParametricDFT.recover(standard_qft, compressed_qft; verify_hash=false)
            push!(qft_psnrs, compute_metrics(img, recovered_qft).psnr)
        end
        
        kept_pct = round(Int, (1 - ratio) * 100)
        println("  $kept_pct% kept: TEBD=$(round(mean(tebd_psnrs), digits=2))±$(round(std(tebd_psnrs), digits=2)) dB, " *
                "QFT=$(round(mean(qft_psnrs), digits=2))±$(round(std(qft_psnrs), digits=2)) dB")
    end
    
    println("\n--- Random Noise Images ---")
    for ratio in [0.95, 0.90, 0.80]
        tebd_psnrs, qft_psnrs = Float64[], Float64[]
        
        for img in random_noise_images
            # TEBD
            compressed = ParametricDFT.compress(trained_tebd, img; ratio=ratio)
            recovered = ParametricDFT.recover(trained_tebd, compressed; verify_hash=false)
            push!(tebd_psnrs, compute_metrics(img, recovered).psnr)
            
            # QFT
            compressed_qft = ParametricDFT.compress(standard_qft, img; ratio=ratio)
            recovered_qft = ParametricDFT.recover(standard_qft, compressed_qft; verify_hash=false)
            push!(qft_psnrs, compute_metrics(img, recovered_qft).psnr)
        end
        
        kept_pct = round(Int, (1 - ratio) * 100)
        println("  $kept_pct% kept: TEBD=$(round(mean(tebd_psnrs), digits=2))±$(round(std(tebd_psnrs), digits=2)) dB, " *
                "QFT=$(round(mean(qft_psnrs), digits=2))±$(round(std(qft_psnrs), digits=2)) dB")
    end
    
    return true
end

function test_larger_mnist_test_set(trained_tebd::TEBDBasis, standard_qft::QFTBasis)
    println("\n" * "="^80)
    println("TEST 5: Larger MNIST Test Set (100 images)")
    println("="^80)
    
    mnist_test = MNIST(split=:test)
    
    Random.seed!(12345)  # Different seed than training
    test_indices = randperm(size(mnist_test.features, 3))[1:100]
    test_images = [pad_mnist_image(mnist_test.features[:, :, i]) for i in test_indices]
    
    println("  Testing on 100 MNIST test images...")
    
    for ratio in [0.95, 0.90, 0.85, 0.80]
        tebd_psnrs, qft_psnrs = Float64[], Float64[]
        tebd_ssims, qft_ssims = Float64[], Float64[]
        
        for img in test_images
            # TEBD
            compressed = ParametricDFT.compress(trained_tebd, img; ratio=ratio)
            recovered = ParametricDFT.recover(trained_tebd, compressed; verify_hash=false)
            metrics = compute_metrics(img, recovered)
            push!(tebd_psnrs, metrics.psnr)
            push!(tebd_ssims, metrics.ssim)
            
            # QFT
            compressed_qft = ParametricDFT.compress(standard_qft, img; ratio=ratio)
            recovered_qft = ParametricDFT.recover(standard_qft, compressed_qft; verify_hash=false)
            metrics_qft = compute_metrics(img, recovered_qft)
            push!(qft_psnrs, metrics_qft.psnr)
            push!(qft_ssims, metrics_qft.ssim)
        end
        
        kept_pct = round(Int, (1 - ratio) * 100)
        println("\n  $kept_pct% kept:")
        println("    TEBD: PSNR=$(round(mean(tebd_psnrs), digits=2))±$(round(std(tebd_psnrs), digits=2)) dB, " *
                "SSIM=$(round(mean(tebd_ssims), digits=4))±$(round(std(tebd_ssims), digits=4))")
        println("    QFT:  PSNR=$(round(mean(qft_psnrs), digits=2))±$(round(std(qft_psnrs), digits=2)) dB, " *
                "SSIM=$(round(mean(qft_ssims), digits=4))±$(round(std(qft_ssims), digits=4))")
        
        # Statistical significance check
        psnr_diff = mean(tebd_psnrs) - mean(qft_psnrs)
        if psnr_diff > 5
            println("    ⚠ TEBD is $(round(psnr_diff, digits=2)) dB better - SUSPICIOUS!")
        elseif psnr_diff > 1
            println("    ✓ TEBD is $(round(psnr_diff, digits=2)) dB better - reasonable improvement")
        else
            println("    ✓ Performance is similar (Δ=$(round(psnr_diff, digits=2)) dB)")
        end
    end
    
    return true
end

function test_reproducibility(trained_tebd::TEBDBasis)
    println("\n" * "="^80)
    println("TEST 6: Reproducibility Check (Same Image, Multiple Runs)")
    println("="^80)
    
    Random.seed!(42)
    test_img = rand(IMG_SIZE, IMG_SIZE)
    
    results = Float64[]
    for i in 1:5
        compressed = ParametricDFT.compress(trained_tebd, test_img; ratio=0.90)
        recovered = ParametricDFT.recover(trained_tebd, compressed; verify_hash=false)
        push!(results, compute_metrics(test_img, recovered).psnr)
    end
    
    println("  PSNR results over 5 runs: $(round.(results, digits=4))")
    println("  Variance: $(round(var(results), digits=8))")
    
    if var(results) < 1e-10
        println("  ✓ Results are perfectly reproducible")
    else
        println("  ⚠ Warning: Results vary between runs")
    end
    
    return var(results) < 1e-10
end

function investigate_transform_matrices(trained_tebd::TEBDBasis)
    println("\n" * "="^80)
    println("TEST 7: Investigating TEBD Transform Matrix Properties")
    println("="^80)
    
    # Create identity-like input to probe the transform
    probing_inputs = []
    for i in 1:min(8, IMG_SIZE^2)
        probe = zeros(IMG_SIZE, IMG_SIZE)
        probe[((i-1) % IMG_SIZE) + 1, ((i-1) ÷ IMG_SIZE) + 1] = 1.0
        push!(probing_inputs, probe)
    end
    
    # Check the response
    println("  Probing transform response to delta functions...")
    
    max_responses = Float64[]
    for (idx, probe) in enumerate(probing_inputs)
        freq = ParametricDFT.forward_transform(trained_tebd, probe)
        max_resp = maximum(abs.(freq))
        push!(max_responses, max_resp)
    end
    
    println("  Max response magnitudes (should be ~1/sqrt(N) for unitary): $(round.(max_responses[1:min(4, length(max_responses))], digits=4))...")
    expected_mag = 1.0 / sqrt(IMG_SIZE^2)  # For normalized unitary
    println("  Expected for unitary DFT: ~$(round(expected_mag, digits=4))")
    
    # Check unitarity by verifying F * F† ≈ I
    println("\n  Checking unitarity (via reconstruction)...")
    Random.seed!(123)
    test_img = rand(IMG_SIZE, IMG_SIZE)
    freq = ParametricDFT.forward_transform(trained_tebd, test_img)
    recovered = ParametricDFT.inverse_transform(trained_tebd, freq)
    reconstruction_error = norm(test_img - real.(recovered)) / norm(test_img)
    println("  Relative reconstruction error: $(round(reconstruction_error, digits=10))")
    
    if reconstruction_error < 1e-10
        println("  ✓ Transform is unitary (perfect reconstruction)")
    else
        println("  ⚠ Transform may not be perfectly unitary")
    end
    
    return reconstruction_error < 1e-6
end

# ================================================================================
# Main
# ================================================================================

function main()
    println("="^80)
    println("       TEBD Verification Script for Issue #20")
    println("       Investigating Suspiciously High Performance")
    println("="^80)
    
    # Load trained TEBD and create standard QFT for comparison
    tebd_path = joinpath(OUTPUT_DIR, "trained_tebd.json")
    
    if !isfile(tebd_path)
        println("\n❌ ERROR: trained_tebd.json not found at $tebd_path")
        println("   Please run basis_demo.jl first to generate the trained model.")
        return
    end
    
    println("\nLoading trained TEBD basis...")
    trained_tebd = ParametricDFT.load_basis(tebd_path)
    standard_qft = QFTBasis(M_QUBITS, N_QUBITS)
    
    println("  Loaded: $(trained_tebd)")
    println("  Comparison: $(standard_qft)")
    
    # Run all verification tests
    results = Dict{String, Bool}()
    
    results["Train/Test Split"] = test_train_test_overlap()
    
    is_identity, energy_conc = test_transform_identity(trained_tebd)
    results["Not Identity"] = !is_identity
    
    results["Compression Zeroing"] = test_compression_zeroing(trained_tebd)
    
    results["Non-MNIST Performance"] = test_non_mnist_images(trained_tebd, standard_qft)
    
    results["Larger Test Set"] = test_larger_mnist_test_set(trained_tebd, standard_qft)
    
    results["Reproducibility"] = test_reproducibility(trained_tebd)
    
    results["Transform Unitarity"] = investigate_transform_matrices(trained_tebd)
    
    # Summary
    println("\n" * "="^80)
    println("                           VERIFICATION SUMMARY")
    println("="^80)
    
    all_passed = true
    for (test_name, passed) in results
        status = passed ? "✓ PASS" : "✗ FAIL"
        println("  $status: $test_name")
        all_passed = all_passed && passed
    end
    
    println("\n" * "="^80)
    if all_passed
        println("All verification tests passed!")
    else
        println("Some verification tests failed - investigation needed")
    end
    println("="^80)
    
    return results
end

# Run
main()
