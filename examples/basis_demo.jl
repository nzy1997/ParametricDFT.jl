# ================================================================================
# Unified Basis Demo: QFT, Entangled QFT, and TEBD Comparison
# ================================================================================
# This example demonstrates all basis types in ParametricDFT:
#   1. Standard QFT Basis (untrained and trained)
#   2. Entangled QFT Basis (untrained and trained)
#   3. TEBD Basis (untrained and trained)
#
# The demo trains all bases on a configurable dataset, compares compression 
# quality, and outputs a comprehensive comparison table.
#
# Run with:
#   julia --project=examples examples/basis_demo.jl              # Default MNIST (auto-detects GPU)
#   julia --project=examples examples/basis_demo.jl quickdraw    # Quick Draw (auto-detects GPU)
#   CUDA_VISIBLE_DEVICES=1 julia --project=examples examples/basis_demo.jl  # Force specific GPU
#
# Quick Draw dataset requires downloading numpy files first:
#   mkdir -p data/quickdraw
#   curl -LO 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/cat.npy'
#   (repeat for other categories: dog, airplane, apple, bicycle)
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
using NPZ
using CairoMakie  # Used for training loss visualization
using CUDA         # GPU support (gracefully handled if no GPU available)

# ================================================================================
# Dataset Configuration
# ================================================================================

# Parse command line argument for dataset selection
const DATASET = length(ARGS) > 0 ? lowercase(ARGS[1]) : "mnist"
@assert DATASET in ["mnist", "quickdraw"] "Unknown dataset: $DATASET. Use 'mnist' or 'quickdraw'"

# Configuration based on dataset
# Both datasets use 28x28 images, padded to 32x32
const M_QUBITS = 5           # 2^5 = 32 rows
const N_QUBITS = 5           # 2^5 = 32 columns
const IMG_SIZE = 32

# Training configuration
const NUM_TRAINING_IMAGES = 20
const TRAINING_EPOCHS = 4
const STEPS_PER_IMAGE = 50
const NUM_TEST_IMAGES = 5

# Compression ratios to test
const COMPRESSION_RATIOS = [0.95, 0.90, 0.85, 0.80]  # Keep 5%, 10%, 15%, 20%

# Output directory (dataset-specific)
const OUTPUT_DIR = joinpath(@__DIR__, DATASET == "mnist" ? "BasisDemo" : "BasisDemo_QuickDraw")

# Quick Draw configuration
const QUICKDRAW_CATEGORIES = ["cat", "dog", "airplane", "apple", "bicycle"]
const QUICKDRAW_DATA_DIR = joinpath(@__DIR__, "..", "data", "quickdraw")

# CUDA auto-detection
const HAS_CUDA = CUDA.functional()

# ================================================================================
# Utility Functions
# ================================================================================

"""Pad 28×28 image to 32×32 (center-padded)."""
function pad_image(raw_img::AbstractMatrix)
    padded = zeros(Float64, IMG_SIZE, IMG_SIZE)
    padded[3:30, 3:30] = Float64.(raw_img)
    return padded
end

# Alias for backward compatibility
const pad_mnist_image = pad_image

"""Load Quick Draw dataset from numpy files."""
function load_quickdraw_data(; categories=QUICKDRAW_CATEGORIES, max_per_category=nothing)
    all_images = Matrix{Float64}[]
    all_labels = String[]
    
    for category in categories
        filepath = joinpath(QUICKDRAW_DATA_DIR, "$(category).npy")
        if !isfile(filepath)
            @warn "Quick Draw file not found: $filepath. Skipping category: $category"
            continue
        end
        
        # Load numpy array (N x 784, UInt8)
        data = npzread(filepath)
        n_images = size(data, 1)
        
        # Limit per category if specified
        n_to_load = max_per_category !== nothing ? min(n_images, max_per_category) : n_images
        
        for i in 1:n_to_load
            # Reshape to 28x28 and normalize to [0, 1]
            img = reshape(Float64.(data[i, :]), 28, 28) ./ 255.0
            push!(all_images, img)
            push!(all_labels, category)
        end
        
        @info "Loaded $n_to_load images from $category"
    end
    
    return all_images, all_labels
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

"""Build list of (label, optimizer_symbol, device, batch_size) configs."""
function build_training_configs()
    configs = Tuple{String, Symbol, Symbol, Int}[]
    for (opt_label, opt_sym) in [("GD", :gradient_descent), ("Adam", :adam)]
        for bs in [1, 4, 8]
            push!(configs, (opt_label, opt_sym, :cpu, bs))
        end
    end
    if HAS_CUDA
        for (opt_label, opt_sym) in [("GD", :gradient_descent), ("Adam", :adam)]
            for bs in [1, 4, 8]
                push!(configs, (opt_label, opt_sym, :gpu, bs))
            end
        end
    end
    return configs
end

"""Train a basis type across all optimizer configs. Returns vector of named tuples + best trained basis."""
function train_basis_sweep(
    ::Type{BT}, training_images, configs, k;
    basis_kwargs...
) where {BT <: AbstractSparseBasis}
    results = []
    for (opt_label, opt_sym, dev, bs) in configs
        bs_clamped = min(bs, length(training_images))
        print("  $(nameof(BT)) | $opt_label | $dev | bs=$bs_clamped ... ")
        flush(stdout)
        Random.seed!(42)
        elapsed = @elapsed begin
            basis, hist = train_basis(
                BT, training_images;
                m=M_QUBITS, n=N_QUBITS,
                loss=ParametricDFT.MSELoss(k),
                epochs=TRAINING_EPOCHS,
                steps_per_image=STEPS_PER_IMAGE,
                validation_split=0.2,
                shuffle=true,
                optimizer=opt_sym,
                device=dev,
                batch_size=bs_clamped,
                basis_kwargs...
            )
        end
        final_loss = hist.step_train_losses[end]
        @printf("%.1fs  loss=%.6f\n", elapsed, final_loss)
        push!(results, (;
            basis_type=nameof(BT), label=opt_label, optimizer=opt_sym,
            device=dev, batch_size=bs_clamped,
            final_loss, elapsed,
            step_losses=copy(hist.step_train_losses),
            train_losses=copy(hist.train_losses),
            val_losses=copy(hist.val_losses),
            basis, hist
        ))
    end
    # Select best config (lowest final loss)
    best_idx = argmin([r.final_loss for r in results])
    return results, results[best_idx]
end

# ================================================================================
# Main Demo
# ================================================================================

function main()
    println("="^100)
    println("       Unified Basis Demo: QFT, Entangled QFT, and TEBD Comparison")
    println("       Dataset: $(uppercase(DATASET))")
    println("="^100)
    
    # Create output directory
    if !isdir(OUTPUT_DIR)
        mkpath(OUTPUT_DIR)
        println("\n📁 Created output directory: $OUTPUT_DIR")
    end
    
    # ============================================================================
    # Step 1: Load Dataset
    # ============================================================================
    
    println("\n" * "="^80)
    println("Step 1: Loading $(uppercase(DATASET)) Dataset")
    println("="^80)
    
    training_images, test_images, test_labels = if DATASET == "mnist"
        # Load MNIST
        mnist_train = MNIST(split=:train)
        mnist_test = MNIST(split=:test)
        
        println("  Training pool: $(size(mnist_train.features, 3)) images")
        println("  Test pool: $(size(mnist_test.features, 3)) images")
        
        # Prepare datasets
        Random.seed!(42)
        train_indices = randperm(size(mnist_train.features, 3))[1:NUM_TRAINING_IMAGES]
        test_indices = randperm(size(mnist_test.features, 3))[1:NUM_TEST_IMAGES]
        
        train_imgs = [pad_image(mnist_train.features[:, :, i]) for i in train_indices]
        test_imgs = [pad_image(mnist_test.features[:, :, i]) for i in test_indices]
        labels = [string(mnist_test.targets[i]) for i in test_indices]
        
        (train_imgs, test_imgs, labels)
    else
        # Load Quick Draw
        println("  Loading Quick Draw categories: $(QUICKDRAW_CATEGORIES)")
        println("  Data directory: $(QUICKDRAW_DATA_DIR)")
        
        # Check if data exists
        if !isdir(QUICKDRAW_DATA_DIR)
            error("""
            Quick Draw data directory not found: $QUICKDRAW_DATA_DIR
            
            Please download the Quick Draw numpy files first:
              mkdir -p data/quickdraw
              cd data/quickdraw
              curl -LO 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/cat.npy'
              curl -LO 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/dog.npy'
              curl -LO 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/airplane.npy'
              curl -LO 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/apple.npy'
              curl -LO 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/bicycle.npy'
            """)
        end
        
        # Load all Quick Draw images
        all_images, all_labels = load_quickdraw_data(max_per_category=2000)
        
        println("  Total images loaded: $(length(all_images))")
        
        # Split into train/test (use different categories or random split)
        Random.seed!(42)
        indices = randperm(length(all_images))
        
        # Take more images for training diversity
        n_train = min(NUM_TRAINING_IMAGES, length(indices) ÷ 2)
        n_test = min(NUM_TEST_IMAGES, length(indices) ÷ 2)
        
        train_indices = indices[1:n_train]
        test_indices = indices[end-n_test+1:end]
        
        train_imgs = [pad_image(all_images[i]) for i in train_indices]
        test_imgs = [pad_image(all_images[i]) for i in test_indices]
        labels = [all_labels[i] for i in test_indices]
        
        (train_imgs, test_imgs, labels)
    end
    
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
    entangled_qft = EntangledQFTBasis(M_QUBITS, N_QUBITS; entangle_position=:back)
    tebd_default = TEBDBasis(M_QUBITS, N_QUBITS)
    
    println("  Standard QFT:     $(num_parameters(standard_qft)) parameters")
    println("  Entangled QFT:    $(num_parameters(entangled_qft)) parameters ($(entangled_qft.n_entangle) entangle gates)")
    println("  TEBD:             $(num_parameters(tebd_default)) parameters ($(num_gates(tebd_default)) ring gates)")
    
    # ============================================================================
    # Step 3: Train All Bases (Full Optimizer Sweep)
    # ============================================================================

    println("\n" * "="^80)
    println("Step 3: Training All Bases (Full Optimizer Sweep)")
    println("="^80)

    total_coefficients = IMG_SIZE * IMG_SIZE
    k = round(Int, total_coefficients * 0.1)

    configs = build_training_configs()
    n_configs = length(configs)

    println("\nTraining configuration:")
    println("  Images: $NUM_TRAINING_IMAGES | Epochs: $TRAINING_EPOCHS | Steps/image: $STEPS_PER_IMAGE")
    println("  Loss: MSELoss($k) | Configs per basis: $n_configs")
    println("  Devices: CPU" * (HAS_CUDA ? " + GPU" : ""))

    # Train QFT
    println("\n--- QFTBasis ---")
    qft_results, best_qft = train_basis_sweep(QFTBasis, training_images, configs, k)
    trained_qft = best_qft.basis

    # Train Entangled QFT
    println("\n--- EntangledQFTBasis ---")
    entangled_results, best_entangled = train_basis_sweep(
        EntangledQFTBasis, training_images, configs, k;
        entangle_position=:back
    )
    trained_entangled = best_entangled.basis

    # Train TEBD
    println("\n--- TEBDBasis ---")
    tebd_results, best_tebd = train_basis_sweep(TEBDBasis, training_images, configs, k)
    trained_tebd = best_tebd.basis

    all_results = [qft_results; entangled_results; tebd_results]

    # Print per-basis comparison tables
    for (basis_name, results_vec, best) in [("QFTBasis", qft_results, best_qft), ("EntangledQFTBasis", entangled_results, best_entangled), ("TEBDBasis", tebd_results, best_tebd)]
        println("\n" * "-"^75)
        println("  $basis_name Comparison")
        println("-"^75)
        @printf("  %-6s  %-6s  %5s  %14s  %10s\n", "Optim", "Device", "Batch", "Final Loss", "Time")
        println("  " * "-"^50)
        for r in results_vec
            marker = r === best ? " *" : ""
            @printf("  %-6s  %-6s  %5d  %14.6f  %9.1fs%s\n",
                    r.label, r.device, r.batch_size, r.final_loss, r.elapsed, marker)
        end
    end

    println("\nBest configs:")
    println("  QFT:       $(best_qft.label) $(best_qft.device) bs=$(best_qft.batch_size) (loss=$(round(best_qft.final_loss, digits=6)))")
    println("  Entangled: $(best_entangled.label) $(best_entangled.device) bs=$(best_entangled.batch_size) (loss=$(round(best_entangled.final_loss, digits=6)))")
    println("  TEBD:      $(best_tebd.label) $(best_tebd.device) bs=$(best_tebd.batch_size) (loss=$(round(best_tebd.final_loss, digits=6)))")

    # ============================================================================
    # Step 3.5: Visualize Training Losses
    # ============================================================================

    println("\n" * "="^80)
    println("Step 3.5: Visualizing Training Losses")
    println("="^80)

    # Best-config histories for per-basis comparison
    histories = [
        TrainingHistory(best_qft.train_losses, best_qft.val_losses, best_qft.step_losses, "QFT (best)"),
        TrainingHistory(best_entangled.train_losses, best_entangled.val_losses, best_entangled.step_losses, "Entangled QFT (best)"),
        TrainingHistory(best_tebd.train_losses, best_tebd.val_losses, best_tebd.step_losses, "TEBD (best)"),
    ]

    plots_dir = joinpath(OUTPUT_DIR, "plots")
    println("\nGenerating training loss plots...")
    saved_plots = save_training_plots(histories, plots_dir)

    println("\nTraining loss plots saved to: $plots_dir")
    for plot_path in saved_plots
        println("  $(relpath(plot_path, plots_dir))")
    end

    # Optimizer comparison plot: one subplot per basis type
    println("\nGenerating optimizer comparison plot...")
    _colors = [:blue, :red, :green, :orange, :purple, :cyan, :magenta, :brown, :pink, :gray, :olive, :teal]
    _styles = [:solid, :dash, :dot, :dashdot, :solid, :dash, :dot, :dashdot, :solid, :dash, :dot, :dashdot]

    fig = CairoMakie.Figure(size=(1400, 500))
    for (col, (basis_name, results_vec)) in enumerate([("QFT", qft_results), ("Entangled QFT", entangled_results), ("TEBD", tebd_results)])
        ax = CairoMakie.Axis(fig[1, col]; title=basis_name, xlabel="Step", ylabel="Loss", yscale=log10)
        for (idx, r) in enumerate(results_vec)
            ci = mod1(idx, length(_colors))
            si = mod1(idx, length(_styles))
            lbl = "$(r.label) $(r.device) bs=$(r.batch_size)"
            CairoMakie.lines!(ax, 1:length(r.step_losses), r.step_losses;
                   color=_colors[ci], linestyle=_styles[si], label=lbl)
        end
        if col == 3
            CairoMakie.Legend(fig[1, 4], ax; framevisible=true, labelsize=9)
        end
    end
    optim_plot_path = joinpath(plots_dir, "optimizer_comparison.png")
    CairoMakie.save(optim_plot_path, fig; px_per_unit=2)
    println("  Saved: $optim_plot_path")

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
    original_path = joinpath(OUTPUT_DIR, "original_$(sample_label).png")
    Images.save(original_path, Gray.(sample_img))
    println("  ✓ original_$(sample_label).png")
    
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
    
    dataset_name = DATASET == "mnist" ? "MNIST" : "Quick Draw"
    summary_text = """
    
    ✅ Comparison completed on $dataset_name dataset
    
    Configuration:
    ├─ Dataset:             $dataset_name
    ├─ Training images:     $NUM_TRAINING_IMAGES
    ├─ Test images:         $NUM_TEST_IMAGES  
    ├─ Image size:          $(IMG_SIZE)×$(IMG_SIZE)
    ├─ Compression ratios:  $(join(["$(round(Int, (1-r)*100))%" for r in COMPRESSION_RATIOS], ", "))
    └─ Training epochs:     $TRAINING_EPOCHS (optimizer sweep: $n_configs configs/basis)

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
    ├─ original_$(sample_label).png          : Original test image
    ├─ recovered_*.png               : Recovered images for each basis
    └─ plots/                        : Training loss visualizations
       ├─ optimizer_comparison.png   : GD vs Adam sweep (per-basis subplots, log-scale)
       ├─ log/                       : Log-scale (y-axis) plots
       │  ├─ *_loss.png              : Per-epoch loss curves
       │  ├─ *_step_loss.png         : Per-step loss curves
       │  ├─ comparison_*.png        : Comparison plots
       │  └─ grid.png                : Grid view of all loss curves
       └─ linear/                    : Linear-scale (y-axis) plots
          ├─ *_loss.png              : Per-epoch loss curves
          ├─ *_step_loss.png         : Per-step loss curves
          ├─ comparison_*.png        : Comparison plots
          └─ grid.png                : Grid view of all loss curves
    """
    
    println(summary_text)
    
    # ============================================================================
    # Step 8: Generalization Test (Non-MNIST Images)
    # ============================================================================
    
    println("\n" * "="^80)
    println("Step 8: Generalization Test (Detecting Overfitting)")
    println("="^80)
    
    # Create synthetic test images
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
    
    synthetic_images = [create_synthetic_image(i + 1000) for i in 1:20]
    generalization_results = Dict{Tuple{String, Float64}, NamedTuple}()
    
    println("\nTesting generalization on 20 synthetic images (gradients + circles)...")
    
    test_bases = Dict(
        "Standard QFT" => standard_qft,
        "Trained QFT" => trained_qft,
        "Trained TEBD" => trained_tebd,
    )
    
    for ratio in [0.90, 0.80]  # Test 10% and 20% kept
        for (basis_name, basis) in test_bases
            mse_vals, psnr_vals = Float64[], Float64[]
            
            for img in synthetic_images
                compressed = compress(basis, img; ratio=ratio)
                recovered = recover(basis, compressed)
                metrics = compute_metrics(img, recovered)
                push!(mse_vals, metrics.mse)
                push!(psnr_vals, metrics.psnr)
            end
            
            generalization_results[(basis_name, ratio)] = (
                mse=mean(mse_vals), psnr=mean(psnr_vals),
                mse_std=std(mse_vals), psnr_std=std(psnr_vals)
            )
        end
        
        # Also test Classical FFT
        mse_vals, psnr_vals = Float64[], Float64[]
        for img in synthetic_images
            recovered = fft_compress(img, ratio)
            metrics = compute_metrics(img, recovered)
            push!(mse_vals, metrics.mse)
            push!(psnr_vals, metrics.psnr)
        end
        generalization_results[("Classical FFT", ratio)] = (
            mse=mean(mse_vals), psnr=mean(psnr_vals),
            mse_std=std(mse_vals), psnr_std=std(psnr_vals)
        )
        
        kept_pct = round(Int, (1-ratio)*100)
        println("\n  $kept_pct% kept:")
        for basis_name in ["Standard QFT", "Trained QFT", "Trained TEBD", "Classical FFT"]
            r = generalization_results[(basis_name, ratio)]
            @printf("    %-20s PSNR: %.2f ± %.2f dB\n", basis_name, r.psnr, r.psnr_std)
        end
    end
    
    # Check for overfitting
    mnist_tebd_psnr = results[("Trained TEBD", 0.90)].psnr
    synth_tebd_psnr = generalization_results[("Trained TEBD", 0.90)].psnr
    synth_qft_psnr = generalization_results[("Standard QFT", 0.90)].psnr
    
    println("\n" * "-"^80)
    println("⚠️  GENERALIZATION CHECK (10% kept):")
    println("-"^80)
    @printf("  TEBD on MNIST:     %.2f dB\n", mnist_tebd_psnr)
    @printf("  TEBD on Synthetic: %.2f dB\n", synth_tebd_psnr)
    @printf("  QFT on Synthetic:  %.2f dB\n", synth_qft_psnr)
    
    if synth_tebd_psnr < synth_qft_psnr - 3.0
        println("\n  ⚠️  WARNING: TEBD performs significantly worse than QFT on non-MNIST images!")
        println("  ⚠️  This indicates OVERFITTING to the MNIST training data.")
        println("  ⚠️  The high MNIST PSNR does NOT indicate general compression quality.")
    elseif synth_tebd_psnr > synth_qft_psnr + 1.0
        println("\n  ✓ TEBD generalizes well to synthetic images.")
    else
        println("\n  ✓ TEBD performs comparably to QFT on synthetic images.")
    end
    println("-"^80)

    # ============================================================================
    # Step 10: Write Summary to Markdown File
    # ============================================================================
    
    summary_path = joinpath(OUTPUT_DIR, "summary.md")
    
    # Check if TEBD is overfit
    tebd_overfit = synth_tebd_psnr < synth_qft_psnr - 3.0
    
    # Build markdown content
    overfit_warning = tebd_overfit ? """
## ⚠️ WARNING: TEBD Overfitting Detected

The trained TEBD shows high PSNR on $dataset_name but **does not generalize** to other image types:

| Image Type | Trained TEBD | Standard QFT |
|------------|--------------|--------------|
| $dataset_name (10% kept) | $(round(mnist_tebd_psnr, digits=2)) dB | $(round(results[("Standard QFT", 0.90)].psnr, digits=2)) dB |
| Synthetic (10% kept) | $(round(synth_tebd_psnr, digits=2)) dB | $(round(synth_qft_psnr, digits=2)) dB |

**Conclusion:** The TEBD has overfit to $dataset_name images. Use Standard/Trained QFT for general compression.

---

""" : ""
    
    md_content = """
# Basis Comparison Summary ($dataset_name)

$overfit_warning## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | $dataset_name |
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

## Results ($dataset_name Test Set)

### 🏆 Best at 10% kept: **$best_basis** (PSNR: $(round(best_psnr, digits=2)) dB)$(tebd_overfit ? " ⚠️ (overfit)" : "")

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

## Generalization Test (Synthetic Images)

| Basis | 10% kept | 20% kept |
|-------|----------|----------|
| Standard QFT | $(round(generalization_results[("Standard QFT", 0.90)].psnr, digits=2)) dB | $(round(generalization_results[("Standard QFT", 0.80)].psnr, digits=2)) dB |
| Trained QFT | $(round(generalization_results[("Trained QFT", 0.90)].psnr, digits=2)) dB | $(round(generalization_results[("Trained QFT", 0.80)].psnr, digits=2)) dB |
| Trained TEBD | $(round(generalization_results[("Trained TEBD", 0.90)].psnr, digits=2)) dB | $(round(generalization_results[("Trained TEBD", 0.80)].psnr, digits=2)) dB |
| Classical FFT | $(round(generalization_results[("Classical FFT", 0.90)].psnr, digits=2)) dB | $(round(generalization_results[("Classical FFT", 0.80)].psnr, digits=2)) dB |

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
- `trained_tebd.json` - Trained TEBD basis$(tebd_overfit ? " ⚠️ (overfit)" : "")
- `original_$(sample_label).png` - Original test image
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
