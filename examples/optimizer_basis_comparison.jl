# ================================================================================
# Optimizer x Basis x Device x Dataset Comparison
# ================================================================================
# Comprehensive comparison of Riemannian optimizers across basis types,
# compute devices, and datasets.
#
# Dimensions:
#   Optimizers: Riemannian Gradient Descent, Riemannian Adam
#   Devices:    CPU, GPU (if available)
#   Bases:      QFT, Entangled QFT, TEBD
#   Datasets:   MNIST, QuickDraw (if data available)
#
# Total: 2 optimizers x 2 devices x 3 bases x 2 datasets = 24 configurations
#
# Run:
#   julia --project=examples examples/optimizer_basis_comparison.jl          # Small run
#   julia --project=examples examples/optimizer_basis_comparison.jl --full   # Full run
#
# The small run uses 5 training / 3 test images, 10 steps, 1 epoch.
# The full run uses 100 training / 100 test images, 500 steps, 3 epochs.
# ================================================================================

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

# ================================================================================
# CUDA Loading (must be before ParametricDFT for extension activation)
# ================================================================================

using CUDA
const GPU_AVAILABLE = CUDA.functional()
if GPU_AVAILABLE
    @info "GPU detected: $(CUDA.name(CUDA.device()))"
else
    @info "No GPU detected, GPU experiments will be skipped"
end

# ================================================================================
# Dependencies
# ================================================================================

using ParametricDFT
using MLDatasets
using Images
using ImageQualityIndexes
using Random
using Statistics
using Printf
using FFTW
using NPZ

# ================================================================================
# Configuration
# ================================================================================

const FULL_RUN = "--full" in ARGS

const M_QUBITS = 5        # 2^5 = 32 rows
const N_QUBITS = 5        # 2^5 = 32 columns
const IMG_SIZE = 32

if FULL_RUN
    const NUM_TRAIN  = 100
    const NUM_TEST   = 100
    const EPOCHS     = 3
    const STEPS      = 500
else
    const NUM_TRAIN  = 5
    const NUM_TEST   = 3
    const EPOCHS     = 1
    const STEPS      = 10
end

const BATCH_SIZE = 1  # Same for all configurations for fair comparison
const CHECKPOINT_INTERVAL = 50  # Save basis + loss every N steps (0 = disabled)

# Compression ratios for evaluation
const COMPRESSION_RATIOS = FULL_RUN ? [0.95, 0.90, 0.85, 0.80] : [0.90]

# QuickDraw data
const QUICKDRAW_CATEGORIES = ["cat", "dog", "airplane", "apple", "bicycle"]
const QUICKDRAW_DATA_DIR = joinpath(@__DIR__, "..", "data", "quickdraw")
const QUICKDRAW_AVAILABLE = isdir(QUICKDRAW_DATA_DIR) &&
    all(isfile(joinpath(QUICKDRAW_DATA_DIR, "$(c).npy")) for c in QUICKDRAW_CATEGORIES)

# Output
const OUTPUT_DIR = joinpath(@__DIR__, "FullComparison")

# ================================================================================
# Utility Functions
# ================================================================================

"""Pad 28x28 image to 32x32 (center-padded)."""
function pad_image(raw_img::AbstractMatrix)
    padded = zeros(Float64, IMG_SIZE, IMG_SIZE)
    padded[3:30, 3:30] = Float64.(raw_img)
    return padded
end

"""Load MNIST dataset."""
function load_mnist_data(n_train, n_test)
    mnist_train = MNIST(split=:train)
    mnist_test = MNIST(split=:test)

    Random.seed!(42)
    train_idx = randperm(size(mnist_train.features, 3))[1:n_train]
    test_idx = randperm(size(mnist_test.features, 3))[1:n_test]

    train_imgs = [pad_image(mnist_train.features[:, :, i]) for i in train_idx]
    test_imgs = [pad_image(mnist_test.features[:, :, i]) for i in test_idx]
    return train_imgs, test_imgs
end

"""Load QuickDraw dataset from numpy files."""
function load_quickdraw_data(n_train, n_test)
    all_images = Matrix{Float64}[]
    per_cat = ceil(Int, (n_train + n_test) / length(QUICKDRAW_CATEGORIES)) + 10

    for category in QUICKDRAW_CATEGORIES
        filepath = joinpath(QUICKDRAW_DATA_DIR, "$(category).npy")
        data = npzread(filepath)
        n_load = min(size(data, 1), per_cat)
        for i in 1:n_load
            img = reshape(Float64.(data[i, :]), 28, 28) ./ 255.0
            push!(all_images, img)
        end
    end

    Random.seed!(42)
    indices = randperm(length(all_images))
    nt = min(n_train, length(indices) ÷ 2)
    ne = min(n_test, length(indices) - nt)

    train_imgs = [pad_image(all_images[i]) for i in indices[1:nt]]
    test_imgs = [pad_image(all_images[i]) for i in indices[nt+1:nt+ne]]
    return train_imgs, test_imgs
end

"""Compute quality metrics between original and recovered images."""
function compute_metrics(original::AbstractMatrix, recovered::AbstractMatrix)
    recovered_clamped = clamp.(real.(recovered), 0.0, 1.0)
    mse_val = mean((original .- recovered_clamped).^2)
    psnr_val = mse_val > 0 ? 10 * log10(1.0 / mse_val) : Inf
    ssim_val = assess_ssim(Gray.(original), Gray.(recovered_clamped))
    return (mse=mse_val, psnr=psnr_val, ssim=ssim_val)
end

"""Classical FFT compression baseline."""
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

"""Evaluate a trained basis on a test set at a given compression ratio."""
function evaluate_basis(basis, test_imgs, ratio)
    psnr_vals, ssim_vals, mse_vals = Float64[], Float64[], Float64[]
    for img in test_imgs
        compressed = compress(basis, img; ratio=ratio)
        recovered = recover(basis, compressed)
        met = compute_metrics(img, recovered)
        push!(psnr_vals, met.psnr)
        push!(ssim_vals, met.ssim)
        push!(mse_vals, met.mse)
    end
    return (psnr=mean(psnr_vals), ssim=mean(ssim_vals), mse=mean(mse_vals))
end

"""Evaluate classical FFT on a test set."""
function evaluate_fft(test_imgs, ratio)
    psnr_vals, ssim_vals, mse_vals = Float64[], Float64[], Float64[]
    for img in test_imgs
        recovered = fft_compress(img, ratio)
        met = compute_metrics(img, recovered)
        push!(psnr_vals, met.psnr)
        push!(ssim_vals, met.ssim)
        push!(mse_vals, met.mse)
    end
    return (psnr=mean(psnr_vals), ssim=mean(ssim_vals), mse=mean(mse_vals))
end

"""Format a result value, or N/A if missing."""
fmt_psnr(r) = r === nothing || r.failed ? "   N/A   " : @sprintf("%8.2f", r.psnr)
fmt_ssim(r) = r === nothing || r.failed ? "   N/A   " : @sprintf("%8.4f", r.ssim)
fmt_time(r) = r === nothing || r.failed ? "   N/A   " : @sprintf("%8.1f", r.time)
fmt_loss(r) = r === nothing || r.failed ? "   N/A   " : @sprintf("%8.4f", r.train_loss)

# ================================================================================
# Main
# ================================================================================

function main()
    println("=" ^ 120)
    println("    Optimizer x Basis x Device x Dataset Comparison")
    println("    Mode: $(FULL_RUN ? "FULL RUN (100 images, 500 steps, 3 epochs)" : "SMALL RUN (5 images, 10 steps, 1 epoch)")")
    println("=" ^ 120)

    println("\nConfiguration:")
    println("  Training images:  $NUM_TRAIN")
    println("  Test images:      $NUM_TEST")
    println("  Epochs:           $EPOCHS")
    println("  Steps/batch:      $STEPS")
    println("  Batch size:       $BATCH_SIZE")
    println("  GPU available:    $GPU_AVAILABLE")
    println("  QuickDraw data:   $QUICKDRAW_AVAILABLE")
    println("  Checkpoint every: $(CHECKPOINT_INTERVAL > 0 ? "$CHECKPOINT_INTERVAL steps" : "disabled")")

    mkpath(OUTPUT_DIR)
    loss_dir = joinpath(OUTPUT_DIR, "loss_history")
    mkpath(loss_dir)
    basis_dir = joinpath(OUTPUT_DIR, "trained_bases")
    mkpath(basis_dir)

    # ========================================================================
    # Step 1: Load Datasets
    # ========================================================================

    println("\n" * "=" ^ 80)
    println("Step 1: Loading Datasets")
    println("=" ^ 80)

    println("\nLoading MNIST...")
    mnist_train, mnist_test = load_mnist_data(NUM_TRAIN, NUM_TEST)
    println("  Training: $(length(mnist_train)) images, Test: $(length(mnist_test)) images")

    datasets = [("MNIST", mnist_train, mnist_test)]

    if QUICKDRAW_AVAILABLE
        println("\nLoading QuickDraw...")
        qd_train, qd_test = load_quickdraw_data(NUM_TRAIN, NUM_TEST)
        println("  Training: $(length(qd_train)) images, Test: $(length(qd_test)) images")
        push!(datasets, ("QuickDraw", qd_train, qd_test))
    else
        println("\nQuickDraw data not found, skipping.")
        println("  Download with: mkdir -p data/quickdraw && cd data/quickdraw")
        for c in QUICKDRAW_CATEGORIES
            println("    curl -LO 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/$c.npy'")
        end
    end

    # ========================================================================
    # Step 2: Define Experiment Grid
    # ========================================================================

    optimizers = [
        (:gradient_descent, "GD"),
        (:adam,             "Adam"),
    ]

    devices = [(:cpu, "CPU")]
    if GPU_AVAILABLE
        push!(devices, (:gpu, "GPU"))
    end

    basis_types = [
        (QFTBasis,          "QFT",          Dict{Symbol,Any}()),
        (EntangledQFTBasis, "EntangledQFT", Dict{Symbol,Any}(:entangle_position => :back)),
        (TEBDBasis,         "TEBD",         Dict{Symbol,Any}()),
    ]

    # Column headers for tables: "CPU GD", "CPU Adam", "GPU GD", "GPU Adam"
    col_keys = [(d, o) for (_, d) in devices for (_, o) in optimizers]
    col_headers = ["$d $o" for (d, o) in col_keys]

    total_configs = length(datasets) * length(optimizers) * length(devices) * length(basis_types)
    println("\n  Total configurations: $total_configs")
    println("  Columns: $(join(col_headers, ", "))")

    # ========================================================================
    # Step 3: Run All Experiments
    # ========================================================================

    println("\n" * "=" ^ 80)
    println("Step 3: Running Experiments")
    println("=" ^ 80)

    total_coefficients = IMG_SIZE * IMG_SIZE
    k = round(Int, total_coefficients * 0.1)  # Keep 10% for MSE loss

    # Results: (dataset, basis, device, optimizer) -> NamedTuple
    results = Dict{NTuple{4, String}, NamedTuple}()

    # FFT baselines per dataset
    fft_results = Dict{String, NamedTuple}()

    config_num = 0
    for (ds_name, train_imgs, test_imgs) in datasets
        # Compute FFT baseline once per dataset
        fft_met = evaluate_fft(test_imgs, 0.90)
        fft_results[ds_name] = fft_met
        @printf("  [FFT baseline] %s: PSNR %.2f dB | SSIM %.4f\n",
                ds_name, fft_met.psnr, fft_met.ssim)

        for (opt_sym, opt_name) in optimizers
            for (dev_sym, dev_name) in devices
                for (BType, basis_name, extra_kwargs) in basis_types
                    config_num += 1
                    config_label = "$ds_name | $opt_name | $dev_name | $basis_name"

                    println("\n" * "-" ^ 80)
                    @printf("[%d/%d] %s\n", config_num, total_configs, config_label)
                    println("-" ^ 80)

                    # Consistent random seed for data splitting within train_basis
                    Random.seed!(123)

                    config_tag = "$(lowercase(ds_name))_$(opt_sym)_$(dev_sym)_$(lowercase(basis_name))"
                    loss_path = joinpath(loss_dir, "$(config_tag)_loss.json")

                    # Per-config checkpoint directory
                    ckpt_dir = CHECKPOINT_INTERVAL > 0 ?
                        joinpath(OUTPUT_DIR, "checkpoints", config_tag) : nothing

                    local trained_basis, history
                    train_time = NaN

                    try
                        train_time = @elapsed begin
                            trained_basis, history = train_basis(
                                BType, train_imgs;
                                m=M_QUBITS, n=N_QUBITS,
                                loss=ParametricDFT.MSELoss(k),
                                epochs=EPOCHS,
                                steps_per_image=STEPS,
                                validation_split=0.2,
                                early_stopping_patience=max(EPOCHS, 5),
                                verbose=true,
                                optimizer=opt_sym,
                                batch_size=BATCH_SIZE,
                                device=dev_sym,
                                save_loss_path=loss_path,
                                checkpoint_interval=CHECKPOINT_INTERVAL,
                                checkpoint_dir=ckpt_dir,
                                extra_kwargs...
                            )
                        end
                    catch e
                        @warn "FAILED: $config_label" exception=(e, catch_backtrace())
                        results[(ds_name, basis_name, dev_name, opt_name)] = (
                            psnr=NaN, ssim=NaN, mse=NaN,
                            train_loss=NaN, val_loss=NaN, time=NaN,
                            failed=true
                        )
                        continue
                    end

                    # Save trained basis
                    basis_path = joinpath(basis_dir,
                        "$(lowercase(ds_name))_$(opt_sym)_$(dev_sym)_$(lowercase(basis_name)).json")
                    save_basis(basis_path, trained_basis)
                    println("  Basis saved: $(relpath(basis_path, OUTPUT_DIR))")

                    # Evaluate on test set at 10% kept
                    met = evaluate_basis(trained_basis, test_imgs, 0.90)

                    final_train = history.train_losses[end]
                    final_val = history.val_losses[end]

                    results[(ds_name, basis_name, dev_name, opt_name)] = (
                        psnr=met.psnr, ssim=met.ssim, mse=met.mse,
                        train_loss=final_train, val_loss=final_val, time=train_time,
                        failed=false
                    )

                    @printf("  Result: PSNR %.2f dB | SSIM %.4f | Train loss %.6f | Time %.1f s\n",
                            met.psnr, met.ssim, final_train, train_time)
                end
            end
        end
    end

    # ========================================================================
    # Step 4: Print Results Tables
    # ========================================================================

    println("\n" * "=" ^ 120)
    println("    RESULTS")
    println("=" ^ 120)

    basis_names = ["QFT", "EntangledQFT", "TEBD"]

    # Column width
    cw = 12
    bw = 18  # basis column width

    for (ds_name, _, _) in datasets
        println("\n" * "=" ^ 100)
        println("  Dataset: $ds_name")
        println("=" ^ 100)

        # --- PSNR Table ---
        println("\n  PSNR (dB) at 10% kept - higher is better:")
        println("  " * "-" ^ (bw + (cw + 3) * length(col_headers)))
        @printf("  %-*s", bw, "Basis")
        for h in col_headers
            @printf(" | %*s", cw, h)
        end
        println()
        println("  " * "-" ^ (bw + (cw + 3) * length(col_headers)))

        for bn in basis_names
            @printf("  %-*s", bw, bn)
            for (d, o) in col_keys
                r = get(results, (ds_name, bn, d, o), nothing)
                @printf(" | %*s", cw, fmt_psnr(r))
            end
            println()
        end
        # FFT baseline
        fft_r = fft_results[ds_name]
        @printf("  %-*s", bw, "Classical FFT")
        for _ in col_keys
            @printf(" | %*s", cw, @sprintf("%8.2f", fft_r.psnr))
        end
        println()
        println("  " * "-" ^ (bw + (cw + 3) * length(col_headers)))

        # --- SSIM Table ---
        println("\n  SSIM at 10% kept - higher is better:")
        println("  " * "-" ^ (bw + (cw + 3) * length(col_headers)))
        @printf("  %-*s", bw, "Basis")
        for h in col_headers
            @printf(" | %*s", cw, h)
        end
        println()
        println("  " * "-" ^ (bw + (cw + 3) * length(col_headers)))

        for bn in basis_names
            @printf("  %-*s", bw, bn)
            for (d, o) in col_keys
                r = get(results, (ds_name, bn, d, o), nothing)
                @printf(" | %*s", cw, fmt_ssim(r))
            end
            println()
        end
        fft_r = fft_results[ds_name]
        @printf("  %-*s", bw, "Classical FFT")
        for _ in col_keys
            @printf(" | %*s", cw, @sprintf("%8.4f", fft_r.ssim))
        end
        println()
        println("  " * "-" ^ (bw + (cw + 3) * length(col_headers)))

        # --- Training Time Table ---
        println("\n  Training Time (seconds):")
        println("  " * "-" ^ (bw + (cw + 3) * length(col_headers)))
        @printf("  %-*s", bw, "Basis")
        for h in col_headers
            @printf(" | %*s", cw, h)
        end
        println()
        println("  " * "-" ^ (bw + (cw + 3) * length(col_headers)))

        for bn in basis_names
            @printf("  %-*s", bw, bn)
            for (d, o) in col_keys
                r = get(results, (ds_name, bn, d, o), nothing)
                @printf(" | %*s", cw, fmt_time(r))
            end
            println()
        end
        println("  " * "-" ^ (bw + (cw + 3) * length(col_headers)))

        # --- Final Training Loss Table ---
        println("\n  Final Training Loss:")
        println("  " * "-" ^ (bw + (cw + 3) * length(col_headers)))
        @printf("  %-*s", bw, "Basis")
        for h in col_headers
            @printf(" | %*s", cw, h)
        end
        println()
        println("  " * "-" ^ (bw + (cw + 3) * length(col_headers)))

        for bn in basis_names
            @printf("  %-*s", bw, bn)
            for (d, o) in col_keys
                r = get(results, (ds_name, bn, d, o), nothing)
                @printf(" | %*s", cw, fmt_loss(r))
            end
            println()
        end
        println("  " * "-" ^ (bw + (cw + 3) * length(col_headers)))
    end

    # ========================================================================
    # Step 5: Cross-Dataset Comparison
    # ========================================================================

    if length(datasets) > 1
        println("\n" * "=" ^ 120)
        println("    CROSS-DATASET COMPARISON (PSNR dB at 10% kept)")
        println("=" ^ 120)

        ds_names = [d[1] for d in datasets]
        for (d, o) in col_keys
            println("\n  $d $o:")
            println("  " * "-" ^ 60)
            @printf("  %-18s", "Basis")
            for ds in ds_names
                @printf(" | %12s", ds)
            end
            println()
            println("  " * "-" ^ 60)

            for bn in basis_names
                @printf("  %-18s", bn)
                for ds in ds_names
                    r = get(results, (ds, bn, d, o), nothing)
                    @printf(" | %12s", fmt_psnr(r))
                end
                println()
            end
            println("  " * "-" ^ 60)
        end
    end

    # ========================================================================
    # Step 6: Best Configuration Summary
    # ========================================================================

    println("\n" * "=" ^ 120)
    println("    BEST CONFIGURATIONS")
    println("=" ^ 120)

    for (ds_name, _, _) in datasets
        println("\n  Dataset: $ds_name")
        println("  " * "-" ^ 80)

        best_key = nothing
        best_psnr = -Inf
        for bn in basis_names
            for (d, o) in col_keys
                r = get(results, (ds_name, bn, d, o), nothing)
                if r !== nothing && !r.failed && r.psnr > best_psnr
                    best_psnr = r.psnr
                    best_key = (bn, d, o)
                end
            end
        end

        if best_key !== nothing
            bn, d, o = best_key
            r = results[(ds_name, bn, d, o)]
            @printf("  Best:   %-14s with %-6s on %-4s  PSNR: %.2f dB  SSIM: %.4f  Time: %.1f s\n",
                    bn, o, d, r.psnr, r.ssim, r.time)
        end

        # Best per basis
        for bn in basis_names
            local_best_key = nothing
            local_best_psnr = -Inf
            for (d, o) in col_keys
                r = get(results, (ds_name, bn, d, o), nothing)
                if r !== nothing && !r.failed && r.psnr > local_best_psnr
                    local_best_psnr = r.psnr
                    local_best_key = (d, o)
                end
            end
            if local_best_key !== nothing
                d, o = local_best_key
                r = results[(ds_name, bn, d, o)]
                @printf("    %-14s best: %-6s on %-4s  PSNR: %.2f dB  Time: %.1f s\n",
                        bn, o, d, r.psnr, r.time)
            end
        end

        # FFT reference
        fft_r = fft_results[ds_name]
        @printf("  FFT baseline:                              PSNR: %.2f dB\n", fft_r.psnr)
    end

    # ========================================================================
    # Step 7: Save Markdown Summary
    # ========================================================================

    summary_path = joinpath(OUTPUT_DIR, "summary.md")

    md = IOBuffer()
    println(md, "# Optimizer x Basis x Device x Dataset Comparison")
    println(md, "")
    println(md, "## Configuration")
    println(md, "")
    println(md, "| Parameter | Value |")
    println(md, "|-----------|-------|")
    println(md, "| Mode | $(FULL_RUN ? "Full" : "Small") |")
    println(md, "| Training images | $NUM_TRAIN |")
    println(md, "| Test images | $NUM_TEST |")
    println(md, "| Epochs | $EPOCHS |")
    println(md, "| Steps/batch | $STEPS |")
    println(md, "| Batch size | $BATCH_SIZE |")
    println(md, "| GPU available | $GPU_AVAILABLE |")
    println(md, "")
    println(md, "## Experiment Grid")
    println(md, "")
    println(md, "- **Optimizers:** Riemannian Gradient Descent (GD), Riemannian Adam")
    println(md, "- **Devices:** $(join([d for (_, d) in devices], ", "))")
    println(md, "- **Bases:** QFT, Entangled QFT, TEBD")
    println(md, "- **Datasets:** $(join([d[1] for d in datasets], ", "))")
    println(md, "- **Total configurations:** $total_configs")
    println(md, "")

    for (ds_name, _, _) in datasets
        println(md, "## $ds_name Results")
        println(md, "")

        # PSNR table
        println(md, "### PSNR (dB) at 10% kept")
        println(md, "")
        print(md, "| Basis |")
        for h in col_headers
            print(md, " $h |")
        end
        println(md, "")
        print(md, "|-------|")
        for _ in col_headers
            print(md, "------|")
        end
        println(md, "")

        for bn in basis_names
            print(md, "| $bn |")
            for (d, o) in col_keys
                r = get(results, (ds_name, bn, d, o), nothing)
                if r !== nothing && !r.failed
                    @printf(md, " %.2f |", r.psnr)
                else
                    print(md, " N/A |")
                end
            end
            println(md, "")
        end
        fft_r = fft_results[ds_name]
        print(md, "| Classical FFT |")
        for _ in col_keys
            @printf(md, " %.2f |", fft_r.psnr)
        end
        println(md, "")
        println(md, "")

        # SSIM table
        println(md, "### SSIM at 10% kept")
        println(md, "")
        print(md, "| Basis |")
        for h in col_headers
            print(md, " $h |")
        end
        println(md, "")
        print(md, "|-------|")
        for _ in col_headers
            print(md, "------|")
        end
        println(md, "")

        for bn in basis_names
            print(md, "| $bn |")
            for (d, o) in col_keys
                r = get(results, (ds_name, bn, d, o), nothing)
                if r !== nothing && !r.failed
                    @printf(md, " %.4f |", r.ssim)
                else
                    print(md, " N/A |")
                end
            end
            println(md, "")
        end
        println(md, "")

        # Training time table
        println(md, "### Training Time (seconds)")
        println(md, "")
        print(md, "| Basis |")
        for h in col_headers
            print(md, " $h |")
        end
        println(md, "")
        print(md, "|-------|")
        for _ in col_headers
            print(md, "------|")
        end
        println(md, "")

        for bn in basis_names
            print(md, "| $bn |")
            for (d, o) in col_keys
                r = get(results, (ds_name, bn, d, o), nothing)
                if r !== nothing && !r.failed
                    @printf(md, " %.1f |", r.time)
                else
                    print(md, " N/A |")
                end
            end
            println(md, "")
        end
        println(md, "")
    end

    # Best configs
    println(md, "## Best Configurations")
    println(md, "")
    for (ds_name, _, _) in datasets
        println(md, "### $ds_name")
        println(md, "")

        best_key = nothing
        best_psnr = -Inf
        for bn in basis_names
            for (d, o) in col_keys
                r = get(results, (ds_name, bn, d, o), nothing)
                if r !== nothing && !r.failed && r.psnr > best_psnr
                    best_psnr = r.psnr
                    best_key = (bn, d, o)
                end
            end
        end

        if best_key !== nothing
            bn, d, o = best_key
            r = results[(ds_name, bn, d, o)]
            println(md, "- **Overall best:** $bn with $o on $d " *
                    "(PSNR: $(@sprintf("%.2f", r.psnr)) dB, " *
                    "SSIM: $(@sprintf("%.4f", r.ssim)), " *
                    "Time: $(@sprintf("%.1f", r.time)) s)")
        end
        println(md, "- **FFT baseline:** PSNR $(@sprintf("%.2f", fft_results[ds_name].psnr)) dB")
        println(md, "")
    end

    open(summary_path, "w") do io
        write(io, String(take!(md)))
    end
    println("\n  Summary saved to: $summary_path")

    # ========================================================================
    # Final
    # ========================================================================

    println("\n" * "=" ^ 120)
    println("    COMPARISON COMPLETE")
    println("    Mode: $(FULL_RUN ? "FULL RUN" : "SMALL RUN")")
    println("    Configs run: $config_num / $total_configs")
    println("    Output: $OUTPUT_DIR")
    if !FULL_RUN
        println("\n    To run the full comparison:")
        println("      julia --project=examples examples/optimizer_basis_comparison.jl --full")
    end
    println("=" ^ 120)

    return results
end

# Run
main()
