# Multi-Dataset Benchmark Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular benchmark suite that trains all 4 basis types (QFT, EntangledQFT, TEBD, MERA) + FFT baseline on 3 datasets (Quick Draw, DIV2K, ATD-12K) and produces publication-quality results.

**Architecture:** Modular scripts under `examples/benchmark/` with shared config, data loading, and evaluation utilities. Each dataset has its own run script. A report generator aggregates results across datasets.

**Tech Stack:** Julia, ParametricDFT.jl, CUDA.jl, CairoMakie, ImageQualityIndexes.jl, NPZ.jl, FFTW.jl, JSON3.jl

**Spec:** `docs/superpowers/specs/2026-03-17-multi-dataset-benchmark-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `examples/benchmark/Project.toml` | Dependencies with path dep to ParametricDFT |
| `examples/benchmark/config.jl` | Training presets, dataset configs, shared constants |
| `examples/benchmark/data_loading.jl` | Dataset loaders: Quick Draw, DIV2K, ATD-12K |
| `examples/benchmark/evaluation.jl` | Metrics, training wrapper, FFT baseline, result I/O |
| `examples/benchmark/run_quickdraw.jl` | Train + evaluate all bases on Quick Draw |
| `examples/benchmark/run_div2k.jl` | Train + evaluate all bases on DIV2K |
| `examples/benchmark/run_atd12k.jl` | Train + evaluate all bases on ATD-12K |
| `examples/benchmark/generate_report.jl` | Cross-dataset summary tables and plots |

---

## Chunk 1: Project Setup and Config

### Task 1: Create Project.toml

**Files:**
- Create: `examples/benchmark/Project.toml`

- [ ] **Step 1: Create the benchmark Project.toml**

```toml
[deps]
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
ImageQualityIndexes = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
NPZ = "15e1cf62-19b3-5cfa-8e77-841668bca605"
ParametricDFT = "cc2eb9de-5297-4754-b0bd-fdc80c6df40d"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[sources]
ParametricDFT = {path = "../.."}
```

- [ ] **Step 2: Verify the project resolves**

Run: `julia --project=examples/benchmark -e 'using Pkg; Pkg.instantiate(); Pkg.status()'`
Expected: All packages resolve, ParametricDFT appears as a dev dependency.

- [ ] **Step 3: Commit**

```bash
git add examples/benchmark/Project.toml
git commit -m "feat(benchmark): add Project.toml for benchmark suite"
```

### Task 2: Create config.jl

**Files:**
- Create: `examples/benchmark/config.jl`

- [ ] **Step 1: Write config.jl**

```julia
# ============================================================================
# Benchmark Configuration
# ============================================================================
# Shared constants, training presets, and dataset configurations for all
# benchmark scripts. Include this file at the top of each run_*.jl script.
# ============================================================================

using ParametricDFT
using Random

# ============================================================================
# Training Presets
# ============================================================================

const TRAINING_PRESETS = Dict(
    :moderate => (
        epochs = 10,
        steps_per_image = 100,
        n_train = 50,
        n_test = 10,
        patience = 3,
        optimizer = :adam,
        validation_split = 0.2,
        device = :gpu,
    ),
    :heavy => (
        epochs = 50,
        steps_per_image = 200,
        n_train = 100,
        n_test = 20,
        patience = 5,
        optimizer = :adam,
        validation_split = 0.2,
        device = :gpu,
    ),
)

# ============================================================================
# Dataset Configurations
# ============================================================================

const DATASET_CONFIGS = Dict(
    :quickdraw => (m = 5, n = 5, img_size = 32),
    :div2k     => (m = 10, n = 10, img_size = 1024),
    :atd12k    => (m = 9, n = 9, img_size = 512),
)

# ============================================================================
# Evaluation Settings
# ============================================================================

const KEEP_RATIOS = [0.05, 0.10, 0.15, 0.20]
const BASIS_TYPES = [QFTBasis, EntangledQFTBasis, TEBDBasis, MERABasis]
const BASIS_NAMES = Dict(
    QFTBasis => "qft",
    EntangledQFTBasis => "entangled_qft",
    TEBDBasis => "tebd",
    MERABasis => "mera",
)

# ============================================================================
# Paths
# ============================================================================

const DATA_DIR = joinpath(@__DIR__, "data")
const RESULTS_DIR = joinpath(@__DIR__, "results")
```

- [ ] **Step 2: Verify config loads**

Run: `julia --project=examples/benchmark -e 'include("examples/benchmark/config.jl"); println("Presets: ", keys(TRAINING_PRESETS)); println("Datasets: ", keys(DATASET_CONFIGS))'`
Expected: Prints preset and dataset keys without error.

- [ ] **Step 3: Commit**

```bash
git add examples/benchmark/config.jl
git commit -m "feat(benchmark): add shared config with presets and dataset configs"
```

---

## Chunk 2: Data Loading

### Task 3: Create data_loading.jl

**Files:**
- Create: `examples/benchmark/data_loading.jl`

This file depends on `config.jl` being included first. It provides three loader functions that all return the same interface: `(train_images, test_images, test_labels)`.

- [ ] **Step 1: Write data_loading.jl with all three loaders**

```julia
# ============================================================================
# Dataset Loading
# ============================================================================
# Provides loader functions for Quick Draw, DIV2K, and ATD-12K datasets.
# All loaders return the same interface:
#   (train_images::Vector{Matrix{Float64}},
#    test_images::Vector{Matrix{Float64}},
#    test_labels::Vector{String})
# All images are normalized to [0, 1] Float64 grayscale.
# ============================================================================

using NPZ
using Downloads
using Images
using FileIO

# ============================================================================
# Quick Draw Loader
# ============================================================================

const QUICKDRAW_CATEGORIES = ["cat", "dog", "airplane", "apple", "bicycle"]
const QUICKDRAW_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

"""
    pad_to_power_of_two(img::AbstractMatrix, target_size::Int)

Center-pad image to target_size x target_size.
"""
function pad_to_power_of_two(img::AbstractMatrix, target_size::Int)
    h, w = size(img)
    padded = zeros(Float64, target_size, target_size)
    y_offset = (target_size - h) ÷ 2 + 1
    x_offset = (target_size - w) ÷ 2 + 1
    padded[y_offset:y_offset+h-1, x_offset:x_offset+w-1] = Float64.(img)
    return padded
end

"""
    load_quickdraw_dataset(; n_train, n_test, img_size=32, seed=42)

Load Quick Draw numpy bitmaps. Auto-downloads if not present.

Returns `(train_images, test_images, test_labels)`.
"""
function load_quickdraw_dataset(; n_train::Int, n_test::Int, img_size::Int = 32, seed::Int = 42)
    quickdraw_dir = joinpath(DATA_DIR, "quickdraw")
    mkpath(quickdraw_dir)

    # Download missing category files
    for category in QUICKDRAW_CATEGORIES
        filepath = joinpath(quickdraw_dir, "$(category).npy")
        if !isfile(filepath)
            url = "$(QUICKDRAW_BASE_URL)/$(category).npy"
            @info "Downloading Quick Draw category: $category" url
            Downloads.download(url, filepath)
        end
    end

    # Load all images
    all_images = Matrix{Float64}[]
    all_labels = String[]

    for category in QUICKDRAW_CATEGORIES
        filepath = joinpath(quickdraw_dir, "$(category).npy")
        data = npzread(filepath)
        n_available = size(data, 1)
        n_to_load = min(n_available, (n_train + n_test) ÷ length(QUICKDRAW_CATEGORIES) + 1)

        for i in 1:n_to_load
            img = reshape(Float64.(data[i, :]), 28, 28) ./ 255.0
            push!(all_images, pad_to_power_of_two(img, img_size))
            push!(all_labels, category)
        end
        @info "Loaded $n_to_load images from $category"
    end

    # Shuffle and split
    Random.seed!(seed)
    indices = randperm(length(all_images))
    train_indices = indices[1:min(n_train, length(indices))]
    test_indices = indices[n_train+1:min(n_train + n_test, length(indices))]

    train_images = all_images[train_indices]
    test_images = all_images[test_indices]
    test_labels = all_labels[test_indices]

    @info "Quick Draw dataset ready" n_train=length(train_images) n_test=length(test_images)
    return train_images, test_images, test_labels
end

# ============================================================================
# DIV2K Loader
# ============================================================================

"""
    center_crop_square(img::AbstractMatrix)

Center-crop image to the largest square that fits.
"""
function center_crop_square(img::AbstractMatrix)
    h, w = size(img)
    side = min(h, w)
    y_start = (h - side) ÷ 2 + 1
    x_start = (w - side) ÷ 2 + 1
    return img[y_start:y_start+side-1, x_start:x_start+side-1]
end

"""
    resize_image(img::AbstractMatrix, target_size::Int)

Resize image to target_size x target_size using bilinear interpolation via Images.jl.
"""
function resize_image(img::AbstractMatrix, target_size::Int)
    return Float64.(imresize(img, (target_size, target_size)))
end

"""
    load_grayscale_image(path::String, target_size::Int)

Load an image file, convert to grayscale, center-crop to square, resize.
"""
function load_grayscale_image(path::String, target_size::Int)
    img = FileIO.load(path)
    gray = Gray.(img)
    gray_matrix = Float64.(channelview(gray))
    cropped = center_crop_square(gray_matrix)
    return resize_image(cropped, target_size)
end

"""
    load_div2k_dataset(; n_train, n_test, img_size=1024, seed=42)

Load DIV2K HR images. Expects data in `data/DIV2K_train_HR/` and/or `data/DIV2K_valid_HR/`.

Returns `(train_images, test_images, test_labels)`.
"""
function load_div2k_dataset(; n_train::Int, n_test::Int, img_size::Int = 1024, seed::Int = 42)
    # Check for DIV2K directories
    train_dir = joinpath(DATA_DIR, "DIV2K_train_HR")
    valid_dir = joinpath(DATA_DIR, "DIV2K_valid_HR")

    all_files = String[]
    for dir in [train_dir, valid_dir]
        if isdir(dir)
            append!(all_files, sort(filter(
                f -> endswith(lowercase(f), ".png"),
                readdir(dir; join = true)
            )))
        end
    end

    if isempty(all_files)
        error("""
        DIV2K dataset not found. Please download:
          mkdir -p $(DATA_DIR)
          cd $(DATA_DIR)
          # Training set (800 images):
          curl -LO https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
          unzip DIV2K_train_HR.zip
          # Validation set (100 images):
          curl -LO https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
          unzip DIV2K_valid_HR.zip
        """)
    end

    @assert length(all_files) >= n_train + n_test "Need $(n_train + n_test) DIV2K images, found $(length(all_files))"

    Random.seed!(seed)
    selected = all_files[randperm(length(all_files))[1:(n_train + n_test)]]

    images = Matrix{Float64}[]
    labels = String[]
    for path in selected
        push!(images, load_grayscale_image(path, img_size))
        push!(labels, basename(path))
    end

    train_images = images[1:n_train]
    test_images = images[n_train+1:end]
    test_labels = labels[n_train+1:end]

    @info "DIV2K dataset ready" n_train=length(train_images) n_test=length(test_images) img_size
    return train_images, test_images, test_labels
end

# ============================================================================
# ATD-12K Loader
# ============================================================================

"""
    load_atd12k_dataset(; n_train, n_test, img_size=512, seed=42)

Load ATD-12K animation frames. Expects data in `data/ATD-12K/`.
Takes the middle frame from each triplet directory.

Returns `(train_images, test_images, test_labels)`.
"""
function load_atd12k_dataset(; n_train::Int, n_test::Int, img_size::Int = 512, seed::Int = 42)
    atd_dir = joinpath(DATA_DIR, "ATD-12K")

    if !isdir(atd_dir)
        error("""
        ATD-12K dataset not found. Please download from:
          https://github.com/lisiyao21/AnimeInterp
        Extract to: $(atd_dir)
        Expected structure: $(atd_dir)/test_2k_540p/<triplet_dir>/{frame1,frame2,frame3}.png
        """)
    end

    # Find triplet directories and select the middle frame from each.
    # Each triplet directory contains 3 frames sorted alphabetically;
    # the middle one (index 2) is the target interpolation frame.
    all_image_files = String[]
    for (root, dirs, files) in walkdir(atd_dir)
        image_files = sort(filter(
            f -> endswith(lowercase(f), ".png") || endswith(lowercase(f), ".jpg"),
            files,
        ))
        if length(image_files) == 3
            # Triplet directory — take middle frame
            push!(all_image_files, joinpath(root, image_files[2]))
        elseif length(image_files) >= 1 && isempty(dirs)
            # Leaf directory with images but not a triplet — take all
            for f in image_files
                push!(all_image_files, joinpath(root, f))
            end
        end
    end
    sort!(all_image_files)

    if isempty(all_image_files)
        error("No image files found in $(atd_dir). Check dataset structure.")
    end

    @assert length(all_image_files) >= n_train + n_test "Need $(n_train + n_test) ATD-12K images, found $(length(all_image_files))"

    Random.seed!(seed)
    selected = all_image_files[randperm(length(all_image_files))[1:(n_train + n_test)]]

    images = Matrix{Float64}[]
    labels = String[]
    for path in selected
        push!(images, load_grayscale_image(path, img_size))
        push!(labels, basename(path))
    end

    train_images = images[1:n_train]
    test_images = images[n_train+1:end]
    test_labels = labels[n_train+1:end]

    @info "ATD-12K dataset ready" n_train=length(train_images) n_test=length(test_images) img_size
    return train_images, test_images, test_labels
end
```

- [ ] **Step 2: Verify data_loading.jl parses without error**

Run: `julia --project=examples/benchmark -e 'include("examples/benchmark/config.jl"); include("examples/benchmark/data_loading.jl"); println("Data loading module OK")'`
Expected: Prints "Data loading module OK" (won't load data, just verifies syntax).

- [ ] **Step 3: Commit**

```bash
git add examples/benchmark/data_loading.jl
git commit -m "feat(benchmark): add dataset loaders for Quick Draw, DIV2K, ATD-12K"
```

---

## Chunk 3: Evaluation Utilities

### Task 4: Create evaluation.jl

**Files:**
- Create: `examples/benchmark/evaluation.jl`

This file depends on `config.jl`. It provides metrics computation, training wrapper, FFT baseline, and result I/O.

- [ ] **Step 1: Write evaluation.jl**

```julia
# ============================================================================
# Benchmark Evaluation Utilities
# ============================================================================
# Provides metrics computation, training wrapper with timing, FFT baseline,
# and result serialization. Include after config.jl.
# ============================================================================

using ImageQualityIndexes: assess_psnr, assess_ssim
using FFTW
using JSON3
using Statistics
using Printf

# ============================================================================
# Metrics
# ============================================================================

"""
    compute_metrics(original::AbstractMatrix, recovered::AbstractMatrix)

Compute MSE, PSNR, and SSIM between original and recovered images.
Returns a NamedTuple `(mse, psnr, ssim)`.
"""
function compute_metrics(original::AbstractMatrix, recovered::AbstractMatrix)
    recovered_clamped = clamp.(real.(recovered), 0.0, 1.0)
    mse = mean((original .- recovered_clamped) .^ 2)
    psnr = mse > 0 ? 10 * log10(1.0 / mse) : Inf
    ssim = assess_ssim(Gray.(original), Gray.(recovered_clamped))
    return (mse = mse, psnr = psnr, ssim = ssim)
end

# ============================================================================
# Basis Evaluation
# ============================================================================

"""
    evaluate_basis(basis, test_images, keep_ratios)

Evaluate a trained basis at multiple compression ratios on a test set.

Returns `Dict(keep_ratio => (mean_mse, std_mse, mean_psnr, std_psnr, mean_ssim, std_ssim))`.

Note: `keep_ratio` is the fraction of coefficients *kept*.
The `compress()` function's `ratio` parameter is the fraction *discarded*,
so we pass `ratio = 1.0 - keep_ratio`.
"""
function evaluate_basis(basis, test_images::Vector{<:AbstractMatrix}, keep_ratios::Vector{Float64})
    results = Dict{Float64,NamedTuple}()

    for keep_ratio in keep_ratios
        discard_ratio = 1.0 - keep_ratio
        mse_vals, psnr_vals, ssim_vals = Float64[], Float64[], Float64[]

        for img in test_images
            compressed = compress(basis, img; ratio = discard_ratio)
            recovered = recover(basis, compressed)
            metrics = compute_metrics(img, recovered)
            push!(mse_vals, metrics.mse)
            push!(psnr_vals, metrics.psnr)
            push!(ssim_vals, metrics.ssim)
        end

        results[keep_ratio] = (
            mean_mse = mean(mse_vals), std_mse = std(mse_vals),
            mean_psnr = mean(psnr_vals), std_psnr = std(psnr_vals),
            mean_ssim = mean(ssim_vals), std_ssim = std(ssim_vals),
        )
    end

    return results
end

# ============================================================================
# FFT Baseline
# ============================================================================

"""
    fft_compress_recover(img::AbstractMatrix, keep_ratio::Float64)

Compress and recover an image using classical FFT.
"""
function fft_compress_recover(img::AbstractMatrix, keep_ratio::Float64)
    freq = fftshift(fft(img))
    total = length(freq)
    keep = max(1, round(Int, total * keep_ratio))

    flat = vec(freq)
    idx = partialsortperm(abs.(flat), 1:keep, rev = true)
    compressed = zeros(ComplexF64, size(freq))
    compressed[idx] = freq[idx]

    return real.(ifft(ifftshift(compressed)))
end

"""
    evaluate_fft_baseline_timed(test_images, keep_ratios)

Evaluate classical FFT baseline at multiple ratios. Returns `(metrics_dict, elapsed_seconds)`.
"""
function evaluate_fft_baseline_timed(test_images::Vector{<:AbstractMatrix}, keep_ratios::Vector{Float64})
    elapsed = @elapsed begin
        results = Dict{Float64,NamedTuple}()
        for keep_ratio in keep_ratios
            mse_vals, psnr_vals, ssim_vals = Float64[], Float64[], Float64[]
            for img in test_images
                recovered = fft_compress_recover(img, keep_ratio)
                metrics = compute_metrics(img, recovered)
                push!(mse_vals, metrics.mse)
                push!(psnr_vals, metrics.psnr)
                push!(ssim_vals, metrics.ssim)
            end
            results[keep_ratio] = (
                mean_mse = mean(mse_vals), std_mse = std(mse_vals),
                mean_psnr = mean(psnr_vals), std_psnr = std(psnr_vals),
                mean_ssim = mean(ssim_vals), std_ssim = std(ssim_vals),
            )
        end
    end
    return results, elapsed
end

# ============================================================================
# Training Wrapper
# ============================================================================

"""
    train_and_time(BasisType, dataset, dataset_config, preset)

Train a basis with timing. Returns `(trained_basis, history, elapsed_seconds)`.

Sets `Random.seed!(42)` before training for reproducibility.
Computes `k = round(Int, 0.10 * img_size^2)` for MSELoss.
"""
function train_and_time(
    BasisType::Type{<:AbstractSparseBasis},
    dataset::Vector{<:AbstractMatrix},
    dataset_config::NamedTuple,
    preset::NamedTuple;
    save_loss_path::Union{Nothing,String} = nothing,
)
    m, n = dataset_config.m, dataset_config.n
    k = round(Int, 0.10 * dataset_config.img_size^2)

    Random.seed!(42)
    elapsed = @elapsed begin
        basis, history = train_basis(
            BasisType, dataset;
            m = m, n = n,
            loss = MSELoss(k),
            epochs = preset.epochs,
            steps_per_image = preset.steps_per_image,
            validation_split = preset.validation_split,
            early_stopping_patience = preset.patience,
            optimizer = preset.optimizer,
            device = preset.device,
            save_loss_path = save_loss_path,
        )
    end

    return basis, history, elapsed
end

# ============================================================================
# Result I/O
# ============================================================================

"""
    save_benchmark_results(path, results_dict)

Save benchmark results (metrics + timing) to JSON.
"""
function save_benchmark_results(path::String, results_dict::Dict)
    mkpath(dirname(path))
    # Convert to serializable format
    serializable = Dict{String,Any}()
    for (name, data) in results_dict
        entry = Dict{String,Any}()
        if haskey(data, :metrics)
            # Convert metric keys from Float64 to String for JSON
            metrics_serializable = Dict{String,Any}()
            for (ratio, vals) in data[:metrics]
                metrics_serializable[string(ratio)] = Dict(pairs(vals))
            end
            entry["metrics"] = metrics_serializable
        end
        if haskey(data, :time)
            entry["time"] = data[:time]
        end
        if haskey(data, :history)
            h = data[:history]
            entry["history"] = Dict(
                "train_losses" => h.train_losses,
                "val_losses" => h.val_losses,
                "step_train_losses" => h.step_train_losses,
                "basis_name" => h.basis_name,
            )
        end
        serializable[name] = entry
    end

    open(path, "w") do io
        JSON3.pretty(io, serializable)
    end
    @info "Results saved to $path"
end

"""
    load_benchmark_results(path)

Load benchmark results from JSON.
"""
function load_benchmark_results(path::String)
    return JSON3.read(read(path, String))
end

# ============================================================================
# Summary Printing
# ============================================================================

"""
    print_dataset_summary(results, keep_ratios)

Print a formatted comparison table for a single dataset.
"""
function print_dataset_summary(results::Dict, keep_ratios::Vector{Float64})
    println("\n" * "=" ^ 100)
    println("COMPRESSION QUALITY COMPARISON")
    println("=" ^ 100)

    # Collect basis names
    basis_names = sort(collect(keys(results)))

    # PSNR table
    println("\nPSNR (dB) — higher is better:")
    println("-" ^ 100)
    @printf("%-25s", "Basis")
    for ratio in keep_ratios
        @printf(" | %10s", "$(round(Int, ratio * 100))%% kept")
    end
    @printf(" | %12s\n", "Train time")
    println("-" ^ 100)

    for name in basis_names
        data = results[name]
        @printf("%-25s", name)
        for ratio in keep_ratios
            if haskey(data[:metrics], ratio)
                @printf(" | %10.2f", data[:metrics][ratio].mean_psnr)
            else
                @printf(" | %10s", "N/A")
            end
        end
        @printf(" | %10.1fs\n", data[:time])
    end

    # SSIM table
    println("\nSSIM — higher is better:")
    println("-" ^ 100)
    @printf("%-25s", "Basis")
    for ratio in keep_ratios
        @printf(" | %10s", "$(round(Int, ratio * 100))%% kept")
    end
    println()
    println("-" ^ 100)

    for name in basis_names
        data = results[name]
        @printf("%-25s", name)
        for ratio in keep_ratios
            if haskey(data[:metrics], ratio)
                @printf(" | %10.4f", data[:metrics][ratio].mean_ssim)
            else
                @printf(" | %10s", "N/A")
            end
        end
        println()
    end
    println("=" ^ 100)
end
```

- [ ] **Step 2: Verify evaluation.jl parses without error**

Run: `julia --project=examples/benchmark -e 'include("examples/benchmark/config.jl"); include("examples/benchmark/evaluation.jl"); println("Evaluation module OK")'`
Expected: Prints "Evaluation module OK".

- [ ] **Step 3: Commit**

```bash
git add examples/benchmark/evaluation.jl
git commit -m "feat(benchmark): add evaluation utilities — metrics, FFT baseline, result I/O"
```

---

## Chunk 4: Run Scripts

### Task 5: Create run_quickdraw.jl

**Files:**
- Create: `examples/benchmark/run_quickdraw.jl`

- [ ] **Step 1: Write run_quickdraw.jl**

```julia
# ============================================================================
# Benchmark: Quick Draw Dataset
# ============================================================================
# Train all 4 basis types on Quick Draw and evaluate compression quality.
#
# Usage:
#   julia --project=examples/benchmark examples/benchmark/run_quickdraw.jl moderate
#   julia --project=examples/benchmark examples/benchmark/run_quickdraw.jl heavy
# ============================================================================

include("config.jl")
include("data_loading.jl")
include("evaluation.jl")

const DATASET_NAME = :quickdraw
const DATASET_CONFIG = DATASET_CONFIGS[DATASET_NAME]

# Parse CLI preset
preset_name = length(ARGS) > 0 ? Symbol(ARGS[1]) : :moderate
@assert haskey(TRAINING_PRESETS, preset_name) "Unknown preset: $preset_name. Use :moderate or :heavy"
preset = TRAINING_PRESETS[preset_name]

println("=" ^ 80)
println("Benchmark: Quick Draw | Preset: $preset_name")
println("Image size: $(DATASET_CONFIG.img_size)x$(DATASET_CONFIG.img_size) | Qubits: m=$(DATASET_CONFIG.m), n=$(DATASET_CONFIG.n)")
println("=" ^ 80)

# ============================================================================
# Step 1: Load Dataset
# ============================================================================

println("\nStep 1: Loading Quick Draw dataset...")
train_images, test_images, test_labels = load_quickdraw_dataset(;
    n_train = preset.n_train,
    n_test = preset.n_test,
    img_size = DATASET_CONFIG.img_size,
)

# ============================================================================
# Step 2: Train and Evaluate All Bases
# ============================================================================

output_dir = joinpath(RESULTS_DIR, string(DATASET_NAME))
loss_dir = joinpath(output_dir, "loss_history")
mkpath(loss_dir)

results = Dict{String,Dict{Symbol,Any}}()

for BasisType in BASIS_TYPES
    basis_name = BASIS_NAMES[BasisType]
    basis_path = joinpath(output_dir, "trained_$(basis_name).json")

    # Resume: skip if already trained
    if isfile(basis_path)
        @info "Skipping $basis_name — already trained at $basis_path"
        loaded_basis = load_basis(basis_path)
        metrics = evaluate_basis(loaded_basis, test_images, KEEP_RATIOS)
        results[basis_name] = Dict(:metrics => metrics, :time => 0.0)
        continue
    end

    println("\n--- Training $(BasisType) ---")
    loss_path = joinpath(loss_dir, "$(basis_name)_loss.json")

    basis, history, elapsed = train_and_time(
        BasisType, train_images, DATASET_CONFIG, preset;
        save_loss_path = loss_path,
    )

    # Save trained basis
    mkpath(output_dir)
    save_basis(basis_path, basis)
    @info "Saved trained $basis_name" path=basis_path time=round(elapsed; digits=1)

    # Evaluate
    metrics = evaluate_basis(basis, test_images, KEEP_RATIOS)
    results[basis_name] = Dict(:metrics => metrics, :time => elapsed, :history => history)
end

# ============================================================================
# Step 3: FFT Baseline
# ============================================================================

println("\n--- FFT Baseline ---")
fft_metrics, fft_time = evaluate_fft_baseline_timed(test_images, KEEP_RATIOS)
results["fft"] = Dict(:metrics => fft_metrics, :time => fft_time)

# ============================================================================
# Step 4: Save and Print Results
# ============================================================================

save_benchmark_results(joinpath(output_dir, "metrics.json"), results)
print_dataset_summary(results, KEEP_RATIOS)

println("\nBenchmark complete. Results saved to: $output_dir")
```

- [ ] **Step 2: Commit**

```bash
git add examples/benchmark/run_quickdraw.jl
git commit -m "feat(benchmark): add Quick Draw run script"
```

### Task 6: Create run_div2k.jl

**Files:**
- Create: `examples/benchmark/run_div2k.jl`

- [ ] **Step 1: Write run_div2k.jl**

Same structure as `run_quickdraw.jl` but with:
- `DATASET_NAME = :div2k`
- Calls `load_div2k_dataset()`

```julia
# ============================================================================
# Benchmark: DIV2K Dataset
# ============================================================================
# Train all 4 basis types on DIV2K and evaluate compression quality.
#
# Usage:
#   julia --project=examples/benchmark examples/benchmark/run_div2k.jl moderate
#   julia --project=examples/benchmark examples/benchmark/run_div2k.jl heavy
# ============================================================================

include("config.jl")
include("data_loading.jl")
include("evaluation.jl")

const DATASET_NAME = :div2k
const DATASET_CONFIG = DATASET_CONFIGS[DATASET_NAME]

preset_name = length(ARGS) > 0 ? Symbol(ARGS[1]) : :moderate
@assert haskey(TRAINING_PRESETS, preset_name) "Unknown preset: $preset_name. Use :moderate or :heavy"
preset = TRAINING_PRESETS[preset_name]

println("=" ^ 80)
println("Benchmark: DIV2K | Preset: $preset_name")
println("Image size: $(DATASET_CONFIG.img_size)x$(DATASET_CONFIG.img_size) | Qubits: m=$(DATASET_CONFIG.m), n=$(DATASET_CONFIG.n)")
println("=" ^ 80)

# Load dataset
println("\nStep 1: Loading DIV2K dataset...")
train_images, test_images, test_labels = load_div2k_dataset(;
    n_train = preset.n_train,
    n_test = preset.n_test,
    img_size = DATASET_CONFIG.img_size,
)

# Train and evaluate
output_dir = joinpath(RESULTS_DIR, string(DATASET_NAME))
loss_dir = joinpath(output_dir, "loss_history")
mkpath(loss_dir)

results = Dict{String,Dict{Symbol,Any}}()

for BasisType in BASIS_TYPES
    basis_name = BASIS_NAMES[BasisType]
    basis_path = joinpath(output_dir, "trained_$(basis_name).json")

    if isfile(basis_path)
        @info "Skipping $basis_name — already trained at $basis_path"
        loaded_basis = load_basis(basis_path)
        metrics = evaluate_basis(loaded_basis, test_images, KEEP_RATIOS)
        results[basis_name] = Dict(:metrics => metrics, :time => 0.0)
        continue
    end

    println("\n--- Training $(BasisType) ---")
    loss_path = joinpath(loss_dir, "$(basis_name)_loss.json")

    basis, history, elapsed = train_and_time(
        BasisType, train_images, DATASET_CONFIG, preset;
        save_loss_path = loss_path,
    )

    mkpath(output_dir)
    save_basis(basis_path, basis)
    @info "Saved trained $basis_name" path=basis_path time=round(elapsed; digits=1)

    metrics = evaluate_basis(basis, test_images, KEEP_RATIOS)
    results[basis_name] = Dict(:metrics => metrics, :time => elapsed, :history => history)
end

# FFT Baseline
println("\n--- FFT Baseline ---")
fft_metrics, fft_time = evaluate_fft_baseline_timed(test_images, KEEP_RATIOS)
results["fft"] = Dict(:metrics => fft_metrics, :time => fft_time)

# Save and print
save_benchmark_results(joinpath(output_dir, "metrics.json"), results)
print_dataset_summary(results, KEEP_RATIOS)
println("\nBenchmark complete. Results saved to: $output_dir")
```

- [ ] **Step 2: Commit**

```bash
git add examples/benchmark/run_div2k.jl
git commit -m "feat(benchmark): add DIV2K run script"
```

### Task 7: Create run_atd12k.jl

**Files:**
- Create: `examples/benchmark/run_atd12k.jl`

- [ ] **Step 1: Write run_atd12k.jl**

Same structure, with `DATASET_NAME = :atd12k` and `load_atd12k_dataset()`:

```julia
# ============================================================================
# Benchmark: ATD-12K Dataset
# ============================================================================
# Train all 4 basis types on ATD-12K and evaluate compression quality.
#
# Usage:
#   julia --project=examples/benchmark examples/benchmark/run_atd12k.jl moderate
#   julia --project=examples/benchmark examples/benchmark/run_atd12k.jl heavy
# ============================================================================

include("config.jl")
include("data_loading.jl")
include("evaluation.jl")

const DATASET_NAME = :atd12k
const DATASET_CONFIG = DATASET_CONFIGS[DATASET_NAME]

preset_name = length(ARGS) > 0 ? Symbol(ARGS[1]) : :moderate
@assert haskey(TRAINING_PRESETS, preset_name) "Unknown preset: $preset_name. Use :moderate or :heavy"
preset = TRAINING_PRESETS[preset_name]

println("=" ^ 80)
println("Benchmark: ATD-12K | Preset: $preset_name")
println("Image size: $(DATASET_CONFIG.img_size)x$(DATASET_CONFIG.img_size) | Qubits: m=$(DATASET_CONFIG.m), n=$(DATASET_CONFIG.n)")
println("=" ^ 80)

# Load dataset
println("\nStep 1: Loading ATD-12K dataset...")
train_images, test_images, test_labels = load_atd12k_dataset(;
    n_train = preset.n_train,
    n_test = preset.n_test,
    img_size = DATASET_CONFIG.img_size,
)

# Train and evaluate
output_dir = joinpath(RESULTS_DIR, string(DATASET_NAME))
loss_dir = joinpath(output_dir, "loss_history")
mkpath(loss_dir)

results = Dict{String,Dict{Symbol,Any}}()

for BasisType in BASIS_TYPES
    basis_name = BASIS_NAMES[BasisType]
    basis_path = joinpath(output_dir, "trained_$(basis_name).json")

    if isfile(basis_path)
        @info "Skipping $basis_name — already trained at $basis_path"
        loaded_basis = load_basis(basis_path)
        metrics = evaluate_basis(loaded_basis, test_images, KEEP_RATIOS)
        results[basis_name] = Dict(:metrics => metrics, :time => 0.0)
        continue
    end

    println("\n--- Training $(BasisType) ---")
    loss_path = joinpath(loss_dir, "$(basis_name)_loss.json")

    basis, history, elapsed = train_and_time(
        BasisType, train_images, DATASET_CONFIG, preset;
        save_loss_path = loss_path,
    )

    mkpath(output_dir)
    save_basis(basis_path, basis)
    @info "Saved trained $basis_name" path=basis_path time=round(elapsed; digits=1)

    metrics = evaluate_basis(basis, test_images, KEEP_RATIOS)
    results[basis_name] = Dict(:metrics => metrics, :time => elapsed, :history => history)
end

# FFT Baseline
println("\n--- FFT Baseline ---")
fft_metrics, fft_time = evaluate_fft_baseline_timed(test_images, KEEP_RATIOS)
results["fft"] = Dict(:metrics => fft_metrics, :time => fft_time)

# Save and print
save_benchmark_results(joinpath(output_dir, "metrics.json"), results)
print_dataset_summary(results, KEEP_RATIOS)
println("\nBenchmark complete. Results saved to: $output_dir")
```

- [ ] **Step 2: Commit**

```bash
git add examples/benchmark/run_atd12k.jl
git commit -m "feat(benchmark): add ATD-12K run script"
```

---

## Chunk 5: Report Generation

### Task 8: Create generate_report.jl

**Files:**
- Create: `examples/benchmark/generate_report.jl`

- [ ] **Step 1: Write generate_report.jl**

```julia
# ============================================================================
# Benchmark Report Generator
# ============================================================================
# Loads results from all dataset benchmarks and produces:
# 1. Rate-distortion tables (CSV)
# 2. Training loss curves (PNG)
# 3. Visual comparison grids (PNG)
# 4. Cross-dataset summary table (CSV)
# 5. Timing table (CSV)
#
# Usage:
#   julia --project=examples/benchmark examples/benchmark/generate_report.jl
# ============================================================================

include("config.jl")
include("data_loading.jl")
include("evaluation.jl")

using CairoMakie
using ParametricDFT

const DATASET_NAMES = [:quickdraw, :div2k, :atd12k]
const DISPLAY_NAMES = Dict(
    :quickdraw => "Quick Draw",
    :div2k => "DIV2K",
    :atd12k => "ATD-12K",
)
const BASIS_DISPLAY_NAMES = Dict(
    "qft" => "QFT",
    "entangled_qft" => "Entangled QFT",
    "tebd" => "TEBD",
    "mera" => "MERA",
    "fft" => "Classical FFT",
)
const BASIS_COLORS = Dict(
    "qft" => :blue,
    "entangled_qft" => :red,
    "tebd" => :green,
    "mera" => :purple,
    "fft" => :black,
)

# ============================================================================
# Load Results
# ============================================================================

function load_all_results()
    all_results = Dict{Symbol,Any}()
    for dataset_name in DATASET_NAMES
        metrics_path = joinpath(RESULTS_DIR, string(dataset_name), "metrics.json")
        if isfile(metrics_path)
            all_results[dataset_name] = load_benchmark_results(metrics_path)
            @info "Loaded results for $(DISPLAY_NAMES[dataset_name])"
        else
            @warn "No results found for $(DISPLAY_NAMES[dataset_name]) at $metrics_path"
        end
    end
    return all_results
end

# ============================================================================
# 1. Rate-Distortion Tables (CSV)
# ============================================================================

function generate_rate_distortion_csv(all_results)
    for (dataset_name, results) in all_results
        output_dir = joinpath(RESULTS_DIR, string(dataset_name))
        mkpath(output_dir)

        for metric_name in ["psnr", "ssim", "mse"]
            csv_path = joinpath(output_dir, "rate_distortion_$(metric_name).csv")
            open(csv_path, "w") do io
                # Header
                print(io, "Basis")
                for ratio in KEEP_RATIOS
                    print(io, ",$(round(Int, ratio * 100))%_kept")
                end
                println(io)

                # Rows
                for basis_name in ["qft", "entangled_qft", "tebd", "mera", "fft"]
                    if haskey(results, basis_name)
                        print(io, BASIS_DISPLAY_NAMES[basis_name])
                        basis_data = results[basis_name]
                        metrics = basis_data["metrics"]
                        for ratio in KEEP_RATIOS
                            ratio_key = string(ratio)
                            if haskey(metrics, ratio_key)
                                val = metrics[ratio_key]["mean_$(metric_name)"]
                                print(io, ",$(val)")
                            else
                                print(io, ",N/A")
                            end
                        end
                        println(io)
                    end
                end
            end
            @info "Saved $csv_path"
        end
    end
end

# ============================================================================
# 2. Training Loss Curves
# ============================================================================

function generate_training_curves(all_results)
    for (dataset_name, results) in all_results
        plots_dir = joinpath(RESULTS_DIR, string(dataset_name), "plots")
        mkpath(plots_dir)

        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1];
            xlabel = "Epoch",
            ylabel = "Validation Loss",
            title = "Training Convergence — $(DISPLAY_NAMES[dataset_name])",
            yscale = log10,
        )

        for basis_name in ["qft", "entangled_qft", "tebd", "mera"]
            if haskey(results, basis_name) && haskey(results[basis_name], "history")
                history = results[basis_name]["history"]
                val_losses = Float64.(history["val_losses"])
                if !isempty(val_losses)
                    lines!(ax, 1:length(val_losses), val_losses;
                        label = BASIS_DISPLAY_NAMES[basis_name],
                        color = BASIS_COLORS[basis_name],
                    )
                end
            end
        end

        axislegend(ax; position = :rt)
        save(joinpath(plots_dir, "training_curves.png"), fig; px_per_unit = 2)
        @info "Saved training curves for $(DISPLAY_NAMES[dataset_name])"
    end
end

# ============================================================================
# 3. Visual Comparison Grids
# ============================================================================

function generate_reconstruction_grids(all_results)
    for (dataset_name, results) in all_results
        plots_dir = joinpath(RESULTS_DIR, string(dataset_name), "plots")
        mkpath(plots_dir)
        output_dir = joinpath(RESULTS_DIR, string(dataset_name))

        # Load first test image using the appropriate loader
        dataset_config = DATASET_CONFIGS[dataset_name]
        # We need a test image — load just 1
        test_images = try
            if dataset_name == :quickdraw
                _, test, _ = load_quickdraw_dataset(; n_train = 1, n_test = 1)
                test
            elseif dataset_name == :div2k
                _, test, _ = load_div2k_dataset(; n_train = 1, n_test = 1)
                test
            else
                _, test, _ = load_atd12k_dataset(; n_train = 1, n_test = 1)
                test
            end
        catch e
            @warn "Could not load test image for $dataset_name: $e"
            continue
        end

        sample_img = test_images[1]
        basis_order = ["qft", "entangled_qft", "tebd", "mera", "fft"]

        # Load trained bases
        trained_bases = Dict{String,Any}()
        for basis_name in ["qft", "entangled_qft", "tebd", "mera"]
            basis_path = joinpath(output_dir, "trained_$(basis_name).json")
            if isfile(basis_path)
                trained_bases[basis_name] = load_basis(basis_path)
            end
        end

        n_rows = 1 + length(basis_order)  # original + each basis
        n_cols = length(KEEP_RATIOS)

        fig = Figure(size = (250 * n_cols, 200 * n_rows))

        # Column headers
        for (j, ratio) in enumerate(KEEP_RATIOS)
            Label(fig[0, j], "$(round(Int, ratio * 100))% kept"; fontsize = 14)
        end

        # Original row
        Label(fig[1, 0], "Original"; fontsize = 12, rotation = pi / 2)
        for j in 1:n_cols
            ax = Axis(fig[1, j]; aspect = DataAspect())
            hidedecorations!(ax)
            heatmap!(ax, rotr90(sample_img); colormap = :grays)
        end

        # Basis rows
        for (i, basis_name) in enumerate(basis_order)
            row = i + 1
            Label(fig[row, 0], get(BASIS_DISPLAY_NAMES, basis_name, basis_name);
                fontsize = 12, rotation = pi / 2)

            for (j, keep_ratio) in enumerate(KEEP_RATIOS)
                ax = Axis(fig[row, j]; aspect = DataAspect())
                hidedecorations!(ax)

                recovered = if basis_name == "fft"
                    fft_compress_recover(sample_img, keep_ratio)
                elseif haskey(trained_bases, basis_name)
                    basis = trained_bases[basis_name]
                    compressed = compress(basis, sample_img; ratio = 1.0 - keep_ratio)
                    real.(recover(basis, compressed))
                else
                    zeros(size(sample_img))
                end

                heatmap!(ax, rotr90(clamp.(recovered, 0.0, 1.0)); colormap = :grays)
            end
        end

        save(joinpath(plots_dir, "reconstruction_grid.png"), fig; px_per_unit = 2)
        @info "Saved reconstruction grid for $(DISPLAY_NAMES[dataset_name])"
    end
end

# ============================================================================
# 4. Cross-Dataset Summary Table
# ============================================================================

function generate_cross_dataset_summary(all_results)
    csv_path = joinpath(RESULTS_DIR, "cross_dataset_summary.csv")

    open(csv_path, "w") do io
        # Header
        print(io, "Basis")
        for dataset_name in DATASET_NAMES
            if haskey(all_results, dataset_name)
                print(io, ",$(DISPLAY_NAMES[dataset_name]) PSNR@10%")
            end
        end
        println(io, ",Avg Rank")

        basis_order = ["qft", "entangled_qft", "tebd", "mera", "fft"]

        # Compute ranks per dataset
        ranks = Dict{String,Vector{Float64}}()
        for basis_name in basis_order
            ranks[basis_name] = Float64[]
        end

        for dataset_name in DATASET_NAMES
            haskey(all_results, dataset_name) || continue
            results = all_results[dataset_name]

            # Get PSNR@10% for each basis
            psnr_values = Dict{String,Float64}()
            for basis_name in basis_order
                if haskey(results, basis_name)
                    metrics = results[basis_name]["metrics"]
                    if haskey(metrics, "0.1")
                        psnr_values[basis_name] = Float64(metrics["0.1"]["mean_psnr"])
                    end
                end
            end

            # Rank by PSNR (higher = better = lower rank)
            sorted = sort(collect(psnr_values); by = x -> -x[2])
            for (rank, (name, _)) in enumerate(sorted)
                push!(ranks[name], Float64(rank))
            end
        end

        # Write rows
        for basis_name in basis_order
            print(io, BASIS_DISPLAY_NAMES[basis_name])
            for dataset_name in DATASET_NAMES
                if haskey(all_results, dataset_name) && haskey(all_results[dataset_name], basis_name)
                    metrics = all_results[dataset_name][basis_name]["metrics"]
                    if haskey(metrics, "0.1")
                        print(io, ",$(Float64(metrics["0.1"]["mean_psnr"]))")
                    else
                        print(io, ",N/A")
                    end
                else
                    print(io, ",N/A")
                end
            end
            avg_rank = isempty(ranks[basis_name]) ? NaN : mean(ranks[basis_name])
            println(io, ",$avg_rank")
        end
    end

    @info "Saved cross-dataset summary to $csv_path"

    # Also print to console
    println("\n" * "=" ^ 80)
    println("CROSS-DATASET SUMMARY (PSNR @ 10% kept)")
    println("=" ^ 80)
    println(read(csv_path, String))
end

# ============================================================================
# 5. Timing Table
# ============================================================================

function generate_timing_table(all_results)
    csv_path = joinpath(RESULTS_DIR, "timing_summary.csv")

    open(csv_path, "w") do io
        print(io, "Basis")
        for dataset_name in DATASET_NAMES
            if haskey(all_results, dataset_name)
                print(io, ",$(DISPLAY_NAMES[dataset_name]) Time(s)")
            end
        end
        println(io)

        for basis_name in ["qft", "entangled_qft", "tebd", "mera", "fft"]
            print(io, BASIS_DISPLAY_NAMES[basis_name])
            for dataset_name in DATASET_NAMES
                if haskey(all_results, dataset_name) && haskey(all_results[dataset_name], basis_name)
                    t = all_results[dataset_name][basis_name]["time"]
                    @printf(io, ",%.1f", Float64(t))
                else
                    print(io, ",N/A")
                end
            end
            println(io)
        end
    end

    @info "Saved timing summary to $csv_path"
    println("\n" * "=" ^ 80)
    println("TIMING SUMMARY")
    println("=" ^ 80)
    println(read(csv_path, String))
end

# ============================================================================
# Cross-Dataset Plots
# ============================================================================

function generate_cross_dataset_plots(all_results)
    plots_dir = joinpath(RESULTS_DIR, "plots")
    mkpath(plots_dir)

    basis_order = ["qft", "entangled_qft", "tebd", "mera", "fft"]
    available_datasets = [d for d in DATASET_NAMES if haskey(all_results, d)]

    for (metric_name, ylabel, higher_better) in [
        ("psnr", "PSNR (dB)", true),
        ("ssim", "SSIM", true),
    ]
        fig = Figure(size = (800, 500))
        ax = Axis(fig[1, 1];
            xlabel = "Dataset",
            ylabel = ylabel,
            title = "Cross-Dataset Comparison — $(uppercase(metric_name)) @ 10% kept",
            xticks = (1:length(available_datasets), [DISPLAY_NAMES[d] for d in available_datasets]),
        )

        n_bases = length(basis_order)
        bar_width = 0.15

        for (bi, basis_name) in enumerate(basis_order)
            values = Float64[]
            positions = Float64[]
            for (di, dataset_name) in enumerate(available_datasets)
                if haskey(all_results[dataset_name], basis_name)
                    metrics = all_results[dataset_name][basis_name]["metrics"]
                    if haskey(metrics, "0.1")
                        push!(values, Float64(metrics["0.1"]["mean_$(metric_name)"]))
                        push!(positions, di + (bi - (n_bases + 1) / 2) * bar_width)
                    end
                end
            end
            if !isempty(values)
                barplot!(ax, positions, values;
                    width = bar_width,
                    color = BASIS_COLORS[basis_name],
                    label = BASIS_DISPLAY_NAMES[basis_name],
                )
            end
        end

        axislegend(ax; position = :rt)
        save(joinpath(plots_dir, "cross_dataset_$(metric_name).png"), fig; px_per_unit = 2)
        @info "Saved cross-dataset $(metric_name) plot"
    end
end

# ============================================================================
# Main
# ============================================================================

function main()
    println("=" ^ 80)
    println("Generating Benchmark Report")
    println("=" ^ 80)

    all_results = load_all_results()

    if isempty(all_results)
        error("No results found. Run the benchmark scripts first.")
    end

    generate_rate_distortion_csv(all_results)
    generate_training_curves(all_results)
    generate_reconstruction_grids(all_results)
    generate_cross_dataset_summary(all_results)
    generate_cross_dataset_plots(all_results)
    generate_timing_table(all_results)

    println("\n" * "=" ^ 80)
    println("Report generation complete!")
    println("Results in: $RESULTS_DIR")
    println("=" ^ 80)
end

main()
```

- [ ] **Step 2: Verify generate_report.jl parses without error**

Run: `julia --project=examples/benchmark -e 'include("examples/benchmark/config.jl"); include("examples/benchmark/data_loading.jl"); include("examples/benchmark/evaluation.jl"); println("All modules OK")'`
Expected: Prints "All modules OK".

- [ ] **Step 3: Commit**

```bash
git add examples/benchmark/generate_report.jl
git commit -m "feat(benchmark): add report generator — tables, plots, cross-dataset summary"
```

---

## Chunk 6: Git Ignore and Final Verification

### Task 9: Update .gitignore and verify full pipeline parses

**Files:**
- Modify: `examples/.gitignore` (or create `examples/benchmark/.gitignore`)

- [ ] **Step 1: Create .gitignore for benchmark outputs**

```gitignore
# Benchmark data and results (large files, not tracked)
data/
results/
```

Save to `examples/benchmark/.gitignore`.

- [ ] **Step 2: Verify all scripts parse**

Run each in sequence:
```bash
julia --project=examples/benchmark -e '
    include("examples/benchmark/config.jl")
    include("examples/benchmark/data_loading.jl")
    include("examples/benchmark/evaluation.jl")
    println("All modules loaded successfully")
    println("Training presets: ", keys(TRAINING_PRESETS))
    println("Dataset configs: ", keys(DATASET_CONFIGS))
    println("Basis types: ", BASIS_TYPES)
    println("Keep ratios: ", KEEP_RATIOS)
'
```
Expected: All modules load, constants print correctly.

- [ ] **Step 3: Commit all remaining files**

```bash
git add examples/benchmark/.gitignore
git commit -m "feat(benchmark): add .gitignore for data and results"
```

- [ ] **Step 4: Final commit with all benchmark files**

Verify all files are tracked:
```bash
git status
git log --oneline -10
```
