# Multi-Dataset Benchmark Design

## Goal

Produce publication-quality benchmark results comparing all 4 basis types (QFT, EntangledQFT, TEBD, MERA) plus a classical FFT baseline across 3 diverse image datasets. The benchmark measures compression quality (PSNR, SSIM, MSE), training convergence, and timing.

**All training runs use GPU** (via `device=:gpu`). CPU-only benchmarking is already covered by `examples/optimizer_benchmark.jl`.

## Datasets

| Dataset | Source | Native Size | Target Size | Qubits (m=n) | Color Handling |
|---------|--------|------------|-------------|--------------|----------------|
| Quick Draw | Google numpy bitmaps (cat, dog, airplane, apple, bicycle) | 28x28 | 32x32 (center-padded) | 5 | Already grayscale |
| DIV2K | HR validation set (100 images) + train set (800 images) | ~2K shortest side | 1024x1024 (center-crop to square, resize) | 10 | Convert RGB to grayscale |
| ATD-12K | Test set (2K triplets, middle frame) | 1280x720 or 1920x1080 | 512x512 (center-crop to square, resize) | 9 | Convert RGB to grayscale |

## Basis Types

All 4 learned bases plus classical FFT baseline:

1. **QFTBasis** — separable QFT circuit, Hadamard + controlled-phase gates
2. **EntangledQFTBasis** — QFT with XY entanglement gates between row/column qubits
3. **TEBDBasis** — nearest-neighbor ring topology with learnable phases
4. **MERABasis** — hierarchical multi-scale connectivity with learnable phases
5. **Classical FFT** — baseline, no training

## Compression Ratios

Evaluate at 4 levels of coefficient retention: **5%, 10%, 15%, 20%** kept.

## Training Configuration

Two named presets, selected via CLI argument:

| Parameter | Moderate | Heavy |
|-----------|----------|-------|
| Epochs | 10 | 50 |
| Steps per image | 100 | 200 |
| Training images | 50 | 100 |
| Test images | 10 | 20 |
| Early stopping patience | 3 | 5 |
| Optimizer | `:adam` | `:adam` |
| Validation split | 0.2 | 0.2 |

Early stopping is already implemented in the training pipeline (`early_stopping_patience` parameter). Training snapshots the best parameters based on validation loss.

Plan: run moderate first, then heavy.

All training uses `device=:gpu`. Set `Random.seed!(42)` before each `train_basis()` call for reproducibility.

## File Structure

```
examples/benchmark/
├── Project.toml              # Dependencies; uses path = "../.." for ParametricDFT
│                             #   Also: CUDA, Images, ImageQualityIndexes, CairoMakie,
│                             #   NPZ, FFTW, JSON3, FileIO, Downloads
├── config.jl                 # Training presets, dataset configs, shared constants
├── data_loading.jl           # load_quickdraw_dataset(), load_div2k_dataset(), load_atd12k_dataset()
├── evaluation.jl             # compute_metrics(), evaluate_basis(), evaluate_fft_baseline_timed(),
│                             #   train_and_time(), save/load_benchmark_results(), print functions
├── run_quickdraw.jl          # Train + evaluate all bases on Quick Draw
├── run_div2k.jl              # Train + evaluate all bases on DIV2K
├── run_atd12k.jl             # Train + evaluate all bases on ATD-12K
└── generate_report.jl        # Cross-dataset summary, publication plots
```

## Module Details

### config.jl

Shared constants and configuration:

```julia
const TRAINING_PRESETS = Dict(
    :moderate => (epochs=10, steps_per_image=100, n_train=50, n_test=10,
                  patience=3, optimizer=:adam, validation_split=0.2, device=:gpu),
    :heavy    => (epochs=50, steps_per_image=200, n_train=100, n_test=20,
                  patience=5, optimizer=:adam, validation_split=0.2, device=:gpu),
)

const DATASET_CONFIGS = Dict(
    :quickdraw => (m=5, n=5, img_size=32),
    :div2k     => (m=10, n=10, img_size=1024),
    :atd12k    => (m=9, n=9, img_size=512),
)

const KEEP_RATIOS = [0.05, 0.10, 0.15, 0.20]
const BASIS_TYPES = [QFTBasis, EntangledQFTBasis, TEBDBasis, MERABasis]
const RESULTS_DIR = joinpath(@__DIR__, "results")
```

### data_loading.jl

Three loader functions, all returning the same interface:

```julia
(train_images::Vector{Matrix{Float64}}, test_images::Vector{Matrix{Float64}}, test_labels::Vector{String})
```

All images normalized to [0, 1] Float64 grayscale.

- **`load_quickdraw_dataset(; n_train, n_test, img_size=32, seed=42)`**
  - Downloads .npy files from Google storage if not present in `data/quickdraw/`
  - Categories: cat, dog, airplane, apple, bicycle
  - 28x28 native, center-padded to 32x32
  - Shuffled mix across categories

- **`load_div2k_dataset(; n_train, n_test, img_size=1024, seed=42)`**
  - Expects DIV2K HR images in `data/DIV2K_valid_HR/` and `data/DIV2K_train_HR/`
  - Center-crop to square, resize to 1024x1024
  - RGB to grayscale conversion
  - 800 train + 100 val available

- **`load_atd12k_dataset(; n_train, n_test, img_size=512, seed=42)`**
  - Uses test_2k set, takes middle frame from each triplet
  - Center-crop to square (min dimension), resize to 512x512
  - RGB to grayscale conversion

### evaluation.jl

Core evaluation functions:

- **`compute_metrics(original, recovered)`** — returns `(mse, psnr, ssim)` for a single image pair
- **`evaluate_basis(basis, test_images, keep_ratios)`** — evaluates one basis at all ratios, returns `Dict(ratio => (mean_mse, std_mse, mean_psnr, std_psnr, mean_ssim, std_ssim))`
- **`evaluate_fft_baseline_timed(test_images, keep_ratios)`** — same as above but for classical FFT, also returns elapsed time
- **`train_and_time(BasisType, dataset, dataset_config, preset)`** — wraps `train_basis()` with `@elapsed`, returns `(trained_basis, history, elapsed_seconds)`. Sets `Random.seed!(42)` before training for reproducibility. Passes `device=:gpu`.
- **`save_benchmark_results(path, results_dict)`** — serialize metrics + timing to JSON
- **`load_benchmark_results(path)`** — deserialize
- **`print_dataset_summary(results, keep_ratios)`** — formatted console table

### run_*.jl Scripts

Each follows the same pattern:

1. Parse CLI preset argument (default: `:moderate`)
2. Load dataset via corresponding loader
3. For each basis type: train with timing, evaluate at all ratios, save checkpoint
4. Evaluate FFT baseline with timing
5. Save all results to `results/<dataset>/metrics.json`
6. Print summary table

Key behaviors:
- **Checkpoints after each basis** — crash-resilient
- **Deterministic** — `Random.seed!(42)` for data loading and training
- **Resumable** — skip retraining if saved basis file already exists

Usage:
```bash
julia --project=examples/benchmark examples/benchmark/run_quickdraw.jl moderate
julia --project=examples/benchmark examples/benchmark/run_div2k.jl moderate
julia --project=examples/benchmark examples/benchmark/run_atd12k.jl moderate
```

### generate_report.jl

Loads all `metrics.json` files from `results/` and produces 5 outputs:

**1. Rate-distortion tables** (printed + saved as CSV per dataset)
- PSNR, SSIM, MSE at each compression ratio for each basis

**2. Training loss curves** (one plot per dataset)
- All 4 bases on same axes, per-epoch validation loss
- Uses existing `plot_training_comparison()` from `src/visualization.jl`
- Saved as `results/<dataset>/plots/training_curves.png`

**3. Visual comparison grids** (one per dataset)
- Rows: original + 4 learned bases + FFT (6 rows)
- Columns: 4 compression ratios (5%, 10%, 15%, 20%)
- Uses first test image as sample
- Saved as `results/<dataset>/plots/reconstruction_grid.png`

**4. Cross-dataset summary table** (printed + saved as CSV)
- Per-basis PSNR@10% across all 3 datasets with average rank

**5. Timing table** (printed + saved as CSV)
- Training time per basis per dataset
- FFT inference time per dataset

All plots use CairoMakie.

Usage:
```bash
julia --project=examples/benchmark examples/benchmark/generate_report.jl
```

## Output Structure

```
examples/benchmark/results/
├── quickdraw/
│   ├── trained_qft.json
│   ├── trained_entangled_qft.json
│   ├── trained_tebd.json
│   ├── trained_mera.json
│   ├── metrics.json
│   ├── loss_history/
│   │   ├── qft_loss.json
│   │   ├── entangled_qft_loss.json
│   │   ├── tebd_loss.json
│   │   └── mera_loss.json
│   └── plots/
│       ├── training_curves.png
│       └── reconstruction_grid.png
├── div2k/
│   └── (same structure)
├── atd12k/
│   └── (same structure)
├── cross_dataset_summary.csv
├── timing_summary.csv
└── plots/
    ├── cross_dataset_psnr.png
    └── cross_dataset_ssim.png
```

## Dependencies

The benchmark scripts need these packages (in `examples/benchmark/Project.toml` with `path = "../.."`):

- ParametricDFT (path dependency to parent project)
- CUDA (GPU support)
- Images, ImageQualityIndexes (PSNR, SSIM)
- CairoMakie (plotting)
- NPZ (Quick Draw .npy files)
- FFTW (classical FFT baseline)
- JSON3 (result serialization)
- FileIO (image loading for DIV2K/ATD-12K)
- Downloads (auto-download Quick Draw data)
- Random, Statistics, Printf (stdlib)

## Implementation Notes

- **GPU required**: All training uses `device=:gpu`. CUDA.jl must be available.
- **Qubit scaling**: Quick Draw uses 5 qubits (32x32), DIV2K uses 10 qubits (1024x1024), ATD-12K uses 9 qubits (512x512). Parameter counts differ across datasets — document in results.
- **Loss function**: `MSELoss(k)` where `k` is 10% of total coefficients (matching existing examples). Training always truncates at 10% regardless of evaluation ratios. This is a deliberate choice: training at 10% provides a good trade-off for evaluating across the 5-20% range.
- **Compression ratio semantics**: The `compress()` function's `ratio` parameter is the fraction *discarded*. To keep 5% of coefficients, pass `ratio=0.95`. The `evaluate_basis()` function accepts `keep_ratios` and handles this conversion internally.
- **All images are grayscale**. Color handling is out of scope.
- **No core library changes needed**. The benchmark uses the existing `train_basis()` API with early stopping.
- **MERA first use**: This benchmark is the first full training/evaluation pipeline for MERABasis. The implementation exists and is tested in `test/mera_tests.jl`, but edge cases may surface at larger qubit counts.
- **Reproducibility**: Set `Random.seed!(42)` before each `train_basis()` call and before data loading.
- **Data download**: Quick Draw .npy files are auto-downloaded via `Downloads.download()` if not present (~100MB per category). DIV2K and ATD-12K require manual download; loaders should print clear error messages with download instructions if data is missing.
- **Project.toml**: The benchmark uses its own `examples/benchmark/Project.toml` with `path = "../.."` for the ParametricDFT dependency. Run with `--project=examples/benchmark`.
