# Benchmark & Paper Coherence Design

**Date:** 2026-04-02
**Goal:** Align the benchmark submodule, ParametricDFT code, and paper results into a single coherent state where the paper reports fresh MSE-trained results from the current optimized codebase.

## Context

The paper argues MSELoss is the superior training objective for compression. The codebase now includes:
- Corrected yao2einsum leg convention (all 4 circuit types)
- Magnitude-only topk truncation (removed frequency bias)
- Optimized `_topk_mask` using quickselect (3-5x faster)
- Batched inverse einsum for MSELoss training

Old benchmark results were generated with frequency-biased topk and the previous leg convention. They are no longer valid. Fresh runs are needed.

## Decisions

- **Primary loss:** MSELoss with k = 10% of total coefficients
- **Datasets:** Quick Draw (5×5), DIV2K 8q (8×8), CLIC (9×9)
- **Preset:** moderate (10 epochs, 15 steps/image, batch_size=16, Adam optimizer)
- **Old results:** archived, not deleted
- **Submodule:** committed and pushed with MSE support, parent pointer updated

## Repository Cleanup

### Benchmark submodule results structure

```
results/
├── archive/
│   ├── l1_moderate/          # Current L1 results (quickdraw, clic, div2k, div2k_7q, div2k_8q)
│   ├── mse_old/              # Pre-fix MSE results (freq-biased topk)
│   └── logs/                 # All *.log files
├── quickdraw/                # Fresh MSE moderate (canonical)
├── div2k_8q/                 # Fresh MSE moderate (canonical)
├── clic/                     # Fresh MSE moderate (canonical)
├── cross_dataset_summary.csv # Regenerated
└── timing_summary.csv        # Regenerated
```

### What goes to archive

**To `archive/l1_moderate/`:** current `quickdraw/`, `clic/`, `div2k/`, `div2k_7q/`, `div2k_8q/` (all L1-trained)

**To `archive/mse_old/`:** `quickdraw_moderate_mse_old/`, `clic_mse_old/`, `div2k_mse_old/`

**To `archive/logs/`:** all `*.log` files in results root

**Delete (not archive):** `quickdraw_old/`, `quickdraw_prev/`, `quickdraw_smoke/`, `quickdraw_mse_smoke/`, `quickdraw_mse_moderate/`, `clic_partial/`, `div2k_l1_failed/`, `smoke/`, `moderate/`, `heavy/`, `plots/`

### Parent repo commits

1. Commit `src/loss.jl` (optimized topk with `_topk_mask`)
2. Commit `src/training.jl` (batched inverse code for MSELoss)
3. Update submodule pointer after benchmark is committed

## Benchmark Outputs Per Dataset

Each canonical directory contains:

```
<dataset>/
├── metrics.json                      # PSNR, SSIM, MSE at keep ratios [0.05, 0.10, 0.15, 0.20]
├── trained_qft.json                  # Saved basis parameters
├── trained_entangled_qft.json
├── trained_tebd.json
├── trained_mera.json                 # DIV2K 8q only
├── loss_history/
│   ├── qft_loss.json
│   ├── entangled_qft_loss.json
│   ├── tebd_loss.json
│   └── mera_loss.json                # DIV2K 8q only
└── plots/
    ├── training_curves.png           # Train/val loss per epoch
    ├── step_training_losses.png      # Per-step loss
    ├── reconstruction_grid_1.png     # Original vs reconstructed at different keep ratios
    ├── reconstruction_grid_2.png
    ├── reconstruction_grid_3.png
    ├── reconstruction_grid_4.png
    ├── reconstruction_grid_5.png
    └── rate_distortion_psnr.csv      # PSNR vs keep ratio table
```

Baselines (FFT, DCT) are computed at evaluation time and included in `metrics.json`.

## Run Configuration

| Dataset | Qubits | Image size | k (10%) | Bases | Est. time |
|---------|--------|-----------|---------|-------|-----------|
| Quick Draw | 5×5 | 32×32 | 102 | QFT, EQFT, TEBD | ~45 min |
| DIV2K 8q | 8×8 | 256×256 | 6554 | QFT, EQFT, TEBD, MERA | ~12-15 hours |
| CLIC | 9×9 | 512×512 | 26214 | QFT, EQFT, TEBD | ~15-20 hours |

**Total estimated: ~30-36 hours**

All runs use: `CUDA_VISIBLE_DEVICES=1`, `moderate` preset, `MSELoss(k)`, `run_mse.jl`.

Run order: Quick Draw → DIV2K 8q → CLIC (sequential, same GPU).

After each dataset completes, run `generate_report.jl` to produce plots.

## Execution Steps

1. **Commit parent repo changes** — `src/loss.jl`, `src/training.jl`, `test/training_tests.jl`
2. **Archive old results** in benchmark submodule
3. **Commit benchmark submodule** — evaluation.jl changes, run_mse.jl, archive
4. **Push benchmark submodule** to remote
5. **Update parent submodule pointer** and commit
6. **Run fresh MSE benchmarks** — Quick Draw, DIV2K 8q, CLIC (sequential nohup)
7. **Generate plots** via `generate_report.jl` after each dataset
8. **Commit and push results** to benchmark submodule
9. **Final parent pointer update**

## Success Criteria

- All three datasets have fresh MSE results in canonical directories
- Each directory has metrics.json, trained bases, loss history, and plots
- No stale results outside `archive/`
- Submodule remote matches local state
- Parent repo points to final benchmark commit
- Paper can reference results directly from the canonical directories
