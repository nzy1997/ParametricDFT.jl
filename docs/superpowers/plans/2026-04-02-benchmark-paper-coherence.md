# Benchmark & Paper Coherence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Archive stale benchmark results, run fresh MSE-trained benchmarks on 3 datasets, generate plots, and align submodule + parent repo into a clean canonical state.

**Architecture:** The benchmark submodule gets cleaned up (archive old, delete junk), `run_mse.jl` is updated to save to canonical directories, `generate_report.jl` is updated for the new dataset list, then fresh runs produce the final results. Parent repo commits the optimized `loss.jl` and `training.jl`, then updates the submodule pointer.

**Tech Stack:** Julia, ParametricDFT.jl, OMEinsum, CUDA, CairoMakie, Git submodules

---

### Task 1: Commit parent repo code changes

**Files:**
- Stage: `src/loss.jl`, `src/training.jl`

These files already contain the optimizations (topk_mask, batched inverse). They just need to be committed.

- [ ] **Step 1: Verify uncommitted changes are correct**

Run:
```bash
cd /home/claude-user/ParametricDFT-fresh && git diff --stat
```

Expected: `src/loss.jl` and `src/training.jl` modified.

- [ ] **Step 2: Run tests to confirm everything passes**

Run:
```bash
cd /home/claude-user/ParametricDFT-fresh && CUDA_VISIBLE_DEVICES=1 julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: All 1314 tests pass.

- [ ] **Step 3: Commit the optimizations**

```bash
cd /home/claude-user/ParametricDFT-fresh
git add src/loss.jl src/training.jl
git commit -m "perf: optimize topk_truncate with quickselect and batch inverse einsum

- Replace partialsortperm with partialsort! quickselect (3-5x faster)
- Extract _topk_mask helper, reuse mask in rrule pullback
- Create batched_inverse_code in _train_basis_core for MSELoss
- Reduces MSE training time by batching inverse transforms

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Verify commit**

```bash
git fsck --connectivity-only
git log --oneline -3
```

---

### Task 2: Archive old results in benchmark submodule

**Files:**
- Modify: `examples/benchmark/results/` (move directories)

- [ ] **Step 1: Create archive directories**

```bash
cd /home/claude-user/ParametricDFT-fresh/examples/benchmark/results
mkdir -p archive/l1_moderate archive/mse_old archive/logs
```

- [ ] **Step 2: Move L1 results to archive**

```bash
# Current L1-trained results
mv quickdraw archive/l1_moderate/quickdraw 2>/dev/null
mv clic archive/l1_moderate/clic 2>/dev/null
mv div2k archive/l1_moderate/div2k 2>/dev/null
mv div2k_7q archive/l1_moderate/div2k_7q 2>/dev/null
mv div2k_8q archive/l1_moderate/div2k_8q 2>/dev/null
```

- [ ] **Step 3: Move old MSE results to archive**

```bash
mv quickdraw_moderate_mse_old archive/mse_old/quickdraw 2>/dev/null
mv clic_mse_old archive/mse_old/clic 2>/dev/null
mv div2k_mse_old archive/mse_old/div2k 2>/dev/null
```

- [ ] **Step 4: Move log files to archive**

```bash
mv *.log archive/logs/ 2>/dev/null
mv cross_dataset_summary.csv archive/ 2>/dev/null
mv timing_summary.csv archive/ 2>/dev/null
```

- [ ] **Step 5: Delete junk directories**

```bash
rm -rf quickdraw_old quickdraw_prev quickdraw_smoke quickdraw_mse_smoke quickdraw_mse_moderate
rm -rf clic_partial div2k_l1_failed
rm -rf smoke moderate heavy plots
```

- [ ] **Step 6: Verify clean state**

```bash
ls results/
```

Expected: only `archive/` directory remains (plus any `.gitkeep` or similar).

---

### Task 3: Update run_mse.jl to save to canonical directories

**Files:**
- Modify: `examples/benchmark/run_mse.jl`

The current `run_mse.jl` saves to `<dataset>_mse_<preset>/`. It should save to `<dataset>/` directly since MSE is now the canonical loss.

- [ ] **Step 1: Update output directory in run_mse.jl**

Change line 45 from:
```julia
output_dir = joinpath(RESULTS_DIR, "$(dataset_arg)_mse_$(preset_name)")
```
to:
```julia
output_dir = joinpath(RESULTS_DIR, string(dataset_arg))
```

- [ ] **Step 2: Verify the change**

```bash
cd /home/claude-user/ParametricDFT-fresh/examples/benchmark
grep "output_dir" run_mse.jl
```

Expected: `output_dir = joinpath(RESULTS_DIR, string(dataset_arg))`

---

### Task 4: Update generate_report.jl for new dataset list

**Files:**
- Modify: `examples/benchmark/generate_report.jl`

The report generator references `[:quickdraw, :div2k, :clic]` but we now use `:div2k_8q` instead of `:div2k`. It also only generates 1 reconstruction grid image; we want up to 5.

- [ ] **Step 1: Update DATASET_NAMES and DISPLAY_NAMES**

Change line 23 from:
```julia
const DATASET_NAMES = [:quickdraw, :div2k, :clic]
```
to:
```julia
const DATASET_NAMES = [:quickdraw, :div2k_8q, :clic]
```

Add to DISPLAY_NAMES (line 24-28):
```julia
const DISPLAY_NAMES = Dict(
    :quickdraw => "Quick Draw",
    :div2k => "DIV2K",
    :div2k_8q => "DIV2K (8q)",
    :clic => "CLIC",
)
```

- [ ] **Step 2: Update reconstruction grid to generate up to 5 images**

In `generate_reconstruction_grids`, change the test image loading to load up to 5 images:

Replace (lines 215-229):
```julia
        test_images = try
            if dataset_name == :quickdraw
                _, test, _ = load_quickdraw_dataset(; n_train = 1, n_test = 1)
                test
            elseif dataset_name == :div2k
                _, test, _ = load_div2k_dataset(; n_train = 1, n_test = 1)
                test
            else
                _, test, _ = load_clic_dataset(; n_train = 1, n_test = 1)
                test
            end
        catch e
            @warn "Could not load test image for $dataset_name: $e"
            continue
        end

        sample_img = test_images[1]
```

With:
```julia
        n_grid_images = 5
        test_images = try
            if dataset_name == :quickdraw
                _, test, _ = load_quickdraw_dataset(; n_train = 1, n_test = n_grid_images, img_size = dataset_config.img_size)
                test
            elseif dataset_name in (:div2k, :div2k_7q, :div2k_8q)
                _, test, _ = load_div2k_dataset(; n_train = 1, n_test = n_grid_images, img_size = dataset_config.img_size)
                test
            else
                _, test, _ = load_clic_dataset(; n_train = 1, n_test = n_grid_images, img_size = dataset_config.img_size)
                test
            end
        catch e
            @warn "Could not load test image for $dataset_name: $e"
            continue
        end
```

- [ ] **Step 3: Wrap grid generation in a loop over test images**

Replace the single-image grid generation (from `sample_img = test_images[1]` through `save(...)`) by wrapping it in a loop. Replace lines 231-308:

```julia
        for (img_idx, sample_img) in enumerate(test_images)
            basis_order = ["qft", "entangled_qft", "tebd", "mera", "fft", "dct"]

            # Load trained bases
            trained_bases = Dict{String,Any}()
            for basis_name in ["qft", "entangled_qft", "tebd", "mera"]
                basis_path = joinpath(output_dir, "trained_$(basis_name).json")
                if isfile(basis_path)
                    trained_bases[basis_name] = load_basis(basis_path)
                end
            end

            available_bases = [b for b in ["qft", "entangled_qft", "tebd", "mera"]
                               if haskey(trained_bases, b)]
            push!(available_bases, "fft")
            push!(available_bases, "dct")

            n_rows = 1 + length(available_bases)
            n_cols = length(KEEP_RATIOS)
            cell_size = 180

            fig = Figure(size = (cell_size * n_cols + 80, cell_size * n_rows + 40);
                figure_padding = 10)

            for (j, ratio) in enumerate(KEEP_RATIOS)
                Label(fig[0, j], "$(round(Int, ratio * 100))% kept"; fontsize = 14)
            end

            Label(fig[1, 0], "Original"; fontsize = 12, rotation = pi / 2)
            for j in 1:n_cols
                ax = Axis(fig[1, j]; aspect = 1)
                hidedecorations!(ax)
                heatmap!(ax, rotr90(sample_img); colormap = :grays, colorrange = (0.0, 1.0))
            end

            for (i, basis_name) in enumerate(available_bases)
                row = i + 1
                Label(fig[row, 0], get(BASIS_DISPLAY_NAMES, basis_name, basis_name);
                    fontsize = 12, rotation = pi / 2)

                for (j, keep_ratio) in enumerate(KEEP_RATIOS)
                    ax = Axis(fig[row, j]; aspect = 1)
                    hidedecorations!(ax)

                    recovered = if basis_name == "fft"
                        fft_compress_recover(sample_img, keep_ratio)
                    elseif basis_name == "dct"
                        dct_compress_recover(sample_img, keep_ratio)
                    elseif haskey(trained_bases, basis_name)
                        basis = trained_bases[basis_name]
                        compressed = compress(basis, sample_img; ratio = 1.0 - keep_ratio)
                        real.(recover(basis, compressed))
                    else
                        zeros(size(sample_img))
                    end

                    heatmap!(ax, rotr90(clamp.(recovered, 0.0, 1.0)); colormap = :grays,
                        colorrange = (0.0, 1.0))
                end
            end

            for row in 1:n_rows
                rowsize!(fig.layout, row, CairoMakie.Fixed(cell_size))
            end
            for col in 1:n_cols
                colsize!(fig.layout, col, CairoMakie.Fixed(cell_size))
            end
            colgap!(fig.layout, 5)
            rowgap!(fig.layout, 5)

            save(joinpath(plots_dir, "reconstruction_grid_$(img_idx).png"), fig; px_per_unit = 2)
            @info "Saved reconstruction grid $(img_idx) for $(DISPLAY_NAMES[dataset_name])"
        end
```

---

### Task 5: Commit benchmark submodule changes

**Files:**
- Stage in submodule: `evaluation.jl`, `run_mse.jl`, `generate_report.jl`, archived results

- [ ] **Step 1: Stage all changes in the benchmark submodule**

```bash
cd /home/claude-user/ParametricDFT-fresh/examples/benchmark
git add evaluation.jl run_mse.jl generate_report.jl
git add results/archive/
git add -u results/  # stages deletions of moved files
```

- [ ] **Step 2: Commit the submodule**

```bash
git commit -m "feat: add MSELoss support, archive old results, update report generator

- Add loss kwarg to train_and_time and run_all_bases
- Add run_mse.jl for MSELoss benchmarks
- Archive L1 and old MSE results to results/archive/
- Delete stale/duplicate result directories
- Update generate_report.jl for div2k_8q dataset and multi-image grids

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: Push benchmark submodule to remote**

```bash
git push origin main
```

- [ ] **Step 4: Update parent repo submodule pointer**

```bash
cd /home/claude-user/ParametricDFT-fresh
git add examples/benchmark
git commit -m "chore: update benchmark submodule — MSE support, archived old results

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Run fresh MSE benchmarks (Quick Draw, DIV2K 8q, CLIC)

**Files:**
- Output to: `examples/benchmark/results/quickdraw/`, `results/div2k_8q/`, `results/clic/`

All three datasets run sequentially on GPU 1 via nohup.

- [ ] **Step 1: Launch the sequential benchmark run**

```bash
cd /home/claude-user/ParametricDFT-fresh/examples/benchmark
CUDA_VISIBLE_DEVICES=1 nohup bash -c '
echo "=== Quick Draw MSE moderate ===" && date
julia --project=. run_mse.jl quickdraw moderate
echo "=== Quick Draw done ===" && date

echo "=== DIV2K 8q MSE moderate ===" && date
julia --project=. run_mse.jl div2k_8q moderate
echo "=== DIV2K 8q done ===" && date

echo "=== CLIC MSE moderate ===" && date
julia --project=. run_mse.jl clic moderate
echo "=== CLIC done ===" && date
' > results/mse_run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"
```

- [ ] **Step 2: Monitor progress**

```bash
# Check which dataset is training
ls results/*/trained_*.json 2>/dev/null
# Watch log
tail -f results/mse_run_*.log
# Check GPU
nvidia-smi -i 1
```

Expected completion: ~30-36 hours total.

- [ ] **Step 3: Verify all results exist after completion**

```bash
for d in quickdraw div2k_8q clic; do
    echo "=== $d ==="
    ls results/$d/metrics.json results/$d/trained_*.json results/$d/loss_history/*.json 2>/dev/null
done
```

Expected: each dataset has `metrics.json`, trained basis files, and loss history files.

---

### Task 7: Generate plots and reports

**Files:**
- Output to: `results/*/plots/`, `results/cross_dataset_summary.csv`, `results/timing_summary.csv`

- [ ] **Step 1: Run the report generator**

```bash
cd /home/claude-user/ParametricDFT-fresh/examples/benchmark
CUDA_VISIBLE_DEVICES=1 julia --project=. generate_report.jl
```

Expected: generates training curves, step losses, reconstruction grids (5 per dataset), rate-distortion CSVs, cross-dataset summary, and timing table.

- [ ] **Step 2: Verify plots exist**

```bash
for d in quickdraw div2k_8q clic; do
    echo "=== $d plots ==="
    ls results/$d/plots/ 2>/dev/null
done
echo "=== cross-dataset ==="
ls results/plots/ results/cross_dataset_summary.csv results/timing_summary.csv 2>/dev/null
```

Expected: each dataset has `training_curves.png`, `step_training_losses.png`, `reconstruction_grid_1.png` through `reconstruction_grid_5.png`. Root has cross-dataset plots and CSVs.

---

### Task 8: Commit and push final results

**Files:**
- Stage in submodule: all new results and plots

- [ ] **Step 1: Commit results in benchmark submodule**

```bash
cd /home/claude-user/ParametricDFT-fresh/examples/benchmark
git add results/quickdraw/ results/div2k_8q/ results/clic/
git add results/plots/ results/cross_dataset_summary.csv results/timing_summary.csv
git add results/mse_run_*.log
git commit -m "results: fresh MSE moderate benchmarks — Quick Draw, DIV2K 8q, CLIC

Trained with MSELoss(k=10%), magnitude-only topk, corrected leg convention,
batched inverse einsum. Includes metrics, trained bases, loss history, and plots.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 2: Push benchmark submodule**

```bash
git push origin main
```

- [ ] **Step 3: Final parent repo submodule update**

```bash
cd /home/claude-user/ParametricDFT-fresh
git add examples/benchmark
git commit -m "chore: update benchmark submodule — fresh MSE results for paper

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Verify git integrity**

```bash
git fsck --connectivity-only
```

---

### Task 9: Validate results for paper

- [ ] **Step 1: Print the final cross-dataset summary**

```bash
cat /home/claude-user/ParametricDFT-fresh/examples/benchmark/results/cross_dataset_summary.csv
```

Verify: all three datasets have PSNR values for QFT, Entangled QFT, TEBD, FFT, DCT. DIV2K 8q should also have MERA.

- [ ] **Step 2: Print timing summary**

```bash
cat /home/claude-user/ParametricDFT-fresh/examples/benchmark/results/timing_summary.csv
```

- [ ] **Step 3: Spot-check reconstruction grids exist and are non-empty**

```bash
for d in quickdraw div2k_8q clic; do
    echo "$d grid sizes:"
    ls -lh results/$d/plots/reconstruction_grid_*.png 2>/dev/null | awk '{print $5, $NF}'
done
```

Expected: each PNG is at least 50KB.

- [ ] **Step 4: Verify no stale results outside archive**

```bash
cd /home/claude-user/ParametricDFT-fresh/examples/benchmark/results
ls -d */ | grep -v -E '^(archive|quickdraw|div2k_8q|clic|plots)/$'
```

Expected: no output (all non-canonical directories are gone or archived).
