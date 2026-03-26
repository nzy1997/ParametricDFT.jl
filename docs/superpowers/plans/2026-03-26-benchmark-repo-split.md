# Benchmark Repo Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `examples/benchmark/` from ParametricDFT.jl into a standalone repo `zazabap/ParametricDFT-Benchmarks.jl`, preserving git history, and replace the original directory with a git submodule.

**Architecture:** Three-phase approach: (1) clone + filter-repo to extract benchmark history, (2) update files for standalone operation, (3) replace original directory with submodule. All work happens in a temporary clone for Phase 1-2; Phase 3 modifies the parent repo.

**Tech Stack:** git, git-filter-repo, gh CLI, bash

**Spec:** `docs/superpowers/specs/2026-03-26-benchmark-repo-split-design.md`

---

### Task 1: Verify Prerequisites

**Files:** None (verification only)

- [ ] **Step 1: Verify git-filter-repo is installed**

Run: `git filter-repo --version`
Expected: Version number printed (e.g., `git-filter-repo 2.x.x`). If missing, install with `pip install git-filter-repo`.

- [ ] **Step 2: Verify gh CLI is authenticated**

Run: `gh auth status`
Expected: Shows `Logged in to github.com account zazabap`.

- [ ] **Step 3: Verify repo name is available**

Run: `gh repo view zazabap/ParametricDFT-Benchmarks.jl 2>&1 || echo "AVAILABLE"`
Expected: `AVAILABLE` (repo does not exist yet).

- [ ] **Step 4: Commit**

No commit needed — this is a verification-only task.

---

### Task 2: Clone and Extract Benchmark History

**Files:** Working in `/tmp/benchmark-extract/` (temporary clone)

- [ ] **Step 1: Clone the repo to a temporary directory**

```bash
git clone https://github.com/nzy1997/ParametricDFT.jl.git /tmp/benchmark-extract
```

- [ ] **Step 2: Checkout the feature branch**

```bash
cd /tmp/benchmark-extract
git checkout feature/multi-dataset-benchmark
```

- [ ] **Step 3: Run git filter-repo**

```bash
cd /tmp/benchmark-extract
git filter-repo --subdirectory-filter examples/benchmark --force
```

This promotes `examples/benchmark/` files to the repo root and removes all non-benchmark commits.

- [ ] **Step 4: Rename the branch to main**

```bash
cd /tmp/benchmark-extract
git branch -m main
```

`filter-repo` does not rename branches — the current branch is still `feature/multi-dataset-benchmark`. Rename it to `main` for the new repo.

- [ ] **Step 5: Verify the extraction**

```bash
cd /tmp/benchmark-extract
ls -la
```

Expected: `config.jl`, `data_loading.jl`, `evaluation.jl`, `generate_report.jl`, `run_quickdraw.jl`, `run_div2k.jl`, `run_clic.jl`, `run_all.sh`, `Project.toml`, `README.md`, `.gitignore` all at repo root. No `src/`, `test/`, or other ParametricDFT.jl files. (`Manifest.toml` is not tracked and will not be present.)

```bash
git log --oneline | head -20
```

Expected: Only commits that touched `examples/benchmark/` files.

---

### Task 3: Create GitHub Repo and Push

**Files:** Working in `/tmp/benchmark-extract/`

- [ ] **Step 1: Create the GitHub repo**

```bash
gh repo create zazabap/ParametricDFT-Benchmarks.jl --public --description "Benchmark suite for ParametricDFT.jl — trains QFT, EntangledQFT, TEBD, MERA bases on QuickDraw, DIV2K, CLIC datasets"
```

- [ ] **Step 2: Add remote and push**

```bash
cd /tmp/benchmark-extract
git remote add origin https://github.com/zazabap/ParametricDFT-Benchmarks.jl.git
git push -u origin main
```

Note: `filter-repo` removes the original remote, so `origin` is available. The branch was renamed to `main` in Task 2, Step 4.

- [ ] **Step 3: Verify on GitHub**

Run: `gh repo view zazabap/ParametricDFT-Benchmarks.jl`
Expected: Repo exists with the benchmark files.

---

### Task 4: Update Project.toml

**Files:**
- Modify: `/tmp/benchmark-extract/Project.toml`

- [ ] **Step 1: Update the ParametricDFT dependency from path to git URL**

Change the `[sources]` section from:
```toml
[sources]
ParametricDFT = {path = "../.."}
```

To:
```toml
[sources]
ParametricDFT = {url = "https://github.com/nzy1997/ParametricDFT.jl", rev = "main"}
```

Keep the `[deps]` section with the UUID unchanged.

- [ ] **Step 2: Commit**

```bash
cd /tmp/benchmark-extract
git add Project.toml
git commit -m "fix: update ParametricDFT dependency from local path to git URL"
```

---

### Task 5: Update .gitignore and Remove Manifest.toml

**Files:**
- Modify: `/tmp/benchmark-extract/.gitignore`
- Remove: `/tmp/benchmark-extract/Manifest.toml` (if tracked)

- [ ] **Step 1: Replace .gitignore contents**

Replace the entire file with:
```
data/
Manifest.toml
```

This removes `results/` from the ignore list and adds `Manifest.toml`.

- [ ] **Step 2: Verify Manifest.toml is not tracked**

`Manifest.toml` was not tracked in the original repo, so `filter-repo` will not have carried it over. Confirm:

```bash
cd /tmp/benchmark-extract
git ls-files Manifest.toml
```

Expected: empty output (not tracked). If it is tracked, run `git rm Manifest.toml`.

- [ ] **Step 3: Commit**

```bash
cd /tmp/benchmark-extract
git add .gitignore
git commit -m "chore: update .gitignore — track results, ignore Manifest.toml"
```

---

### Task 6: Update run_all.sh

**Files:**
- Modify: `/tmp/benchmark-extract/run_all.sh`

- [ ] **Step 1: Update usage comments (lines 6-7)**

Change:
```bash
#   CUDA_VISIBLE_DEVICES=0 nohup bash examples/benchmark/run_all.sh > benchmark_run.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=0 nohup bash examples/benchmark/run_all.sh moderate > benchmark_run.log 2>&1 &
```

To:
```bash
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_all.sh > benchmark_run.log 2>&1 &
#   CUDA_VISIBLE_DEVICES=0 nohup bash run_all.sh moderate > benchmark_run.log 2>&1 &
```

- [ ] **Step 2: Update results path comment (line 12)**

Change:
```bash
# Results are preserved under examples/benchmark/results/<preset>/<dataset>/
```

To:
```bash
# Results are preserved under results/<preset>/<dataset>/
```

- [ ] **Step 3: Update dataset data path comments (lines 17-18)**

Change:
```bash
#   - div2k: requires manual download to examples/benchmark/data/DIV2K_train_HR/
#   - clic: requires manual download to examples/benchmark/data/professional_train_2020/ etc.
```

To:
```bash
#   - div2k: requires manual download to data/DIV2K_train_HR/
#   - clic: requires manual download to data/professional_train_2020/ etc.
```

- [ ] **Step 4: Update REPO_DIR and cd logic (lines 26-27)**

Change:
```bash
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"
```

To:
```bash
cd "$SCRIPT_DIR"
```

- [ ] **Step 5: Update Julia invocation (line 79)**

Change:
```bash
        if julia --project=examples/benchmark "examples/benchmark/$SCRIPT" "$PRESET" 2>&1 | tee "$RESULTS_BASE/${PRESET}_${DATASET}_${TIMESTAMP}.log"; then
```

To:
```bash
        if julia --project=. "$SCRIPT" "$PRESET" 2>&1 | tee "$RESULTS_BASE/${PRESET}_${DATASET}_${TIMESTAMP}.log"; then
```

- [ ] **Step 6: Commit**

```bash
cd /tmp/benchmark-extract
git add run_all.sh
git commit -m "fix: update run_all.sh paths for standalone repo"
```

---

### Task 7: Update README.md

**Files:**
- Modify: `/tmp/benchmark-extract/README.md`

- [ ] **Step 1: Add link to parent repo and update Julia version requirement**

At the top, after the title, add:
```markdown
> Companion benchmark suite for [ParametricDFT.jl](https://github.com/nzy1997/ParametricDFT.jl).
```

Change `Julia 1.10+` to `Julia 1.11+` (required for `[sources]` in Project.toml).

- [ ] **Step 2: Update Setup section**

Change:
```bash
# From the repository root
julia --project=examples/benchmark -e 'using Pkg; Pkg.instantiate()'
```

To:
```bash
# From the benchmark repo root
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

- [ ] **Step 3: Update Running a Single Benchmark section**

Change all `julia --project=examples/benchmark examples/benchmark/` to `julia --project=. ./`. For example:
```bash
julia --project=. run_quickdraw.jl moderate
CUDA_VISIBLE_DEVICES=0 julia --project=. run_quickdraw.jl heavy
```

- [ ] **Step 4: Update run_all.sh usage**

Change:
```bash
CUDA_VISIBLE_DEVICES=0 nohup bash examples/benchmark/run_all.sh > benchmark_run.log 2>&1 &
```

To:
```bash
CUDA_VISIBLE_DEVICES=0 nohup bash run_all.sh > benchmark_run.log 2>&1 &
```

- [ ] **Step 5: Update Results and Resuming paths**

Change all `examples/benchmark/results` references to `results`. Change `examples/benchmark/data/` to `data/`.

- [ ] **Step 6: Update generate_report.jl command**

Change:
```bash
julia --project=examples/benchmark examples/benchmark/generate_report.jl
```

To:
```bash
julia --project=. generate_report.jl
```

- [ ] **Step 7: Update available run scripts list**

Change `run_atd12k.jl` reference (line 36) to `run_clic.jl` and update the description to match current datasets.

- [ ] **Step 8: Update Notes section — ATD-12K → CLIC**

The Notes section (line 114) references ATD-12K which no longer exists as a dataset. Change:
```
MERA requires power-of-2 qubit counts. It is automatically skipped for Quick Draw (m=5, n=5) and ATD-12K (m=9, n=9).
```

To:
```
MERA requires power-of-2 qubit counts. It is automatically skipped for Quick Draw (m=5, n=5) and CLIC (m=9, n=9).
```

- [ ] **Step 9: Verify no hardcoded paths remain in Julia source files**

```bash
cd /tmp/benchmark-extract
grep -r "examples/benchmark" *.jl *.sh || echo "No hardcoded paths found"
```

Expected: No matches (all Julia files use `@__DIR__`-relative paths via `config.jl`).

- [ ] **Step 10: Commit**

```bash
cd /tmp/benchmark-extract
git add README.md
git commit -m "docs: update README for standalone repo"
```

---

### Task 8: Copy Results and Push

**Files:**
- Add: `/tmp/benchmark-extract/results/` (copied from original working tree)

- [ ] **Step 1: Copy results from original working tree**

```bash
cp -r /home/claude-user/ParametricDFT-fresh/examples/benchmark/results/ /tmp/benchmark-extract/results/
```

- [ ] **Step 2: Add and commit results**

```bash
cd /tmp/benchmark-extract
git add results/
git commit -m "data: add benchmark results (smoke, moderate, heavy presets)"
```

- [ ] **Step 3: Push all changes to GitHub**

```bash
cd /tmp/benchmark-extract
git push origin main
```

- [ ] **Step 4: Verify on GitHub**

Run: `gh repo view zazabap/ParametricDFT-Benchmarks.jl --web` or check that files appear via `gh api repos/zazabap/ParametricDFT-Benchmarks.jl/contents/`.

---

### Task 9: Replace Benchmark Directory with Submodule in Parent Repo

**Files:**
- Remove: `/home/claude-user/ParametricDFT-fresh/examples/benchmark/` (directory)
- Create: `/home/claude-user/ParametricDFT-fresh/.gitmodules`

- [ ] **Step 1: Remove the benchmark directory from the parent repo**

```bash
cd /home/claude-user/ParametricDFT-fresh
git rm -rf examples/benchmark/
```

- [ ] **Step 2: Commit the removal**

```bash
git commit -m "refactor: remove examples/benchmark/ in preparation for submodule"
```

- [ ] **Step 3: Add the new repo as a submodule**

```bash
cd /home/claude-user/ParametricDFT-fresh
git submodule add https://github.com/zazabap/ParametricDFT-Benchmarks.jl examples/benchmark
```

- [ ] **Step 4: Commit the submodule addition**

```bash
git add .gitmodules examples/benchmark
git commit -m "feat: add ParametricDFT-Benchmarks.jl as submodule at examples/benchmark"
```

- [ ] **Step 5: Verify the submodule works**

```bash
cd /home/claude-user/ParametricDFT-fresh
git submodule status
```

Expected: Shows the submodule commit hash and path `examples/benchmark`.

- [ ] **Step 6: Push to remote**

```bash
cd /home/claude-user/ParametricDFT-fresh
git push origin feature/multi-dataset-benchmark
```

---

### Task 10: Final Verification

**Files:** None (verification only)

- [ ] **Step 1: Verify new repo is complete**

```bash
gh repo view zazabap/ParametricDFT-Benchmarks.jl
```

Check: repo exists, has commits, has results/ tracked.

- [ ] **Step 2: Test a fresh clone with submodule**

```bash
cd /tmp
git clone --recurse-submodules https://github.com/nzy1997/ParametricDFT.jl.git /tmp/verify-clone
ls /tmp/verify-clone/examples/benchmark/
```

Expected: Benchmark files present via submodule.

- [ ] **Step 3: Clean up temporary directories**

```bash
rm -rf /tmp/benchmark-extract /tmp/verify-clone
```
