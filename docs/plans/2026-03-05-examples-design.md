# Examples Cleanup Design

**Date**: 2026-03-05
**Branch**: feat/examples (PR#49)

## Goal

Replace the sprawling example scripts with three clean, focused examples that verify library performance and demonstrate the API.

## Approach

Start from PR#49's `feat/examples` branch and refine: simplify `circuit_visualization.jl`, keep `optimizer_benchmark.jl`, add new minimal `basis_demo.jl`.

## Scripts

### 1. `basis_demo.jl` (~80 lines) — NEW

Minimal showcase of the three basis types:
- Create `QFTBasis`, `EntangledQFTBasis`, `TEBDBasis` for small sizes (m=3, n=3)
- Generate a small random test image
- Run `forward_transform` -> `topk_truncate` -> `inverse_transform` on each
- Print reconstruction error and verify round-trip works
- No training, no external data, no plotting

### 2. `circuit_visualization.jl` (~250 lines) — SIMPLIFIED from 891 lines

Replace the abstract specification system with direct plotting:
- One function per circuit type: `draw_qft!`, `draw_entangled_qft!`, `draw_tebd!`
- Each function directly draws wires + gates on a CairoMakie axis
- Color-coded gates (keep PR#49's color scheme)
- Save PNG output for QFT (1D, 2D), Entangled QFT, and TEBD
- No abstract types, no dispatch

### 3. `optimizer_benchmark.jl` (~700 lines) — REFINED from PR#49

Keep PR#49's structure with minor cleanup:
- Benchmarks: Manopt-GD vs PDFT-GD (cpu/gpu) vs PDFT-Adam (cpu/gpu)
- DIV2K 512x512 images, download instructions in assert message
- Loss curves, PSNR/SSIM metrics, JSON results output
- GPU support via CUDA.jl with automatic fallback to CPU

## File Structure

```
examples/
├── Project.toml              # Dependencies (from PR#49)
├── .gitignore                # Output dirs
├── basis_demo.jl             # Minimal basis showcase
├── circuit_visualization.jl  # CairoMakie circuit diagrams
└── optimizer_benchmark.jl    # Manopt + GPU benchmark
```

## Removed Files

- `entangle_position_demo.jl`
- `verify_tebd.jl`
- `BasisDemo/` and `BasisDemo_QuickDraw/` directories
- `cat.png`

## Dependencies

Keep PR#49's `Project.toml`: CUDA, CairoMakie, Manopt, Manifolds, ManifoldDiff, Images, ImageQualityIndexes, JSON3, FileIO, OMEinsum, RecursiveArrayTools, Zygote, ADTypes.
