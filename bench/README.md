# ParametricDFT Benchmark Suite

This benchmark suite evaluates the **Parametric Quantum Fourier Transform** implementation, inspired by the methodology in:

> **Dao, T., et al. (2019).** *Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations.* ICML 2019.  
> [Paper Link](https://proceedings.mlr.press/v97/dao19a)

---

## Overview

The Dao et al. paper demonstrates that structured linear transforms (like the DFT) can be **learned automatically** using butterfly factorizations, achieving:
- Recovery of the Cooley-Tukey FFT to **machine precision**
- **O(N log N)** parameters instead of O(N²) for dense matrices
- **4× faster** inference with **40× fewer** parameters

This benchmark suite tests whether our Parametric QFT achieves similar properties.

---

## Benchmark Components

### 1. DFT Recovery Test (`dft_recovery.jl`)
**Purpose:** Test if the parametric DFT can recover the standard DFT matrix.

**Key Metrics:**
- Frobenius norm relative error: `||Learned - DFT||_F / ||DFT||_F`
- Maximum element-wise error
- Recovery to machine precision (error < 1e-10)

**Dao et al. Result:** Recovered FFT to ~1e-15 error for N ≤ 1024.

---

### 2. Parameter Efficiency (`parameter_count.jl`)
**Purpose:** Compare parameter count between methods.

#### Methods Compared

| Method | Parameters | Complexity | Learnable? | Source |
|--------|-----------|------------|------------|--------|
| **Dense DFT Matrix** | 2N² | O(N²) | ✗ Fixed | Baseline |
| **Cooley-Tukey FFT** | 0 (algorithm) | O(N log N) | ✗ Fixed | [Cooley & Tukey, 1965] |
| **Butterfly Factorization** | O(N log N) | O(N log N) | ✓ Yes | [Dao et al., 2019] |
| **Parametric QFT (Ours)** | O(log² N) | O(N log N)* | ✓ Yes | This work |
| **Sparse FFT** | O(k log N) | O(k log N) | ✗ Fixed | [Hassanieh et al., 2012] |
| **Low-rank (SVD)** | 2Nr | O(Nr) | ✗ Fixed | Classical |

> *\* Naive tensor network contraction: O(N log² N). Optimized by exploiting diagonal structure of controlled-phase gates: **O(N log N)** — same as FFT.*

#### Detailed Method Descriptions

**1. Dense DFT Matrix**
- Stores full N×N complex matrix
- Parameters: 2N² (real + imaginary parts)
- No structure exploited; serves as upper bound

**2. Cooley-Tukey FFT (FFTW)**
- Fixed algorithm, no learnable parameters
- O(N log N) operations via divide-and-conquer
- Optimal for standard DFT; cannot adapt to data

**3. Butterfly Factorization (Dao et al., 2019)**
- Factorizes transform as product of sparse butterfly matrices
- Parameters: O(N log N) learnable entries
- Can recover FFT to machine precision
- Generalizes to learn data-adaptive transforms

**4. Parametric QFT (This Work)**
- Quantum circuit ansatz with unitary gate parameters
- Parameters: O(n²) = O(log² N) where N = 2^n
  - n Hadamard gates (2×2 unitary): 4n parameters
  - n(n-1)/2 controlled-phase gates (diagonal): ~n² parameters
- Optimized on Riemannian manifold (preserves unitarity)
- **Complexity**: 
  - Naive tensor network contraction: O(N log² N)
  - Optimized (exploiting diagonal controlled-phase): **O(N log N)**
- Key insight: controlled-phase gates are diagonal matrices that can be merged

**5. Sparse FFT**
- For signals with k-sparse frequency representation
- Only recovers k largest coefficients
- Not directly comparable (different problem setting)

**6. Low-rank Approximation (SVD)**
- Approximates transform via rank-r factorization
- Parameters: 2Nr for rank-r approximation
- Quality degrades significantly for full-rank transforms

#### Parameter Count Formulas

For an image of size 2^m × 2^n (total N = 2^(m+n) pixels), let k = m + n (total qubits):

```
Dense Matrix:        2 × N × N = 2N²
Butterfly (Dao):     N × log₂(N) × c₁ ≈ N log N  
Parametric QFT:      k Hadamard gates + k(k-1)/2 controlled-phase gates
                   = 4k + 2 × k(k-1)/2 = 4k + k² - k = k² + 3k
                   = O(k²) = O(log² N)
```

**Example (512×512 image, N = 262,144, k = 18 qubits):**
| Method | Parameters | Ratio vs Dense |
|--------|-----------|----------------|
| Dense | 137 billion | 1× |
| Butterfly | ~4.7 million | ~29,000× smaller |
| Parametric QFT | ~378 | **~363 million× smaller** |

**Calculation for Parametric QFT (k=18):**
- Hadamard gates: 18 × 4 = 72 parameters
- Controlled-phase gates: 18 × 17 / 2 × 2 = 306 parameters  
- Total: **378 parameters**

#### Key Insight

The Parametric QFT achieves **extreme parameter efficiency** while maintaining **optimal runtime**:

**Why O(log² N) Parameters?**

**Step 1: How many qubits?**
- To represent a vector of size N, we need **n = log₂(N) qubits**
- Example: N = 1,048,576 (1024×1024 image) → n = 20 qubits

**Step 2: How many gates in QFT circuit?**
The standard QFT circuit structure for n qubits:
```
Qubit 0: H — CP(1) — CP(2) — ... — CP(n-1)     → 1 + (n-1) gates
Qubit 1:     H    — CP(1) — ... — CP(n-2)     → 1 + (n-2) gates
Qubit 2:           H    — ... — CP(n-3)       → 1 + (n-3) gates
  ⋮
Qubit n-1:                      H              → 1 gate
```

Total gates:
- Hadamard gates: **n**
- Controlled-phase gates: **(n-1) + (n-2) + ... + 1 = n(n-1)/2**
- Total: **n + n(n-1)/2 = n(n+1)/2 = O(n²)**

**Step 3: Parameters per gate**
- Each gate is a 2×2 unitary matrix with O(1) parameters

**Step 4: Total parameters**
```
Total = O(n²) gates × O(1) params/gate = O(n²) = O((log₂ N)²) = O(log² N)
```

**Concrete Example:**
| Image Size | N | n = log₂(N) | Gates ≈ n²/2 | Parameters |
|------------|---|-------------|--------------|------------|
| 64×64 | 4,096 | 12 | ~78 | ~312 |
| 256×256 | 65,536 | 16 | ~136 | ~544 |
| 512×512 | 262,144 | 18 | ~171 | ~684 |
| 1024×1024 | 1,048,576 | 20 | ~210 | ~840 |

Compare to Butterfly (Dao et al.): N log N parameters
| Image Size | N | Butterfly Params | QFT Params | Ratio |
|------------|---|-----------------|------------|-------|
| 512×512 | 262,144 | ~4.7 million | ~684 | **6,900×** fewer |

**Parameter Efficiency (O(log² N)):**
1. QFT circuit for n qubits has n Hadamard gates + n(n-1)/2 controlled-phase gates
2. Each gate has O(1) parameters (2×2 matrix)
3. Total: O(n²) = O(log² N) parameters

**Runtime Efficiency (O(N log N)):**
1. Naive tensor network contraction: O(N log² N)
2. **Key optimization**: Controlled-phase gates are diagonal matrices
3. Diagonal gates can be merged/fused during contraction
4. Optimized complexity: **O(N log N)** — same as Cooley-Tukey FFT!

**Comparison with Butterfly Factorizations:**
| Aspect | Butterfly (Dao et al.) | Parametric QFT (Ours) |
|--------|------------------------|----------------------|
| Parameters | O(N log N) | **O(log² N)** |
| Runtime | O(N log N) | O(N log N)* |
| Expressiveness | Arbitrary structured | Unitary only |
| Optimization | Standard gradient descent | Riemannian GD |

> *With diagonal gate optimization

The tradeoff: Butterfly can represent more general (non-unitary) transforms, while QFT is constrained to unitary operations — but this constraint provides implicit regularization and dramatic parameter reduction.

---

### 3. Scaling Analysis (`scaling_analysis.jl`)
**Purpose:** Verify O(N log N) computational complexity.

**Test Sizes:**
- 16×16 (N = 256)
- 32×32 (N = 1,024)
- 64×64 (N = 4,096)
- 128×128 (N = 16,384)
- 256×256 (N = 65,536)
- 512×512 (N = 262,144)
- 1024×1024 (N = 1,048,576)

**Expected:** Time/N should scale as O(log N).

---

### 4. Rate-Distortion Curves (`rate_distortion.jl`)
**Purpose:** Compare reconstruction quality vs. compression ratio.

**Compression Ratios Tested:**
- 50%, 60%, 70%, 80%, 90%, 95%, 99%

**Quality Metrics:**
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- MSE (Mean Squared Error)
- MS-SSIM (Multi-Scale SSIM)

---

### 5. Convergence Analysis (`convergence_analysis.jl`)
**Purpose:** Track training dynamics and optimization behavior.

**Key Metrics:**
- Loss curve over iterations
- Gradient norm decay
- Convergence rate estimation
- Early stopping behavior

---

### 6. FLOPs Analysis (`flops_analysis.jl`)
**Purpose:** Estimate computational cost in floating-point operations.

**Theoretical Complexity:**
| Method | FLOPs | Notes |
|--------|-------|-------|
| Classical FFT (FFTW) | 5N log₂ N | Highly optimized C implementation |
| Parametric DFT (naive) | O(N log² N) | Direct tensor network contraction |
| Parametric DFT (optimized) | O(N log N) | Exploiting diagonal controlled-phase gates |

**Current Implementation Status:**
- The current OMEinsum-based implementation uses general tensor contraction
- Optimizing to O(N log N) requires exploiting the diagonal structure of controlled-phase gates
- This optimization is described in `note/main.typ` (Section: Tensor network representation)

---

## Running the Benchmarks

### Run All Benchmarks
```bash
cd /workspace
julia --project=. bench/run_all_benchmarks.jl
```

### Run Individual Benchmarks
```bash
# DFT Recovery Test
julia --project=. bench/dft_recovery.jl

# Scaling Analysis
julia --project=. bench/scaling_analysis.jl

# Rate-Distortion Curves
julia --project=. bench/rate_distortion.jl
```

---

## Output Files

Results are saved to `bench/results/`:

```
bench/results/
├── dft_recovery_results.csv       # DFT recovery metrics
├── parameter_count_results.csv    # Parameter efficiency data
├── scaling_results.csv            # Timing vs. problem size
├── rate_distortion_curves.csv     # Quality vs. compression
├── convergence_logs/              # Training convergence data
│   └── convergence_<timestamp>.csv
└── summary_report.md              # Human-readable summary
```

---

## Comparison with Dao et al. (2019)

| Benchmark | Dao et al. Result | Our Target | Notes |
|-----------|------------------|------------|-------|
| DFT Recovery | ~1e-15 error | < 1e-10 | Machine precision |
| Parameter Count | O(N log N) | **O(log² N)** | Ours is more efficient! |
| Inference Speed | 4× faster than dense | Compare to FFTW | Both achieve O(N log N) |
| Scaling | O(N log N) | O(N log N)* | *With diagonal optimization |
| Image Compression | +3.9% on CIFAR-10 | SSIM/PSNR improvement | Different application |

**Key Advantage:** Our Parametric QFT achieves **dramatically fewer parameters** (O(log² N) vs O(N log N)) while maintaining the same asymptotic runtime complexity when optimized.

---

## Key Differences from Dao et al.

| Aspect | Dao et al. (Butterfly) | This Work (Parametric QFT) |
|--------|------------------------|---------------------------|
| **Architecture** | Butterfly matrices | QFT circuits as tensor networks |
| **Parameters** | O(N log N) | **O(log² N)** |
| **Constraints** | Unconstrained entries | Unitary gates |
| **Optimization** | Standard gradient descent | Riemannian GD on U(2) manifold |
| **Runtime** | O(N log N) | O(N log N)* |
| **Application** | General transforms + NN compression | Image compression |
| **Implementation** | Custom CUDA kernels | OMEinsum.jl tensor contraction |

> *Requires optimization exploiting diagonal structure of controlled-phase gates

**Why fewer parameters?**
- Butterfly: Each of log N layers has N entries → O(N log N) total
- QFT: n qubits need n² gates, each with O(1) params → O(n²) = O(log² N) total
- The unitary constraint acts as implicit regularization

---

## Dependencies

```julia
# Core
ParametricDFT.jl      # Main package
FFTW.jl               # Classical FFT baseline

# Benchmarking
BenchmarkTools.jl     # Accurate timing
Statistics.jl         # Statistical analysis
DelimitedFiles.jl     # CSV output

# Image Processing
Images.jl             # Image I/O
ImageQualityIndexes.jl # SSIM, PSNR metrics
```

---

## References

1. Dao, T., Gu, A., Eichhorn, M., Rudra, A., & Ré, C. (2019). *Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations.* ICML 2019.

2. Cooley, J. W., & Tukey, J. W. (1965). *An algorithm for the machine calculation of complex Fourier series.* Mathematics of Computation.

3. Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information.* Cambridge University Press.

---

## Contributing

To add new benchmarks:

1. Create a new file in `bench/` following the naming convention
2. Implement the benchmark function with proper documentation
3. Add CSV output to `bench/results/`
4. Update this README with the new benchmark description
5. Add the benchmark to `run_all_benchmarks.jl`

