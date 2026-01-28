# Basis Comparison Summary

## ⚠️ IMPORTANT: TEBD Results May Be Misleading

**Issue #20 Investigation Results:**

The trained TEBD shows extraordinarily high PSNR on MNIST test images, but this is due to **overfitting to the MNIST domain**, not a genuine improvement in compression quality. Evidence:

| Image Type | TEBD (10% kept) | QFT (10% kept) | Winner |
|------------|-----------------|----------------|--------|
| MNIST digits | 33.18 dB | 21.21 dB | TEBD ✗ (overfit) |
| Synthetic images | 13.28 dB | 23.88 dB | QFT ✓ |
| Random noise | 7.76 dB | 12.60 dB | QFT ✓ |

**Conclusion:** The TEBD transform has learned to specifically concentrate energy for MNIST-like images (28×28 digits padded to 32×32 in a specific position). This does NOT generalize to other image types.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Training images | 20 |
| Test images | 5 |
| Image size | 32×32 |
| Compression ratios | 5%, 10%, 15%, 20% |
| Training epochs | 2 |

## Basis Architectures

| Basis | Architecture |
|-------|-------------|
| QFT | 5 + 5 qubits (2D separable) |
| Entangled QFT | 5 + 5 qubits + 5 entangle gates |
| TEBD | 5 + 5 qubits (2D ring: 5 row + 5 col gates) |

## Results (MNIST Only - Overfitting Warning)

### ⚠️ Best at 10% kept: **Trained TEBD** (PSNR: 27.59 dB) - BUT THIS IS OVERFIT

### Compression Quality Comparison (PSNR in dB)

| Basis | 5% kept | 10% kept | 15% kept | 20% kept |
|-------|------|------|------|------|
| Standard QFT | 17.14 | 20.18 | 22.27 | 23.89 |
| Trained QFT | 17.43 | 20.43 | 22.48 | 24.14 |
| Standard Entangled QFT | 17.14 | 20.18 | 22.27 | 23.89 |
| Trained Entangled QFT | 17.37 | 20.36 | 22.44 | 24.17 |
| Standard TEBD | 16.37 | 18.70 | 20.60 | 22.18 |
| Trained TEBD | 20.23 | 27.59 | 36.61 | 52.93 |
| Classical FFT | 17.09 | 20.12 | 22.23 | 23.83 |

### Compression Quality Comparison (SSIM)

| Basis | 5% kept | 10% kept | 15% kept | 20% kept |
|-------|------|------|------|------|
| Standard QFT | 0.3989 | 0.4942 | 0.5352 | 0.5754 |
| Trained QFT | 0.4258 | 0.5243 | 0.5726 | 0.6122 |
| Standard Entangled QFT | 0.3989 | 0.4942 | 0.5352 | 0.5754 |
| Trained Entangled QFT | 0.4128 | 0.5121 | 0.5767 | 0.6201 |
| Standard TEBD | 0.3809 | 0.4431 | 0.4647 | 0.4901 |
| Trained TEBD | 0.9188 | 0.9805 | 0.9935 | 0.9962 |
| Classical FFT | 0.3941 | 0.4777 | 0.5149 | 0.5626 |

### Compression Quality Comparison (MSE)

| Basis | 5% kept | 10% kept | 15% kept | 20% kept |
|-------|------|------|------|------|
| Standard QFT | 0.019404 | 0.009793 | 0.006077 | 0.004166 |
| Trained QFT | 0.018183 | 0.009227 | 0.005763 | 0.003921 |
| Standard Entangled QFT | 0.019404 | 0.009793 | 0.006077 | 0.004166 |
| Trained Entangled QFT | 0.018474 | 0.009364 | 0.005800 | 0.003885 |
| Standard TEBD | 0.023402 | 0.013654 | 0.008812 | 0.006127 |
| Trained TEBD | 0.010882 | 0.001966 | 0.000285 | 0.000015 |
| Classical FFT | 0.019664 | 0.009912 | 0.006132 | 0.004230 |

## Learned Parameters

### Entanglement Phases
```
[-0.284, -0.0845, -0.0214, -0.0118, -0.007]
```

### TEBD Phases
```
[0.0476, -0.0012, -0.0771, -0.0186, 0.0204, 0.0571, -0.1754, 0.0583, 0.1169, -0.0122]
```

## Generalization Test Results (Issue #20 Verification)

Run `julia --project=examples examples/verify_tebd.jl` to reproduce these tests.

### Key Findings

1. **Train/Test Split**: ✓ No overlap (different MNIST splits)
2. **Compression Zeroing**: ✓ Coefficients correctly zeroed
3. **Transform Unitarity**: ✓ Perfect reconstruction without compression
4. **Reproducibility**: ✓ Results are deterministic
5. **Generalization**: ✗ **FAILS** - TEBD does not generalize to non-MNIST images

### Recommendations

1. **Do not use the trained TEBD for general image compression** - it only works well on MNIST-like images
2. **The standard QFT or Trained QFT are better choices** for general-purpose compression
3. **To fix TEBD overfitting**, consider:
   - Training on diverse image datasets
   - Adding regularization (e.g., weight decay on phases)
   - Using data augmentation (rotations, translations, scaling)
   - Longer training with larger batch sizes

## Output Files

- `trained_qft.json` - Trained QFT basis
- `trained_entangled_qft.json` - Trained Entangled QFT basis
- `trained_tebd.json` - Trained TEBD basis (overfit warning)
- `original_digit_5.png` - Original test image
- `recovered_*.png` - Recovered images for each basis
