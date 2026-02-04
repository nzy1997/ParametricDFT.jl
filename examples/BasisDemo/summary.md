# Basis Comparison Summary (MNIST)

## ⚠️ WARNING: TEBD Overfitting Detected

The trained TEBD shows high PSNR on MNIST but **does not generalize** to other image types:

| Image Type | Trained TEBD | Standard QFT |
|------------|--------------|--------------|
| MNIST (10% kept) | 27.59 dB | 20.18 dB |
| Synthetic (10% kept) | 13.33 dB | 23.95 dB |

**Conclusion:** The TEBD has overfit to MNIST images. Use Standard/Trained QFT for general compression.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST |
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

## Results (MNIST Test Set)

### 🏆 Best at 10% kept: **Trained TEBD** (PSNR: 27.59 dB) ⚠️ (overfit)

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

## Generalization Test (Synthetic Images)

| Basis | 10% kept | 20% kept |
|-------|----------|----------|
| Standard QFT | 23.95 dB | 26.07 dB |
| Trained QFT | 23.88 dB | 26.33 dB |
| Trained TEBD | 13.33 dB | 20.97 dB |
| Classical FFT | 23.91 dB | 25.98 dB |

## Learned Parameters

### Entanglement Phases
```
[-0.284, -0.0845, -0.0214, -0.0118, -0.007]
```

### TEBD Phases
```
[0.0476, -0.0012, -0.0771, -0.0186, 0.0204, 0.0571, -0.1754, 0.0583, 0.1169, -0.0122]
```

## Output Files

- `trained_qft.json` - Trained QFT basis
- `trained_entangled_qft.json` - Trained Entangled QFT basis
- `trained_tebd.json` - Trained TEBD basis ⚠️ (overfit)
- `original_5.png` - Original test image
- `recovered_*.png` - Recovered images for each basis
