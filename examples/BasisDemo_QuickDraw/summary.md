# Basis Comparison Summary (Quick Draw)

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | Quick Draw |
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

## Results (Quick Draw Test Set)

### 🏆 Best at 10% kept: **Trained QFT** (PSNR: 20.83 dB)

### Compression Quality Comparison (PSNR in dB)

| Basis | 5% kept | 10% kept | 15% kept | 20% kept |
|-------|------|------|------|------|
| Standard QFT | 14.78 | 16.71 | 18.30 | 19.75 |
| Trained QFT | 17.06 | 20.83 | 24.37 | 27.75 |
| Standard Entangled QFT | 14.78 | 16.71 | 18.30 | 19.75 |
| Trained Entangled QFT | 16.87 | 20.75 | 24.30 | 27.64 |
| Standard TEBD | 14.88 | 16.86 | 18.44 | 19.89 |
| Trained TEBD | 17.12 | 20.76 | 24.08 | 27.38 |
| Classical FFT | 14.74 | 16.66 | 18.26 | 19.69 |

### Compression Quality Comparison (SSIM)

| Basis | 5% kept | 10% kept | 15% kept | 20% kept |
|-------|------|------|------|------|
| Standard QFT | 0.3718 | 0.4393 | 0.4873 | 0.5215 |
| Trained QFT | 0.8102 | 0.8975 | 0.9458 | 0.9625 |
| Standard Entangled QFT | 0.3718 | 0.4393 | 0.4873 | 0.5215 |
| Trained Entangled QFT | 0.7776 | 0.8664 | 0.9066 | 0.9203 |
| Standard TEBD | 0.3832 | 0.4779 | 0.5009 | 0.5502 |
| Trained TEBD | 0.7339 | 0.8012 | 0.8401 | 0.8669 |
| Classical FFT | 0.3580 | 0.4340 | 0.4829 | 0.5190 |

### Compression Quality Comparison (MSE)

| Basis | 5% kept | 10% kept | 15% kept | 20% kept |
|-------|------|------|------|------|
| Standard QFT | 0.034050 | 0.021861 | 0.015158 | 0.010853 |
| Trained QFT | 0.020947 | 0.009038 | 0.004229 | 0.002032 |
| Standard Entangled QFT | 0.034050 | 0.021861 | 0.015158 | 0.010853 |
| Trained Entangled QFT | 0.021948 | 0.009304 | 0.004410 | 0.002185 |
| Standard TEBD | 0.033061 | 0.021120 | 0.014716 | 0.010535 |
| Trained TEBD | 0.020545 | 0.009202 | 0.004494 | 0.002203 |
| Classical FFT | 0.034381 | 0.022148 | 0.015287 | 0.011014 |

## Generalization Test (Synthetic Images)

| Basis | 10% kept | 20% kept |
|-------|----------|----------|
| Standard QFT | 23.95 dB | 26.07 dB |
| Trained QFT | 23.25 dB | 26.67 dB |
| Trained TEBD | 22.63 dB | 26.46 dB |

## Learned Parameters

### Entanglement Phases
```
[-0.3643, -0.1076, -0.0547, 0.0262, 0.0134]
```

### TEBD Phases
```
[0.0034, 0.1542, -0.0856, 0.0734, -0.0132, 0.045, 0.0523, -0.1221, -0.0529, -0.0033]
```

## Output Files

- `trained_qft.json` - Trained QFT basis
- `trained_entangled_qft.json` - Trained Entangled QFT basis
- `trained_tebd.json` - Trained TEBD basis
- `original_bicycle.png` - Original test image
- `recovered_*.png` - Recovered images for each basis
