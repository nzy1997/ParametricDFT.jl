# ================================================================================
# Image Processing Example with ParametricDFT.jl
# ================================================================================
# This example demonstrates the use of ParametricDFT.jl for image compression
# by learning a parametric quantum Fourier transform and truncating in the
# frequency domain.

using Images
using FFTW
using ParametricDFT
using DelimitedFiles

# ================================================================================
# Section 1: Utility Functions for Image Processing
# ================================================================================

"""
    img2gray(img)

Convert an RGB image to grayscale by extracting the green channel.
"""
function img2gray(img)
    return [Gray(img[i, j].g) for i in 1:size(img, 1), j in 1:size(img, 2)]
end

"""
    img2mat(img)

Convert a grayscale image to a matrix of Float64 values.
"""
function img2mat(img)
    return [img[i, j].val for i in 1:size(img, 1), j in 1:size(img, 2)]
end

"""
    mat2img(mat)

Convert a matrix of Float64 values back to a grayscale image.
"""
function mat2img(mat)
    return Gray.(mat)
end

"""
    compress_by_ratio(freq_domain, compression_ratio)

Compress frequency domain data by keeping only the largest coefficients.

# Arguments
- `freq_domain`: Complex frequency domain data (2D array)
- `compression_ratio`: Compression ratio (0.0 = no compression, 1.0 = full compression)

# Returns
- `compressed_data`: Compressed frequency domain data
- `keep_count`: Number of coefficients kept
- `total_coeffs`: Total number of coefficients
"""
function compress_by_ratio(freq_domain, compression_ratio)
    # Get all coefficients and sort by absolute value
    all_coeffs = vec(freq_domain)
    sorted_indices = sortperm(abs.(all_coeffs), rev=true)
    
    # Calculate how many coefficients to keep
    total_coeffs = length(all_coeffs)
    keep_count = Int(round(total_coeffs * (1 - compression_ratio)))
    
    # Create compressed version by keeping only the largest coefficients
    compressed_data = zeros(Complex{Float64}, size(freq_domain))
    keep_indices = sorted_indices[1:keep_count]
    compressed_data[keep_indices] = freq_domain[keep_indices]
    
    return compressed_data, keep_count, total_coeffs
end

# ================================================================================
# Section 2: Load and Prepare Image
# ================================================================================

# Load image and downsample to 64×64 pixels
# test_img = Images.load("examples/cat.png")
test_img = Images.load("data/cat_005.jpg")
# test_img = test_img[101:4:356, 301:4:556]
test_img = test_img[1:512, 1:512]

# Convert to grayscale and then to matrix
img_gray = img2gray(test_img)
mat_gray = img2mat(img_gray)

println("Image size: $(size(mat_gray))")

# ================================================================================
# Section 3: Parametric Quantum DFT Training (using ParametricDFT.jl)
# ================================================================================

# Prepare image matrix for processing - convert to complex for quantum circuit
img_matrix = Complex{Float64}.(mat_gray)
m, n = 9, 9  # 2^9 × 2^9 = 512×512 pixels

println("\nTraining parametric 2D DFT with $m × $n qubits...")
println("This will take several minutes (50 steps)...\n")

# Set k for MSE loss: keep top 30% of coefficients (matching compression ratio)
k = Int(round(2^(m+n) * 0.30))
println("MSE loss: keeping top $k coefficients out of $(2^(m+n)) total")

# Train the parametric DFT circuit to minimize MSE reconstruction error
@time theta = ParametricDFT.fft_with_training(
    m, n,
    img_matrix, 
    ParametricDFT.MSELoss(k);
    steps = 50
)

# Convert trained parameters to tensors and construct DFT einsum code
optcode, initial_tensors = ParametricDFT.qft_code(m, n)
M = ParametricDFT.generate_manifold(initial_tensors)
tensors = ParametricDFT.point2tensors(theta, M)

println("Parametric DFT training completed")

# ================================================================================
# Section 4: Classical FFT for Comparison (using FFTW.jl)
# ================================================================================

# Compute FFT using classical FFTW
@time fft_result = fftshift(fft(mat_gray))
fft_forward_time = @elapsed fftshift(fft(mat_gray))

# Visualize the frequency domain (normalized for display)
fft_magnitude = abs.(fft_result)
fft_normalized = fft_magnitude ./ maximum(fft_magnitude)
# display(Gray.(fft_normalized))

# ================================================================================
# Section 5: Image Compression with Fixed Compression Ratio
# ================================================================================

# Set compression ratio
# `compression_ratio = 0.95  # 95% compression (keep only 5% of coefficients)
compression_ratio = 0.70  # 70% compression (keep only 30% of coefficients)

# Truncate frequency domain by fixed compression ratio
fft_truncated, keep_count_fft, total_coeffs_fft = compress_by_ratio(fft_result, compression_ratio)

# Compute normalized truncated frequency domain for visualization
fft_truncated_magnitude = abs.(fft_truncated)
fft_truncated_normalized = fft_truncated_magnitude ./ maximum(fft_truncated_magnitude)

# Inverse FFT to get compressed image
@time img_fft_compressed_raw = real.(ifft(ifftshift(fft_truncated)))
fft_inverse_time = @elapsed real.(ifft(ifftshift(fft_truncated)))
# Clamp values to [0, 1] range for valid image storage
img_fft_compressed = Gray.(clamp.(img_fft_compressed_raw, 0.0, 1.0))

println("Classical FFT compression completed")

# ================================================================================
# Section 6: Parametric DFT Application and Compression
# ================================================================================

# Apply the learned 2D DFT
@time ft_img = reshape(optcode(tensors..., reshape(img_matrix, fill(2, m+n)...)), 2^m, 2^n)
parametric_forward_time = @elapsed reshape(optcode(tensors..., reshape(img_matrix, fill(2, m+n)...)), 2^m, 2^n)

# Visualize the frequency domain representation
ft_img_magnitude = abs.(ft_img)
ft_img_normalized = ft_img_magnitude ./ maximum(ft_img_magnitude)
# display(Gray.(ft_img_normalized))

# Truncate in frequency domain by fixed compression ratio
ft_img_truncated, keep_count, total_coeffs = compress_by_ratio(ft_img, compression_ratio)
zero_count = total_coeffs - keep_count

# Compute normalized truncated frequency domain for visualization
ft_img_truncated_magnitude = abs.(ft_img_truncated)
ft_img_truncated_normalized = ft_img_truncated_magnitude ./ maximum(ft_img_truncated_magnitude)

println("\nFixed compression ratio:")
println("  Total coefficients: $total_coeffs")
println("  Coefficients to keep: $keep_count")
println("  Coefficients to zero: $zero_count")
println("  Compression ratio: $(round(100 * compression_ratio, digits=1))%")

# Inverse DFT to reconstruct the compressed image
optcode_inv, tensors_inv = ParametricDFT.qft_code(m, n; inverse=true)
@time img_reconstructed = ParametricDFT.ift_mat(conj.(tensors), optcode_inv, m, n, ft_img_truncated)
parametric_inverse_time = @elapsed ParametricDFT.ift_mat(conj.(tensors), optcode_inv, m, n, ft_img_truncated)
# Clamp values to [0, 1] range for valid image storage
img_parametric_compressed = Gray.(clamp.(real.(img_reconstructed), 0.0, 1.0))

# ================================================================================
# Section 7: Display Results and Save Images
# ================================================================================

# Create results directory if it doesn't exist
results_dir = joinpath(@__DIR__, "results")
if !isdir(results_dir)
    mkdir(results_dir)
    println("Created results directory: $results_dir")
end

# Save spatial domain images
println("\n" * "="^60)
println("Saving images to results folder...")
println("="^60)

# Save original image
save(joinpath(results_dir, "original_image.png"), img_gray)
println("Saved: original_image.png")

# Save classical FFT compressed image
save(joinpath(results_dir, "classical_fft_compressed.png"), img_fft_compressed)
println("Saved: classical_fft_compressed.png")

# Save parametric DFT compressed image
save(joinpath(results_dir, "parametric_dft_compressed.png"), img_parametric_compressed)
println("Saved: parametric_dft_compressed.png")

# Prepare and save frequency domain images (normalized for visualization)
println("\nSaving frequency domain images...")

# Original frequency domain (classical FFT) - reuse existing fft_normalized
save(joinpath(results_dir, "frequency_domain_original_fft.png"), Gray.(fft_normalized))
println("Saved: frequency_domain_original_fft.png")

# Truncated frequency domain (classical FFT) - reuse existing fft_truncated_normalized
save(joinpath(results_dir, "frequency_domain_truncated_fft.png"), Gray.(fft_truncated_normalized))
println("Saved: frequency_domain_truncated_fft.png")

# Parametric DFT frequency domain - reuse existing ft_img_normalized
save(joinpath(results_dir, "frequency_domain_parametric_dft.png"), Gray.(ft_img_normalized))
println("Saved: frequency_domain_parametric_dft.png")

# Truncated parametric DFT frequency domain - reuse existing ft_img_truncated_normalized
save(joinpath(results_dir, "frequency_domain_truncated_parametric_dft.png"), Gray.(ft_img_truncated_normalized))
println("Saved: frequency_domain_truncated_parametric_dft.png")

println("\nAll images saved to: $results_dir")
println("\nExample completed!")
# ================================================================================
# Section 8: Image Quality Assessment
# ================================================================================

println("\n" * "="^60)
println("Image Quality Assessment Results:")
println("="^60)

# Calculate metrics
ssim_fft = assess_ssim(img_gray, img_fft_compressed)
ssim_parametric = assess_ssim(img_gray, img_parametric_compressed)

mse_fft = sum(abs2.(img_gray .- img_fft_compressed)) / length(img_gray)
mse_parametric = sum(abs2.(img_gray .- img_parametric_compressed)) / length(img_gray)

psnr_fft = assess_psnr(img_gray, img_fft_compressed)
psnr_parametric = assess_psnr(img_gray, img_parametric_compressed)

msssim_fft = assess_msssim(img_gray, img_fft_compressed)
msssim_parametric = assess_msssim(img_gray, img_parametric_compressed)

# Create and display results table
println("\n┌─────────────────┬─────────────────┬─────────────────┐")
println("│ Metric          │ Classical FFT   │ Parametric DFT  │")
println("├─────────────────┼─────────────────┼─────────────────┤")
println("│ SSIM            │ $(lpad(round(ssim_fft, digits=4), 15)) │ $(lpad(round(ssim_parametric, digits=4), 15)) │")
println("│ MSE             │ $(lpad(round(mse_fft, digits=4), 15)) │ $(lpad(round(mse_parametric, digits=4), 15)) │")
println("│ PSNR (dB)       │ $(lpad(round(psnr_fft, digits=2), 15)) │ $(lpad(round(psnr_parametric, digits=2), 15)) │")
println("│ MS-SSIM         │ $(lpad(round(msssim_fft, digits=4), 15)) │ $(lpad(round(msssim_parametric, digits=4), 15)) │")
println("│ Forward FFT (s) │ $(lpad(round(fft_forward_time, digits=6), 15)) │ $(lpad(round(parametric_forward_time, digits=6), 15)) │")
println("│ Inverse FFT (s) │ $(lpad(round(fft_inverse_time, digits=6), 15)) │ $(lpad(round(parametric_inverse_time, digits=6), 15)) │")
println("└─────────────────┴─────────────────┴─────────────────┘")

println("\nLegend:")
println("• SSIM: Structural Similarity Index (0-1, higher is better)")
println("• MSE: Mean Squared Error (lower is better)")
println("• PSNR: Peak Signal-to-Noise Ratio in dB (higher is better)")
println("• MS-SSIM: Multi-Scale Structural Similarity (0-1, higher is better)")
println("• Forward FFT (s): Time for forward FFT transformation (lower is better)")
println("• Inverse FFT (s): Time for inverse FFT transformation (lower is better)")