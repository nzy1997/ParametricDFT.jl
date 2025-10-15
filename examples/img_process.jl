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

# ================================================================================
# Section 2: Load and Prepare Image
# ================================================================================

# Load image and downsample to 64×64 pixels
test_img = Images.load("examples/cat.png")
test_img = test_img[101:4:356, 301:4:556]

# Convert to grayscale and then to matrix
img_gray = img2gray(test_img)
mat_gray = img2mat(img_gray)

println("Image size: $(size(mat_gray))")

# ================================================================================
# Section 3: Classical FFT for Comparison (using FFTW.jl)
# ================================================================================

# Compute FFT using classical FFTW
fft_result = fftshift(fft(mat_gray))

# Visualize the frequency domain (normalized for display)
fft_magnitude = abs.(fft_result)
fft_normalized = fft_magnitude ./ maximum(fft_magnitude)
display(Gray.(fft_normalized))

# Truncate frequency domain (keep only low frequencies)
fft_truncated = copy(fft_result)
mid = 32
band_size = 10

# Zero out high frequencies
fft_truncated[1:mid-band_size, :] .= 0
fft_truncated[mid+band_size:end, :] .= 0
fft_truncated[:, 1:mid-band_size] .= 0
fft_truncated[:, mid+band_size:end] .= 0

# Inverse FFT to get compressed image
img_fft_compressed = Gray.(real.(ifft(ifftshift(fft_truncated))))

println("Classical FFT compression completed")

# ================================================================================
# Section 4: Parametric Quantum DFT (using ParametricDFT.jl)
# ================================================================================

# Prepare image vector for processing
img_vector = vec(mat_gray)
qubit_num = 12  # 2^12 = 4096 pixels (64×64)

println("\nTraining parametric DFT with $qubit_num qubits...")
println("This will take several minutes (~9 minutes for 200 steps)...\n")

# Train the parametric DFT circuit to minimize L1 norm
@time theta = ParametricDFT.fft_with_training(
    qubit_num, 
    img_vector, 
    ParametricDFT.L1Norm();
    steps = 200
)

# Convert trained parameters to tensors and construct DFT matrix
tensors = ParametricDFT.point2tensors(theta, qubit_num)
optcode = ParametricDFT.qft_code(qubit_num)[1]
dft_matrix = ParametricDFT.ft_mat(tensors, optcode, qubit_num)

# Optional: Save the trained matrix for later use
# using DelimitedFiles
# writedlm("examples/matrices/epoch200.txt", dft_matrix)

# Optional: Load a pre-trained matrix to skip training
# dft_matrix = readdlm("examples/matrices/epoch200.txt", '\t', Complex{Float64})

# Visualize the learned DFT matrix
display(Gray.(abs.(dft_matrix) ./ maximum(abs.(dft_matrix))))

# ================================================================================
# Section 5: Image Compression with Parametric DFT
# ================================================================================

# Apply the learned DFT to the image
ft_img_vector = dft_matrix * img_vector

# Visualize the frequency domain representation
ft_img_reshaped = reshape(ft_img_vector, 64, 64)
display(Gray.(abs.(ft_img_reshaped)))

# Truncate in frequency domain by thresholding small coefficients
cut_threshold = 0.25
num_zeros = count(x -> abs(x) < cut_threshold, ft_img_vector)
num_kept = count(x -> abs(x) >= cut_threshold, ft_img_vector)

println("\nCompression ratio:")
println("  Coefficients set to zero: $num_zeros / $(length(ft_img_vector))")
println("  Coefficients kept: $num_kept / $(length(ft_img_vector))")
println("  Compression: $(round(100 * num_zeros / length(ft_img_vector), digits=1))%")

# Apply threshold: zero out small coefficients
ft_img_truncated = copy(ft_img_vector)
ft_img_truncated[findall(x -> abs(x) < cut_threshold, ft_img_vector)] .= 0

# Inverse DFT to reconstruct the compressed image
img_reconstructed = dft_matrix' * ft_img_truncated
img_parametric_compressed = Gray.(reshape(img_reconstructed, 64, 64))

# ================================================================================
# Section 6: Display Results
# ================================================================================

println("\n" * "="^60)
println("Results:")
println("="^60)
println("Original image:")
display(img_gray)

println("\nClassical FFT compression:")
display(img_fft_compressed)

println("\nParametric DFT compression:")
display(img_parametric_compressed)

println("\nExample completed!")