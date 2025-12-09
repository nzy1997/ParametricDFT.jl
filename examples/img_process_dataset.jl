# ================================================================================
# Dataset Image Processing Example with ParametricDFT.jl
# ================================================================================
# This example demonstrates batch processing of a dataset using ParametricDFT.jl
# for image compression. It processes all images in the data folder, trains a
# parametric quantum Fourier transform, and compares results with classical FFT.

# Activate the project environment to ensure all dependencies are available
import Pkg
script_dir = @__DIR__
Pkg.activate(script_dir)

# Set up paths relative to script location
workspace_root = dirname(script_dir)  # Go up from examples/ to workspace root

using Images
using FFTW
using ParametricDFT
using DelimitedFiles
using ImageQualityIndexes
using Statistics
using ManifoldDiff
using Manopt
using ADTypes

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
    resize_to_power_of_2(img, target_size)

Resize an image to a power-of-2 size (e.g., 512x512).
"""
function resize_to_power_of_2(img, target_size::Int)
    return imresize(img, (target_size, target_size))
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

"""
    process_single_image(img_path, tensors, optcode, optcode_inv, m, n, compression_ratio)

Process a single image through both FFT and parametric DFT compression.

# Returns
A dictionary with all metrics and timing information.
"""
function process_single_image(img_path, tensors, optcode, optcode_inv, m, n, compression_ratio)
    try
        # Load and preprocess image
        img = Images.load(img_path)
        img = resize_to_power_of_2(img, 2^m)
        img_gray = img2gray(img)
        mat_gray = img2mat(img_gray)
        img_matrix = Complex{Float64}.(mat_gray)
        
        # ===== Classical FFT Processing =====
        fft_forward_time = @elapsed fft_result = fftshift(fft(mat_gray))
        fft_truncated, _, _ = compress_by_ratio(fft_result, compression_ratio)
        fft_inverse_time = @elapsed img_fft_compressed = Gray.(real.(ifft(ifftshift(fft_truncated))))
        
        # ===== Parametric DFT Processing =====
        parametric_forward_time = @elapsed ft_img = reshape(
            optcode(tensors..., reshape(img_matrix, fill(2, m+n)...)), 
            2^m, 2^n
        )
        ft_img_truncated, _, _ = compress_by_ratio(ft_img, compression_ratio)
        parametric_inverse_time = @elapsed img_reconstructed = ParametricDFT.ift_mat(
            conj.(tensors), optcode_inv, m, n, ft_img_truncated
        )
        img_parametric_compressed = Gray.(real.(img_reconstructed))
        
        # ===== Calculate Quality Metrics =====
        ssim_fft = assess_ssim(img_gray, img_fft_compressed)
        ssim_parametric = assess_ssim(img_gray, img_parametric_compressed)
        
        mse_fft = sum(abs2.(img_gray .- img_fft_compressed)) / length(img_gray)
        mse_parametric = sum(abs2.(img_gray .- img_parametric_compressed)) / length(img_gray)
        
        psnr_fft = assess_psnr(img_gray, img_fft_compressed)
        psnr_parametric = assess_psnr(img_gray, img_parametric_compressed)
        
        msssim_fft = assess_msssim(img_gray, img_fft_compressed)
        msssim_parametric = assess_msssim(img_gray, img_parametric_compressed)
        
        return Dict(
            "filename" => basename(img_path),
            "ssim_fft" => ssim_fft,
            "ssim_parametric" => ssim_parametric,
            "mse_fft" => mse_fft,
            "mse_parametric" => mse_parametric,
            "psnr_fft" => psnr_fft,
            "psnr_parametric" => psnr_parametric,
            "msssim_fft" => msssim_fft,
            "msssim_parametric" => msssim_parametric,
            "fft_forward_time" => fft_forward_time,
            "fft_inverse_time" => fft_inverse_time,
            "parametric_forward_time" => parametric_forward_time,
            "parametric_inverse_time" => parametric_inverse_time,
            "success" => true
        )
    catch e
        println("Error processing $(basename(img_path)): $e")
        return Dict(
            "filename" => basename(img_path),
            "success" => false,
            "error" => string(e)
        )
    end
end

# ================================================================================
# Section 2: Load Dataset
# ================================================================================

println("="^80)
println("Dataset Image Processing with ParametricDFT.jl")
println("="^80)

# Configuration
data_dir = joinpath(workspace_root, "data")
target_size = 512  # 2^9 = 512
m, n = 9, 9  # 2^9 × 2^9 = 512×512 pixels
# compression_ratio = 0.95  # 95% compression (keep only 5% of coefficients)
# compression_ratio = 0.70  # 70% compression (keep only 30% of coefficients)
# compression_ratio = 0.80  # 70% compression (keep only 30% of coefficients)
compression_ratio = 0.90  # 70% compression (keep only 30% of coefficients)

# Enhanced training configuration (TEST VERSION - reduced for faster testing)
training_steps_per_image = 200  # Reduced steps per image for testing
num_epochs = 3  # Single epoch for testing
use_reconstruction_loss = true  # Use reconstruction loss instead of L1 norm
validation_split = 0.2  # 20% of images for validation
shuffle_images = false  # Disable shuffling for faster testing
learning_rate = 0.01  # Initial learning rate (will be used by gradient_descent)
convergence_threshold = 1e-5  # Stop if loss change is below this

# Find all image files in the data directory
image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
all_files = readdir(data_dir)
image_files = String[]
for file in all_files
    file_lower = lowercase(file)
    if any(endswith(file_lower, ext_lower) for ext_lower in lowercase.(image_extensions))
        push!(image_files, joinpath(data_dir, file))
    end
end

# Sort files for consistent processing order
sort!(image_files)

# Limit to first N images for testing (TEST VERSION - reduced for faster testing)
max_images = 50
if length(image_files) > max_images
    image_files = image_files[1:max_images]
    println("\nFound $(length(image_files)) images in $data_dir (limited to first $max_images for testing)")
else
    println("\nFound $(length(image_files)) images in $data_dir")
end

if length(image_files) == 0
    error("No images found in $data_dir")
end

# Split into training and validation sets
using Random
Random.seed!(42)  # For reproducibility
if shuffle_images
    shuffle!(image_files)
end

num_validation = max(1, Int(round(length(image_files) * validation_split)))
validation_files = image_files[1:num_validation]
training_files = image_files[num_validation+1:end]

println("\nDataset split:")
println("  Training images: $(length(training_files))")
println("  Validation images: $(length(validation_files))")

# ================================================================================
# Section 3: Enhanced Training with Multiple Epochs and Validation
# ================================================================================

println("\n" * "="^80)
println("Enhanced Training Parametric 2D DFT with $m × $n qubits")
println("Training configuration:")
println("  Loss function: $(use_reconstruction_loss ? "Reconstruction MSE" : "L1 Norm (frequency domain)")")
println("  Epochs: $num_epochs")
println("  Steps per image: $training_steps_per_image")
println("  Training images: $(length(training_files))")
println("  Validation images: $(length(validation_files))")
println("  Learning rate: $learning_rate")
println("This will take some time...")
println("="^80)

# Initialize with QFT code and get initial tensors
optcode, initial_tensors = ParametricDFT.qft_code(m, n)
M = ParametricDFT.generate_manifold(initial_tensors)
current_theta = ParametricDFT.tensors2point(initial_tensors, M)
# Use Ref to handle scoping in nested loops
current_theta_ref = Ref(current_theta)

# Prepare inverse transform (needed for reconstruction loss)
optcode_inv, tensors_inv = ParametricDFT.qft_code(m, n; inverse=true)

# Helper function to compute reconstruction loss
function compute_reconstruction_loss(img_matrix, tensors, optcode, optcode_inv, compression_ratio)
    # Forward transform
    ft_img = reshape(optcode(tensors..., reshape(img_matrix, fill(2, m+n)...)), 2^m, 2^n)
    
    # Apply compression (simulate compression during training)
    ft_img_truncated, _, _ = compress_by_ratio(ft_img, compression_ratio)
    
    # Inverse transform
    img_reconstructed = ParametricDFT.ift_mat(conj.(tensors), optcode_inv, m, n, ft_img_truncated)
    
    # Compute MSE between original and reconstructed
    mse = sum(abs2.(img_matrix .- img_reconstructed)) / length(img_matrix)
    return mse
end

# Helper function to train on a single image starting from current parameters
function train_on_image(img_matrix, current_theta, steps, optcode, optcode_inv, M, compression_ratio, use_recon_loss)
    # Define loss function
    if use_recon_loss
        # Reconstruction loss: optimize for end-to-end compression quality
        f(M, p) = begin
            tensors = ParametricDFT.point2tensors(p, M)
            return compute_reconstruction_loss(img_matrix, tensors, optcode, optcode_inv, compression_ratio)
        end
    else
        # Original L1 norm loss: encourages sparsity in frequency domain
        f(M, p) = begin
            tensors = ParametricDFT.point2tensors(p, M)
            fft_pic = reshape(optcode(tensors..., reshape(img_matrix, fill(2, m+n)...)), 2^m, 2^n)
            return sum(abs.(fft_pic))
        end
    end
    
    # Define gradient function
    grad_f2(M, p) = ManifoldDiff.gradient(
        M, x->f(M, x), p, 
        RiemannianProjectionBackend(AutoZygote())
    )
    
    # Run gradient descent
    result = gradient_descent(
        M, f, grad_f2, current_theta;
        debug = [],
        stopping_criterion = StopAfterIteration(steps) | StopWhenGradientNormLess(1e-5)
    )
    
    return result
end

# Helper function to evaluate validation loss
function evaluate_validation_loss(validation_files, current_theta, optcode, optcode_inv, M, compression_ratio)
    total_loss = 0.0
    count = 0
    
    for img_path in validation_files
        try
            img = Images.load(img_path)
            img = resize_to_power_of_2(img, target_size)
            img_gray = img2gray(img)
            mat_gray = img2mat(img_gray)
            img_matrix = Complex{Float64}.(mat_gray)
            
            tensors = ParametricDFT.point2tensors(current_theta, M)
            loss = compute_reconstruction_loss(img_matrix, tensors, optcode, optcode_inv, compression_ratio)
            total_loss += loss
            count += 1
        catch e
            # Skip failed images
        end
    end
    
    return count > 0 ? total_loss / count : Inf
end

# Train with multiple epochs and validation
function train_with_validation!(
    current_theta_ref, training_files, validation_files, optcode, optcode_inv, M,
    num_epochs, training_steps_per_image, compression_ratio, use_reconstruction_loss,
    shuffle_images, target_size
)
    # Evaluate initial validation loss
    initial_val_loss = evaluate_validation_loss(validation_files, current_theta_ref[], optcode, optcode_inv, M, compression_ratio)
    println("\nInitial validation loss: $(round(initial_val_loss, digits=8))")
    
    # Local variables (no global needed in function scope)
    best_val_loss = initial_val_loss
    best_theta = current_theta_ref[]
    patience = 2  # Stop if validation doesn't improve for this many epochs
    patience_counter = 0
    
    # Train for multiple epochs
    for epoch in 1:num_epochs
        println("\n" * "="^80)
        println("Epoch $epoch/$num_epochs")
        println("="^80)
        
        # Shuffle training images each epoch
        if shuffle_images && epoch > 1
            shuffle!(training_files)
            println("Shuffled training images")
        end
        
        epoch_losses = Float64[]
        successful_training = 0
        
        # Train on each image in training set
        for (idx, img_path) in enumerate(training_files)
            try
                # Load and preprocess image
                img = Images.load(img_path)
                img = resize_to_power_of_2(img, target_size)
                img_gray = img2gray(img)
                mat_gray = img2mat(img_gray)
                img_matrix = Complex{Float64}.(mat_gray)
                
                # Train on this image starting from current parameters
                print("  [$(idx)/$(length(training_files))] Training on $(basename(img_path))...\r")
                # Update theta using Ref to avoid scoping issues
                current_theta_ref[] = train_on_image(
                    img_matrix, current_theta_ref[], training_steps_per_image, 
                    optcode, optcode_inv, M, compression_ratio, use_reconstruction_loss
                )
                
                # Compute loss for this image
                tensors = ParametricDFT.point2tensors(current_theta_ref[], M)
                if use_reconstruction_loss
                    loss = compute_reconstruction_loss(img_matrix, tensors, optcode, optcode_inv, compression_ratio)
                else
                    fft_pic = reshape(optcode(tensors..., reshape(img_matrix, fill(2, m+n)...)), 2^m, 2^n)
                    loss = sum(abs.(fft_pic))
                end
                push!(epoch_losses, loss)
                successful_training += 1
                
            catch e
                println("\n  Warning: Failed to train on $(basename(img_path)): $e")
            end
        end
        println()  # New line after progress
        
        if successful_training == 0
            error("No images were successfully used for training in epoch $epoch")
        end
        
        # Evaluate validation loss
        val_loss = evaluate_validation_loss(validation_files, current_theta_ref[], optcode, optcode_inv, M, compression_ratio)
        avg_train_loss = mean(epoch_losses)
        
        println("\nEpoch $epoch results:")
        println("  Average training loss: $(round(avg_train_loss, digits=8))")
        println("  Validation loss: $(round(val_loss, digits=8))")
        
        # Check if validation improved
        if val_loss < best_val_loss
            improvement = (best_val_loss - val_loss) / best_val_loss * 100
            println("  ✓ Validation improved by $(round(improvement, digits=2))%")
            best_val_loss = val_loss
            best_theta = current_theta_ref[]
            patience_counter = 0
        else
            patience_counter += 1
            println("  ✗ Validation did not improve (patience: $patience_counter/$patience)")
            if patience_counter >= patience && epoch > 1
                println("\nEarly stopping: validation loss not improving")
                current_theta_ref[] = best_theta
                break
            end
        end
    end
    
    # Return best parameters
    return best_theta, best_val_loss
end

# Run training
final_best_theta, final_best_val_loss = train_with_validation!(
    current_theta_ref, training_files, validation_files, optcode, optcode_inv, M,
    num_epochs, training_steps_per_image, compression_ratio, use_reconstruction_loss,
    shuffle_images, target_size
)

# Use best parameters found during training
current_theta = final_best_theta
println("\n✓ Training completed. Best validation loss: $(round(final_best_val_loss, digits=8))")

# Convert final trained parameters to tensors
tensors = ParametricDFT.point2tensors(current_theta, M)

println("\n✓ Parametric DFT training completed")

# ================================================================================
# Section 4: Process All Images in Dataset (Similar to img_process.jl lines 117-229)
# ================================================================================

# Process each image (use all images for final evaluation)
all_eval_files = vcat(training_files, validation_files)
sort!(all_eval_files)  # Sort for consistent output

println("\n" * "="^10)
println("Processing $(length(all_eval_files)) images for comparison...")
println("="^10)

# Storage for all results
all_results = Dict[]

for (idx, img_path) in enumerate(all_eval_files)
    println("\n" * "-"^10)
    println("[$(idx)/$(length(all_eval_files))] Processing: $(basename(img_path))")
    println("-"^10)
    
    try
        # Load and preprocess image
        img = Images.load(img_path)
        img = resize_to_power_of_2(img, 2^m)
        img_gray = img2gray(img)
        mat_gray = img2mat(img_gray)
        img_matrix = Complex{Float64}.(mat_gray)
        
        println("Image size: $(size(mat_gray))")
        
        # ===== Classical FFT Processing (similar to img_process.jl lines 121-144) =====
        println("\nComputing classical FFT...")
        @time fft_result = fftshift(fft(mat_gray))
        fft_forward_time = @elapsed fftshift(fft(mat_gray))
        
        # Visualize the frequency domain (normalized for display)
        fft_magnitude = abs.(fft_result)
        fft_normalized = fft_magnitude ./ maximum(fft_magnitude)
        
        # Truncate frequency domain by fixed compression ratio
        fft_truncated, keep_count_fft, total_coeffs_fft = compress_by_ratio(fft_result, compression_ratio)
        
        # Inverse FFT to get compressed image
        @time img_fft_compressed = Gray.(real.(ifft(ifftshift(fft_truncated))))
        fft_inverse_time = @elapsed real.(ifft(ifftshift(fft_truncated)))
        
        println("Classical FFT compression completed")
        
        # ===== Parametric DFT Processing (similar to img_process.jl lines 150-171) =====
        println("\nApplying parametric DFT...")
        
        # Apply the learned 2D DFT
        @time ft_img = reshape(optcode(tensors..., reshape(img_matrix, fill(2, m+n)...)), 2^m, 2^n)
        parametric_forward_time = @elapsed reshape(optcode(tensors..., reshape(img_matrix, fill(2, m+n)...)), 2^m, 2^n)
        
        # Visualize the frequency domain representation
        ft_img_normalized = abs.(ft_img) ./ maximum(abs.(ft_img))
        
        # Truncate in frequency domain by fixed compression ratio
        ft_img_truncated, keep_count, total_coeffs = compress_by_ratio(ft_img, compression_ratio)
        zero_count = total_coeffs - keep_count
        
        println("\nFixed compression ratio:")
        println("  Total coefficients: $total_coeffs")
        println("  Coefficients to keep: $keep_count")
        println("  Coefficients to zero: $zero_count")
        println("  Compression ratio: $(round(100 * compression_ratio, digits=1))%")
        
        # Inverse DFT to reconstruct the compressed image
        @time img_reconstructed = ParametricDFT.ift_mat(conj.(tensors), optcode_inv, m, n, ft_img_truncated)
        parametric_inverse_time = @elapsed ParametricDFT.ift_mat(conj.(tensors), optcode_inv, m, n, ft_img_truncated)
        img_parametric_compressed = Gray.(real.(img_reconstructed))
        
        # ===== Calculate Quality Metrics (similar to img_process.jl lines 198-209) =====
        ssim_fft = assess_ssim(img_gray, img_fft_compressed)
        ssim_parametric = assess_ssim(img_gray, img_parametric_compressed)
        
        mse_fft = sum(abs2.(img_gray .- img_fft_compressed)) / length(img_gray)
        mse_parametric = sum(abs2.(img_gray .- img_parametric_compressed)) / length(img_gray)
        
        psnr_fft = assess_psnr(img_gray, img_fft_compressed)
        psnr_parametric = assess_psnr(img_gray, img_parametric_compressed)
        
        msssim_fft = assess_msssim(img_gray, img_fft_compressed)
        msssim_parametric = assess_msssim(img_gray, img_parametric_compressed)
        
        # Display results table for this image (similar to img_process.jl lines 211-221)
        println("\n" * "="^60)
        println("Results for $(basename(img_path)):")
        println("="^60)
        println("┌─────────────────┬─────────────────┬─────────────────┐")
        println("│ Metric          │ Classical FFT   │ Parametric DFT  │")
        println("├─────────────────┼─────────────────┼─────────────────┤")
        println("│ SSIM            │ $(lpad(round(ssim_fft, digits=4), 15)) │ $(lpad(round(ssim_parametric, digits=4), 15)) │")
        println("│ MSE             │ $(lpad(round(mse_fft, digits=4), 15)) │ $(lpad(round(mse_parametric, digits=4), 15)) │")
        println("│ PSNR (dB)       │ $(lpad(round(psnr_fft, digits=2), 15)) │ $(lpad(round(psnr_parametric, digits=2), 15)) │")
        println("│ MS-SSIM         │ $(lpad(round(msssim_fft, digits=4), 15)) │ $(lpad(round(msssim_parametric, digits=4), 15)) │")
        println("│ Forward FFT (s) │ $(lpad(round(fft_forward_time, digits=6), 15)) │ $(lpad(round(parametric_forward_time, digits=6), 15)) │")
        println("│ Inverse FFT (s) │ $(lpad(round(fft_inverse_time, digits=6), 15)) │ $(lpad(round(parametric_inverse_time, digits=6), 15)) │")
        println("└─────────────────┴─────────────────┴─────────────────┘")
        
        # Store results
        push!(all_results, Dict(
            "filename" => basename(img_path),
            "ssim_fft" => ssim_fft,
            "ssim_parametric" => ssim_parametric,
            "mse_fft" => mse_fft,
            "mse_parametric" => mse_parametric,
            "psnr_fft" => psnr_fft,
            "psnr_parametric" => psnr_parametric,
            "msssim_fft" => msssim_fft,
            "msssim_parametric" => msssim_parametric,
            "fft_forward_time" => fft_forward_time,
            "fft_inverse_time" => fft_inverse_time,
            "parametric_forward_time" => parametric_forward_time,
            "parametric_inverse_time" => parametric_inverse_time,
            "success" => true
        ))
        
    catch e
        println("Error processing $(basename(img_path)): $e")
        push!(all_results, Dict(
            "filename" => basename(img_path),
            "success" => false,
            "error" => string(e)
        ))
    end
end

# Filter successful results
successful_results = filter(r -> get(r, "success", false), all_results)
failed_count = length(all_results) - length(successful_results)

println("\n" * "="^80)
println("Processing Summary:")
println("="^80)
println("Total images: $(length(all_eval_files))")
println("Successfully processed: $(length(successful_results))")
println("Failed: $failed_count")

# ================================================================================
# Section 5: Aggregate Statistics
# ================================================================================

if length(successful_results) > 0
    println("\n" * "="^80)
    println("Aggregate Statistics Across Dataset:")
    println("="^80)
    
    # Extract metrics
    ssim_fft_vals = [r["ssim_fft"] for r in successful_results]
    ssim_parametric_vals = [r["ssim_parametric"] for r in successful_results]
    mse_fft_vals = [r["mse_fft"] for r in successful_results]
    mse_parametric_vals = [r["mse_parametric"] for r in successful_results]
    psnr_fft_vals = [r["psnr_fft"] for r in successful_results]
    psnr_parametric_vals = [r["psnr_parametric"] for r in successful_results]
    msssim_fft_vals = [r["msssim_fft"] for r in successful_results]
    msssim_parametric_vals = [r["msssim_parametric"] for r in successful_results]
    fft_forward_times = [r["fft_forward_time"] for r in successful_results]
    fft_inverse_times = [r["fft_inverse_time"] for r in successful_results]
    parametric_forward_times = [r["parametric_forward_time"] for r in successful_results]
    parametric_inverse_times = [r["parametric_inverse_time"] for r in successful_results]
    
    # Calculate statistics
    function stats_summary(vals)
        return Dict(
            "mean" => mean(vals),
            "std" => std(vals),
            "min" => minimum(vals),
            "max" => maximum(vals),
            "median" => median(vals)
        )
    end
    
    ssim_fft_stats = stats_summary(ssim_fft_vals)
    ssim_parametric_stats = stats_summary(ssim_parametric_vals)
    mse_fft_stats = stats_summary(mse_fft_vals)
    mse_parametric_stats = stats_summary(mse_parametric_vals)
    psnr_fft_stats = stats_summary(psnr_fft_vals)
    psnr_parametric_stats = stats_summary(psnr_parametric_vals)
    msssim_fft_stats = stats_summary(msssim_fft_vals)
    msssim_parametric_stats = stats_summary(msssim_parametric_vals)
    fft_forward_stats = stats_summary(fft_forward_times)
    fft_inverse_stats = stats_summary(fft_inverse_times)
    parametric_forward_stats = stats_summary(parametric_forward_times)
    parametric_inverse_stats = stats_summary(parametric_inverse_times)
    
    # Display results table
    println("\n┌─────────────────────┬───────────────────────────────────┬───────────────────────────────────┐")
    println("│ Metric              │ Classical FFT (Mean ± Std)         │ Parametric DFT (Mean ± Std)       │")
    println("├─────────────────────┼───────────────────────────────────┼───────────────────────────────────┤")
    println("│ SSIM                │ $(lpad(round(ssim_fft_stats["mean"], digits=4), 8)) ± $(lpad(round(ssim_fft_stats["std"], digits=4), 8))      │ $(lpad(round(ssim_parametric_stats["mean"], digits=4), 8)) ± $(lpad(round(ssim_parametric_stats["std"], digits=4), 8))      │")
    println("│ MSE                 │ $(lpad(round(mse_fft_stats["mean"], digits=6), 8)) ± $(lpad(round(mse_fft_stats["std"], digits=6), 8))      │ $(lpad(round(mse_parametric_stats["mean"], digits=6), 8)) ± $(lpad(round(mse_parametric_stats["std"], digits=6), 8))      │")
    println("│ PSNR (dB)           │ $(lpad(round(psnr_fft_stats["mean"], digits=2), 8)) ± $(lpad(round(psnr_fft_stats["std"], digits=2), 8))      │ $(lpad(round(psnr_parametric_stats["mean"], digits=2), 8)) ± $(lpad(round(psnr_parametric_stats["std"], digits=2), 8))      │")
    println("│ MS-SSIM             │ $(lpad(round(msssim_fft_stats["mean"], digits=4), 8)) ± $(lpad(round(msssim_fft_stats["std"], digits=4), 8))      │ $(lpad(round(msssim_parametric_stats["mean"], digits=4), 8)) ± $(lpad(round(msssim_parametric_stats["std"], digits=4), 8))      │")
    println("│ Forward Time (s)   │ $(lpad(round(fft_forward_stats["mean"], digits=6), 8)) ± $(lpad(round(fft_forward_stats["std"], digits=6), 8))      │ $(lpad(round(parametric_forward_stats["mean"], digits=6), 8)) ± $(lpad(round(parametric_forward_stats["std"], digits=6), 8))      │")
    println("│ Inverse Time (s)   │ $(lpad(round(fft_inverse_stats["mean"], digits=6), 8)) ± $(lpad(round(fft_inverse_stats["std"], digits=6), 8))      │ $(lpad(round(parametric_inverse_stats["mean"], digits=6), 8)) ± $(lpad(round(parametric_inverse_stats["std"], digits=6), 8))      │")
    println("└─────────────────────┴───────────────────────────────────┴───────────────────────────────────┘")
    
    # Detailed statistics
    println("\nDetailed Statistics:")
    println("\nSSIM:")
    println("  Classical FFT:    mean=$(round(ssim_fft_stats["mean"], digits=4)), std=$(round(ssim_fft_stats["std"], digits=4)), min=$(round(ssim_fft_stats["min"], digits=4)), max=$(round(ssim_fft_stats["max"], digits=4)), median=$(round(ssim_fft_stats["median"], digits=4))")
    println("  Parametric DFT:   mean=$(round(ssim_parametric_stats["mean"], digits=4)), std=$(round(ssim_parametric_stats["std"], digits=4)), min=$(round(ssim_parametric_stats["min"], digits=4)), max=$(round(ssim_parametric_stats["max"], digits=4)), median=$(round(ssim_parametric_stats["median"], digits=4))")
    
    println("\nPSNR (dB):")
    println("  Classical FFT:    mean=$(round(psnr_fft_stats["mean"], digits=2)), std=$(round(psnr_fft_stats["std"], digits=2)), min=$(round(psnr_fft_stats["min"], digits=2)), max=$(round(psnr_fft_stats["max"], digits=2)), median=$(round(psnr_fft_stats["median"], digits=2))")
    println("  Parametric DFT:   mean=$(round(psnr_parametric_stats["mean"], digits=2)), std=$(round(psnr_parametric_stats["std"], digits=2)), min=$(round(psnr_parametric_stats["min"], digits=2)), max=$(round(psnr_parametric_stats["max"], digits=2)), median=$(round(psnr_parametric_stats["median"], digits=2))")
    
    println("\nTiming (seconds):")
    println("  Classical FFT Forward:    mean=$(round(fft_forward_stats["mean"], digits=6)), std=$(round(fft_forward_stats["std"], digits=6))")
    println("  Classical FFT Inverse:    mean=$(round(fft_inverse_stats["mean"], digits=6)), std=$(round(fft_inverse_stats["std"], digits=6))")
    println("  Parametric DFT Forward:   mean=$(round(parametric_forward_stats["mean"], digits=6)), std=$(round(parametric_forward_stats["std"], digits=6))")
    println("  Parametric DFT Inverse:   mean=$(round(parametric_inverse_stats["mean"], digits=6)), std=$(round(parametric_inverse_stats["std"], digits=6))")
    
    # ================================================================================
    # Section 6: Save Results to CSV
    # ================================================================================
    
    println("\n" * "="^80)
    println("Saving results to CSV...")
    println("="^80)
    
    # Prepare data for CSV
    csv_data = Matrix{Any}(undef, length(all_results), 14)
    csv_data[:, 1] = [r["filename"] for r in all_results]
    csv_data[:, 2] = [get(r, "success", false) ? "✓" : "✗" for r in all_results]
    csv_data[:, 3] = [get(r, "ssim_fft", NaN) for r in all_results]
    csv_data[:, 4] = [get(r, "ssim_parametric", NaN) for r in all_results]
    csv_data[:, 5] = [get(r, "mse_fft", NaN) for r in all_results]
    csv_data[:, 6] = [get(r, "mse_parametric", NaN) for r in all_results]
    csv_data[:, 7] = [get(r, "psnr_fft", NaN) for r in all_results]
    csv_data[:, 8] = [get(r, "psnr_parametric", NaN) for r in all_results]
    csv_data[:, 9] = [get(r, "msssim_fft", NaN) for r in all_results]
    csv_data[:, 10] = [get(r, "msssim_parametric", NaN) for r in all_results]
    csv_data[:, 11] = [get(r, "fft_forward_time", NaN) for r in all_results]
    csv_data[:, 12] = [get(r, "fft_inverse_time", NaN) for r in all_results]
    csv_data[:, 13] = [get(r, "parametric_forward_time", NaN) for r in all_results]
    csv_data[:, 14] = [get(r, "parametric_inverse_time", NaN) for r in all_results]
    
    header = [
        "filename", "success", 
        "ssim_fft", "ssim_parametric",
        "mse_fft", "mse_parametric",
        "psnr_fft", "psnr_parametric",
        "msssim_fft", "msssim_parametric",
        "fft_forward_time", "fft_inverse_time",
        "parametric_forward_time", "parametric_inverse_time"
    ]
    
    compression_str = string(Int(round(compression_ratio * 100)))
    output_file = joinpath(workspace_root, "dataset_compression_results_$(compression_str).csv")
    open(output_file, "w") do io
        writedlm(io, [header], ',')
        writedlm(io, csv_data, ',')
    end
    
    println("✓ Results saved to: $output_file")
    
    println("\n" * "="^80)
    println("Legend:")
    println("="^80)
    println("• SSIM: Structural Similarity Index (0-1, higher is better)")
    println("• MSE: Mean Squared Error (lower is better)")
    println("• PSNR: Peak Signal-to-Noise Ratio in dB (higher is better)")
    println("• MS-SSIM: Multi-Scale Structural Similarity (0-1, higher is better)")
    println("• Forward/Inverse Time: Time for transformation in seconds (lower is better)")
    
    println("\n✓ Dataset processing completed!")
else
    println("\n✗ No images were successfully processed. Please check the error messages above.")
end

