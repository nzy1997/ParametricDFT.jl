# ================================================================================
# Sparse Basis Demo with MNIST Dataset
# ================================================================================
# This example demonstrates the complete workflow of the sparse basis features:
#   1. Loading the MNIST dataset
#   2. Training and obtaining a sparse basis
#   3. Dumping and storing the sparse basis to JSON
#   4. Loading the sparse basis from JSON
#   5. Compressing and decompressing images using the loaded basis
#
# Run with: julia --project=examples examples/sparse_basis_demo.jl
# ================================================================================

# Auto-accept dataset downloads (for MLDatasets)
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using ParametricDFT
using MLDatasets
using Images
using ImageQualityIndexes
using Random
using Statistics

# ================================================================================
# Configuration
# ================================================================================

# Image dimensions: 32×32 (5 qubits each dimension)
# MNIST images are 28×28, we'll pad to 32×32 for power-of-2 requirement
const M_QUBITS = 5  # 2^5 = 32 rows
const N_QUBITS = 5  # 2^5 = 32 columns
const IMG_SIZE = 32

# Training configuration
const NUM_TRAINING_IMAGES = 20      # Number of images to train on
const TRAINING_EPOCHS = 2           # Number of training epochs
const STEPS_PER_IMAGE = 50          # Gradient descent steps per image
const COMPRESSION_RATIO = 0.9       # Keep only 10% of coefficients

# Output paths
const OUTPUT_DIR = joinpath(@__DIR__, "SparseBasicDemo")
const BASIS_PATH = joinpath(OUTPUT_DIR, "mnist_basis.json")
const COMPRESSED_PATH = joinpath(OUTPUT_DIR, "compressed_image.json")

# ================================================================================
# Utility Functions
# ================================================================================

"""
    pad_mnist_image(raw_img)

Pad a 28×28 MNIST image to 32×32 by centering it.
"""
function pad_mnist_image(raw_img::AbstractMatrix)
    padded = zeros(Float64, IMG_SIZE, IMG_SIZE)
    # Center the 28×28 image in the 32×32 canvas (2 pixels padding on each side)
    padded[3:30, 3:30] = Float64.(raw_img)
    return padded
end

"""
    display_comparison(original, recovered, title)

Display a simple text-based comparison of image quality.
"""
function display_comparison(original::AbstractMatrix, recovered::AbstractMatrix, title::String)
    # Clamp recovered values to valid range
    recovered_clamped = clamp.(recovered, 0.0, 1.0)
    
    # Calculate quality metrics
    mse = mean((original .- recovered_clamped).^2)
    psnr = mse > 0 ? 10 * log10(1.0 / mse) : Inf
    
    # SSIM using ImageQualityIndexes
    original_gray = Gray.(original)
    recovered_gray = Gray.(recovered_clamped)
    ssim = assess_ssim(original_gray, recovered_gray)
    
    println("\n$title")
    println("  MSE:  $(round(mse, digits=6))")
    println("  PSNR: $(round(psnr, digits=2)) dB")
    println("  SSIM: $(round(ssim, digits=4))")
end

# ================================================================================
# Main Demo
# ================================================================================

function main()
    println("="^70)
    println("Sparse Basis Demo with MNIST Dataset")
    println("="^70)
    
    # Create output directory
    if !isdir(OUTPUT_DIR)
        mkpath(OUTPUT_DIR)
        println("\nCreated output directory: $OUTPUT_DIR")
    end
    
    # ============================================================================
    # Step 1: Load the MNIST Dataset
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 1: Loading MNIST Dataset")
    println("="^70)
    
    # Load MNIST training set
    mnist_train = MNIST(split=:train)
    mnist_test = MNIST(split=:test)
    
    println("MNIST dataset loaded:")
    println("  Training images: $(size(mnist_train.features, 3))")
    println("  Test images: $(size(mnist_test.features, 3))")
    println("  Original image size: 28×28")
    println("  Padded image size: $(IMG_SIZE)×$(IMG_SIZE)")
    
    # Prepare training dataset (pad images to 32×32)
    println("\nPreparing training dataset...")
    Random.seed!(42)
    train_indices = randperm(size(mnist_train.features, 3))[1:NUM_TRAINING_IMAGES]
    
    training_images = Matrix{Float64}[]
    for idx in train_indices
        raw_img = mnist_train.features[:, :, idx]
        padded_img = pad_mnist_image(raw_img)
        push!(training_images, padded_img)
    end
    
    println("  Prepared $(length(training_images)) training images")
    println("  Image dimensions: $(size(training_images[1]))")
    
    # ============================================================================
    # Step 2: Train and Obtain Sparse Basis
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 2: Training Sparse Basis")
    println("="^70)
    
    # Calculate k for MSE loss (keep 10% of coefficients during training)
    total_coefficients = IMG_SIZE * IMG_SIZE
    k = round(Int, total_coefficients * (1.0 - COMPRESSION_RATIO))
    
    println("Training configuration:")
    println("  Qubits: $(M_QUBITS)×$(N_QUBITS)")
    println("  Image size: $(IMG_SIZE)×$(IMG_SIZE)")
    println("  Training images: $(NUM_TRAINING_IMAGES)")
    println("  Epochs: $(TRAINING_EPOCHS)")
    println("  Steps per image: $(STEPS_PER_IMAGE)")
    println("  Coefficients to keep (k): $k / $total_coefficients")
    println("\nTraining in progress...")
    
    # Train the basis using MSE loss
    @time trained_basis = train_basis(
        QFTBasis, training_images;
        m=M_QUBITS,
        n=N_QUBITS,
        loss=ParametricDFT.MSELoss(k),
        epochs=TRAINING_EPOCHS,
        steps_per_image=STEPS_PER_IMAGE,
        validation_split=0.2,
        early_stopping_patience=2,
        verbose=true
    )
    
    println("\n✓ Training completed!")
    println("  Trained basis: $trained_basis")
    println("  Parameters: $(num_parameters(trained_basis))")
    println("  Basis hash: $(basis_hash(trained_basis)[1:16])...")
    
    # ============================================================================
    # Step 3: Dump and Store the Sparse Basis
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 3: Saving Sparse Basis to JSON")
    println("="^70)
    
    println("Saving basis to: $BASIS_PATH")
    @time save_basis(BASIS_PATH, trained_basis)
    
    # Show file size
    file_size = filesize(BASIS_PATH)
    println("✓ Basis saved successfully!")
    println("  File size: $(round(file_size / 1024, digits=2)) KB")
    
    # ============================================================================
    # Step 4: Load the Sparse Basis
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 4: Loading Sparse Basis from JSON")
    println("="^70)
    
    println("Loading basis from: $BASIS_PATH")
    @time loaded_basis = load_basis(BASIS_PATH)
    
    println("✓ Basis loaded successfully!")
    println("  Loaded basis: $loaded_basis")
    println("  Parameters: $(num_parameters(loaded_basis))")
    println("  Basis hash: $(basis_hash(loaded_basis)[1:16])...")
    
    # Verify the loaded basis matches the trained one
    if basis_hash(trained_basis) == basis_hash(loaded_basis)
        println("  ✓ Hash verification: PASSED")
    else
        println("  ✗ Hash verification: FAILED")
    end
    
    # ============================================================================
    # Step 5: Compress and Decompress a Random Image
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 5: Compress and Decompress Image")
    println("="^70)
    
    # Select a random test image
    Random.seed!(123)
    test_idx = rand(1:size(mnist_test.features, 3))
    test_label = mnist_test.targets[test_idx]
    raw_test_img = mnist_test.features[:, :, test_idx]
    test_image = pad_mnist_image(raw_test_img)
    
    println("Selected test image:")
    println("  Index: $test_idx")
    println("  Label: $test_label (digit)")
    println("  Size: $(size(test_image))")
    
    # Compress the image using the loaded basis
    println("\nCompressing image...")
    @time compressed = compress(loaded_basis, test_image; ratio=COMPRESSION_RATIO)
    
    stats = compression_stats(compressed)
    println("✓ Image compressed!")
    println("  $compressed")
    println("  Kept coefficients: $(stats.kept_coefficients) / $(stats.total_coefficients)")
    println("  Compression ratio: $(round(stats.compression_ratio * 100, digits=1))%")
    println("  Storage reduction: $(round(stats.storage_reduction, digits=2))×")
    
    # Save compressed image
    println("\nSaving compressed image to: $COMPRESSED_PATH")
    @time save_compressed(COMPRESSED_PATH, compressed)
    compressed_file_size = filesize(COMPRESSED_PATH)
    println("  Compressed file size: $(round(compressed_file_size / 1024, digits=2)) KB")
    
    # Load compressed image back
    println("\nLoading compressed image from: $COMPRESSED_PATH")
    @time loaded_compressed = load_compressed(COMPRESSED_PATH)
    println("✓ Compressed image loaded!")
    
    # Decompress (recover) the image
    println("\nRecovering image...")
    @time recovered_image = recover(loaded_basis, loaded_compressed)
    
    println("✓ Image recovered!")
    println("  Recovered size: $(size(recovered_image))")
    
    # ============================================================================
    # Quality Comparison
    # ============================================================================
    
    println("\n" * "="^70)
    println("Quality Comparison")
    println("="^70)
    
    # Compare with trained basis
    display_comparison(test_image, recovered_image, "Parametric DFT (Trained Basis):")
    
    # Compare with default (untrained) basis
    default_basis = QFTBasis(M_QUBITS, N_QUBITS)
    compressed_default = compress(default_basis, test_image; ratio=COMPRESSION_RATIO)
    recovered_default = recover(default_basis, compressed_default)
    display_comparison(test_image, recovered_default, "Standard QFT (Default Basis):")
    
    # ============================================================================
    # Save Images
    # ============================================================================
    
    println("\n" * "="^70)
    println("Saving Output Images")
    println("="^70)
    
    # Save original image
    original_path = joinpath(OUTPUT_DIR, "original_digit_$(test_label).png")
    Images.save(original_path, Gray.(test_image))
    println("  Saved: original_digit_$(test_label).png")
    
    # Save recovered image (trained basis)
    recovered_path = joinpath(OUTPUT_DIR, "recovered_trained.png")
    Images.save(recovered_path, Gray.(clamp.(recovered_image, 0.0, 1.0)))
    println("  Saved: recovered_trained.png")
    
    # Save recovered image (default basis)
    default_path = joinpath(OUTPUT_DIR, "recovered_default.png")
    Images.save(default_path, Gray.(clamp.(recovered_default, 0.0, 1.0)))
    println("  Saved: recovered_default.png")
    
    # ============================================================================
    # Summary
    # ============================================================================
    
    println("\n" * "="^70)
    println("Demo Summary")
    println("="^70)
    println("""
    This demo demonstrated the complete sparse basis workflow:
    
    1. ✓ Loaded MNIST dataset ($(NUM_TRAINING_IMAGES) training images)
    2. ✓ Trained QFTBasis using MSE loss ($(num_parameters(trained_basis)) parameters)
    3. ✓ Saved basis to JSON ($(round(file_size / 1024, digits=2)) KB)
    4. ✓ Loaded basis from JSON (hash verified)
    5. ✓ Compressed and decompressed test image (digit $test_label)
    
    Output files saved to: $OUTPUT_DIR
    - mnist_basis.json          : Trained sparse basis
    - compressed_image.json     : Compressed image data
    - original_digit_$test_label.png  : Original test image
    - recovered_trained.png     : Recovered with trained basis
    - recovered_default.png     : Recovered with default basis
    
    Key insight: The trained basis should produce better reconstruction
    quality compared to the default QFT basis for this dataset.
    """)
    
    println("="^70)
    println("Demo completed successfully!")
    println("="^70)
end

# Run the demo
main()

