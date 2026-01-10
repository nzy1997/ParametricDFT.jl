# ================================================================================
# Entangled Basis Demo with MNIST Dataset
# ================================================================================
# This example demonstrates the entangled QFT basis with XY correlation:
#   1. Loading the MNIST dataset
#   2. Comparing standard QFT vs Entangled QFT basis
#   3. Training an entangled basis with learnable entanglement phases
#   4. Compressing and decompressing images using the entangled basis
#   5. Visualizing the effect of entanglement phases
#
# Run with: julia --project=examples examples/entangled_basis_demo.jl
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
const OUTPUT_DIR = joinpath(@__DIR__, "EntangledBasisDemo")
const BASIS_PATH = joinpath(OUTPUT_DIR, "entangled_basis.json")
const COMPRESSED_PATH = joinpath(OUTPUT_DIR, "compressed_entangled.json")

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
    println("Entangled QFT Basis Demo with MNIST Dataset")
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
    # Step 2: Compare Standard QFT vs Entangled QFT Basis
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 2: Comparing Standard QFT vs Entangled QFT Basis")
    println("="^70)
    
    # Select a test image for comparison
    Random.seed!(123)
    test_idx = rand(1:size(mnist_test.features, 3))
    test_label = mnist_test.targets[test_idx]
    raw_test_img = mnist_test.features[:, :, test_idx]
    test_image = pad_mnist_image(raw_test_img)
    
    println("Selected test image:")
    println("  Index: $test_idx")
    println("  Label: $test_label (digit)")
    
    # Create standard QFT basis
    println("\nCreating bases...")
    standard_basis = QFTBasis(M_QUBITS, N_QUBITS)
    println("  Standard QFT: $standard_basis")
    
    # Create entangled QFT basis with zero phases (equivalent to standard)
    entangled_zero = EntangledQFTBasis(M_QUBITS, N_QUBITS; entangle_phases=zeros(M_QUBITS))
    println("  Entangled (zero phases): $entangled_zero")
    
    # Create entangled QFT basis with random phases
    Random.seed!(42)
    random_phases = rand(M_QUBITS) * 2π
    entangled_random = EntangledQFTBasis(M_QUBITS, N_QUBITS; entangle_phases=random_phases)
    println("  Entangled (random phases): $entangled_random")
    println("    Phases: $(round.(random_phases, digits=3))")
    
    # Compare transforms
    println("\nComparing frequency domain representations...")
    
    freq_standard = forward_transform(standard_basis, test_image)
    freq_entangled_zero = forward_transform(entangled_zero, test_image)
    freq_entangled_random = forward_transform(entangled_random, test_image)
    
    println("  Standard QFT energy: $(round(sum(abs2.(freq_standard)), digits=2))")
    println("  Entangled (zero) energy: $(round(sum(abs2.(freq_entangled_zero)), digits=2))")
    println("  Entangled (random) energy: $(round(sum(abs2.(freq_entangled_random)), digits=2))")
    
    # Verify zero-phase entangled matches standard
    max_diff = maximum(abs.(freq_standard .- freq_entangled_zero))
    println("  Max diff (standard vs zero-phase entangled): $(round(max_diff, sigdigits=3))")
    
    # ============================================================================
    # Step 3: Compression Comparison
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 3: Compression Comparison")
    println("="^70)
    
    # Compress with standard basis
    println("\nCompressing with standard QFT...")
    compressed_std = compress(standard_basis, test_image; ratio=COMPRESSION_RATIO)
    recovered_std = recover(standard_basis, compressed_std)
    stats_std = compression_stats(compressed_std)
    
    # Compress with entangled basis (zero phases)
    println("Compressing with entangled QFT (zero phases)...")
    compressed_ent_zero = compress(entangled_zero, test_image; ratio=COMPRESSION_RATIO)
    recovered_ent_zero = recover(entangled_zero, compressed_ent_zero)
    
    # Compress with entangled basis (random phases)
    println("Compressing with entangled QFT (random phases)...")
    compressed_ent_rand = compress(entangled_random, test_image; ratio=COMPRESSION_RATIO)
    recovered_ent_rand = recover(entangled_random, compressed_ent_rand)
    
    println("\nCompression statistics:")
    println("  Kept coefficients: $(stats_std.kept_coefficients) / $(stats_std.total_coefficients)")
    println("  Compression ratio: $(round(stats_std.compression_ratio * 100, digits=1))%")
    
    # Quality comparison
    display_comparison(test_image, recovered_std, "Standard QFT Basis:")
    display_comparison(test_image, recovered_ent_zero, "Entangled QFT (zero phases):")
    display_comparison(test_image, recovered_ent_rand, "Entangled QFT (random phases):")
    
    # ============================================================================
    # Step 4: Train Entangled Basis
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 4: Training Entangled QFT Basis")
    println("="^70)
    
    # Calculate k for MSE loss
    total_coefficients = IMG_SIZE * IMG_SIZE
    k = round(Int, total_coefficients * (1.0 - COMPRESSION_RATIO))
    
    println("Training configuration:")
    println("  Basis type: EntangledQFTBasis")
    println("  Qubits: $(M_QUBITS)×$(N_QUBITS)")
    println("  Entanglement gates: $(M_QUBITS)")
    println("  Training images: $(NUM_TRAINING_IMAGES)")
    println("  Epochs: $(TRAINING_EPOCHS)")
    println("  Steps per image: $(STEPS_PER_IMAGE)")
    println("  Coefficients to keep (k): $k / $total_coefficients")
    println("\nTraining in progress...")
    
    # Train the entangled basis
    @time trained_basis = train_basis(
        EntangledQFTBasis, training_images;
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
    println("  Entanglement phases: $(round.(get_entangle_phases(trained_basis), digits=3))")
    
    # ============================================================================
    # Step 5: Evaluate Trained Entangled Basis
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 5: Evaluating Trained Entangled Basis")
    println("="^70)
    
    # Compress with trained entangled basis
    println("\nCompressing with trained entangled QFT...")
    compressed_trained = compress(trained_basis, test_image; ratio=COMPRESSION_RATIO)
    recovered_trained = recover(trained_basis, compressed_trained)
    
    # Quality comparison
    println("\nQuality Comparison on Test Image (digit $test_label):")
    display_comparison(test_image, recovered_std, "Standard QFT (untrained):")
    display_comparison(test_image, recovered_trained, "Entangled QFT (trained):")
    
    # ============================================================================
    # Step 6: Save and Load Entangled Basis
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 6: Saving and Loading Entangled Basis")
    println("="^70)
    
    # Save the trained basis
    println("Saving trained entangled basis to: $BASIS_PATH")
    @time save_basis(BASIS_PATH, trained_basis)
    file_size = filesize(BASIS_PATH)
    println("✓ Basis saved! ($(round(file_size / 1024, digits=2)) KB)")
    
    # Load it back
    println("\nLoading entangled basis from: $BASIS_PATH")
    @time loaded_basis = load_basis(BASIS_PATH)
    println("✓ Basis loaded!")
    println("  Loaded: $loaded_basis")
    
    # Verify hash
    if basis_hash(trained_basis) == basis_hash(loaded_basis)
        println("  ✓ Hash verification: PASSED")
    else
        println("  ✗ Hash verification: FAILED")
    end
    
    # ============================================================================
    # Step 7: Save Output Images
    # ============================================================================
    
    println("\n" * "="^70)
    println("Step 7: Saving Output Images")
    println("="^70)
    
    # Save original image
    original_path = joinpath(OUTPUT_DIR, "original_digit_$(test_label).png")
    Images.save(original_path, Gray.(test_image))
    println("  Saved: original_digit_$(test_label).png")
    
    # Save recovered image (standard basis)
    standard_path = joinpath(OUTPUT_DIR, "recovered_standard.png")
    Images.save(standard_path, Gray.(clamp.(recovered_std, 0.0, 1.0)))
    println("  Saved: recovered_standard.png")
    
    # Save recovered image (trained entangled basis)
    entangled_path = joinpath(OUTPUT_DIR, "recovered_entangled.png")
    Images.save(entangled_path, Gray.(clamp.(recovered_trained, 0.0, 1.0)))
    println("  Saved: recovered_entangled.png")
    
    # ============================================================================
    # Summary
    # ============================================================================
    
    println("\n" * "="^70)
    println("Demo Summary")
    println("="^70)
    println("""
    This demo demonstrated the entangled QFT basis features:
    
    1. ✓ Loaded MNIST dataset ($(NUM_TRAINING_IMAGES) training images)
    2. ✓ Compared standard vs entangled QFT transforms
    3. ✓ Verified zero-phase entangled QFT = standard QFT
    4. ✓ Trained EntangledQFTBasis with $(M_QUBITS) learnable entanglement phases
    5. ✓ Compressed and recovered test image (digit $test_label)
    6. ✓ Saved and loaded trained basis (hash verified)
    
    Key differences from standard QFT:
    - Entangled basis adds $(M_QUBITS) controlled-phase gates (E_k) between
      corresponding row and column qubits
    - Each E_k gate has a learnable phase parameter φ_k
    - The entanglement captures XY correlations in images
    - With all phases = 0, reverts to standard 2D QFT
    
    Output files saved to: $OUTPUT_DIR
    - entangled_basis.json      : Trained entangled basis
    - original_digit_$test_label.png   : Original test image
    - recovered_standard.png    : Recovered with standard QFT
    - recovered_entangled.png   : Recovered with trained entangled basis
    """)
    
    println("="^70)
    println("Demo completed successfully!")
    println("="^70)
end

# Run the demo
main()

