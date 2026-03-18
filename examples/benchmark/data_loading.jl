# ============================================================================
# Dataset Loading
# ============================================================================
# Provides loader functions for Quick Draw, DIV2K, and ATD-12K datasets.
# All loaders return the same interface:
#   (train_images::Vector{Matrix{Float64}},
#    test_images::Vector{Matrix{Float64}},
#    test_labels::Vector{String})
# All images are normalized to [0, 1] Float64 grayscale.
# ============================================================================

using NPZ
using Downloads
using Images
using FileIO

# ============================================================================
# Quick Draw Loader
# ============================================================================

const QUICKDRAW_CATEGORIES = ["cat", "dog", "airplane", "apple", "bicycle"]
const QUICKDRAW_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

"""
    pad_to_power_of_two(img::AbstractMatrix, target_size::Int)

Center-pad image to target_size x target_size.
"""
function pad_to_power_of_two(img::AbstractMatrix, target_size::Int)
    h, w = size(img)
    padded = zeros(Float64, target_size, target_size)
    y_offset = (target_size - h) ÷ 2 + 1
    x_offset = (target_size - w) ÷ 2 + 1
    padded[y_offset:y_offset+h-1, x_offset:x_offset+w-1] = Float64.(img)
    return padded
end

"""
    load_quickdraw_dataset(; n_train, n_test, img_size=32, seed=42)

Load Quick Draw numpy bitmaps. Auto-downloads if not present.

Returns `(train_images, test_images, test_labels)`.
"""
function load_quickdraw_dataset(; n_train::Int, n_test::Int, img_size::Int = 32, seed::Int = 42)
    quickdraw_dir = joinpath(DATA_DIR, "quickdraw")
    mkpath(quickdraw_dir)

    # Download missing category files
    for category in QUICKDRAW_CATEGORIES
        filepath = joinpath(quickdraw_dir, "$(category).npy")
        if !isfile(filepath)
            url = "$(QUICKDRAW_BASE_URL)/$(category).npy"
            @info "Downloading Quick Draw category: $category" url
            Downloads.download(url, filepath)
        end
    end

    # Load all images
    all_images = Matrix{Float64}[]
    all_labels = String[]

    for category in QUICKDRAW_CATEGORIES
        filepath = joinpath(quickdraw_dir, "$(category).npy")
        data = npzread(filepath)
        n_available = size(data, 1)
        n_to_load = min(n_available, (n_train + n_test) ÷ length(QUICKDRAW_CATEGORIES) + 1)

        for i in 1:n_to_load
            img = reshape(Float64.(data[i, :]), 28, 28) ./ 255.0
            push!(all_images, pad_to_power_of_two(img, img_size))
            push!(all_labels, category)
        end
        @info "Loaded $n_to_load images from $category"
    end

    # Shuffle and split
    Random.seed!(seed)
    indices = randperm(length(all_images))
    train_indices = indices[1:min(n_train, length(indices))]
    test_indices = indices[n_train+1:min(n_train + n_test, length(indices))]

    train_images = all_images[train_indices]
    test_images = all_images[test_indices]
    test_labels = all_labels[test_indices]

    @info "Quick Draw dataset ready" n_train=length(train_images) n_test=length(test_images)
    return train_images, test_images, test_labels
end

# ============================================================================
# DIV2K Loader
# ============================================================================

"""
    center_crop_square(img::AbstractMatrix)

Center-crop image to the largest square that fits.
"""
function center_crop_square(img::AbstractMatrix)
    h, w = size(img)
    side = min(h, w)
    y_start = (h - side) ÷ 2 + 1
    x_start = (w - side) ÷ 2 + 1
    return img[y_start:y_start+side-1, x_start:x_start+side-1]
end

"""
    resize_image(img::AbstractMatrix, target_size::Int)

Resize image to target_size x target_size using bilinear interpolation via Images.jl.
"""
function resize_image(img::AbstractMatrix, target_size::Int)
    return Float64.(imresize(img, (target_size, target_size)))
end

"""
    load_grayscale_image(path::String, target_size::Int)

Load an image file, convert to grayscale, center-crop to square, resize.
"""
function load_grayscale_image(path::String, target_size::Int)
    img = FileIO.load(path)
    gray = Gray.(img)
    gray_matrix = Float64.(channelview(gray))
    cropped = center_crop_square(gray_matrix)
    return resize_image(cropped, target_size)
end

"""
    load_div2k_dataset(; n_train, n_test, img_size=1024, seed=42)

Load DIV2K HR images. Expects data in `data/DIV2K_train_HR/` and/or `data/DIV2K_valid_HR/`.

Returns `(train_images, test_images, test_labels)`.
"""
function load_div2k_dataset(; n_train::Int, n_test::Int, img_size::Int = 1024, seed::Int = 42)
    # Check for DIV2K directories
    train_dir = joinpath(DATA_DIR, "DIV2K_train_HR")
    valid_dir = joinpath(DATA_DIR, "DIV2K_valid_HR")

    all_files = String[]
    for dir in [train_dir, valid_dir]
        if isdir(dir)
            append!(all_files, sort(filter(
                f -> endswith(lowercase(f), ".png"),
                readdir(dir; join = true)
            )))
        end
    end

    if isempty(all_files)
        error("""
        DIV2K dataset not found. Please download:
          mkdir -p $(DATA_DIR)
          cd $(DATA_DIR)
          # Training set (800 images):
          curl -LO https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
          unzip DIV2K_train_HR.zip
          # Validation set (100 images):
          curl -LO https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
          unzip DIV2K_valid_HR.zip
        """)
    end

    @assert length(all_files) >= n_train + n_test "Need $(n_train + n_test) DIV2K images, found $(length(all_files))"

    Random.seed!(seed)
    selected = all_files[randperm(length(all_files))[1:(n_train + n_test)]]

    images = Matrix{Float64}[]
    labels = String[]
    for path in selected
        push!(images, load_grayscale_image(path, img_size))
        push!(labels, basename(path))
    end

    train_images = images[1:n_train]
    test_images = images[n_train+1:end]
    test_labels = labels[n_train+1:end]

    @info "DIV2K dataset ready" n_train=length(train_images) n_test=length(test_images) img_size
    return train_images, test_images, test_labels
end

# ============================================================================
# CLIC Loader
# ============================================================================

"""
    load_clic_dataset(; n_train, n_test, img_size=512, seed=42)

Load CLIC (Challenge on Learned Image Compression) images.
Expects data in `data/professional_train_2020/`, `data/professional_valid_2020/`,
and/or `data/mobile_train_2020/`.

Returns `(train_images, test_images, test_labels)`.
"""
function load_clic_dataset(; n_train::Int, n_test::Int, img_size::Int = 512, seed::Int = 42)
    clic_dirs = [
        joinpath(DATA_DIR, "professional_train_2020"),
        joinpath(DATA_DIR, "professional_valid_2020"),
        joinpath(DATA_DIR, "mobile_train_2020"),
    ]

    all_files = String[]
    for dir in clic_dirs
        if isdir(dir)
            append!(all_files, sort(filter(
                f -> endswith(lowercase(f), ".png") || endswith(lowercase(f), ".jpg"),
                readdir(dir; join = true)
            )))
        end
    end

    if isempty(all_files)
        error("""
        CLIC dataset not found. Please download:
          mkdir -p $(DATA_DIR)
          cd $(DATA_DIR)
          curl -LO https://data.vision.ee.ethz.ch/cvl/clic/professional_train_2020.zip
          curl -LO https://data.vision.ee.ethz.ch/cvl/clic/professional_valid_2020.zip
          curl -LO https://data.vision.ee.ethz.ch/cvl/clic/mobile_train_2020.zip
          # Extract all zips
        """)
    end

    @assert length(all_files) >= n_train + n_test "Need $(n_train + n_test) CLIC images, found $(length(all_files))"

    Random.seed!(seed)
    selected = all_files[randperm(length(all_files))[1:(n_train + n_test)]]

    images = Matrix{Float64}[]
    labels = String[]
    for path in selected
        push!(images, load_grayscale_image(path, img_size))
        push!(labels, basename(path))
    end

    train_images = images[1:n_train]
    test_images = images[n_train+1:end]
    test_labels = labels[n_train+1:end]

    @info "CLIC dataset ready" n_train=length(train_images) n_test=length(test_images) img_size
    return train_images, test_images, test_labels
end
