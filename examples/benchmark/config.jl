# ============================================================================
# Benchmark Configuration
# ============================================================================
# Shared constants, training presets, and dataset configurations for all
# benchmark scripts. Include this file at the top of each run_*.jl script.
# ============================================================================

using ParametricDFT
using CUDA
using Random

# ============================================================================
# Training Presets
# ============================================================================

const TRAINING_PRESETS = Dict(
    :smoke => (
        epochs = 2,
        steps_per_image = 10,
        n_train = 5,
        n_test = 2,
        patience = 2,
        optimizer = :adam,
        validation_split = 0.2,
        device = :gpu,
        batch_size = 16,
    ),
    :moderate => (
        epochs = 20,
        steps_per_image = 50,
        n_train = 20,
        n_test = 50,
        patience = 10,
        optimizer = :adam,
        validation_split = 0.2,
        device = :gpu,
        batch_size = 16,
    ),
    :heavy => (
        epochs = 20,
        steps_per_image = 100,
        n_train = 50,
        n_test = 100,
        patience = 10,
        optimizer = :adam,
        validation_split = 0.2,
        device = :gpu,
        batch_size = 16,
    ),
)

# ============================================================================
# Dataset Configurations
# ============================================================================

const DATASET_CONFIGS = Dict(
    :quickdraw => (m = 5, n = 5, img_size = 32),
    :div2k     => (m = 10, n = 10, img_size = 1024),
    :clic      => (m = 9, n = 9, img_size = 512),
)

# ============================================================================
# Evaluation Settings
# ============================================================================

const KEEP_RATIOS = [0.05, 0.10, 0.15, 0.20]
const BASIS_TYPES = [QFTBasis, EntangledQFTBasis, TEBDBasis, MERABasis]
const BASIS_NAMES = Dict(
    QFTBasis => "qft",
    EntangledQFTBasis => "entangled_qft",
    TEBDBasis => "tebd",
    MERABasis => "mera",
)

# ============================================================================
# Paths
# ============================================================================

const DATA_DIR = joinpath(@__DIR__, "data")
const RESULTS_DIR = joinpath(@__DIR__, "results")
