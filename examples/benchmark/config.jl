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
    ),
    :moderate => (
        epochs = 5,
        steps_per_image = 20,
        n_train = 10,
        n_test = 5,
        patience = 3,
        optimizer = :adam,
        validation_split = 0.2,
        device = :gpu,
    ),
    :heavy => (
        epochs = 10,
        steps_per_image = 50,
        n_train = 20,
        n_test = 10,
        patience = 5,
        optimizer = :adam,
        validation_split = 0.2,
        device = :gpu,
    ),
)

# ============================================================================
# Dataset Configurations
# ============================================================================

const DATASET_CONFIGS = Dict(
    :quickdraw => (m = 5, n = 5, img_size = 32),
    :div2k     => (m = 10, n = 10, img_size = 1024),
    :atd12k    => (m = 9, n = 9, img_size = 512),
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
