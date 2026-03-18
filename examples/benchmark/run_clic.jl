# ============================================================================
# Benchmark: CLIC Dataset
# ============================================================================
# Train all 4 basis types on CLIC and evaluate compression quality.
#
# Usage:
#   julia --project=examples/benchmark examples/benchmark/run_clic.jl moderate
#   julia --project=examples/benchmark examples/benchmark/run_clic.jl heavy
# ============================================================================

include("config.jl")
include("data_loading.jl")
include("evaluation.jl")

const DATASET_NAME = :clic
const DATASET_CONFIG = DATASET_CONFIGS[DATASET_NAME]

preset_name = length(ARGS) > 0 ? Symbol(ARGS[1]) : :moderate
@assert haskey(TRAINING_PRESETS, preset_name) "Unknown preset: $preset_name. Use :moderate or :heavy"
preset = TRAINING_PRESETS[preset_name]

println("=" ^ 80)
println("Benchmark: CLIC | Preset: $preset_name")
println("Image size: $(DATASET_CONFIG.img_size)x$(DATASET_CONFIG.img_size) | Qubits: m=$(DATASET_CONFIG.m), n=$(DATASET_CONFIG.n)")
println("=" ^ 80)

# Load dataset
println("\nStep 1: Loading CLIC dataset...")
train_images, test_images, test_labels = load_clic_dataset(;
    n_train = preset.n_train,
    n_test = preset.n_test,
    img_size = DATASET_CONFIG.img_size,
)

# Train, evaluate, save
output_dir = joinpath(RESULTS_DIR, string(DATASET_NAME))
results = run_all_bases(train_images, test_images, DATASET_CONFIG, preset, output_dir)
save_benchmark_results(joinpath(output_dir, "metrics.json"), results)
print_dataset_summary(results, KEEP_RATIOS)
println("\nBenchmark complete. Results saved to: $output_dir")
