# ============================================================================
# Benchmark: ATD-12K Dataset
# ============================================================================
# Train all 4 basis types on ATD-12K and evaluate compression quality.
#
# Usage:
#   julia --project=examples/benchmark examples/benchmark/run_atd12k.jl moderate
#   julia --project=examples/benchmark examples/benchmark/run_atd12k.jl heavy
# ============================================================================

include("config.jl")
include("data_loading.jl")
include("evaluation.jl")

const DATASET_NAME = :atd12k
const DATASET_CONFIG = DATASET_CONFIGS[DATASET_NAME]

preset_name = length(ARGS) > 0 ? Symbol(ARGS[1]) : :moderate
@assert haskey(TRAINING_PRESETS, preset_name) "Unknown preset: $preset_name. Use :moderate or :heavy"
preset = TRAINING_PRESETS[preset_name]

println("=" ^ 80)
println("Benchmark: ATD-12K | Preset: $preset_name")
println("Image size: $(DATASET_CONFIG.img_size)x$(DATASET_CONFIG.img_size) | Qubits: m=$(DATASET_CONFIG.m), n=$(DATASET_CONFIG.n)")
println("=" ^ 80)

# Load dataset
println("\nStep 1: Loading ATD-12K dataset...")
train_images, test_images, test_labels = load_atd12k_dataset(;
    n_train = preset.n_train,
    n_test = preset.n_test,
    img_size = DATASET_CONFIG.img_size,
)

# Train and evaluate
output_dir = joinpath(RESULTS_DIR, string(DATASET_NAME))
loss_dir = joinpath(output_dir, "loss_history")
mkpath(loss_dir)

results = Dict{String,Dict{Symbol,Any}}()

for BasisType in BASIS_TYPES
    basis_name = BASIS_NAMES[BasisType]
    basis_path = joinpath(output_dir, "trained_$(basis_name).json")

    if isfile(basis_path)
        @info "Skipping $basis_name — already trained at $basis_path"
        loaded_basis = load_basis(basis_path)
        metrics = evaluate_basis(loaded_basis, test_images, KEEP_RATIOS)
        results[basis_name] = Dict(:metrics => metrics, :time => 0.0)
        continue
    end

    println("\n--- Training $(BasisType) ---")
    loss_path = joinpath(loss_dir, "$(basis_name)_loss.json")

    basis, history, elapsed = train_and_time(
        BasisType, train_images, DATASET_CONFIG, preset;
        save_loss_path = loss_path,
    )

    mkpath(output_dir)
    save_basis(basis_path, basis)
    @info "Saved trained $basis_name" path=basis_path time=round(elapsed; digits=1)

    metrics = evaluate_basis(basis, test_images, KEEP_RATIOS)
    results[basis_name] = Dict(:metrics => metrics, :time => elapsed, :history => history)
end

# FFT Baseline
println("\n--- FFT Baseline ---")
fft_metrics, fft_time = evaluate_fft_baseline_timed(test_images, KEEP_RATIOS)
results["fft"] = Dict(:metrics => fft_metrics, :time => fft_time)

# Save and print
save_benchmark_results(joinpath(output_dir, "metrics.json"), results)
print_dataset_summary(results, KEEP_RATIOS)
println("\nBenchmark complete. Results saved to: $output_dir")
