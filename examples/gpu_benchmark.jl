# GPU vs CPU benchmark with L1Norm (no topk_truncate overhead)
# Run: CUDA_VISIBLE_DEVICES=1 julia --project=examples examples/gpu_benchmark.jl

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using ParametricDFT
using CUDA
using MLDatasets
using Random
using Printf

println("CUDA functional: ", CUDA.functional())
println("GPU: ", CUDA.name(CUDA.device()))
println()

# Load MNIST and pad to 32x32
mnist = MNIST(split=:train)
Random.seed!(42)
indices = randperm(size(mnist.features, 3))[1:20]
training_images = [begin
    padded = zeros(Float64, 32, 32)
    padded[3:30, 3:30] = Float64.(mnist.features[:, :, i])
    padded
end for i in indices]

m, n = 5, 5  # 32x32

println("="^70)
println("  GPU vs CPU Benchmark — L1Norm (no topk_truncate)")
println("="^70)
println("  Images: 20 | Epochs: 4 | Steps/image: 50 | Image: 32×32")
println()

configs = [
    ("GD",   :gradient_descent, :cpu, 1),
    ("GD",   :gradient_descent, :cpu, 4),
    ("GD",   :gradient_descent, :cpu, 8),
    ("Adam", :adam,              :cpu, 1),
    ("Adam", :adam,              :cpu, 4),
    ("Adam", :adam,              :cpu, 8),
    ("GD",   :gradient_descent, :gpu, 1),
    ("GD",   :gradient_descent, :gpu, 4),
    ("GD",   :gradient_descent, :gpu, 8),
    ("Adam", :adam,              :gpu, 1),
    ("Adam", :adam,              :gpu, 4),
    ("Adam", :adam,              :gpu, 8),
]

results = []

for (label, opt_sym, dev, bs) in configs
    bs_clamped = min(bs, length(training_images))
    print("  $label | $dev | bs=$bs_clamped ... ")
    flush(stdout)
    Random.seed!(42)
    elapsed = @elapsed begin
        basis, hist = train_basis(
            QFTBasis, training_images;
            m=m, n=n,
            loss=ParametricDFT.L1Norm(),
            epochs=4,
            steps_per_image=50,
            validation_split=0.2,
            shuffle=true,
            optimizer=opt_sym,
            device=dev,
            batch_size=bs_clamped,
        )
    end
    final_loss = hist.step_train_losses[end]
    @printf("%.1fs  loss=%.6f\n", elapsed, final_loss)
    push!(results, (; label, device=dev, batch_size=bs_clamped, elapsed, final_loss))
end

# Summary table
println("\n" * "="^70)
println("  Summary")
println("="^70)
@printf("  %-6s  %-6s  %5s  %10s  %14s\n", "Optim", "Device", "Batch", "Time", "Final Loss")
println("  " * "-"^50)
for r in results
    @printf("  %-6s  %-6s  %5d  %9.1fs  %14.6f\n",
            r.label, r.device, r.batch_size, r.elapsed, r.final_loss)
end

# CPU vs GPU speedup
println("\n  Speedup (CPU/GPU):")
for bs in [1, 4, 8]
    for opt in ["GD", "Adam"]
        cpu_r = filter(r -> r.label == opt && r.device == :cpu && r.batch_size == bs, results)
        gpu_r = filter(r -> r.label == opt && r.device == :gpu && r.batch_size == bs, results)
        if !isempty(cpu_r) && !isempty(gpu_r)
            ratio = cpu_r[1].elapsed / gpu_r[1].elapsed
            @printf("    %s bs=%d: %.2fx %s\n", opt, bs, ratio, ratio > 1 ? "(GPU faster)" : "(CPU faster)")
        end
    end
end
println("="^70)
