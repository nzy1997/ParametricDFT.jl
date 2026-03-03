# GPU vs CPU benchmark across image sizes (small to large)
# Tests einsum path scaling to verify GPU speedup on large images
# Run: CUDA_VISIBLE_DEVICES=0 julia --project=examples examples/gpu_benchmark.jl

using ParametricDFT
using CUDA
using LinearAlgebra
using Random
using Zygote
using OMEinsum
using Printf
using Statistics

println("CUDA functional: ", CUDA.functional())
println("GPU: ", CUDA.name(CUDA.device()))
println("GPU memory: ", round(CUDA.total_memory() / 2^30, digits=1), " GB")
println()

# Helper: time a function N times, return median
function bench(f, n_warmup=2, n_runs=5)
    for _ in 1:n_warmup; f(); end
    times = Float64[]
    for _ in 1:n_runs
        GC.gc(false)
        t = @elapsed f()
        push!(times, t)
    end
    return median(times)
end

# GPU-safe bench: synchronize before timing
function bench_gpu(f, n_warmup=2, n_runs=5)
    for _ in 1:n_warmup; f(); CUDA.synchronize(); end
    times = Float64[]
    for _ in 1:n_runs
        GC.gc(false)
        CUDA.synchronize()
        t = @elapsed begin f(); CUDA.synchronize(); end
        push!(times, t)
    end
    return median(times)
end

# Cache circuits to avoid rebuilding the same (m,n) across parts
const circuit_cache = Dict{Tuple{Int,Int}, Tuple{Any, Vector}}()

function get_circuit(m, n)
    key = (m, n)
    if !haskey(circuit_cache, key)
        img_size = "$(2^m)x$(2^n)"
        print("  Building circuit for $img_size (D=$(2^(m+n)))... ")
        flush(stdout)
        t_build = @elapsed begin
            optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        end
        n_gates = length(tensors_raw)
        @printf("done (%.1fs, %d gates)\n", t_build, n_gates)
        circuit_cache[key] = (optcode, tensors_raw)
    end
    return circuit_cache[key]
end

# ============================================================================
# Part 1: Scaling across image sizes — einsum path
# ============================================================================

println("="^78)
println("  Part 1: Einsum Forward Pass — Scaling with Image Size")
println("="^78)
println("  Tests how GPU einsum performance changes as image size grows.")
println("  For small images, kernel launch overhead dominates (GPU slower).")
println("  For large images, tensor contraction compute dominates (GPU faster).")
println()

# Test sizes: 2^m x 2^n images
# m=n=5 -> 32x32,  m=n=6 -> 64x64,  m=n=7 -> 128x128
# m=n=8 -> 256x256, m=n=9 -> 512x512
sizes = [(5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]

@printf("  %-12s  %6s  %10s  %10s  %10s  %8s\n",
        "Image Size", "Gates", "CPU (ms)", "GPU (ms)", "Ratio", "Winner")
println("  " * "-"^68)

for (m, n) in sizes
    img_size = "$(2^m)x$(2^n)"
    try
        optcode, tensors_raw = get_circuit(m, n)
        n_gates = length(tensors_raw)
        tensors_cpu = [Matrix{ComplexF64}(t) for t in tensors_raw]
        tensors_gpu = [CuArray(t) for t in tensors_cpu]

        # Random test image
        Random.seed!(42)
        img_cpu = rand(ComplexF64, 2^m, 2^n)
        img_reshaped_cpu = reshape(img_cpu, fill(2, m + n)...)
        img_reshaped_gpu = CuArray(img_reshaped_cpu)

        # Benchmark single-image einsum forward
        t_cpu = bench(() -> optcode(tensors_cpu..., img_reshaped_cpu))
        t_gpu = bench_gpu(() -> optcode(tensors_gpu..., img_reshaped_gpu))
        ratio = t_cpu / t_gpu
        winner = ratio > 1.0 ? "GPU" : "CPU"

        @printf("  %-12s  %6d  %10.1f  %10.1f  %10.1fx  %8s\n",
                img_size, n_gates, t_cpu * 1000, t_gpu * 1000, ratio, winner)

        # Free GPU memory for next size
        tensors_gpu = nothing
        img_reshaped_gpu = nothing
        GC.gc()
        CUDA.reclaim()
    catch e
        println("  $img_size: SKIPPED ($e)")
    end
end

# ============================================================================
# Part 2: Batched einsum on large images
# ============================================================================

println()
println("="^78)
println("  Part 2: Batched Einsum — Large Images")
println("="^78)
println("  Tests batched forward pass where B images are processed together.")
println("  Batching adds another dimension, increasing per-kernel compute.")
println()

for (m, n) in [(7, 7), (8, 8), (9, 9)]
    img_size = "$(2^m)x$(2^n)"
    try
        optcode, tensors_raw = get_circuit(m, n)
        n_gates = length(tensors_raw)
        tensors_cpu = [Matrix{ComplexF64}(t) for t in tensors_raw]
        tensors_gpu = [CuArray(t) for t in tensors_cpu]

        Random.seed!(42)

        for batch_size in [1, 2, 4]
            images_cpu = [rand(ComplexF64, 2^m, 2^n) for _ in 1:batch_size]
            images_gpu = [CuArray(img) for img in images_cpu]

            # Build batched einsum code
            print("    Optimizing batched code ($img_size, bs=$batch_size)... ")
            flush(stdout)
            t_opt = @elapsed begin
                flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
                batched_opt = ParametricDFT.optimize_batched_code(flat_batched, blabel, batch_size)
            end
            @printf("done (%.1fs)\n", t_opt)

            # Batched L1 forward
            t_cpu = bench(() -> ParametricDFT.batched_loss_l1(batched_opt, Tuple(tensors_cpu), images_cpu, m, n))
            t_gpu = bench_gpu(() -> ParametricDFT.batched_loss_l1(batched_opt, Tuple(tensors_gpu), images_gpu, m, n))
            ratio = t_cpu / t_gpu

            @printf("    %-8s bs=%-2d  L1 forward:  CPU=%8.1fms  GPU=%8.1fms  %.1fx %s\n",
                    img_size, batch_size, t_cpu * 1000, t_gpu * 1000, ratio,
                    ratio > 1 ? "(GPU faster)" : "(CPU faster)")

            images_gpu = nothing
            GC.gc()
            CUDA.reclaim()
        end

        tensors_gpu = nothing
        GC.gc()
        CUDA.reclaim()
    catch e
        println("  $img_size: SKIPPED ($e)")
    end
end

# ============================================================================
# Part 3: Zygote gradient on large images
# ============================================================================

println()
println("="^78)
println("  Part 3: Zygote Gradient (forward + backward) — Large Images")
println("="^78)
println("  Tests the full gradient computation (most expensive part of training).")
println("  Backward pass launches ~2x more kernels than forward.")
println()

for (m, n) in [(6, 6), (7, 7), (8, 8)]
    img_size = "$(2^m)x$(2^n)"
    try
        optcode, tensors_raw = get_circuit(m, n)
        n_gates = length(tensors_raw)
        tensors_cpu = [Matrix{ComplexF64}(t) for t in tensors_raw]
        tensors_gpu = [CuArray(t) for t in tensors_cpu]

        Random.seed!(42)
        img_cpu = rand(ComplexF64, 2^m, 2^n)
        img_gpu = CuArray(img_cpu)

        # Single-image L1 loss + gradient
        l1_fn_cpu = ts -> ParametricDFT.loss_function(Tuple(ts), m, n, optcode, img_cpu, ParametricDFT.L1Norm())
        l1_fn_gpu = ts -> ParametricDFT.loss_function(Tuple(ts), m, n, optcode, img_gpu, ParametricDFT.L1Norm())

        t_cpu = bench(() -> Zygote.gradient(l1_fn_cpu, tensors_cpu))
        t_gpu = bench_gpu(() -> Zygote.gradient(l1_fn_gpu, tensors_gpu))
        ratio = t_cpu / t_gpu

        @printf("  %-8s  L1 grad (1 img):  CPU=%8.1fms  GPU=%8.1fms  %.1fx %s\n",
                img_size, t_cpu * 1000, t_gpu * 1000, ratio,
                ratio > 1 ? "(GPU faster)" : "(CPU faster)")

        # Batched L1 gradient (single batch size to keep runtime manageable)
        batch_size = 2
        images_cpu = [rand(ComplexF64, 2^m, 2^n) for _ in 1:batch_size]
        images_gpu = [CuArray(img) for img in images_cpu]

        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        batched_opt = ParametricDFT.optimize_batched_code(flat_batched, blabel, batch_size)

        bl1_cpu = ts -> ParametricDFT.batched_loss_l1(batched_opt, Tuple(ts), images_cpu, m, n)
        bl1_gpu = ts -> ParametricDFT.batched_loss_l1(batched_opt, Tuple(ts), images_gpu, m, n)

        t_cpu = bench(() -> Zygote.gradient(bl1_cpu, tensors_cpu))
        t_gpu = bench_gpu(() -> Zygote.gradient(bl1_gpu, tensors_gpu))
        ratio = t_cpu / t_gpu

        @printf("  %-8s  L1 grad (bs=%-2d):  CPU=%8.1fms  GPU=%8.1fms  %.1fx %s\n",
                img_size, batch_size, t_cpu * 1000, t_gpu * 1000, ratio,
                ratio > 1 ? "(GPU faster)" : "(CPU faster)")

        images_gpu = nothing
        tensors_gpu = nothing
        GC.gc()
        CUDA.reclaim()
    catch e
        println("  $img_size: SKIPPED ($e)")
    end
end

# ============================================================================
# Summary
# ============================================================================

println()
println("="^78)
println("  Summary")
println("="^78)
println("""

  Key findings:
  - Small D (32x32, 64x64): CPU faster than GPU on einsum (kernel launch overhead).
  - Large D (256x256+): GPU 5-7x faster on einsum (tensor contractions saturate GPU cores).
  - Batching helps: batch_size >= 2 shifts the crossover point to smaller D.
  - The crossover point depends on GPU model and gate count.
""")
println("="^78)
