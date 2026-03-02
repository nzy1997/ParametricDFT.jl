# Profile GPU vs CPU bottlenecks in training pipeline
# CUDA_VISIBLE_DEVICES=1 julia --project=examples examples/profile_gpu.jl

using ParametricDFT
using CUDA
using LinearAlgebra
using Random
using Zygote
using OMEinsum
using Printf
using Statistics

println("GPU: ", CUDA.name(CUDA.device()))
println()

# Setup: 32x32 images, 5+5 qubits
m, n = 5, 5
Random.seed!(42)

optcode, tensors_raw = ParametricDFT.qft_code(m, n)
inverse_code, _ = ParametricDFT.qft_code(m, n; inverse=true)
tensors_cpu = [Matrix{ComplexF64}(t) for t in tensors_raw]
tensors_gpu = [CuArray(t) for t in tensors_cpu]

n_gates = length(tensors_cpu)
images_cpu = [rand(ComplexF64, 32, 32) for _ in 1:8]
images_gpu = [CuArray(img) for img in images_cpu]

# Batched einsum codes
batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, 8)

k = 102  # MSELoss parameter

# Helper: time a function N times, return median
function bench(f, n_warmup=3, n_runs=10)
    for _ in 1:n_warmup; f(); end
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed f()
        push!(times, t)
    end
    return median(times)
end

# GPU-safe bench: synchronize before timing
function bench_gpu(f, n_warmup=3, n_runs=10)
    for _ in 1:n_warmup; f(); CUDA.synchronize(); end
    times = Float64[]
    for _ in 1:n_runs
        CUDA.synchronize()
        t = @elapsed begin f(); CUDA.synchronize(); end
        push!(times, t)
    end
    return median(times)
end

println("="^70)
println("  Component-level profiling: CPU vs GPU")
println("="^70)
println()

# 1. Single-image forward pass (einsum)
println("--- 1. Single-image einsum forward pass ---")
img_reshaped_cpu = reshape(images_cpu[1], fill(2, m+n)...)
img_reshaped_gpu = CuArray(img_reshaped_cpu)
t_cpu = bench(() -> optcode(tensors_cpu..., img_reshaped_cpu))
t_gpu = bench_gpu(() -> optcode(tensors_gpu..., img_reshaped_gpu))
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

# 2. Batched forward pass (8 images)
println("\n--- 2. Batched einsum forward (8 images) ---")
stacked_cpu = cat([reshape(img, fill(2, m+n)...) for img in images_cpu]...; dims=m+n+1)
stacked_gpu = CuArray(stacked_cpu)
t_cpu = bench(() -> batched_opt(tensors_cpu..., stacked_cpu))
t_gpu = bench_gpu(() -> batched_opt(tensors_gpu..., stacked_gpu))
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

# 3. topk_truncate
println("\n--- 3. topk_truncate (32x32 matrix, k=$k) ---")
coeffs_cpu = rand(ComplexF64, 32, 32)
coeffs_gpu = CuArray(coeffs_cpu)
t_cpu = bench(() -> ParametricDFT.topk_truncate(coeffs_cpu, k))
t_gpu = bench_gpu(() -> ParametricDFT.topk_truncate(coeffs_gpu, k))
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

# 4. L1 loss function (single image)
println("\n--- 4. L1 loss forward (single image) ---")
t_cpu = bench(() -> ParametricDFT.loss_function(tensors_cpu, m, n, optcode, images_cpu[1], ParametricDFT.L1Norm()))
t_gpu = bench_gpu(() -> ParametricDFT.loss_function(tensors_gpu, m, n, optcode, images_gpu[1], ParametricDFT.L1Norm()))
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

# 5. MSE loss function (single image)
println("\n--- 5. MSE loss forward (single image) ---")
t_cpu = bench(() -> ParametricDFT.loss_function(tensors_cpu, m, n, optcode, images_cpu[1], ParametricDFT.MSELoss(k); inverse_code=inverse_code))
t_gpu = bench_gpu(() -> ParametricDFT.loss_function(tensors_gpu, m, n, optcode, images_gpu[1], ParametricDFT.MSELoss(k); inverse_code=inverse_code))
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

# 6. L1 loss + gradient (single image, Zygote)
println("\n--- 6. L1 loss + Zygote gradient (single image) ---")
l1_fn_cpu = ts -> ParametricDFT.loss_function(Tuple(ts), m, n, optcode, images_cpu[1], ParametricDFT.L1Norm())
l1_fn_gpu = ts -> ParametricDFT.loss_function(Tuple(ts), m, n, optcode, images_gpu[1], ParametricDFT.L1Norm())
t_cpu = bench(() -> Zygote.gradient(l1_fn_cpu, tensors_cpu))
t_gpu = bench_gpu(() -> Zygote.gradient(l1_fn_gpu, tensors_gpu))
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

# 7. MSE loss + gradient (single image, Zygote)
println("\n--- 7. MSE loss + Zygote gradient (single image) ---")
mse_fn_cpu = ts -> ParametricDFT.loss_function(Tuple(ts), m, n, optcode, images_cpu[1], ParametricDFT.MSELoss(k); inverse_code=inverse_code)
mse_fn_gpu = ts -> ParametricDFT.loss_function(Tuple(ts), m, n, optcode, images_gpu[1], ParametricDFT.MSELoss(k); inverse_code=inverse_code)
t_cpu = bench(() -> Zygote.gradient(mse_fn_cpu, tensors_cpu))
t_gpu = bench_gpu(() -> Zygote.gradient(mse_fn_gpu, tensors_gpu))
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

# 8. Batched L1 loss + gradient (8 images)
println("\n--- 8. Batched L1 loss + gradient (8 images) ---")
bl1_cpu = ts -> ParametricDFT.batched_loss_l1(batched_opt, Tuple(ts), images_cpu, m, n)
bl1_gpu = ts -> ParametricDFT.batched_loss_l1(batched_opt, Tuple(ts), images_gpu, m, n)
t_cpu = bench(() -> Zygote.gradient(bl1_cpu, tensors_cpu))
t_gpu = bench_gpu(() -> Zygote.gradient(bl1_gpu, tensors_gpu))
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

# 9. Batched MSE loss + gradient (8 images)
println("\n--- 9. Batched MSE loss + gradient (8 images) ---")
bmse_cpu = ts -> ParametricDFT.batched_loss_mse(batched_opt, Tuple(ts), images_cpu, m, n, k, inverse_code)
bmse_gpu = ts -> ParametricDFT.batched_loss_mse(batched_opt, Tuple(ts), images_gpu, m, n, k, inverse_code)
t_cpu = bench(() -> Zygote.gradient(bmse_cpu, tensors_cpu))
t_gpu = bench_gpu(() -> Zygote.gradient(bmse_gpu, tensors_gpu))
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

# 10. Manifold operations (project + retract)
println("\n--- 10. Manifold project + retract (batched 2x2, n=10) ---")
U_cpu = randn(ComplexF64, 2, 2, 10); for k in 1:10; F = qr(U_cpu[:,:,k]); U_cpu[:,:,k] = Matrix(F.Q); end
G_cpu = randn(ComplexF64, 2, 2, 10)
U_gpu = CuArray(U_cpu)
G_gpu = CuArray(G_cpu)
um = ParametricDFT.UnitaryManifold()
t_cpu = bench(() -> begin rg = ParametricDFT.project(um, U_cpu, G_cpu); ParametricDFT.retract(um, U_cpu, rg, 0.01); end)
t_gpu = bench_gpu(() -> begin rg = ParametricDFT.project(um, U_gpu, G_gpu); ParametricDFT.retract(um, U_gpu, rg, 0.01); end)
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

# 11. Full optimize! step (1 iteration, Adam, L1, batch=8)
println("\n--- 11. Full optimize! (1 iter, Adam, L1, batch=8) ---")
opt = ParametricDFT.RiemannianAdam(lr=0.001)
fn_cpu = ts -> ParametricDFT.batched_loss_l1(batched_opt, Tuple(ts), images_cpu, m, n)
fn_gpu = ts -> ParametricDFT.batched_loss_l1(batched_opt, Tuple(ts), images_gpu, m, n)
gfn_cpu = ts -> begin _, back = Zygote.pullback(fn_cpu, ts); back(1.0)[1]; end
gfn_gpu = ts -> begin _, back = Zygote.pullback(fn_gpu, ts); back(1.0)[1]; end
t_cpu = bench(() -> ParametricDFT.optimize!(opt, copy.(tensors_cpu), fn_cpu, gfn_cpu; max_iter=1))
t_gpu = bench_gpu(() -> ParametricDFT.optimize!(opt, copy.(tensors_gpu), fn_gpu, gfn_gpu; max_iter=1))
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

# 12. Full optimize! step (1 iteration, Adam, MSE, batch=8)
println("\n--- 12. Full optimize! (1 iter, Adam, MSE, batch=8) ---")
fn2_cpu = ts -> ParametricDFT.batched_loss_mse(batched_opt, Tuple(ts), images_cpu, m, n, k, inverse_code)
fn2_gpu = ts -> ParametricDFT.batched_loss_mse(batched_opt, Tuple(ts), images_gpu, m, n, k, inverse_code)
gfn2_cpu = ts -> begin _, back = Zygote.pullback(fn2_cpu, ts); back(1.0)[1]; end
gfn2_gpu = ts -> begin _, back = Zygote.pullback(fn2_gpu, ts); back(1.0)[1]; end
t_cpu = bench(() -> ParametricDFT.optimize!(opt, copy.(tensors_cpu), fn2_cpu, gfn2_cpu; max_iter=1))
t_gpu = bench_gpu(() -> ParametricDFT.optimize!(opt, copy.(tensors_gpu), fn2_gpu, gfn2_gpu; max_iter=1))
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

println("\n" * "="^70)
println("  Materialized Unitary Benchmarks")
println("="^70)
println()

# Build materialized unitary
D = 2^(m + n)
flat_u, blabel_u = ParametricDFT.make_batched_code(optcode, n_gates)
unitary_optcode = ParametricDFT.optimize_batched_code(flat_u, blabel_u, D)

println("--- 13. Build circuit unitary (D=$D) ---")
t_cpu = bench(() -> ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors_cpu), m, n))
t_gpu = bench_gpu(() -> ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors_gpu), m, n))
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

U_cpu = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors_cpu), m, n)
U_gpu = CuArray(U_cpu)

println("\n--- 14. Materialized forward (single image, matmul) ---")
img_vec_cpu = vec(images_cpu[1])
img_vec_gpu = CuArray(img_vec_cpu)
t_cpu = bench(() -> U_cpu * img_vec_cpu)
t_gpu = bench_gpu(() -> U_gpu * img_vec_gpu)
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

println("\n--- 15. Materialized forward (batch=8, matmul) ---")
X_cpu = hcat([vec(img) for img in images_cpu]...)
X_gpu = CuArray(X_cpu)
t_cpu = bench(() -> U_cpu * X_cpu)
t_gpu = bench_gpu(() -> U_gpu * X_gpu)
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

println("\n--- 16. Materialized L1 loss + gradient (8 images) ---")
mat_fn_cpu = ts -> begin
    U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(ts), m, n)
    ParametricDFT.materialized_loss_l1(U, images_cpu, m, n)
end
mat_fn_gpu = ts -> begin
    U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(ts), m, n)
    ParametricDFT.materialized_loss_l1(U, images_gpu, m, n)
end
t_cpu = bench(() -> Zygote.gradient(mat_fn_cpu, tensors_cpu))
t_gpu = bench_gpu(() -> Zygote.gradient(mat_fn_gpu, tensors_gpu))
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

println("\n--- 17. Full optimize! (1 iter, Adam, materialized L1, batch=8) ---")
mat_gfn_cpu = ts -> begin _, back = Zygote.pullback(mat_fn_cpu, ts); back(1.0)[1]; end
mat_gfn_gpu = ts -> begin _, back = Zygote.pullback(mat_fn_gpu, ts); back(1.0)[1]; end
t_cpu = bench(() -> ParametricDFT.optimize!(opt, copy.(tensors_cpu), mat_fn_cpu, mat_gfn_cpu; max_iter=1))
t_gpu = bench_gpu(() -> ParametricDFT.optimize!(opt, copy.(tensors_gpu), mat_fn_gpu, mat_gfn_gpu; max_iter=1))
@printf("  CPU: %.3f ms   GPU: %.3f ms   ratio: %.1fx\n", t_cpu*1000, t_gpu*1000, t_cpu/t_gpu)

println("\n" * "="^70)
println("  Analysis")
println("="^70)
println("""
  Key insight: For 32×32 images with 2×2 gate tensors, each GPU kernel
  launch (~5-10μs overhead) costs more than the actual computation.
  The einsum contracts many tiny tensors, each becoming a separate kernel.
  Zygote's AD tape multiplies this: backward pass launches ~2x more kernels.

  The materialized unitary path (benchmarks 13-17) builds the full D×D
  unitary matrix once via batched einsum with identity input, then uses
  a single cuBLAS GEMM (U*X) for the forward pass. This eliminates
  hundreds of kernel launches. Compare benchmark 8 (einsum) vs 16
  (materialized) for the same workload.

  GPU wins when: larger images (128×128+), materialized unitary path,
  or when computation per kernel dominates launch overhead.
""")
