# ============================================================================
# speedup_benchmark.jl — PDFT vs Manopt speedup on MSE top-k
# ============================================================================
# Per the design at docs/superpowers/specs/2026-04-18-speedup-benchmark-design.md.
# Produces two tables:
#   - Main decision table: 5 configs × 4 batch sizes at m = n = 6 (64×64).
#   - Large-image appendix: 2 configs × B=1 at m = n = 9 (512×512).
#
# Usage:
#   julia --project=examples examples/speedup_benchmark.jl
# or with nohup:
#   nohup julia --project=examples examples/speedup_benchmark.jl > /tmp/speedup.log 2>&1 &
#
# Requires the `examples/` project (Project.toml) which brings in Manopt,
# Manifolds, ManifoldDiff, RecursiveArrayTools, and ADTypes alongside
# ParametricDFT, Zygote, and CUDA.
# ============================================================================

using ParametricDFT
using ParametricDFT: qft_code, make_batched_code, optimize_batched_code,
                     stack_image_batch, loss_function,
                     MSELoss, RiemannianGD, RiemannianAdam, optimize!
using CUDA
using Zygote
using Random
using Printf

include(joinpath(@__DIR__, "_manopt_baseline.jl"))

const SEED             = 42
const N_ITER           = 10
const N_WARMUP         = 2
const N_TRIALS         = 3
const TIMEOUT_S        = 300          # 5 min per Manopt cell
const MAIN_M, MAIN_N   = 6, 6         # 64 × 64 images
const APX_M, APX_N     = 9, 9         # 512 × 512 images
const MAIN_KEEP_RATIO  = 0.1
const APX_KEEP_RATIO   = 0.1
const MAIN_BATCH_SIZES = (1, 8, 32, 64)

# ----------------------------------------------------------------------------
# PDFT setup + timing
# ----------------------------------------------------------------------------

"""Build batched loss + gradient closures for PDFT, on CPU or GPU."""
function build_pdft(tensors_raw, imgs_raw, m, n, k, device)
    optcode, _ = qft_code(m, n)
    inverse_code, _ = qft_code(m, n; inverse = true)

    tensors = [ParametricDFT.to_device(Matrix{ComplexF64}(t), device) for t in tensors_raw]
    imgs = [ParametricDFT.to_device(ComplexF64.(img), device) for img in imgs_raw]
    stacked = stack_image_batch(imgs, m, n)

    flat_b, blabel = make_batched_code(optcode, length(tensors_raw))
    batched_optcode = optimize_batched_code(flat_b, blabel, length(imgs))

    flat_inv, blabel_inv = make_batched_code(inverse_code, length(tensors_raw))
    batched_inverse_code = optimize_batched_code(flat_inv, blabel_inv, length(imgs))

    loss = MSELoss(k)

    loss_fn = ts -> loss_function(ts, m, n, optcode, stacked, loss;
                                  inverse_code = inverse_code,
                                  batched_optcode = batched_optcode,
                                  batched_inverse_code = batched_inverse_code)
    grad_fn = ts -> begin
        _, back = Zygote.pullback(loss_fn, ts)
        return back(one(real(eltype(ts[1]))))[1]
    end
    return (; tensors, loss_fn, grad_fn)
end

function bench_pdft(tensors_raw, imgs_raw, m, n, k, opt, device)
    built = build_pdft(tensors_raw, imgs_raw, m, n, k, device)

    sync = () -> device === :gpu && CUDA.synchronize()

    timed_run() = @elapsed begin
        optimize!(opt, copy.(built.tensors), built.loss_fn, built.grad_fn;
                  max_iter = N_ITER, tol = 0.0)
        sync()
    end

    for _ in 1:N_WARMUP
        optimize!(opt, copy.(built.tensors), built.loss_fn, built.grad_fn;
                  max_iter = 1, tol = 0.0)
        sync()
    end

    alloc_run = @timed begin
        optimize!(opt, copy.(built.tensors), built.loss_fn, built.grad_fn;
                  max_iter = N_ITER, tol = 0.0)
        sync()
    end
    allocs = Base.gc_alloc_count(alloc_run.gcstats)
    bytes  = alloc_run.bytes

    times = [timed_run() for _ in 1:N_TRIALS]
    return (; time_s = minimum(times), allocs = allocs, memory_bytes = bytes,
              status = :ok)
end

# ----------------------------------------------------------------------------
# Manopt cell
# ----------------------------------------------------------------------------

function bench_manopt(tensors_raw, imgs_raw, m, n, k)
    optcode, _ = qft_code(m, n)
    inverse_code, _ = qft_code(m, n; inverse = true)

    # Single untimed warm-up call to pay Zygote / Manopt JIT
    manopt_gd_run(tensors_raw, imgs_raw, optcode, inverse_code, m, n, k;
                  steps = 1, timeout_s = TIMEOUT_S)

    # Single trial — Manopt is so much slower than PDFT that min-of-3
    # barely moves the reported number and costs 3× the wall-clock.
    r = manopt_gd_run(tensors_raw, imgs_raw, optcode, inverse_code, m, n, k;
                      steps = N_ITER, timeout_s = TIMEOUT_S)
    return (; time_s = r.time_s, allocs = r.allocs,
              memory_bytes = r.memory_bytes, status = r.status)
end

# ----------------------------------------------------------------------------
# Configuration dispatch
# ----------------------------------------------------------------------------

struct Config
    label::String
    library::Symbol     # :pdft or :manopt
    optimizer::Symbol   # :gd or :adam (:gd only for Manopt)
    device::Symbol      # :cpu or :gpu
end

const MAIN_CONFIGS = (
    Config("Manopt-GD-CPU",   :manopt, :gd,   :cpu),
    Config("PDFT-GD-CPU",     :pdft,   :gd,   :cpu),
    Config("PDFT-GD-GPU",     :pdft,   :gd,   :gpu),
    Config("PDFT-Adam-CPU",   :pdft,   :adam, :cpu),
    Config("PDFT-Adam-GPU",   :pdft,   :adam, :gpu),
)

const APPENDIX_CONFIGS = (
    Config("PDFT-GD-CPU", :pdft, :gd, :cpu),
    Config("PDFT-GD-GPU", :pdft, :gd, :gpu),
)

function bench_cell(config::Config, tensors_raw, imgs_raw, m, n, k)
    if config.library === :manopt
        return bench_manopt(tensors_raw, imgs_raw, m, n, k)
    elseif config.library === :pdft
        opt = config.optimizer === :gd ?
              RiemannianGD(lr = 0.01) :
              RiemannianAdam(lr = 0.001)
        return bench_pdft(tensors_raw, imgs_raw, m, n, k, opt, config.device)
    else
        error("unknown config: $(config.library)")
    end
end

# ----------------------------------------------------------------------------
# Table runners
# ----------------------------------------------------------------------------

function setup_problem(m, n, seed)
    Random.seed!(seed)
    _, tensors = qft_code(m, n)
    base_tensors = [Matrix{ComplexF64}(t) for t in tensors]
    return base_tensors
end

draw_images(m, n, B) = [randn(ComplexF64, 2^m, 2^n) for _ in 1:B]

function run_main_table()
    println("\n=== Main decision table: m = n = $MAIN_M (2^$MAIN_M × 2^$MAIN_N images), $N_ITER iters, k = $(floor(Int, MAIN_KEEP_RATIO * 2^(MAIN_M + MAIN_N))) ===")
    base_tensors = setup_problem(MAIN_M, MAIN_N, SEED)
    k = floor(Int, MAIN_KEEP_RATIO * 2^(MAIN_M + MAIN_N))

    @printf "  %-18s " "config"
    for B in MAIN_BATCH_SIZES
        @printf "%-14s " "B=$B"
    end
    println()

    results = Dict{Tuple{String, Int}, Any}()
    for config in MAIN_CONFIGS
        @printf "  %-18s " config.label
        for B in MAIN_BATCH_SIZES
            imgs = draw_images(MAIN_M, MAIN_N, B)
            r = bench_cell(config, base_tensors, imgs, MAIN_M, MAIN_N, k)
            results[(config.label, B)] = r
            cell = r.status === :timeout ?
                   ">$(TIMEOUT_S)s" :
                   @sprintf("%6.1f ms", r.time_s * 1000)
            @printf "%-14s " cell
            flush(stdout)
        end
        println()
    end
    return results
end

function run_appendix()
    println("\n=== Appendix: m = n = $APX_M (2^$APX_M × 2^$APX_N images), B = 1, k = $(floor(Int, APX_KEEP_RATIO * 2^(APX_M + APX_N))) ===")
    base_tensors = setup_problem(APX_M, APX_N, SEED)
    k = floor(Int, APX_KEEP_RATIO * 2^(APX_M + APX_N))
    imgs = draw_images(APX_M, APX_N, 1)

    @printf "  %-18s %s\n" "config" "time / $N_ITER iters (ms)"
    results = Dict{String, Any}()
    for config in APPENDIX_CONFIGS
        r = bench_cell(config, base_tensors, imgs, APX_M, APX_N, k)
        results[config.label] = r
        cell = r.status === :timeout ?
               ">$(TIMEOUT_S)s" :
               @sprintf("%8.1f", r.time_s * 1000)
        @printf "  %-18s %s\n" config.label cell
        flush(stdout)
    end
    return results
end

# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

function main()
    println("speedup_benchmark — seed=$SEED, N_ITER=$N_ITER")
    if CUDA.functional()
        println("GPU: ", CUDA.name(CUDA.device()))
    else
        error("CUDA not functional; this benchmark requires GPU access.")
    end

    main_results = run_main_table()
    appendix_results = run_appendix()

    println("\n=== Done. ===")
    return (main = main_results, appendix = appendix_results)
end

main()
