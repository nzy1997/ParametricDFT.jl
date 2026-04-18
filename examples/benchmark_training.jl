# ============================================================================
# Benchmark harness for the batched training inner loop
# ============================================================================
# Measures wall-clock time per `optimize!` call at several batch sizes, for
# each (loss, optimizer) combination. Populates the numbers on the
# Performance page of the docs.
#
# Usage:
#   julia --project=. examples/benchmark_training.jl
# ============================================================================

using ParametricDFT
using ParametricDFT: qft_code, make_batched_code, optimize_batched_code,
                     stack_image_batch, loss_function,
                     L1Norm, MSELoss,
                     RiemannianGD, RiemannianAdam, optimize!
using Zygote
using Random
using Printf

"""
    build_fns(B, m, n, loss; seed = 0) -> (tensors, loss_fn, grad_fn)

Build the batched loss and gradient closures for `B` random `(2^m, 2^n)`
images. Deterministic in `seed`.
"""
function build_fns(B::Int, m::Int, n::Int, loss; seed::Int = 0)
    Random.seed!(seed)
    optcode, tensors = qft_code(m, n)
    inverse_code, _ = qft_code(m, n; inverse = true)
    tensors = [Matrix{ComplexF64}(t) for t in tensors]

    flat_b, blabel = make_batched_code(optcode, length(tensors))
    batched_optcode = optimize_batched_code(flat_b, blabel, B)

    batched_inverse_code = nothing
    if loss isa MSELoss
        flat_b_inv, blabel_inv = make_batched_code(inverse_code, length(tensors))
        batched_inverse_code = optimize_batched_code(flat_b_inv, blabel_inv, B)
    end

    imgs = [randn(ComplexF64, 2^m, 2^n) for _ in 1:B]
    stacked = stack_image_batch(imgs, m, n)

    loss_fn = ts -> loss_function(ts, m, n, optcode, stacked, loss;
                                  inverse_code = inverse_code,
                                  batched_optcode = batched_optcode,
                                  batched_inverse_code = batched_inverse_code)
    grad_fn = ts -> begin
        _, back = Zygote.pullback(loss_fn, ts)
        return back(1.0)[1]
    end
    return (; tensors, loss_fn, grad_fn)
end

"""
    bench_one(B, m, n, loss, opt; n_iter, n_trials, n_warmup) -> NamedTuple

Run `optimize!` for `n_iter` inner iterations, `n_trials` times (after
`n_warmup` untimed runs), and return the minimum wall-clock time and
the allocation/memory counts from a single run.
"""
function bench_one(B::Int, m::Int, n::Int, loss, opt;
                   n_iter::Int = 5, n_trials::Int = 3, n_warmup::Int = 2)
    built = build_fns(B, m, n, loss)
    ts = built.tensors
    loss_fn = built.loss_fn
    grad_fn = built.grad_fn

    # Warm up — triggers Zygote / einsum compile
    for _ in 1:n_warmup
        optimize!(opt, copy.(ts), loss_fn, grad_fn; max_iter = 1, tol = 0.0)
    end

    # Measure allocations via @timed once (cheap)
    alloc_run = @timed optimize!(opt, copy.(ts), loss_fn, grad_fn;
                                  max_iter = n_iter, tol = 0.0)
    allocs = alloc_run.value === nothing ? 0 : Base.gc_alloc_count(alloc_run.gcstats)
    bytes = alloc_run.bytes

    # Measure wall-clock as the minimum of `n_trials` untimed runs
    times_s = [@elapsed optimize!(opt, copy.(ts), loss_fn, grad_fn;
                                  max_iter = n_iter, tol = 0.0)
               for _ in 1:n_trials]
    min_time_s = minimum(times_s)

    return (time_ms = min_time_s * 1e3,
            allocs = allocs,
            memory_kib = bytes / 1024)
end

function run_table(m::Int, n::Int, loss, optimizer;
                   batch_sizes, n_iter::Int = 5)
    println("\n=== ", typeof(optimizer).name.name, "  +  ",
            typeof(loss).name.name, "  (m=$m, n=$n, $n_iter iters) ===")
    @printf("  %-5s  %-14s  %-12s  %-14s  %-12s\n",
            "B", "time/5it (ms)", "time/iter/B", "allocs", "memory (KiB)")
    for B in batch_sizes
        r = bench_one(B, m, n, loss, optimizer; n_iter = n_iter)
        per_img_us = (r.time_ms * 1e3) / (n_iter * B)  # μs per iteration per image
        @printf("  %-5d  %-14.2f  %-12.2f  %-14d  %-12.1f\n",
                B, r.time_ms, per_img_us, r.allocs, r.memory_kib)
    end
end

function main()
    m, n = 2, 2
    batch_sizes = (1, 4, 16, 64)

    run_table(m, n, L1Norm(),     RiemannianGD(lr = 0.01);     batch_sizes = batch_sizes)
    run_table(m, n, L1Norm(),     RiemannianAdam(lr = 0.001);  batch_sizes = batch_sizes)
    run_table(m, n, MSELoss(8),   RiemannianGD(lr = 0.01);     batch_sizes = batch_sizes)
    run_table(m, n, MSELoss(8),   RiemannianAdam(lr = 0.001);  batch_sizes = batch_sizes)
end

main()
