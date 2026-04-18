# ============================================================================
# Manopt.jl baseline helper
# ============================================================================
# Single-file extraction of the Manopt setup used by optimizer_benchmark.jl
# so speedup_benchmark.jl can call it without duplicating the ProductManifold
# / ArrayPartition / gradient_descent plumbing.
#
# Public API:
#   manopt_gd_run(tensors, images, optcode, inverse_code, m, n, k;
#                 steps, timeout_s = 300)
#     → (final_tensors, time_s, allocs, memory_bytes, status)
#
# `status` is one of :ok or :timeout. On :timeout, the returned time is the
# elapsed wall-clock up to the guard, not an iteration-count extrapolation.
# ============================================================================

using Manopt, Manifolds, ManifoldDiff
using RecursiveArrayTools: ArrayPartition
using ADTypes: AutoZygote
import Zygote

"""Build a ProductManifold of `Stiefel(2, 2, ℂ)` sized to the QFT gate list."""
function _manopt_manifold(tensors)
    S = Stiefel(2, 2, ℂ)
    return ProductManifold(ntuple(_ -> S, length(tensors))...)
end

_tensors2point(tensors) = ArrayPartition(tensors...)
_point2tensors(p)       = collect(p.x)

"""
    manopt_gd_run(tensors, images, optcode, inverse_code, m, n, k;
                  steps = 10, timeout_s = 300)

Run `Manopt.gradient_descent` on the QFT circuit `tensors` with an MSE-plus-
top-k loss summed over `images`. Returns
`(final_tensors, time_s, allocs, memory_bytes, status)`.

`timeout_s` caps the Manopt call. If exceeded, the partially-completed work
is aborted; the returned `status` is `:timeout` and `final_tensors` is
`nothing`.
"""
function manopt_gd_run(
    tensors, images, optcode, inverse_code, m::Int, n::Int, k::Int;
    steps::Int = 10, timeout_s::Real = 300,
)
    loss = ParametricDFT.MSELoss(k)
    imgs = [ComplexF64.(img) for img in images]
    M = _manopt_manifold(tensors)
    p0 = _tensors2point([Matrix{ComplexF64}(t) for t in tensors])

    f = (_M, p) -> begin
        ts = _point2tensors(p)
        total = sum(
            ParametricDFT.loss_function(ts, m, n, optcode, img, loss;
                                        inverse_code = inverse_code)
            for img in imgs
        )
        return Float64(total / length(imgs))
    end

    grad_f = (_M, p) -> ManifoldDiff.gradient(
        _M, x -> f(_M, x), p,
        ManifoldDiff.RiemannianProjectionBackend(AutoZygote()),
    )

    status = :ok
    final_tensors = nothing
    time_s = 0.0
    allocs = 0
    memory_bytes = 0

    # Simple deadline-based timeout: start a Task and wait up to `timeout_s`.
    result_channel = Channel{Any}(1)
    task = @async begin
        try
            gc_before = Base.gc_num()
            t = @elapsed begin
                res = Manopt.gradient_descent(
                    M, f, grad_f, p0;
                    stopping_criterion = Manopt.StopAfterIteration(steps),
                    return_state = true,
                )
            end
            gc_after = Base.gc_num()
            fp = get_solver_result(res)
            fts = [Matrix{ComplexF64}(t2) for t2 in _point2tensors(fp)]
            put!(result_channel, (; time_s = t, tensors = fts,
                                  allocs = Base.gc_alloc_count(gc_after) -
                                           Base.gc_alloc_count(gc_before),
                                  memory = gc_after.allocd - gc_before.allocd))
        catch e
            put!(result_channel, (; error = e))
        end
    end

    deadline = time() + Float64(timeout_s)
    while !istaskdone(task) && time() < deadline
        sleep(0.5)
    end

    if istaskdone(task)
        r = take!(result_channel)
        if haskey(r, :error)
            rethrow(r.error)
        end
        final_tensors = r.tensors
        time_s = r.time_s
        allocs = r.allocs
        memory_bytes = r.memory
    else
        # Manopt has no cooperative interrupt path; we simply abandon the task.
        # Julia will GC it once it finishes; we return with `:timeout`.
        status = :timeout
        time_s = Float64(timeout_s)
    end

    return (; tensors = final_tensors, time_s = time_s,
              allocs = allocs, memory_bytes = memory_bytes, status = status)
end
