# ============================================================================
# Tests for Riemannian Optimizer New API (optimizers.jl)
# ============================================================================
# Tests RiemannianGD and RiemannianAdam via the optimize! interface,
# verifying loss reduction, cross-optimizer consistency, and manifold constraints.

using Test
using ParametricDFT
using LinearAlgebra
using Random
using OMEinsum
using Zygote
using Manifolds
using Manopt
using RecursiveArrayTools

@testset "Riemannian Optimizers (New API)" begin

    # Shared setup: 4x4 image, QFT circuit with m=2, n=2
    function make_test_problem(; seed=42)
        Random.seed!(seed)
        m, n = 2, 2
        pic = rand(ComplexF64, 2^m, 2^n)
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        optcode_inv, _ = ParametricDFT.qft_code(m, n; inverse=true)
        # Convert to Vector{Matrix{ComplexF64}} for type stability with optimize!
        tensors = Matrix{ComplexF64}[Matrix{ComplexF64}(t) for t in tensors_raw]
        loss_obj = ParametricDFT.MSELoss(4)

        loss_fn = ts -> ParametricDFT.loss_function(ts, m, n, optcode, pic, loss_obj; inverse_code=optcode_inv)

        function grad_fn(ts)
            _, back = Zygote.pullback(loss_fn, ts)
            grads = back(one(Float64))[1]
            return grads
        end

        return tensors, loss_fn, grad_fn
    end

    @testset "RiemannianGD reduces loss" begin
        tensors, loss_fn, grad_fn = make_test_problem(seed=42)
        initial_loss = loss_fn(tensors)

        opt = ParametricDFT.RiemannianGD(lr=0.05)
        optimized = ParametricDFT.optimize!(opt, tensors, loss_fn, grad_fn;
            max_iter=50, tol=1e-10)

        final_loss = loss_fn(optimized)
        @test final_loss < initial_loss
    end

    @testset "RiemannianAdam reduces loss" begin
        tensors, loss_fn, grad_fn = make_test_problem(seed=42)
        initial_loss = loss_fn(tensors)

        opt = ParametricDFT.RiemannianAdam(lr=0.01)
        optimized = ParametricDFT.optimize!(opt, tensors, loss_fn, grad_fn;
            max_iter=50, tol=1e-10)

        final_loss = loss_fn(optimized)
        @test final_loss < initial_loss
    end

    @testset "GD and Adam produce similar results" begin
        # Both should reduce loss by >10%
        tensors_gd, loss_fn_gd, grad_fn_gd = make_test_problem(seed=123)
        tensors_adam, loss_fn_adam, grad_fn_adam = make_test_problem(seed=123)

        initial_loss = loss_fn_gd(tensors_gd)

        opt_gd = ParametricDFT.RiemannianGD(lr=0.05)
        optimized_gd = ParametricDFT.optimize!(opt_gd, tensors_gd, loss_fn_gd, grad_fn_gd;
            max_iter=50, tol=1e-10)
        final_loss_gd = loss_fn_gd(optimized_gd)

        opt_adam = ParametricDFT.RiemannianAdam(lr=0.01)
        optimized_adam = ParametricDFT.optimize!(opt_adam, tensors_adam, loss_fn_adam, grad_fn_adam;
            max_iter=50, tol=1e-10)
        final_loss_adam = loss_fn_adam(optimized_adam)

        # Both should reduce loss by more than 10%
        @test final_loss_gd < 0.9 * initial_loss
        @test final_loss_adam < 0.9 * initial_loss

        # Ratio of final losses should be within a reasonable range
        ratio = final_loss_gd / final_loss_adam
        @test 0.1 < ratio < 10.0
    end

    @testset "Tensors remain on manifolds after optimization" begin
        tensors, loss_fn, grad_fn = make_test_problem(seed=42)

        # Run GD
        opt_gd = ParametricDFT.RiemannianGD(lr=0.05)
        optimized_gd = ParametricDFT.optimize!(opt_gd, tensors, loss_fn, grad_fn;
            max_iter=30, tol=1e-10)

        for (i, t) in enumerate(optimized_gd)
            if ParametricDFT.is_unitary_general(t)
                # Unitary tensor: t' * t should be identity
                @test t' * t ≈ Matrix{eltype(t)}(I, size(t)...) atol=1e-6
            else
                # Phase tensor: each element should have unit magnitude
                for idx in eachindex(t)
                    @test abs(t[idx]) ≈ 1.0 atol=1e-6
                end
            end
        end

        # Run Adam
        tensors2, loss_fn2, grad_fn2 = make_test_problem(seed=42)
        opt_adam = ParametricDFT.RiemannianAdam(lr=0.01)
        optimized_adam = ParametricDFT.optimize!(opt_adam, tensors2, loss_fn2, grad_fn2;
            max_iter=30, tol=1e-10)

        for (i, t) in enumerate(optimized_adam)
            if ParametricDFT.is_unitary_general(t)
                @test t' * t ≈ Matrix{eltype(t)}(I, size(t)...) atol=1e-6
            else
                for idx in eachindex(t)
                    @test abs(t[idx]) ≈ 1.0 atol=1e-6
                end
            end
        end
    end

    @testset "loss_trace records per-iteration losses" begin
        tensors, loss_fn, grad_fn = make_test_problem(seed=42)

        # GD with loss_trace
        opt_gd = ParametricDFT.RiemannianGD(lr=0.05)
        trace_gd = Float64[]
        ParametricDFT.optimize!(opt_gd, tensors, loss_fn, grad_fn;
            max_iter=20, tol=1e-10, loss_trace=trace_gd)

        @test length(trace_gd) > 0
        @test all(isfinite, trace_gd)
        # Armijo line search should produce non-increasing losses
        for i in 2:length(trace_gd)
            @test trace_gd[i] <= trace_gd[i-1] + 1e-10
        end

        # Adam with loss_trace
        tensors2, loss_fn2, grad_fn2 = make_test_problem(seed=42)
        opt_adam = ParametricDFT.RiemannianAdam(lr=0.01)
        trace_adam = Float64[]
        ParametricDFT.optimize!(opt_adam, tensors2, loss_fn2, grad_fn2;
            max_iter=20, tol=1e-10, loss_trace=trace_adam)

        @test length(trace_adam) > 0
        @test all(isfinite, trace_adam)
    end

    # ========================================================================
    # Manopt.jl Cross-Check
    # ========================================================================

    @testset "Cross-check with Manopt.jl" begin
        # Local helpers: build Manifolds.jl ProductManifold and convert tensors
        # (these functions were removed from the main module)
        function _build_manifold(tensors)
            M2 = Manifolds.UnitaryMatrices(2)
            M1 = PowerManifold(Manifolds.UnitaryMatrices(1), 4)
            return ProductManifold(map(t -> isapprox(t * t', I(2), atol=1e-6) ? M2 : M1, tensors)...)
        end

        function _tensors2point(tensors, M::ProductManifold)
            ArrayPartition([
                mi isa Manifolds.UnitaryMatrices ?
                    tensors[j] :
                    [tensors[j][1,1];;; tensors[j][1,2];;; tensors[j][2,1];;; tensors[j][2,2]]
                for (j, mi) in enumerate(M.manifolds)
            ]...)
        end

        function _point2tensors(p, M::ProductManifold)
            [mi isa Manifolds.UnitaryMatrices ? p.x[j] : reshape(p.x[j], 2, 2)
             for (j, mi) in enumerate(M.manifolds)]
        end

        Random.seed!(123)
        m, n = 2, 2
        pic = rand(ComplexF64, 4, 4)
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        optcode_inv, _ = ParametricDFT.qft_code(m, n; inverse=true)
        tensors_init = Matrix{ComplexF64}[Matrix{ComplexF64}(t) for t in tensors_raw]
        loss_obj = ParametricDFT.MSELoss(4)

        loss_fn = ts -> ParametricDFT.loss_function(
            ts, m, n, optcode, pic, loss_obj; inverse_code=optcode_inv)
        grad_fn = ts -> begin
            _, back = Zygote.pullback(loss_fn, ts)
            back(one(Float64))[1]
        end

        initial_loss = loss_fn(tensors_init)
        steps = 50

        # --- Manopt baseline ---
        M = _build_manifold(tensors_init)
        theta0 = _tensors2point(tensors_init, M)
        f_manopt(M, p) = loss_fn(_point2tensors(p, M))
        # Compute Euclidean gradient via Zygote, then project onto tangent space
        function grad_manopt(M, p)
            egrad = Zygote.gradient(x -> f_manopt(M, x), p)[1]
            return Manifolds.project(M, p, egrad)
        end

        theta_opt = gradient_descent(M, f_manopt, grad_manopt, theta0;
            stopping_criterion=StopAfterIteration(steps), debug=[])
        loss_manopt = loss_fn(_point2tensors(theta_opt, M))

        # --- Our RiemannianGD ---
        opt_gd = ParametricDFT.RiemannianGD(lr=0.05)
        optimized_gd = ParametricDFT.optimize!(
            opt_gd, copy.(tensors_init), loss_fn, grad_fn;
            max_iter=steps, tol=1e-10)
        loss_gd = loss_fn(optimized_gd)

        # --- Our RiemannianAdam ---
        opt_adam = ParametricDFT.RiemannianAdam(lr=0.01)
        optimized_adam = ParametricDFT.optimize!(
            opt_adam, copy.(tensors_init), loss_fn, grad_fn;
            max_iter=steps, tol=1e-10)
        loss_adam = loss_fn(optimized_adam)

        # All three should significantly reduce loss
        @test loss_manopt < 0.9 * initial_loss
        @test loss_gd < 0.9 * initial_loss
        @test loss_adam < 0.9 * initial_loss

        # Our optimizers should reach similar loss levels as Manopt (within 5x)
        @test 0.2 < loss_gd / max(loss_manopt, 1e-10) < 5.0
        @test 0.2 < loss_adam / max(loss_manopt, 1e-10) < 5.0
    end

end  # @testset "Riemannian Optimizers (New API)"
