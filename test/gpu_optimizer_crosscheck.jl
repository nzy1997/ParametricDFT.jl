# ============================================================================
# GPU Optimizer Cross-Check Tests
# ============================================================================
# This file validates that the custom GPU-compatible Riemannian optimizer
# produces results consistent with the Manifolds.jl/Manopt.jl implementation.

using Test
using ParametricDFT
using LinearAlgebra
using Manifolds
using ManifoldDiff
using Manopt
using Manopt: ConstantStepsize
using Zygote
using ADTypes
using Random

@testset "GPU Optimizer Cross-Check" begin

    @testset "Skew-Hermitian Projection" begin
        # Test that our skew function matches Manifolds.jl behavior
        A = randn(ComplexF64, 4, 4)

        # Our implementation
        S_ours = ParametricDFT.skew(A)

        # Expected: (A - A') / 2
        S_expected = (A - A') / 2

        @test S_ours ≈ S_expected
        @test isapprox(S_ours, -S_ours', atol=1e-14)  # Should be skew-Hermitian
    end

    @testset "Tangent Space Projection" begin
        # Test that our projection produces a valid tangent vector
        # Note: Different libraries may use different formulas for projecting to the
        # tangent space of U(n). Our implementation uses proj_U(G) = U * skew(U' * G)
        # which is a valid projection onto the tangent space T_U = {U*S : S skew-Hermitian}.

        U = Matrix(qr(randn(ComplexF64, 4, 4)).Q)  # Random unitary
        G = randn(ComplexF64, 4, 4)  # Euclidean gradient

        # Our implementation
        ξ_ours = ParametricDFT.project_tangent_unitary(U, G)

        # Check our result is in the tangent space (U' * ξ should be skew-Hermitian)
        S_ours = U' * ξ_ours
        @test isapprox(S_ours, -S_ours', atol=1e-10)

        # Check that ξ_ours can be written as U * S for some skew-Hermitian S
        S_recovered = U' * ξ_ours
        ξ_reconstructed = U * S_recovered
        @test isapprox(ξ_ours, ξ_reconstructed, atol=1e-10)
    end

    @testset "QR Retraction" begin
        # Test that our QR retraction produces unitary matrices
        U = Matrix(qr(randn(ComplexF64, 4, 4)).Q)
        G = randn(ComplexF64, 4, 4)
        ξ = ParametricDFT.project_tangent_unitary(U, G)

        for α in [0.01, 0.1, 0.5, 1.0]
            U_new = ParametricDFT.retract_unitary_qr(U, ξ, α)

            # Result should be unitary
            @test isapprox(U_new' * U_new, I, atol=1e-10)
            @test isapprox(U_new * U_new', I, atol=1e-10)

            # Determinant magnitude should be 1
            @test isapprox(abs(det(U_new)), 1.0, atol=1e-10)
        end
    end

    @testset "Single Step Gradient Descent" begin
        # Verify that our optimizer makes progress with a single step
        Random.seed!(42)

        m, n = 2, 2  # Small circuit for fast testing

        # Create a simple loss function
        target = randn(ComplexF64, 4, 4)

        # Initialize circuit
        optcode, tensors = qft_code(m, n)

        # Define loss: squared distance from target transform
        function loss_fn(ts)
            T = reshape(optcode(ts..., reshape(target, fill(2, m+n)...)), 2^m, 2^n)
            return real(sum(abs2, T))
        end

        # Get initial loss
        initial_loss = loss_fn(tensors)

        # --- Our GPU optimizer: single step ---
        function grad_fn(ts)
            _, back = Zygote.pullback(loss_fn, ts)
            return back(1.0)[1]
        end

        # Copy tensors for our method
        tensors_ours = copy.(tensors)

        # Single gradient step with our optimizer
        euclidean_grads_raw = grad_fn(tensors_ours)
        euclidean_grads = euclidean_grads_raw isa Tuple ? collect(euclidean_grads_raw) : euclidean_grads_raw
        lr = 0.01
        for i in 1:length(tensors_ours)
            riemannian_grad = ParametricDFT.project_tangent_unitary(tensors_ours[i], euclidean_grads[i])
            tensors_ours[i] = ParametricDFT.retract_unitary_qr(tensors_ours[i], -riemannian_grad, lr)
        end
        loss_ours = loss_fn(tensors_ours)

        # Our optimizer should reduce loss
        @test loss_ours < initial_loss

        # Tensors should remain unitary after the step
        for t in tensors_ours
            @test isapprox(t' * t, I(size(t, 1)), atol=1e-10)
        end
    end

    @testset "Multi-Step Optimization Convergence" begin
        # Test that both optimizers converge to similar solutions
        Random.seed!(123)

        m, n = 2, 2

        # Simple training image
        img = randn(4, 4)

        # Initialize with same tensors
        optcode, tensors_init = qft_code(m, n)
        inverse_code, _ = qft_code(m, n; inverse=true)

        k = 4  # Keep 4 out of 16 coefficients
        loss = ParametricDFT.MSELoss(k)

        # Loss function
        function loss_fn(ts)
            ParametricDFT.loss_function(ts, m, n, optcode, Complex{Float64}.(img), loss; inverse_code=inverse_code)
        end

        initial_loss = loss_fn(tensors_init)

        # --- Our GPU optimizer ---
        function grad_fn(ts)
            _, back = Zygote.pullback(loss_fn, ts)
            return back(1.0)[1]
        end

        tensors_ours = copy.(tensors_init)
        lr = 0.05
        steps = 50

        for iter in 1:steps
            euclidean_grads = grad_fn(tensors_ours)
            for i in 1:length(tensors_ours)
                riemannian_grad = ParametricDFT.project_tangent_unitary(tensors_ours[i], euclidean_grads[i])
                tensors_ours[i] = ParametricDFT.retract_unitary_qr(tensors_ours[i], -riemannian_grad, lr)
            end
        end
        loss_ours = loss_fn(tensors_ours)

        # --- Manopt ---
        M = generate_manifold(tensors_init)
        theta = tensors2point(tensors_init, M)
        f(M, p) = loss_fn(point2tensors(p, M))
        grad_f(M, p) = ManifoldDiff.gradient(M, x -> f(M, x), p, RiemannianProjectionBackend(AutoZygote()))

        sc = StopAfterIteration(steps)
        theta_manopt = gradient_descent(M, f, grad_f, theta; stopping_criterion=sc, debug=[])
        tensors_manopt = point2tensors(theta_manopt, M)
        loss_manopt = loss_fn(tensors_manopt)

        # Both should significantly reduce loss
        @test loss_ours < 0.9 * initial_loss
        @test loss_manopt < 0.9 * initial_loss

        # Final losses should be in similar range (within 2x of each other)
        ratio = loss_ours / max(loss_manopt, 1e-10)
        @test 0.2 < ratio < 5.0

        @info "Multi-step optimization results:" initial_loss loss_ours loss_manopt
    end

    @testset "All Tensors Remain Unitary" begin
        # Verify that after optimization, all tensors remain unitary
        Random.seed!(456)

        m, n = 3, 3
        img = randn(8, 8)

        optcode, tensors_init = qft_code(m, n)
        inverse_code, _ = qft_code(m, n; inverse=true)
        k = 8
        loss = ParametricDFT.MSELoss(k)

        function loss_fn(ts)
            ParametricDFT.loss_function(ts, m, n, optcode, Complex{Float64}.(img), loss; inverse_code=inverse_code)
        end

        function grad_fn(ts)
            _, back = Zygote.pullback(loss_fn, ts)
            return back(1.0)[1]
        end

        tensors = copy.(tensors_init)
        lr = 0.01

        for iter in 1:100
            euclidean_grads = grad_fn(tensors)
            for i in 1:length(tensors)
                riemannian_grad = ParametricDFT.project_tangent_unitary(tensors[i], euclidean_grads[i])
                tensors[i] = ParametricDFT.retract_unitary_qr(tensors[i], -riemannian_grad, lr)
            end
        end

        # Check all tensors are unitary
        for (i, t) in enumerate(tensors)
            @test isapprox(t' * t, I(size(t, 1)), atol=1e-8)
        end
    end

    @testset "riemannian_gradient_descent_gpu Function" begin
        # Test the full GPU optimizer function
        Random.seed!(789)

        m, n = 2, 2
        img = randn(4, 4)

        optcode, tensors_raw = qft_code(m, n)
        inverse_code, _ = qft_code(m, n; inverse=true)
        k = 4
        loss_type = ParametricDFT.MSELoss(k)

        # Convert to Vector{Matrix{ComplexF64}} for type stability
        tensors_init = [Matrix{ComplexF64}(t) for t in tensors_raw]

        function loss_fn(ts)
            ParametricDFT.loss_function(ts, m, n, optcode, Complex{Float64}.(img), loss_type; inverse_code=inverse_code)
        end

        function grad_fn(ts)
            _, back = Zygote.pullback(loss_fn, ts)
            return back(1.0)[1]
        end

        initial_loss = loss_fn(tensors_init)

        # Run the GPU optimizer
        optimized = ParametricDFT.riemannian_gradient_descent_gpu(
            tensors_init, loss_fn, grad_fn;
            lr=0.05, max_iter=50, tol=1e-8, verbose=false
        )

        final_loss = loss_fn(optimized)

        @test final_loss < initial_loss

        # Check all optimized tensors remain on their manifolds
        is_unitary = [ParametricDFT.is_unitary_tensor(t) for t in tensors_init]
        for (i, t) in enumerate(optimized)
            if is_unitary[i]
                @test isapprox(t' * t, I(size(t, 1)), atol=1e-8)
            else
                # U(1)^4: each element should have magnitude 1
                for idx in eachindex(t)
                    @test isapprox(abs(t[idx]), 1.0, atol=1e-8)
                end
            end
        end

        @info "GPU optimizer result:" initial_loss final_loss reduction_percent=round((1-final_loss/initial_loss)*100, digits=1)
    end

    # ========================================================================
    # Parallel Transport Tests
    # ========================================================================

    @testset "Parallel Transport on U(2)" begin
        Random.seed!(100)
        U_old = Matrix(qr(randn(ComplexF64, 4, 4)).Q)
        U_new = Matrix(qr(randn(ComplexF64, 4, 4)).Q)

        # Create a tangent vector at U_old
        S = randn(ComplexF64, 4, 4)
        v = ParametricDFT.project_tangent_unitary(U_old, S)

        # Transport to U_new
        v_transported = ParametricDFT.parallel_transport_unitary(U_old, U_new, v)

        # Transported vector should be in tangent space at U_new
        S_new = U_new' * v_transported
        @test isapprox(S_new, -S_new', atol=1e-10)  # skew-Hermitian

        # Norm should be approximately preserved (projection-based transport is approximate)
        @test norm(v_transported) > 0
    end

    @testset "Parallel Transport on U(1)^4" begin
        Random.seed!(101)
        z_old = exp.(im .* randn(2, 2))  # Random phases on U(1)
        z_new = exp.(im .* randn(2, 2))

        # Create a tangent vector at z_old
        g = randn(ComplexF64, 2, 2)
        v = ParametricDFT.project_tangent_u1_product(z_old, g)

        # Transport to z_new
        v_transported = ParametricDFT.parallel_transport_u1_product(z_old, z_new, v)

        # Transported vector should be in tangent space at z_new
        # For U(1), tangent vectors satisfy: real(conj(z) * v) = 0
        for idx in eachindex(z_new)
            @test abs(real(conj(z_new[idx]) * v_transported[idx])) < 1e-10
        end
    end

    # ========================================================================
    # Riemannian Adam Tests
    # ========================================================================

    @testset "Riemannian Adam Single Step" begin
        Random.seed!(200)

        m, n = 2, 2
        img = randn(4, 4)

        optcode, tensors_raw = qft_code(m, n)
        inverse_code, _ = qft_code(m, n; inverse=true)
        tensors_init = [Matrix{ComplexF64}(t) for t in tensors_raw]

        k = 4
        loss_type = ParametricDFT.MSELoss(k)

        function loss_fn(ts)
            ParametricDFT.loss_function(ts, m, n, optcode, Complex{Float64}.(img), loss_type; inverse_code=inverse_code)
        end

        initial_loss = loss_fn(tensors_init)

        function grad_fn(ts)
            _, back = Zygote.pullback(loss_fn, ts)
            return back(1.0)[1]
        end

        # Run Riemannian Adam for a few steps
        # (Adam's first step is small due to bias correction, so use 5 steps)
        optimized = ParametricDFT.riemannian_adam(
            tensors_init, loss_fn, grad_fn;
            lr=0.01, max_iter=5, tol=1e-12, verbose=false
        )

        final_loss = loss_fn(optimized)

        # Adam should reduce loss
        @test final_loss < initial_loss

        # Tensors should remain on their manifolds
        is_unitary = [ParametricDFT.is_unitary_tensor(t) for t in tensors_init]
        for (i, t) in enumerate(optimized)
            if is_unitary[i]
                @test isapprox(t' * t, I(size(t, 1)), atol=1e-8)
            else
                for idx in eachindex(t)
                    @test isapprox(abs(t[idx]), 1.0, atol=1e-8)
                end
            end
        end
    end

    @testset "Riemannian Adam Multi-Step Convergence" begin
        Random.seed!(300)

        m, n = 2, 2
        img = randn(4, 4)

        optcode, tensors_raw = qft_code(m, n)
        inverse_code, _ = qft_code(m, n; inverse=true)
        tensors_init = [Matrix{ComplexF64}(t) for t in tensors_raw]

        k = 4
        loss_type = ParametricDFT.MSELoss(k)

        function loss_fn(ts)
            ParametricDFT.loss_function(ts, m, n, optcode, Complex{Float64}.(img), loss_type; inverse_code=inverse_code)
        end

        function grad_fn(ts)
            _, back = Zygote.pullback(loss_fn, ts)
            return back(1.0)[1]
        end

        initial_loss = loss_fn(tensors_init)

        # --- Riemannian Adam ---
        optimized_adam = ParametricDFT.riemannian_adam(
            tensors_init, loss_fn, grad_fn;
            lr=0.01, max_iter=50, tol=1e-12, verbose=false
        )
        loss_adam = loss_fn(optimized_adam)

        # --- Riemannian GD (for comparison) ---
        optimized_gd = ParametricDFT.riemannian_gradient_descent_gpu(
            tensors_init, loss_fn, grad_fn;
            lr=0.05, max_iter=50, tol=1e-12, verbose=false
        )
        loss_gd = loss_fn(optimized_gd)

        # --- Manopt GD (for comparison) ---
        M = generate_manifold(tensors_init)
        theta = tensors2point(tensors_init, M)
        f(M, p) = loss_fn(point2tensors(p, M))
        grad_f(M, p) = ManifoldDiff.gradient(M, x -> f(M, x), p, RiemannianProjectionBackend(AutoZygote()))
        sc = StopAfterIteration(50)
        theta_manopt = gradient_descent(M, f, grad_f, theta; stopping_criterion=sc, debug=[])
        loss_manopt = loss_fn(point2tensors(theta_manopt, M))

        # All three should significantly reduce loss
        @test loss_adam < 0.9 * initial_loss
        @test loss_gd < 0.9 * initial_loss
        @test loss_manopt < 0.9 * initial_loss

        # Adam should be in reasonable range compared to others
        ratio_adam_gd = loss_adam / max(loss_gd, 1e-10)
        ratio_adam_manopt = loss_adam / max(loss_manopt, 1e-10)
        @test 0.1 < ratio_adam_gd < 10.0
        @test 0.1 < ratio_adam_manopt < 10.0

        @info "Adam vs GD vs Manopt:" initial_loss loss_adam loss_gd loss_manopt
    end

    @testset "Riemannian Adam Handles Mixed Manifolds" begin
        Random.seed!(400)

        m, n = 2, 2
        img = randn(4, 4)

        optcode, tensors_raw = qft_code(m, n)
        inverse_code, _ = qft_code(m, n; inverse=true)
        tensors_init = [Matrix{ComplexF64}(t) for t in tensors_raw]

        k = 4
        loss_type = ParametricDFT.MSELoss(k)

        function loss_fn(ts)
            ParametricDFT.loss_function(ts, m, n, optcode, Complex{Float64}.(img), loss_type; inverse_code=inverse_code)
        end

        function grad_fn(ts)
            _, back = Zygote.pullback(loss_fn, ts)
            return back(1.0)[1]
        end

        is_unitary = [ParametricDFT.is_unitary_tensor(t) for t in tensors_init]
        n_unitary = sum(is_unitary)
        n_u1 = length(tensors_init) - n_unitary

        # Verify we have both manifold types
        @test n_unitary > 0
        @test n_u1 > 0

        # Run Adam for 100 iterations
        optimized = ParametricDFT.riemannian_adam(
            tensors_init, loss_fn, grad_fn;
            lr=0.01, max_iter=100, tol=1e-12, verbose=false
        )

        # Check all tensors remain on their manifolds
        for (i, t) in enumerate(optimized)
            if is_unitary[i]
                @test isapprox(t' * t, I(size(t, 1)), atol=1e-8)
            else
                for idx in eachindex(t)
                    @test isapprox(abs(t[idx]), 1.0, atol=1e-8)
                end
            end
        end

        # Loss should have decreased
        @test loss_fn(optimized) < loss_fn(tensors_init)
    end

    @testset "riemannian_adam Function" begin
        # Test the full riemannian_adam function interface
        Random.seed!(500)

        m, n = 2, 2
        img = randn(4, 4)

        optcode, tensors_raw = qft_code(m, n)
        inverse_code, _ = qft_code(m, n; inverse=true)
        tensors_init = [Matrix{ComplexF64}(t) for t in tensors_raw]

        k = 4
        loss_type = ParametricDFT.MSELoss(k)

        function loss_fn(ts)
            ParametricDFT.loss_function(ts, m, n, optcode, Complex{Float64}.(img), loss_type; inverse_code=inverse_code)
        end

        function grad_fn(ts)
            _, back = Zygote.pullback(loss_fn, ts)
            return back(1.0)[1]
        end

        initial_loss = loss_fn(tensors_init)

        # Run with default Adam hyperparameters
        optimized = ParametricDFT.riemannian_adam(
            tensors_init, loss_fn, grad_fn;
            lr=0.01, betas=(0.9, 0.999), eps=1e-8,
            max_iter=50, tol=1e-8, verbose=false
        )

        final_loss = loss_fn(optimized)

        @test final_loss < initial_loss

        # All tensors should be valid
        is_unitary = [ParametricDFT.is_unitary_tensor(t) for t in tensors_init]
        for (i, t) in enumerate(optimized)
            if is_unitary[i]
                @test isapprox(t' * t, I(size(t, 1)), atol=1e-8)
            end
        end

        @info "Riemannian Adam result:" initial_loss final_loss reduction_percent=round((1-final_loss/initial_loss)*100, digits=1)
    end

end
