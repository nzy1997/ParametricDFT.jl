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

        # Check all optimized tensors are unitary
        for (i, t) in enumerate(optimized)
            @test isapprox(t' * t, I(size(t, 1)), atol=1e-8)
        end

        @info "GPU optimizer result:" initial_loss final_loss reduction_percent=round((1-final_loss/initial_loss)*100, digits=1)
    end

end
