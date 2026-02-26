# ============================================================================
# Tests for Batched 2×2 Operations (riemannian_optimizers.jl)
# ============================================================================

using Zygote

@testset "Batched 2×2 Operations" begin

    @testset "batched_matmul_2x2" begin
        Random.seed!(123)
        n = 10
        A = randn(ComplexF64, 2, 2, n)
        B = randn(ComplexF64, 2, 2, n)

        C = ParametricDFT.batched_matmul_2x2(A, B)

        # Verify against per-slice matrix multiply
        for k in 1:n
            expected = A[:, :, k] * B[:, :, k]
            @test isapprox(C[:, :, k], expected, atol=1e-12)
        end
    end

    @testset "batched_adjoint_2x2" begin
        Random.seed!(123)
        n = 10
        A = randn(ComplexF64, 2, 2, n)

        Ah = ParametricDFT.batched_adjoint_2x2(A)

        # Verify against per-slice adjoint
        for k in 1:n
            expected = A[:, :, k]'
            @test isapprox(Ah[:, :, k], expected, atol=1e-12)
        end
    end

    @testset "stack_tensors / unstack_tensors! round-trip" begin
        Random.seed!(123)
        tensors = [randn(ComplexF64, 2, 2) for _ in 1:8]
        indices = [1, 3, 5, 7]

        # Stack selected tensors
        batch = ParametricDFT.stack_tensors(tensors, indices)
        @test size(batch) == (2, 2, 4)

        # Verify contents match originals
        for (k, idx) in enumerate(indices)
            @test batch[:, :, k] ≈ tensors[idx]
        end

        # Unstack back — modifies tensors in place
        modified = copy.(tensors)
        # Perturb to verify unstack overwrites
        for idx in indices
            modified[idx] .= 0.0
        end
        ParametricDFT.unstack_tensors!(modified, batch, indices)

        for idx in indices
            @test modified[idx] ≈ tensors[idx]
        end
    end

    @testset "stack_tensors! in-place" begin
        Random.seed!(123)
        tensors = [randn(ComplexF64, 2, 2) for _ in 1:6]
        indices = [2, 4, 6]

        batch = zeros(ComplexF64, 2, 2, 3)
        ParametricDFT.stack_tensors!(batch, tensors, indices)

        for (k, idx) in enumerate(indices)
            @test batch[:, :, k] ≈ tensors[idx]
        end
    end

    @testset "batched_project_unitary" begin
        Random.seed!(123)
        n = 8

        # Create batch of unitary matrices
        U = zeros(ComplexF64, 2, 2, n)
        G = randn(ComplexF64, 2, 2, n)
        for k in 1:n
            U[:, :, k] = Matrix(qr(randn(ComplexF64, 2, 2)).Q)
        end

        proj = ParametricDFT.batched_project_unitary(U, G)

        # Verify against per-tensor projection
        for k in 1:n
            expected = ParametricDFT.project_tangent_unitary(U[:, :, k], G[:, :, k])
            @test isapprox(proj[:, :, k], expected, atol=1e-10)
        end

        # Verify result is in tangent space: U' * proj should be skew-Hermitian
        for k in 1:n
            S = U[:, :, k]' * proj[:, :, k]
            @test isapprox(S, -S', atol=1e-10)
        end
    end

    @testset "batched_retract_unitary_qr" begin
        Random.seed!(123)
        n = 8

        U = zeros(ComplexF64, 2, 2, n)
        G = randn(ComplexF64, 2, 2, n)
        for k in 1:n
            U[:, :, k] = Matrix(qr(randn(ComplexF64, 2, 2)).Q)
        end

        Xi = ParametricDFT.batched_project_unitary(U, G)

        for α in [0.01, 0.1, 0.5]
            Q = ParametricDFT.batched_retract_unitary_qr(U, Xi, α)

            # Verify output is unitary for each slice
            for k in 1:n
                Qk = Q[:, :, k]
                @test isapprox(Qk' * Qk, I(2), atol=1e-10)
                @test isapprox(Qk * Qk', I(2), atol=1e-10)
                @test isapprox(abs(det(Qk)), 1.0, atol=1e-10)
            end
        end
    end

    @testset "batched_project_u1" begin
        Random.seed!(123)
        n = 8

        # Create batch of U(1)^4 elements (unit complex numbers in 2x2 diagonal)
        Z = randn(ComplexF64, 2, 2, n)
        for k in 1:n
            Z[:, :, k] ./= abs.(Z[:, :, k])  # normalize to unit circle
        end
        G = randn(ComplexF64, 2, 2, n)

        proj = ParametricDFT.batched_project_u1(Z, G)

        # Verify against per-tensor projection
        for k in 1:n
            expected = ParametricDFT.project_tangent_u1_product(Z[:, :, k], G[:, :, k])
            @test isapprox(proj[:, :, k], expected, atol=1e-10)
        end
    end

    @testset "batched_retract_u1" begin
        Random.seed!(123)
        n = 8

        Z = randn(ComplexF64, 2, 2, n)
        for k in 1:n
            Z[:, :, k] ./= abs.(Z[:, :, k])
        end
        G = randn(ComplexF64, 2, 2, n)
        Xi = ParametricDFT.batched_project_u1(Z, G)

        for α in [0.01, 0.1, 0.5]
            Z_new = ParametricDFT.batched_retract_u1(Z, Xi, α)

            # Verify output is on unit circle for each element
            for k in 1:n
                for i in 1:2, j in 1:2
                    @test isapprox(abs(Z_new[i, j, k]), 1.0, atol=1e-10)
                end
            end
        end
    end

    @testset "classify_tensors_once" begin
        # Use real QFT tensors to verify classification
        m, n = 3, 3
        _, tensors_raw = ParametricDFT.qft_code(m, n)
        # Convert to Vector{Matrix{ComplexF64}} to match expected signature
        tensors = Matrix{ComplexF64}[Matrix{ComplexF64}(t) for t in tensors_raw]

        unitary_idx, u1_idx = ParametricDFT.classify_tensors_once(tensors)

        # All tensors should be classified
        @test sort(vcat(unitary_idx, u1_idx)) == collect(1:length(tensors))

        # No overlap
        @test isempty(intersect(unitary_idx, u1_idx))

        # For QFT circuit: Hadamard gates are unitary, phase gates are U(1)^4
        # Verify unitary tensors are actually unitary
        for idx in unitary_idx
            t = tensors[idx]
            @test isapprox(t * t', I(2), atol=1e-6)
        end

        # Verify U(1)^4 tensors are NOT full unitary
        for idx in u1_idx
            t = tensors[idx]
            @test !isapprox(t * t', I(2), atol=1e-6)
        end
    end

end
