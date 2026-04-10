@testset "Manifold Abstraction" begin

    @testset "batched_matmul generalized" begin
        Random.seed!(42)
        for d in [2, 3, 4]
            n = 5
            A = randn(ComplexF64, d, d, n)
            B = randn(ComplexF64, d, d, n)
            C = ParametricDFT.batched_matmul(A, B)
            @test size(C) == (d, d, n)
            for k in 1:n
                @test C[:, :, k] ≈ A[:, :, k] * B[:, :, k]
            end
        end
    end

    @testset "batched_matmul rectangular" begin
        Random.seed!(43)
        n = 5
        A = randn(ComplexF64, 3, 4, n)
        B = randn(ComplexF64, 4, 2, n)
        C = ParametricDFT.batched_matmul(A, B)
        @test size(C) == (3, 2, n)
        for k in 1:n
            @test C[:, :, k] ≈ A[:, :, k] * B[:, :, k]
        end
    end

    @testset "batched_matmul adversarial" begin
        @test_throws AssertionError ParametricDFT.batched_matmul(
            randn(ComplexF64, 2, 3, 4),
            randn(ComplexF64, 2, 2, 4)   # inner dim mismatch: 3 ≠ 2
        )
        @test_throws AssertionError ParametricDFT.batched_matmul(
            randn(ComplexF64, 2, 3, 4),
            randn(ComplexF64, 3, 2, 5)   # batch size mismatch: 4 ≠ 5
        )
    end

    @testset "batched_matmul strided batched path" begin
        Random.seed!(53)
        for d in [2, 3, 4]
            n = 20
            A = randn(ComplexF64, d, d, n)
            B = randn(ComplexF64, d, d, n)
            C = ParametricDFT.batched_matmul(A, B)
            @test size(C) == (d, d, n)
            for k in 1:n
                @test C[:, :, k] ≈ A[:, :, k] * B[:, :, k] atol=1e-12
            end
        end
    end

    @testset "batched_adjoint generalized" begin
        Random.seed!(44)
        for d in [2, 3, 4]
            n = 5
            A = randn(ComplexF64, d, d, n)
            Ah = ParametricDFT.batched_adjoint(A)
            @test size(Ah) == (d, d, n)
            for k in 1:n
                @test Ah[:, :, k] ≈ A[:, :, k]'
            end
        end
    end

    @testset "batched_adjoint rectangular" begin
        Random.seed!(45)
        n = 5
        A = randn(ComplexF64, 3, 4, n)
        Ah = ParametricDFT.batched_adjoint(A)
        @test size(Ah) == (4, 3, n)
        for k in 1:n
            @test Ah[:, :, k] ≈ A[:, :, k]'
        end
    end

    @testset "batched_adjoint real-valued" begin
        Random.seed!(46)
        n = 4
        A = randn(Float64, 3, 4, n)
        Ah = ParametricDFT.batched_adjoint(A)
        @test size(Ah) == (4, 3, n)
        for k in 1:n
            @test Ah[:, :, k] ≈ A[:, :, k]'
        end
    end

    @testset "UnitaryManifold project" begin
        Random.seed!(46)
        um = ParametricDFT.UnitaryManifold()
        for d in [2, 3, 4]
            n = 5
            # Generate random unitary matrices
            U = Array{ComplexF64}(undef, d, d, n)
            for k in 1:n
                Q, _ = qr(randn(ComplexF64, d, d))
                U[:, :, k] = Matrix{ComplexF64}(Q)
            end
            G = randn(ComplexF64, d, d, n)
            proj = ParametricDFT.project(um, U, G)
            @test size(proj) == (d, d, n)
            # U' * proj should be skew-Hermitian for each slice
            for k in 1:n
                UhP = U[:, :, k]' * proj[:, :, k]
                @test UhP ≈ -UhP' atol=1e-10
            end
        end
    end

    @testset "UnitaryManifold retract" begin
        Random.seed!(47)
        um = ParametricDFT.UnitaryManifold()
        for d in [2, 3, 4]
            n = 5
            U = Array{ComplexF64}(undef, d, d, n)
            for k in 1:n
                Q, _ = qr(randn(ComplexF64, d, d))
                U[:, :, k] = Matrix{ComplexF64}(Q)
            end
            G = randn(ComplexF64, d, d, n)
            Xi = ParametricDFT.project(um, U, G)
            Q_ret = ParametricDFT.retract(um, U, Xi, 0.1)
            @test size(Q_ret) == (d, d, n)
            # Each slice should be unitary: Q'Q ≈ I
            for k in 1:n
                @test Q_ret[:, :, k]' * Q_ret[:, :, k] ≈ Matrix{ComplexF64}(I, d, d) atol=1e-10
            end
        end
    end

    @testset "batched_inv" begin
        Random.seed!(53)
        for d in [2, 3, 4]
            n = 5
            A = Array{ComplexF64}(undef, d, d, n)
            for k in 1:n
                A[:, :, k] = randn(ComplexF64, d, d) + d * Matrix{ComplexF64}(I, d, d)
            end
            Ainv = ParametricDFT.batched_inv(A)
            @test size(Ainv) == (d, d, n)
            for k in 1:n
                @test Ainv[:, :, k] ≈ inv(A[:, :, k]) atol=1e-10
                @test A[:, :, k] * Ainv[:, :, k] ≈ Matrix{ComplexF64}(I, d, d) atol=1e-10
            end
        end
    end

    @testset "Cayley retract small step" begin
        Random.seed!(54)
        um = ParametricDFT.UnitaryManifold()
        for d in [2, 3, 4]
            n = 5
            U = Array{ComplexF64}(undef, d, d, n)
            for k in 1:n
                Q, _ = qr(randn(ComplexF64, d, d))
                U[:, :, k] = Matrix{ComplexF64}(Q)
            end
            G = randn(ComplexF64, d, d, n)
            Xi = ParametricDFT.project(um, U, G)

            # Small step should stay close to U
            Q_small = ParametricDFT.retract(um, U, Xi, 1e-8)
            for k in 1:n
                @test Q_small[:, :, k] ≈ U[:, :, k] atol=1e-6
            end
        end
    end

    @testset "UnitaryManifold retract with pre-allocated I_batch" begin
        Random.seed!(61)
        um = ParametricDFT.UnitaryManifold()
        d, n = 4, 5
        U = Array{ComplexF64}(undef, d, d, n)
        for k in 1:n
            Q, _ = qr(randn(ComplexF64, d, d))
            U[:, :, k] = Matrix{ComplexF64}(Q)
        end
        G = randn(ComplexF64, d, d, n)
        Xi = ParametricDFT.project(um, U, G)

        I_b = zeros(ComplexF64, d, d, n)
        for k in 1:n, i in 1:d
            I_b[i, i, k] = one(ComplexF64)
        end

        result_with    = ParametricDFT.retract(um, U, Xi, 0.1; I_batch=I_b)
        result_without = ParametricDFT.retract(um, U, Xi, 0.1)

        @test result_with ≈ result_without atol=1e-14
        for k in 1:n
            @test result_with[:, :, k]' * result_with[:, :, k] ≈ Matrix{ComplexF64}(I, d, d) atol=1e-10
        end
    end

    @testset "PhaseManifold project" begin
        Random.seed!(48)
        pm = ParametricDFT.PhaseManifold()
        n = 5
        # Generate random unit complex numbers as (d1, d2, n) arrays
        for (d1, d2) in [(2, 2), (3, 3), (2, 4)]
            angles = randn(d1, d2, n)
            Z = Complex{Float64}.(exp.(im .* angles))
            G = randn(ComplexF64, d1, d2, n)
            proj = ParametricDFT.project(pm, Z, G)
            @test size(proj) == (d1, d2, n)
            # real(conj(z) * proj) ≈ 0 for each element
            for k in 1:n
                for j in 1:d2, i in 1:d1
                    @test real(conj(Z[i, j, k]) * proj[i, j, k]) ≈ 0.0 atol=1e-10
                end
            end
        end
    end

    @testset "PhaseManifold retract" begin
        Random.seed!(49)
        pm = ParametricDFT.PhaseManifold()
        n = 5
        for (d1, d2) in [(2, 2), (3, 3), (2, 4)]
            angles = randn(d1, d2, n)
            Z = Complex{Float64}.(exp.(im .* angles))
            G = randn(ComplexF64, d1, d2, n)
            Xi = ParametricDFT.project(pm, Z, G)
            Z_new = ParametricDFT.retract(pm, Z, Xi, 0.1)
            @test size(Z_new) == (d1, d2, n)
            # Each element should have unit modulus
            for k in 1:n
                for j in 1:d2, i in 1:d1
                    @test abs(Z_new[i, j, k]) ≈ 1.0 atol=1e-10
                end
            end
        end
    end

    @testset "classify_manifold" begin
        Random.seed!(50)
        # Unitary matrix should be classified as UnitaryManifold
        Q, _ = qr(randn(ComplexF64, 2, 2))
        U = Matrix{ComplexF64}(Q)
        @test ParametricDFT.classify_manifold(U) isa ParametricDFT.UnitaryManifold

        # Also works for larger unitary matrices
        Q3, _ = qr(randn(ComplexF64, 3, 3))
        U3 = Matrix{ComplexF64}(Q3)
        @test ParametricDFT.classify_manifold(U3) isa ParametricDFT.UnitaryManifold

        Q4, _ = qr(randn(ComplexF64, 4, 4))
        U4 = Matrix{ComplexF64}(Q4)
        @test ParametricDFT.classify_manifold(U4) isa ParametricDFT.UnitaryManifold

        # Non-unitary matrix should be classified as PhaseManifold
        non_unitary = randn(ComplexF64, 2, 2)
        @test ParametricDFT.classify_manifold(non_unitary) isa ParametricDFT.PhaseManifold
    end

    @testset "group_by_manifold" begin
        Random.seed!(51)
        m, n = 2, 2
        _, tensors_raw = ParametricDFT.qft_code(m, n)
        # Convert to Vector{AbstractMatrix{ComplexF64}} for type compatibility
        tensors = AbstractMatrix{ComplexF64}[Matrix{ComplexF64}(t) for t in tensors_raw]
        groups = ParametricDFT.group_by_manifold(tensors)

        # Should have at most 2 groups
        @test length(groups) <= 2

        # All indices should be covered
        all_indices = Int[]
        for (_, indices) in groups
            append!(all_indices, indices)
        end
        sort!(all_indices)
        @test all_indices == collect(1:length(tensors))
    end

    @testset "stack/unstack round-trip generalized" begin
        Random.seed!(52)
        for d in [2, 3, 4]
            n = 5
            tensors = [randn(ComplexF64, d, d) for _ in 1:n]
            indices = collect(1:n)

            # stack_tensors should produce correct batched array
            batch = ParametricDFT.stack_tensors(tensors, indices)
            @test size(batch) == (d, d, n)
            for k in 1:n
                @test batch[:, :, k] ≈ tensors[k]
            end

            # Unstack back and verify round-trip
            tensors_out = Vector{Matrix{ComplexF64}}(undef, n)
            for k in 1:n
                tensors_out[k] = zeros(ComplexF64, d, d)
            end
            ParametricDFT.unstack_tensors!(tensors_out, batch, indices)
            for k in 1:n
                @test tensors_out[k] ≈ tensors[k]
            end

            # In-place stack_tensors! round-trip
            batch2 = similar(batch)
            ParametricDFT.stack_tensors!(batch2, tensors, indices)
            for k in 1:n
                @test batch2[:, :, k] ≈ tensors[k]
            end
        end
    end

end
