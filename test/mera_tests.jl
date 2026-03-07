# ============================================================================
# Tests for MERA Circuit (mera.jl)
# ============================================================================

using Yao: mat, H

@testset "MERA Circuit" begin

    @testset "mera_code basic construction" begin
        Random.seed!(42)
        m, n = 2, 2
        total = m + n

        optcode, tensors, n_row, n_col = ParametricDFT.mera_code(m, n)
        @test n_row == 2  # 2*(2-1) = 2
        @test n_col == 2  # 2*(2-1) = 2
        n_gates = n_row + n_col
        @test n_gates == 4
        @test length(tensors) == total + n_gates  # 4 Hadamards + 4 phase gates

        # Test that the circuit can be applied
        state = rand(ComplexF64, fill(2, total)...)
        result = optcode(tensors..., state)
        @test size(result) == size(state)
    end

    @testset "mera_code with custom phases" begin
        Random.seed!(42)
        m, n = 2, 2
        total = m + n
        n_gates = 2 + 2  # 2*(2-1) + 2*(2-1)
        phases = [0.1, 0.2, 0.3, 0.4]

        optcode, tensors, n_row, n_col = ParametricDFT.mera_code(m, n; phases=phases)
        @test n_row + n_col == n_gates
        @test length(tensors) == total + n_gates

        state = rand(ComplexF64, fill(2, total)...)
        result = optcode(tensors..., state)
        @test size(result) == size(state)
    end

    @testset "gate count verification" begin
        # For MERA: n_row_gates = 2*(m-1), n_col_gates = 2*(n-1)
        for (m, n) in [(2, 2), (4, 4), (2, 4), (4, 2)]
            total = m + n
            _, tensors, n_row, n_col = ParametricDFT.mera_code(m, n)
            @test n_row == 2 * (m - 1)
            @test n_col == 2 * (n - 1)
            n_gates = n_row + n_col
            @test length(tensors) == total + n_gates
        end
    end

    @testset "forward and inverse transforms" begin
        Random.seed!(42)
        m, n = 2, 2
        total = m + n
        n_gates = 2 * (m - 1) + 2 * (n - 1)
        phases = rand(n_gates) * 2π

        optcode, tensors, _, _ = ParametricDFT.mera_code(m, n; phases=phases)
        optcode_inv, tensors_inv, _, _ = ParametricDFT.mera_code(m, n; phases=phases, inverse=true)

        state = rand(ComplexF64, fill(2, total)...)

        # Forward transform
        result = optcode(tensors..., state)

        # Inverse transform (conjugate tensors for inverse)
        reconstructed = optcode_inv(conj.(tensors)..., result)

        # Should recover original
        @test isapprox(reconstructed, state, rtol=1e-10)
    end

    @testset "norm preservation" begin
        Random.seed!(42)
        m, n = 4, 2
        total = m + n
        n_gates = 2 * (m - 1) + 2 * (n - 1)
        phases = rand(n_gates) * 2π

        optcode, tensors, _, _ = ParametricDFT.mera_code(m, n; phases=phases)

        state = rand(ComplexF64, fill(2, total)...)
        result = optcode(tensors..., state)

        @test isapprox(norm(result), norm(state), rtol=1e-10)
    end

    @testset "get_mera_gate_indices" begin
        m, n = 2, 2
        n_gates = 2 * (m - 1) + 2 * (n - 1)
        phases = [0.1, 0.2, 0.3, 0.4]
        _, tensors, _, _ = ParametricDFT.mera_code(m, n; phases=phases)

        indices = ParametricDFT.get_mera_gate_indices(tensors, n_gates)
        @test length(indices) == n_gates
    end

    @testset "extract_mera_phases" begin
        m, n = 2, 2
        n_gates = 2 * (m - 1) + 2 * (n - 1)
        original_phases = [0.1, 0.2, 0.3, 0.4]
        _, tensors, _, _ = ParametricDFT.mera_code(m, n; phases=original_phases)

        indices = ParametricDFT.get_mera_gate_indices(tensors, n_gates)
        extracted = ParametricDFT.extract_mera_phases(tensors, indices)

        @test length(extracted) == n_gates
        @test isapprox(extracted, original_phases, rtol=1e-10)
    end

    @testset "error handling" begin
        # Wrong phase length
        wrong_phases = [0.1, 0.2]  # Should be length 4 for 2×2 MERA
        @test_throws AssertionError ParametricDFT.mera_code(2, 2; phases=wrong_phases)

        # Non-power-of-2
        @test_throws AssertionError ParametricDFT.mera_code(3, 3)
    end

    @testset "all tensors are 2x2" begin
        m, n = 4, 2
        _, tensors, _, _ = ParametricDFT.mera_code(m, n)

        @test all(t -> size(t) == (2, 2), tensors)
    end

    @testset "Hadamard count" begin
        for (m, n) in [(2, 2), (4, 4), (2, 4)]
            total = m + n
            _, tensors, _, _ = ParametricDFT.mera_code(m, n)
            n_hadamards = count(t -> t ≈ mat(H), tensors)
            @test n_hadamards == total  # One Hadamard per qubit
        end
    end

    @testset "different phases produce different results" begin
        Random.seed!(42)
        m, n = 2, 2
        total = m + n
        n_gates = 2 * (m - 1) + 2 * (n - 1)

        phases1 = zeros(n_gates)
        phases2 = fill(π / 4, n_gates)

        optcode1, tensors1, _, _ = ParametricDFT.mera_code(m, n; phases=phases1)
        optcode2, tensors2, _, _ = ParametricDFT.mera_code(m, n; phases=phases2)

        state = rand(ComplexF64, fill(2, total)...)
        result1 = optcode1(tensors1..., state)
        result2 = optcode2(tensors2..., state)

        @test !isapprox(result1, result2, rtol=1e-5)
    end

    @testset "minimum size m=1, n=1" begin
        m, n = 1, 1
        total = m + n

        optcode, tensors, n_row, n_col = ParametricDFT.mera_code(m, n)
        @test n_row == 0
        @test n_col == 0
        # Only Hadamards, no phase gates
        @test length(tensors) == total

        state = rand(ComplexF64, fill(2, total)...)
        result = optcode(tensors..., state)
        @test size(result) == size(state)
    end

    @testset "larger circuit 4x4" begin
        Random.seed!(42)
        m, n = 4, 4
        total = m + n
        n_row_gates = 2 * (m - 1)  # 6
        n_col_gates = 2 * (n - 1)  # 6
        n_gates = n_row_gates + n_col_gates  # 12

        @test n_gates == 12

        phases = rand(n_gates) * 2π
        optcode, tensors, n_row, n_col = ParametricDFT.mera_code(m, n; phases=phases)
        @test n_row == 6
        @test n_col == 6
        @test length(tensors) == total + n_gates

        # Roundtrip test
        optcode_inv, tensors_inv, _, _ = ParametricDFT.mera_code(m, n; phases=phases, inverse=true)
        state = rand(ComplexF64, fill(2, total)...)
        result = optcode(tensors..., state)
        reconstructed = optcode_inv(conj.(tensors)..., result)
        @test isapprox(reconstructed, state, rtol=1e-10)
    end

end
