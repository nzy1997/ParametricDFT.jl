# ============================================================================
# Tests for TEBD Circuit (tebd.jl)
# ============================================================================

using Yao: mat, H

@testset "TEBD Circuit" begin
    
    @testset "tebd_code basic construction" begin
        Random.seed!(1234)
        m, n = 3, 3  # 3 row qubits + 3 col qubits = 6 total
        total = m + n
        
        optcode, tensors, n_row, n_col = ParametricDFT.tebd_code(m, n)
        @test n_row == m  # 3 row ring gates
        @test n_col == n  # 3 col ring gates
        n_gates = n_row + n_col
        @test n_gates == 6  # Total 6 phase gates
        @test length(tensors) == total + n_gates  # 6 Hadamards + 6 phase gates
        @test length(tensors) > 0
        
        # Test that the circuit can be applied
        state = rand(ComplexF64, fill(2, total)...)
        result = optcode(tensors..., state)
        @test size(result) == size(state)
    end
    
    @testset "tebd_code with custom phases" begin
        m, n = 3, 3
        total = m + n
        n_gates = m + n  # 6 gates for ring topology
        phases = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        optcode, tensors, n_row, n_col = ParametricDFT.tebd_code(m, n; phases=phases)
        @test n_row + n_col == n_gates
        @test length(tensors) == total + n_gates
        
        state = rand(ComplexF64, fill(2, total)...)
        result = optcode(tensors..., state)
        @test size(result) == size(state)
    end
    
    @testset "2D ring topology gate count" begin
        # For m×n topology with rings:
        # - Row ring gates: m
        # - Col ring gates: n
        # - Total phase gates: m+n
        # - Hadamard gates: m+n
        for (m, n) in [(2, 2), (3, 3), (3, 4), (4, 3), (4, 4), (5, 3)]
            total = m + n
            _, tensors, n_row, n_col = ParametricDFT.tebd_code(m, n)
            @test n_row == m
            @test n_col == n
            n_gates = n_row + n_col
            @test n_gates == m + n
            @test length(tensors) == total + n_gates  # Hadamards + phase gates
        end
    end
    
    @testset "forward and inverse transforms" begin
        Random.seed!(1234)
        m, n = 3, 3
        total = m + n
        n_gates = m + n
        phases = rand(n_gates) * 2π
        
        optcode, tensors, _, _ = ParametricDFT.tebd_code(m, n; phases=phases)
        optcode_inv, tensors_inv, _, _ = ParametricDFT.tebd_code(m, n; phases=phases, inverse=true)
        
        state = rand(ComplexF64, fill(2, total)...)
        
        # Forward transform
        result = optcode(tensors..., state)
        
        # Inverse transform (conjugate tensors for inverse)
        reconstructed = optcode_inv(conj.(tensors)..., result)
        
        # Should recover original
        @test isapprox(reconstructed, state, rtol=1e-10)
    end
    
    @testset "norm preservation (unitarity)" begin
        Random.seed!(1234)
        m, n = 3, 4
        total = m + n
        n_gates = m + n
        phases = rand(n_gates) * 2π
        
        optcode, tensors, _, _ = ParametricDFT.tebd_code(m, n; phases=phases)
        
        state = rand(ComplexF64, fill(2, total)...)
        result = optcode(tensors..., state)
        
        @test isapprox(norm(result), norm(state), rtol=1e-10)
    end
    
    @testset "get_tebd_gate_indices" begin
        m, n = 3, 3
        n_gates = m + n
        phases = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        _, tensors, _, _ = ParametricDFT.tebd_code(m, n; phases=phases)
        
        indices = ParametricDFT.get_tebd_gate_indices(tensors, n_gates)
        @test length(indices) == n_gates
    end
    
    @testset "extract_tebd_phases" begin
        m, n = 3, 3
        n_gates = m + n
        original_phases = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        _, tensors, _, _ = ParametricDFT.tebd_code(m, n; phases=original_phases)
        
        indices = ParametricDFT.get_tebd_gate_indices(tensors, n_gates)
        extracted = ParametricDFT.extract_tebd_phases(tensors, indices)
        
        @test length(extracted) == n_gates
        @test isapprox(extracted, original_phases, rtol=1e-10)
    end
    
    @testset "error handling - wrong phase length" begin
        m, n = 3, 3
        wrong_phases = [0.1, 0.2]  # Should be length 6 for 3×3 ring topology
        @test_throws AssertionError ParametricDFT.tebd_code(m, n; phases=wrong_phases)
    end
end

@testset "TEBD Tensor Network Structure" begin
    
    @testset "all tensors are 2×2" begin
        m, n = 3, 4
        _, tensors, _, _ = ParametricDFT.tebd_code(m, n)
        
        @test all(t -> size(t) == (2, 2), tensors)
    end
    
    @testset "tensor count includes Hadamards and phase gates" begin
        # Total tensors = (m+n) Hadamards + (m+n) phase gates
        for (m, n) in [(2, 2), (3, 3), (3, 4), (4, 5), (5, 5)]
            total = m + n
            n_gates = m + n
            _, tensors, n_row, n_col = ParametricDFT.tebd_code(m, n)
            @test length(tensors) == total + n_gates
            @test n_row + n_col == n_gates
        end
    end
    
    @testset "Hadamard count" begin
        for (m, n) in [(2, 3), (3, 3), (4, 4)]
            total = m + n
            _, tensors, _, _ = ParametricDFT.tebd_code(m, n)
            n_hadamards = count(t -> t ≈ mat(H), tensors)
            @test n_hadamards == total  # One Hadamard per qubit
        end
    end
    
    @testset "phase extraction accuracy" begin
        m, n = 3, 3
        n_gates = m + n
        # Use phases in (-π, π] range since angle() returns values in this range
        test_phases = [0.0, π/2, π, -π/2, π/4, -π/4]
        _, tensors, _, _ = ParametricDFT.tebd_code(m, n; phases=test_phases)
        
        indices = ParametricDFT.get_tebd_gate_indices(tensors, n_gates)
        extracted = ParametricDFT.extract_tebd_phases(tensors, indices)
        
        @test length(extracted) == length(test_phases)
        @test isapprox(extracted, test_phases, atol=1e-10)
    end
    
    @testset "parameter count with Hadamards" begin
        # (m+n) Hadamards (4 params each) + (m+n) phase gates (4 params each)
        for (m, n) in [(2, 2), (3, 3), (4, 4)]
            total = m + n
            n_gates = m + n
            _, tensors, _, _ = ParametricDFT.tebd_code(m, n)
            total_params = sum(prod(size(t)) for t in tensors)
            @test total_params == (total + n_gates) * 4
        end
    end
    
    @testset "einsum contraction code validity" begin
        m, n = 3, 4
        optcode, _, _, _ = ParametricDFT.tebd_code(m, n)
        @test optcode isa OMEinsum.AbstractEinsum
    end
    
    @testset "gate identification after perturbation" begin
        # Simulate training by slightly perturbing tensors
        Random.seed!(1234)
        m, n = 3, 3
        n_gates = m + n
        _, tensors, _, _ = ParametricDFT.tebd_code(m, n)
        
        # Find gate indices in original tensors
        indices_before = ParametricDFT.get_tebd_gate_indices(tensors, n_gates)
        @test length(indices_before) == n_gates
        
        # Simulate perturbation (small noise to tensor values, keeping structure)
        perturbed_tensors = copy(tensors)
        for idx in indices_before
            t = perturbed_tensors[idx]
            perturbed_tensors[idx] = t .* (1 .+ 0.01 * randn(ComplexF64, size(t)))
        end
        
        # Should still find the same indices (robust to small perturbations)
        indices_after = ParametricDFT.get_tebd_gate_indices(perturbed_tensors, n_gates)
        @test length(indices_after) == n_gates
        @test indices_after == indices_before
    end
end

@testset "TEBD 2D Ring Connectivity" begin
    
    @testset "different phases produce different results" begin
        Random.seed!(1234)
        m, n = 3, 3
        n_gates = m + n
        
        phases1 = zeros(n_gates)
        phases2 = fill(π/4, n_gates)
        
        optcode1, tensors1, _, _ = ParametricDFT.tebd_code(m, n; phases=phases1)
        optcode2, tensors2, _, _ = ParametricDFT.tebd_code(m, n; phases=phases2)
        
        total = m + n
        state = rand(ComplexF64, fill(2, total)...)
        result1 = optcode1(tensors1..., state)
        result2 = optcode2(tensors2..., state)
        
        # Results should be different with different phases
        @test !isapprox(result1, result2, rtol=1e-5)
    end
    
    @testset "phase periodicity" begin
        Random.seed!(1234)
        m, n = 3, 3
        n_gates = m + n
        
        # Phases differing by 2π should give same result
        phases1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        phases2 = phases1 .+ 2π
        
        optcode1, tensors1, _, _ = ParametricDFT.tebd_code(m, n; phases=phases1)
        optcode2, tensors2, _, _ = ParametricDFT.tebd_code(m, n; phases=phases2)
        
        total = m + n
        state = rand(ComplexF64, fill(2, total)...)
        result1 = optcode1(tensors1..., state)
        result2 = optcode2(tensors2..., state)
        
        @test isapprox(result1, result2, rtol=1e-10)
    end
    
    @testset "small circuit (2×2)" begin
        m, n = 2, 2
        total = m + n
        n_gates = m + n  # 4 gates for ring topology
        phases = rand(n_gates) * 2π
        
        optcode, tensors, n_row, n_col = ParametricDFT.tebd_code(m, n; phases=phases)
        @test n_row == 2
        @test n_col == 2
        @test length(tensors) == total + n_gates  # 4 Hadamards + 4 phase gates
        
        state = rand(ComplexF64, fill(2, total)...)
        result = optcode(tensors..., state)
        @test size(result) == size(state)
        @test isapprox(norm(result), norm(state), rtol=1e-10)
    end
    
    @testset "asymmetric circuit (5×3)" begin
        m, n = 5, 3
        total = m + n
        n_gates = m + n  # 8 gates for ring topology
        phases = rand(n_gates) * 2π
        
        optcode, tensors, n_row, n_col = ParametricDFT.tebd_code(m, n; phases=phases)
        @test n_row == 5
        @test n_col == 3
        @test length(tensors) == total + n_gates  # 8 Hadamards + 8 phase gates
        
        state = rand(ComplexF64, fill(2, total)...)
        result = optcode(tensors..., state)
        @test size(result) == size(state)
        @test isapprox(norm(result), norm(state), rtol=1e-10)
    end
    
    @testset "large circuit (6×6)" begin
        m, n = 6, 6
        total = m + n
        n_gates = m + n  # 12 gates for ring topology
        phases = rand(n_gates) * 2π
        
        optcode, tensors, n_row, n_col = ParametricDFT.tebd_code(m, n; phases=phases)
        @test n_row == 6
        @test n_col == 6
        @test length(tensors) == total + n_gates  # 12 Hadamards + 12 phase gates
        
        state = rand(ComplexF64, fill(2, total)...)
        result = optcode(tensors..., state)
        @test size(result) == size(state)
        @test isapprox(norm(result), norm(state), rtol=1e-10)
    end
end

@testset "TEBDBasis" begin
    
    @testset "construction" begin
        basis = TEBDBasis(3, 3)
        @test basis.m == 3
        @test basis.n == 3
        @test basis.n_row_gates == 3
        @test basis.n_col_gates == 3
        @test length(basis.phases) == 6
    end
    
    @testset "image_size" begin
        basis = TEBDBasis(3, 4)
        @test image_size(basis) == (8, 16)  # 2^3 × 2^4
    end
    
    @testset "num_gates" begin
        basis = TEBDBasis(4, 5)
        @test num_gates(basis) == 9  # 4 + 5 = 9 for ring topology
    end
    
    @testset "forward and inverse transform" begin
        Random.seed!(1234)
        basis = TEBDBasis(3, 3)
        image = rand(8, 8)  # 2^3 × 2^3
        
        freq = forward_transform(basis, image)
        @test size(freq) == size(image)
        
        recovered = inverse_transform(basis, freq)
        @test isapprox(real.(recovered), image, rtol=1e-10)
    end
end
