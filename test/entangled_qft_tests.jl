# ============================================================================
# Tests for Entangled QFT (entangled_qft.jl)
# ============================================================================

@testset "Entangled QFT Circuit" begin
    
    @testset "entangled_qft_code basic construction" begin
        Random.seed!(1234)
        m, n = 3, 3
        
        optcode, tensors, n_entangle = ParametricDFT.entangled_qft_code(m, n)
        @test n_entangle == min(m, n)
        @test length(tensors) > 0
        
        pic = rand(ComplexF64, 2^m, 2^n)
        result = reshape(optcode(tensors..., reshape(pic, fill(2, m+n)...)), 2^m, 2^n)
        @test size(result) == (2^m, 2^n)
    end
    
    @testset "entangled_qft_code with custom phases" begin
        m, n = 3, 3
        phases = [0.1, 0.2, 0.3]
        optcode, tensors, n_entangle = ParametricDFT.entangled_qft_code(m, n; entangle_phases=phases)
        @test n_entangle == 3
        
        pic = rand(ComplexF64, 2^m, 2^n)
        result = reshape(optcode(tensors..., reshape(pic, fill(2, m+n)...)), 2^m, 2^n)
        @test size(result) == (2^m, 2^n)
    end
    
    @testset "zero phases equals standard QFT" begin
        Random.seed!(1234)
        m, n = 3, 3
        
        optcode_std, tensors_std = ParametricDFT.qft_code(m, n)
        optcode_ent, tensors_ent, _ = ParametricDFT.entangled_qft_code(m, n; entangle_phases=zeros(min(m, n)))
        
        pic = rand(ComplexF64, 2^m, 2^n)
        result_std = reshape(optcode_std(tensors_std..., reshape(pic, fill(2, m+n)...)), 2^m, 2^n)
        result_ent = reshape(optcode_ent(tensors_ent..., reshape(pic, fill(2, m+n)...)), 2^m, 2^n)
        
        @test size(result_std) == size(result_ent)
        @test isapprox(result_std, result_ent, rtol=1e-10)
    end
    
    @testset "rectangular images" begin
        m_rect, n_rect = 2, 4
        optcode, tensors, n_entangle = ParametricDFT.entangled_qft_code(m_rect, n_rect)
        @test n_entangle == min(m_rect, n_rect)
        
        pic = rand(ComplexF64, 2^m_rect, 2^n_rect)
        result = reshape(optcode(tensors..., reshape(pic, fill(2, m_rect+n_rect)...)), 2^m_rect, 2^n_rect)
        @test size(result) == (2^m_rect, 2^n_rect)
    end
    
    @testset "forward and inverse transforms" begin
        Random.seed!(1234)
        m, n = 3, 3
        phases = rand(min(m, n)) * 2π
        
        optcode, tensors, _ = ParametricDFT.entangled_qft_code(m, n; entangle_phases=phases)
        optcode_inv, _, _ = ParametricDFT.entangled_qft_code(m, n; entangle_phases=phases, inverse=true)
        
        pic = rand(ComplexF64, 2^m, 2^n)
        
        # Forward transform
        fft_result = reshape(optcode(tensors..., reshape(pic, fill(2, m+n)...)), 2^m, 2^n)
        
        # Inverse transform
        reconstructed = reshape(optcode_inv(conj.(tensors)..., reshape(fft_result, fill(2, m+n)...)), 2^m, 2^n)
        
        # Should recover original
        @test isapprox(reconstructed, pic, rtol=1e-10)
        
        # Norm preservation (unitarity)
        @test isapprox(norm(fft_result), norm(pic), rtol=1e-10)
    end
    
    @testset "entanglement_gate function" begin
        # Test zero phase
        gate_zero = ParametricDFT.entanglement_gate(0.0)
        @test gate_zero ≈ [1 0; 0 1]
        
        # Test π phase
        gate_pi = ParametricDFT.entanglement_gate(π)
        @test gate_pi[1,1] ≈ 1
        @test gate_pi[2,2] ≈ -1
        
        # Test arbitrary phase
        phi = 0.5
        gate = ParametricDFT.entanglement_gate(phi)
        @test gate[1,1] ≈ 1
        @test gate[2,2] ≈ exp(im * phi)
    end
    
    @testset "get_entangle_tensor_indices" begin
        m, n = 3, 3
        _, tensors, n_entangle = ParametricDFT.entangled_qft_code(m, n; entangle_phases=[0.1, 0.2, 0.3])
        
        indices = ParametricDFT.get_entangle_tensor_indices(tensors, n_entangle)
        @test length(indices) == n_entangle
    end
    
    @testset "extract_entangle_phases" begin
        m, n = 3, 3
        original_phases = [0.1, 0.2, 0.3]
        _, tensors, n_entangle = ParametricDFT.entangled_qft_code(m, n; entangle_phases=original_phases)
        
        indices = ParametricDFT.get_entangle_tensor_indices(tensors, n_entangle)
        extracted = ParametricDFT.extract_entangle_phases(tensors, indices)
        
        @test length(extracted) == n_entangle
        @test isapprox(extracted, original_phases, rtol=1e-10)
    end
end

@testset "EntangledQFTBasis" begin
    
    @testset "EntangledQFTBasis construction" begin
        # Test basic construction with default phases
        basis = EntangledQFTBasis(4, 4)
        @test basis isa AbstractSparseBasis
        @test basis isa EntangledQFTBasis
        @test basis.m == 4
        @test basis.n == 4
        @test basis.n_entangle == 4
        @test length(basis.entangle_phases) == 4
        @test all(basis.entangle_phases .== 0)
        
        # Test with custom phases
        phases = [0.1, 0.2, 0.3, 0.4]
        basis_custom = EntangledQFTBasis(4, 4; entangle_phases=phases)
        @test basis_custom.entangle_phases ≈ phases
        
        # Test rectangular
        basis_rect = EntangledQFTBasis(3, 5)
        @test basis_rect.m == 3
        @test basis_rect.n == 5
        @test basis_rect.n_entangle == 3  # min(3, 5)
    end
    
    @testset "image_size" begin
        basis = EntangledQFTBasis(4, 4)
        @test image_size(basis) == (16, 16)
        
        basis2 = EntangledQFTBasis(3, 5)
        @test image_size(basis2) == (8, 32)
    end
    
    @testset "num_parameters" begin
        basis = EntangledQFTBasis(4, 4)
        n_params = num_parameters(basis)
        @test n_params isa Int
        @test n_params > 0
        
        # Entangled basis should have more parameters than standard QFT
        basis_std = QFTBasis(4, 4)
        @test num_parameters(basis) >= num_parameters(basis_std)
    end
    
    @testset "num_entangle_parameters" begin
        basis = EntangledQFTBasis(4, 4)
        @test ParametricDFT.num_entangle_parameters(basis) == 4
        
        basis_rect = EntangledQFTBasis(3, 5)
        @test ParametricDFT.num_entangle_parameters(basis_rect) == 3
    end
    
    @testset "get_entangle_phases" begin
        phases = [0.5, 1.0, 1.5, 2.0]
        basis = EntangledQFTBasis(4, 4; entangle_phases=phases)
        retrieved = ParametricDFT.get_entangle_phases(basis)
        @test retrieved ≈ phases
        # Should return a copy
        retrieved[1] = 999.0
        @test basis.entangle_phases[1] ≈ 0.5
    end
    
    @testset "basis_hash" begin
        basis1 = EntangledQFTBasis(4, 4)
        basis2 = EntangledQFTBasis(4, 4)
        basis3 = EntangledQFTBasis(3, 3)
        
        h1 = basis_hash(basis1)
        @test h1 isa String
        @test length(h1) == 64
        
        @test basis_hash(basis1) == basis_hash(basis1)
        @test basis_hash(basis1) != basis_hash(basis3)
        
        # Different phases should produce different hash
        basis_diff = EntangledQFTBasis(4, 4; entangle_phases=[0.1, 0.2, 0.3, 0.4])
        @test basis_hash(basis1) != basis_hash(basis_diff)
    end
    
    @testset "forward_transform" begin
        Random.seed!(42)
        basis = EntangledQFTBasis(4, 4)
        
        img_real = rand(16, 16)
        freq = forward_transform(basis, img_real)
        @test size(freq) == (16, 16)
        @test eltype(freq) <: Complex
        
        img_complex = rand(ComplexF64, 16, 16)
        freq_complex = forward_transform(basis, img_complex)
        @test size(freq_complex) == (16, 16)
        
        img_wrong = rand(8, 8)
        @test_throws AssertionError forward_transform(basis, img_wrong)
    end
    
    @testset "inverse_transform" begin
        Random.seed!(42)
        basis = EntangledQFTBasis(4, 4)
        
        freq = rand(ComplexF64, 16, 16)
        img = inverse_transform(basis, freq)
        @test size(img) == (16, 16)
        @test eltype(img) <: Complex
        
        freq_wrong = rand(ComplexF64, 8, 8)
        @test_throws AssertionError inverse_transform(basis, freq_wrong)
    end
    
    @testset "forward and inverse are inverses" begin
        Random.seed!(42)
        
        # Test with default phases
        basis = EntangledQFTBasis(4, 4)
        img_original = rand(16, 16)
        freq = forward_transform(basis, img_original)
        img_recovered = inverse_transform(basis, freq)
        @test isapprox(real.(img_recovered), img_original, rtol=1e-10)
        
        # Test with custom phases
        phases = rand(4) * 2π
        basis_custom = EntangledQFTBasis(4, 4; entangle_phases=phases)
        img_complex = rand(ComplexF64, 16, 16)
        freq_complex = forward_transform(basis_custom, img_complex)
        img_complex_recovered = inverse_transform(basis_custom, freq_complex)
        @test isapprox(img_complex_recovered, img_complex, rtol=1e-10)
    end
    
    @testset "norm preservation (unitarity)" begin
        Random.seed!(42)
        phases = rand(4) * 2π
        basis = EntangledQFTBasis(4, 4; entangle_phases=phases)
        
        img = rand(ComplexF64, 16, 16)
        freq = forward_transform(basis, img)
        
        @test isapprox(norm(freq), norm(img), rtol=1e-10)
    end
    
    @testset "zero phases matches standard QFT" begin
        Random.seed!(42)
        basis_entangled = EntangledQFTBasis(4, 4; entangle_phases=zeros(4))
        basis_standard = QFTBasis(4, 4)
        
        img = rand(ComplexF64, 16, 16)
        freq_entangled = forward_transform(basis_entangled, img)
        freq_standard = forward_transform(basis_standard, img)
        
        # With zero entanglement phases, results should match
        @test isapprox(freq_entangled, freq_standard, rtol=1e-10)
    end
    
    @testset "get_manifold" begin
        basis = EntangledQFTBasis(3, 3)
        M = get_manifold(basis)
        @test M isa Manifolds.ProductManifold
    end
    
    @testset "Base.show" begin
        basis = EntangledQFTBasis(4, 4)
        io = IOBuffer()
        show(io, basis)
        str = String(take!(io))
        @test occursin("EntangledQFTBasis", str)
        @test occursin("16×16", str)
        @test occursin("entanglement", str)
    end
    
    @testset "Base.==" begin
        basis1 = EntangledQFTBasis(4, 4)
        basis2 = EntangledQFTBasis(4, 4)
        basis3 = EntangledQFTBasis(3, 3)
        
        @test basis1 == basis1
        @test basis1 == basis2
        @test !(basis1 == basis3)
        
        # Different phases
        basis4 = EntangledQFTBasis(4, 4; entangle_phases=[0.1, 0.2, 0.3, 0.4])
        @test !(basis1 == basis4)
    end
    
    @testset "rectangular images" begin
        # Test m != n
        basis = EntangledQFTBasis(2, 4)
        @test basis.n_entangle == 2  # min(2, 4)
        @test image_size(basis) == (4, 16)
        
        img = rand(4, 16)
        freq = forward_transform(basis, img)
        recovered = inverse_transform(basis, freq)
        @test isapprox(real.(recovered), img, rtol=1e-10)
    end
    
    @testset "small basis" begin
        # Test with single qubit per dimension
        basis = EntangledQFTBasis(1, 1)
        @test basis.n_entangle == 1
        @test image_size(basis) == (2, 2)
        
        img = rand(2, 2)
        freq = forward_transform(basis, img)
        recovered = inverse_transform(basis, freq)
        @test isapprox(real.(recovered), img, rtol=1e-10)
    end
    
    @testset "phase variations" begin
        m, n = 3, 3
        
        # Test with negative phases
        phases_neg = [-0.1, -0.2, -0.3]
        basis_neg = EntangledQFTBasis(m, n; entangle_phases=phases_neg)
        @test basis_neg.entangle_phases ≈ phases_neg
        
        img = rand(8, 8)
        freq = forward_transform(basis_neg, img)
        recovered = inverse_transform(basis_neg, freq)
        @test isapprox(real.(recovered), img, rtol=1e-10)
        
        # Test with large phases (> 2π)
        phases_large = [10.0, 20.0, 30.0]
        basis_large = EntangledQFTBasis(m, n; entangle_phases=phases_large)
        @test basis_large.entangle_phases ≈ phases_large
    end
    
    @testset "error handling" begin
        # Wrong image size for forward transform
        basis = EntangledQFTBasis(4, 4)
        wrong_img = rand(8, 8)
        @test_throws AssertionError forward_transform(basis, wrong_img)
        
        # Wrong size for inverse transform
        @test_throws AssertionError inverse_transform(basis, wrong_img)
        
        # Wrong phase length
        @test_throws AssertionError ParametricDFT.entangled_qft_code(4, 4; entangle_phases=[0.1, 0.2])
    end
end

@testset "Tensor Network Structure" begin
    
    @testset "standard QFT tensor structure" begin
        m, n = 3, 3
        optcode, tensors = ParametricDFT.qft_code(m, n)
        
        # Count tensor types
        H_mat = mat(H)
        hadamard_count = count(t -> t ≈ H_mat, tensors)
        phase_gate_count = count(t -> size(t) == (2, 2) && !(t ≈ H_mat), tensors)
        
        @test hadamard_count == m + n  # One Hadamard per qubit
        @test phase_gate_count > 0     # Should have phase gates
        @test length(tensors) == hadamard_count + phase_gate_count
    end
    
    @testset "entangled QFT has more tensors" begin
        m, n = 3, 3
        _, tensors_std = ParametricDFT.qft_code(m, n)
        _, tensors_ent, n_entangle = ParametricDFT.entangled_qft_code(m, n)
        
        # Entangled QFT should have extra tensors for entanglement gates
        @test length(tensors_ent) >= length(tensors_std)
        @test n_entangle == min(m, n)
    end
    
    @testset "tensor types in entangled QFT" begin
        m, n = 3, 3
        phases = [π/4, π/3, π/6]
        _, tensors, n_entangle = ParametricDFT.entangled_qft_code(m, n; entangle_phases=phases)
        
        H_mat = mat(H)
        hadamard_count = count(t -> t ≈ H_mat, tensors)
        
        # Should have m + n Hadamard gates
        @test hadamard_count == m + n
        
        # All tensors should be 2×2
        @test all(t -> size(t) == (2, 2), tensors)
    end
    
    @testset "entangle gate identification after training simulation" begin
        # Simulate training by slightly perturbing tensors
        m, n = 2, 2
        _, tensors, n_entangle = ParametricDFT.entangled_qft_code(m, n)
        
        # Find entangle indices in original tensors
        indices_before = ParametricDFT.get_entangle_tensor_indices(tensors, n_entangle)
        @test length(indices_before) == n_entangle
        
        # Simulate perturbation (small noise to tensor values, keeping structure)
        perturbed_tensors = copy(tensors)
        for idx in indices_before
            t = perturbed_tensors[idx]
            # Add small perturbation but keep the structure
            perturbed_tensors[idx] = t .* (1 .+ 0.01 * randn(ComplexF64, size(t)))
        end
        
        # Should still find the same indices (robust to small perturbations)
        indices_after = ParametricDFT.get_entangle_tensor_indices(perturbed_tensors, n_entangle)
        @test length(indices_after) == n_entangle
        @test indices_after == indices_before
    end
    
    @testset "phase extraction accuracy" begin
        m, n = 4, 4
        # Use phases in (-π, π] range since angle() returns values in this range
        test_phases = [0.0, π/2, π, -π/2]  # -π/2 is equivalent to 3π/2
        _, tensors, n_entangle = ParametricDFT.entangled_qft_code(m, n; entangle_phases=test_phases)
        
        indices = ParametricDFT.get_entangle_tensor_indices(tensors, n_entangle)
        extracted = ParametricDFT.extract_entangle_phases(tensors, indices)
        
        @test length(extracted) == length(test_phases)
        @test isapprox(extracted, test_phases, atol=1e-10)
        
        # Also test that phases differing by 2π are equivalent
        test_phases_wrapped = [0.0, π/2, π, 3π/2]  # 3π/2 should extract as -π/2
        _, tensors2, _ = ParametricDFT.entangled_qft_code(m, n; entangle_phases=test_phases_wrapped)
        indices2 = ParametricDFT.get_entangle_tensor_indices(tensors2, n_entangle)
        extracted2 = ParametricDFT.extract_entangle_phases(tensors2, indices2)
        
        # Check phase equivalence modulo 2π
        for (ext, orig) in zip(extracted2, test_phases_wrapped)
            # exp(i*ext) should equal exp(i*orig)
            @test isapprox(exp(im * ext), exp(im * orig), atol=1e-10)
        end
    end
    
    @testset "tensor count comparison" begin
        # Compare parameter counts across different sizes
        for (m, n) in [(2, 2), (3, 3), (4, 4), (2, 3)]
            _, tensors_std = ParametricDFT.qft_code(m, n)
            _, tensors_ent, n_ent = ParametricDFT.entangled_qft_code(m, n)
            
            params_std = sum(prod(size(t)) for t in tensors_std)
            params_ent = sum(prod(size(t)) for t in tensors_ent)
            
            # Entangled should have at least as many parameters
            @test params_ent >= params_std
            
            # Number of entanglement gates should be min(m, n)
            @test n_ent == min(m, n)
        end
    end
    
    @testset "einsum contraction code" begin
        m, n = 3, 3
        optcode_std, _ = ParametricDFT.qft_code(m, n)
        optcode_ent, _, _ = ParametricDFT.entangled_qft_code(m, n)
        
        # Both should be valid einsum codes
        @test optcode_std isa OMEinsum.AbstractEinsum
        @test optcode_ent isa OMEinsum.AbstractEinsum
    end
end
