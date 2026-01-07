# ============================================================================
# Tests for Sparse Basis (basis.jl)
# ============================================================================

@testset "AbstractSparseBasis and QFTBasis" begin
    
    @testset "QFTBasis construction" begin
        # Test basic construction
        basis = QFTBasis(4, 4)
        @test basis isa AbstractSparseBasis
        @test basis isa QFTBasis
        @test basis.m == 4
        @test basis.n == 4
        
        # Test different dimensions
        basis_rect = QFTBasis(3, 5)
        @test basis_rect.m == 3
        @test basis_rect.n == 5
    end
    
    @testset "image_size" begin
        basis = QFTBasis(4, 4)
        @test image_size(basis) == (16, 16)
        
        basis2 = QFTBasis(3, 5)
        @test image_size(basis2) == (8, 32)
        
        basis3 = QFTBasis(6, 6)
        @test image_size(basis3) == (64, 64)
    end
    
    @testset "num_parameters" begin
        basis = QFTBasis(4, 4)
        n_params = num_parameters(basis)
        @test n_params isa Int
        @test n_params > 0
        
        # More qubits should mean more parameters
        basis_small = QFTBasis(2, 2)
        basis_large = QFTBasis(5, 5)
        @test num_parameters(basis_small) < num_parameters(basis_large)
    end
    
    @testset "basis_hash" begin
        basis1 = QFTBasis(4, 4)
        basis2 = QFTBasis(4, 4)
        basis3 = QFTBasis(3, 3)
        
        # Hash should be a string
        h1 = basis_hash(basis1)
        @test h1 isa String
        @test length(h1) == 64  # SHA-256 produces 64 hex characters
        
        # Same basis should have same hash (deterministic)
        @test basis_hash(basis1) == basis_hash(basis1)
        
        # Different dimensions should have different hash
        @test basis_hash(basis1) != basis_hash(basis3)
    end
    
    @testset "forward_transform" begin
        Random.seed!(42)
        basis = QFTBasis(4, 4)
        
        # Test with real input
        img_real = rand(16, 16)
        freq = forward_transform(basis, img_real)
        @test size(freq) == (16, 16)
        @test eltype(freq) <: Complex
        
        # Test with complex input
        img_complex = rand(ComplexF64, 16, 16)
        freq_complex = forward_transform(basis, img_complex)
        @test size(freq_complex) == (16, 16)
        
        # Test dimension mismatch error
        img_wrong = rand(8, 8)
        @test_throws AssertionError forward_transform(basis, img_wrong)
    end
    
    @testset "inverse_transform" begin
        Random.seed!(42)
        basis = QFTBasis(4, 4)
        
        freq = rand(ComplexF64, 16, 16)
        img = inverse_transform(basis, freq)
        @test size(img) == (16, 16)
        @test eltype(img) <: Complex
        
        # Test dimension mismatch error
        freq_wrong = rand(ComplexF64, 8, 8)
        @test_throws AssertionError inverse_transform(basis, freq_wrong)
    end
    
    @testset "forward and inverse are inverses" begin
        Random.seed!(42)
        basis = QFTBasis(4, 4)
        
        # Test round-trip for real image
        img_original = rand(16, 16)
        freq = forward_transform(basis, img_original)
        img_recovered = inverse_transform(basis, freq)
        
        @test isapprox(real.(img_recovered), img_original, rtol=1e-10)
        
        # Test round-trip for complex image
        img_complex = rand(ComplexF64, 16, 16)
        freq_complex = forward_transform(basis, img_complex)
        img_complex_recovered = inverse_transform(basis, freq_complex)
        
        @test isapprox(img_complex_recovered, img_complex, rtol=1e-10)
    end
    
    @testset "norm preservation (unitarity)" begin
        Random.seed!(42)
        basis = QFTBasis(4, 4)
        
        img = rand(ComplexF64, 16, 16)
        freq = forward_transform(basis, img)
        
        # Unitary transforms preserve norm
        @test isapprox(norm(freq), norm(img), rtol=1e-10)
    end
    
    @testset "get_manifold" begin
        basis = QFTBasis(3, 3)
        M = get_manifold(basis)
        @test M isa ProductManifold
    end
    
    @testset "QFTBasis with custom tensors" begin
        # Get initial tensors
        m, n = 3, 3
        _, initial_tensors = ParametricDFT.qft_code(m, n)
        
        # Create basis with custom tensors
        basis = QFTBasis(m, n, initial_tensors)
        @test basis.m == m
        @test basis.n == n
        @test length(basis.tensors) == length(initial_tensors)
    end
    
    @testset "Base.show" begin
        basis = QFTBasis(4, 4)
        io = IOBuffer()
        show(io, basis)
        str = String(take!(io))
        @test occursin("QFTBasis", str)
        @test occursin("16×16", str)
    end
    
    @testset "Base.==" begin
        basis1 = QFTBasis(4, 4)
        basis2 = QFTBasis(4, 4)
        basis3 = QFTBasis(3, 3)
        
        @test basis1 == basis1
        @test basis1 == basis2
        @test !(basis1 == basis3)
    end
end

