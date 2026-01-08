# ============================================================================
# Tests for Basis Serialization (serialization.jl)
# ============================================================================

@testset "Basis Serialization" begin
    
    # Create temp directory for test files
    test_dir = mktempdir()
    
    @testset "save_basis and load_basis" begin
        basis = QFTBasis(4, 4)
        
        # Save basis
        path = joinpath(test_dir, "test_basis.json")
        returned_path = save_basis(path, basis)
        @test returned_path == path
        @test isfile(path)
        
        # Load basis
        loaded_basis = load_basis(path)
        @test loaded_basis isa QFTBasis
        @test loaded_basis.m == basis.m
        @test loaded_basis.n == basis.n
        @test length(loaded_basis.tensors) == length(basis.tensors)
        
        # Verify tensors are equal
        for (t1, t2) in zip(basis.tensors, loaded_basis.tensors)
            @test t1 ≈ t2
        end
        
        # Verify hash matches
        @test basis_hash(loaded_basis) == basis_hash(basis)
    end
    
    @testset "save_basis creates valid JSON" begin
        basis = QFTBasis(3, 3)
        path = joinpath(test_dir, "test_json_valid.json")
        save_basis(path, basis)
        
        # Read the JSON content
        json_str = read(path, String)
        @test !isempty(json_str)
        
        # Parse JSON manually to verify structure
        json_data = JSON3.read(json_str)
        @test json_data.type == "QFTBasis"
        @test json_data.version == "1.0"
        @test json_data.m == 3
        @test json_data.n == 3
        @test haskey(json_data, :tensors)
        @test haskey(json_data, :hash)
    end
    
    @testset "round-trip preserves functionality" begin
        Random.seed!(42)
        basis = QFTBasis(4, 4)
        
        # Save and reload
        path = joinpath(test_dir, "test_roundtrip.json")
        save_basis(path, basis)
        loaded_basis = load_basis(path)
        
        # Test that loaded basis produces same transforms
        img = rand(16, 16)
        freq_original = forward_transform(basis, img)
        freq_loaded = forward_transform(loaded_basis, img)
        
        @test freq_original ≈ freq_loaded
        
        # Test inverse transform
        recovered_original = inverse_transform(basis, freq_original)
        recovered_loaded = inverse_transform(loaded_basis, freq_loaded)
        
        @test recovered_original ≈ recovered_loaded
    end
    
    @testset "different basis sizes" begin
        for (m, n) in [(2, 2), (3, 4), (5, 3), (4, 4)]
            basis = QFTBasis(m, n)
            path = joinpath(test_dir, "test_size_$(m)_$(n).json")
            
            save_basis(path, basis)
            loaded = load_basis(path)
            
            @test loaded.m == m
            @test loaded.n == n
            @test image_size(loaded) == (2^m, 2^n)
        end
    end
    
    @testset "basis_to_dict and dict_to_basis" begin
        basis = QFTBasis(4, 4)
        
        # Convert to dict
        d = basis_to_dict(basis)
        @test d isa Dict
        @test d["type"] == "QFTBasis"
        @test d["version"] == "1.0"
        @test d["m"] == 4
        @test d["n"] == 4
        @test haskey(d, "tensors")
        @test haskey(d, "hash")
        
        # Convert back to basis
        loaded = dict_to_basis(d)
        @test loaded isa QFTBasis
        @test loaded.m == basis.m
        @test loaded.n == basis.n
        @test basis_hash(loaded) == basis_hash(basis)
    end
    
    @testset "save trained basis" begin
        Random.seed!(42)
        
        # Create a small trained basis
        m, n = 3, 3
        dataset = [rand(Float64, 8, 8) for _ in 1:3]
        trained_basis = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            epochs=1,
            steps_per_image=2,
            verbose=false
        )
        
        # Save and load
        path = joinpath(test_dir, "trained_basis.json")
        save_basis(path, trained_basis)
        loaded = load_basis(path)
        
        @test loaded isa QFTBasis
        @test basis_hash(loaded) == basis_hash(trained_basis)
        
        # Verify transforms still work
        img = rand(8, 8)
        freq_trained = forward_transform(trained_basis, img)
        freq_loaded = forward_transform(loaded, img)
        @test freq_trained ≈ freq_loaded
    end
    
    @testset "hash verification on load" begin
        basis = QFTBasis(3, 3)
        path = joinpath(test_dir, "test_hash.json")
        save_basis(path, basis)
        
        # Load normally - should not warn
        loaded = load_basis(path)
        @test basis_hash(loaded) == basis_hash(basis)
    end
    
    # Cleanup
    rm(test_dir, recursive=true)
end

