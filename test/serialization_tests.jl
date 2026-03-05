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
        trained_basis, _ = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            epochs=1,
            steps_per_image=2
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
    
    # ============================================================================
    # EntangledQFTBasis Serialization Tests
    # ============================================================================
    
    @testset "EntangledQFTBasis save and load" begin
        phases = [0.1, 0.2, 0.3, 0.4]
        basis = EntangledQFTBasis(4, 4; entangle_phases=phases)
        
        # Save basis
        path = joinpath(test_dir, "test_entangled_basis.json")
        returned_path = save_basis(path, basis)
        @test returned_path == path
        @test isfile(path)
        
        # Load basis
        loaded_basis = load_basis(path)
        @test loaded_basis isa EntangledQFTBasis
        @test loaded_basis.m == basis.m
        @test loaded_basis.n == basis.n
        @test loaded_basis.n_entangle == basis.n_entangle
        @test loaded_basis.entangle_phases ≈ basis.entangle_phases
        
        # Verify tensors are equal
        for (t1, t2) in zip(basis.tensors, loaded_basis.tensors)
            @test t1 ≈ t2
        end
        
        # Verify hash matches
        @test basis_hash(loaded_basis) == basis_hash(basis)
    end
    
    @testset "EntangledQFTBasis JSON structure" begin
        phases = [0.5, 1.0]
        basis = EntangledQFTBasis(2, 2; entangle_phases=phases)
        path = joinpath(test_dir, "test_entangled_json.json")
        save_basis(path, basis)

        # Read and verify JSON structure
        json_str = read(path, String)
        json_data = JSON3.read(json_str)
        @test json_data.type == "EntangledQFTBasis"
        @test json_data.version == "1.1"
        @test json_data.m == 2
        @test json_data.n == 2
        @test json_data.n_entangle == 2
        @test collect(json_data.entangle_phases) ≈ phases
        @test json_data.entangle_position == "back"
        @test haskey(json_data, :tensors)
        @test haskey(json_data, :hash)
    end
    
    @testset "EntangledQFTBasis round-trip preserves functionality" begin
        Random.seed!(42)
        phases = rand(4) * 2π
        basis = EntangledQFTBasis(4, 4; entangle_phases=phases)
        
        # Save and reload
        path = joinpath(test_dir, "test_entangled_roundtrip.json")
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
    
    @testset "EntangledQFTBasis basis_to_dict and dict_to_basis" begin
        phases = [0.1, 0.2, 0.3]
        basis = EntangledQFTBasis(3, 3; entangle_phases=phases)

        # Convert to dict
        d = basis_to_dict(basis)
        @test d isa Dict
        @test d["type"] == "EntangledQFTBasis"
        @test d["version"] == "1.1"
        @test d["m"] == 3
        @test d["n"] == 3
        @test d["n_entangle"] == 3
        @test d["entangle_phases"] ≈ phases
        @test d["entangle_position"] == "back"

        # Convert back to basis
        loaded = dict_to_basis(d)
        @test loaded isa EntangledQFTBasis
        @test loaded.m == basis.m
        @test loaded.n == basis.n
        @test loaded.entangle_phases ≈ basis.entangle_phases
        @test loaded.entangle_position == :back
        @test basis_hash(loaded) == basis_hash(basis)
    end
    
    @testset "EntangledQFTBasis entangle_position serialization" begin
        for pos in [:front, :middle, :back]
            phases = [0.1, 0.2, 0.3]
            basis = EntangledQFTBasis(3, 3; entangle_phases=phases, entangle_position=pos)

            # Save and load
            path = joinpath(test_dir, "test_entangled_pos_$(pos).json")
            save_basis(path, basis)
            loaded_basis = load_basis(path)

            @test loaded_basis isa EntangledQFTBasis
            @test loaded_basis.entangle_position == pos
            @test loaded_basis.entangle_phases ≈ basis.entangle_phases
            @test basis_hash(loaded_basis) == basis_hash(basis)

            # Verify transforms work
            img = rand(8, 8)
            freq_original = forward_transform(basis, img)
            freq_loaded = forward_transform(loaded_basis, img)
            @test freq_original ≈ freq_loaded
        end
    end

    @testset "EntangledQFTBasis entangle_position in JSON" begin
        basis = EntangledQFTBasis(3, 3; entangle_position=:front)
        path = joinpath(test_dir, "test_entangled_pos_json.json")
        save_basis(path, basis)

        json_str = read(path, String)
        json_data = JSON3.read(json_str)
        @test json_data.entangle_position == "front"
    end

    @testset "EntangledQFTBasis entangle_position dict roundtrip" begin
        for pos in [:front, :middle, :back]
            basis = EntangledQFTBasis(3, 3; entangle_position=pos)
            d = basis_to_dict(basis)
            @test d["entangle_position"] == string(pos)

            loaded = dict_to_basis(d)
            @test loaded.entangle_position == pos
            @test basis_hash(loaded) == basis_hash(basis)
        end
    end

    @testset "EntangledQFTBasis backward compat (missing entangle_position)" begin
        # Simulate an old-format dict without entangle_position
        basis = EntangledQFTBasis(3, 3)
        d = basis_to_dict(basis)
        delete!(d, "entangle_position")

        loaded = dict_to_basis(d)
        @test loaded isa EntangledQFTBasis
        @test loaded.entangle_position == :back  # Default
    end

    @testset "EntangledQFTBasis with different sizes" begin
        for (m, n) in [(2, 2), (3, 4), (4, 3)]
            n_entangle = min(m, n)
            phases = rand(n_entangle)
            basis = EntangledQFTBasis(m, n; entangle_phases=phases)
            path = joinpath(test_dir, "test_entangled_size_$(m)_$(n).json")
            
            save_basis(path, basis)
            loaded = load_basis(path)
            
            @test loaded.m == m
            @test loaded.n == n
            @test loaded.n_entangle == n_entangle
            @test image_size(loaded) == (2^m, 2^n)
        end
    end
    
    # ============================================================================
    # TEBDBasis Serialization Tests
    # ============================================================================

    @testset "TEBDBasis save and load" begin
        phases = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        basis = TEBDBasis(3, 3; phases=phases)

        path = joinpath(test_dir, "test_tebd_basis.json")
        returned_path = save_basis(path, basis)
        @test returned_path == path
        @test isfile(path)

        loaded_basis = load_basis(path)
        @test loaded_basis isa TEBDBasis
        @test loaded_basis.m == basis.m
        @test loaded_basis.n == basis.n
        @test loaded_basis.n_row_gates == basis.n_row_gates
        @test loaded_basis.n_col_gates == basis.n_col_gates
        @test loaded_basis.phases ≈ basis.phases

        for (t1, t2) in zip(basis.tensors, loaded_basis.tensors)
            @test t1 ≈ t2
        end

        @test basis_hash(loaded_basis) == basis_hash(basis)
    end

    @testset "TEBDBasis JSON structure" begin
        phases = [0.1, 0.2, 0.3, 0.4]
        basis = TEBDBasis(2, 2; phases=phases)
        path = joinpath(test_dir, "test_tebd_json.json")
        save_basis(path, basis)

        json_str = read(path, String)
        json_data = JSON3.read(json_str)
        @test json_data.type == "TEBDBasis"
        @test json_data.version == "2.0"
        @test json_data.m == 2
        @test json_data.n == 2
        @test json_data.n_row_gates == 2
        @test json_data.n_col_gates == 2
        @test collect(Float64, json_data.phases) ≈ phases
        @test haskey(json_data, :tensors)
        @test haskey(json_data, :hash)
    end

    @testset "TEBDBasis round-trip preserves functionality" begin
        Random.seed!(42)
        phases = rand(6) .* 2π
        basis = TEBDBasis(3, 3; phases=phases)

        path = joinpath(test_dir, "test_tebd_roundtrip.json")
        save_basis(path, basis)
        loaded_basis = load_basis(path)

        img = rand(8, 8)
        freq_original = forward_transform(basis, img)
        freq_loaded = forward_transform(loaded_basis, img)
        @test freq_original ≈ freq_loaded

        recovered_original = inverse_transform(basis, freq_original)
        recovered_loaded = inverse_transform(loaded_basis, freq_loaded)
        @test recovered_original ≈ recovered_loaded
    end

    @testset "TEBDBasis basis_to_dict and dict_to_basis" begin
        phases = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        basis = TEBDBasis(3, 3; phases=phases)

        d = basis_to_dict(basis)
        @test d isa Dict
        @test d["type"] == "TEBDBasis"
        @test d["version"] == "2.0"
        @test d["m"] == 3
        @test d["n"] == 3
        @test d["n_row_gates"] == 3
        @test d["n_col_gates"] == 3
        @test d["phases"] ≈ phases

        loaded = dict_to_basis(d)
        @test loaded isa TEBDBasis
        @test loaded.m == basis.m
        @test loaded.n == basis.n
        @test loaded.phases ≈ basis.phases
        @test basis_hash(loaded) == basis_hash(basis)
    end

    @testset "TEBDBasis different sizes" begin
        for (m, n) in [(2, 2), (3, 4), (4, 3)]
            n_gates = m + n
            phases = rand(n_gates)
            basis = TEBDBasis(m, n; phases=phases)
            path = joinpath(test_dir, "test_tebd_size_$(m)_$(n).json")

            save_basis(path, basis)
            loaded = load_basis(path)

            @test loaded.m == m
            @test loaded.n == n
            @test loaded.n_row_gates == m
            @test loaded.n_col_gates == n
            @test image_size(loaded) == (2^m, 2^n)
        end
    end

    # Cleanup
    rm(test_dir, recursive=true)
end

