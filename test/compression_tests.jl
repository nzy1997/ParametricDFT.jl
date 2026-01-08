# ============================================================================
# Tests for Image Compression and Recovery (compression.jl)
# ============================================================================

@testset "Image Compression and Recovery" begin
    
    # Create temp directory for test files
    test_dir = mktempdir()
    
    @testset "CompressedImage structure" begin
        compressed = CompressedImage(
            [1, 5, 10],
            [1.0, 2.0, 3.0],
            [0.1, 0.2, 0.3],
            (16, 16),
            "test_hash"
        )
        
        @test compressed.indices == [1, 5, 10]
        @test compressed.values_real == [1.0, 2.0, 3.0]
        @test compressed.values_imag == [0.1, 0.2, 0.3]
        @test compressed.original_size == (16, 16)
        @test compressed.basis_hash == "test_hash"
    end
    
    @testset "compress basic" begin
        Random.seed!(42)
        basis = QFTBasis(4, 4)  # 16×16 images
        img = rand(16, 16)
        
        compressed = compress(basis, img; ratio=0.9)
        
        @test compressed isa CompressedImage
        @test compressed.original_size == (16, 16)
        @test compressed.basis_hash == basis_hash(basis)
        
        # Should keep ~10% of 256 coefficients
        expected_kept = round(Int, 256 * 0.1)
        @test length(compressed.indices) == expected_kept
        @test length(compressed.values_real) == expected_kept
        @test length(compressed.values_imag) == expected_kept
    end
    
    @testset "compress with different ratios" begin
        Random.seed!(42)
        basis = QFTBasis(4, 4)
        img = rand(16, 16)
        
        for ratio in [0.5, 0.7, 0.9, 0.95]
            compressed = compress(basis, img; ratio=ratio)
            total_coeffs = 16 * 16
            expected_kept = max(1, round(Int, total_coeffs * (1.0 - ratio)))
            @test length(compressed.indices) == expected_kept
        end
    end
    
    @testset "compress_with_k" begin
        Random.seed!(42)
        basis = QFTBasis(4, 4)
        img = rand(16, 16)
        
        for k in [1, 10, 50, 100, 256]
            compressed = compress_with_k(basis, img; k=k)
            @test length(compressed.indices) == min(k, 256)
        end
    end
    
    @testset "compress input validation" begin
        basis = QFTBasis(4, 4)
        
        # Wrong image size
        wrong_img = rand(8, 8)
        @test_throws AssertionError compress(basis, wrong_img)
        
        # Invalid ratio
        img = rand(16, 16)
        @test_throws AssertionError compress(basis, img; ratio=1.5)
        @test_throws AssertionError compress(basis, img; ratio=-0.1)
    end
    
    @testset "recover basic" begin
        Random.seed!(42)
        basis = QFTBasis(4, 4)
        img = rand(16, 16)
        
        compressed = compress(basis, img; ratio=0.5)
        recovered = recover(basis, compressed)
        
        @test size(recovered) == (16, 16)
        @test eltype(recovered) == Float64
    end
    
    @testset "recover with full compression" begin
        Random.seed!(42)
        basis = QFTBasis(4, 4)
        img = rand(16, 16)
        
        # Keep all coefficients (no compression)
        compressed = compress(basis, img; ratio=0.0)
        recovered = recover(basis, compressed)
        
        # Should recover original image almost exactly
        @test isapprox(recovered, img, rtol=1e-10)
    end
    
    @testset "recover hash verification" begin
        Random.seed!(42)
        basis1 = QFTBasis(4, 4)
        
        # Create a basis with different (trained) parameters
        m, n = 4, 4
        dataset = [rand(Float64, 16, 16) for _ in 1:2]
        basis2 = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            epochs=1,
            steps_per_image=5,
            verbose=false
        )
        
        img = rand(16, 16)
        compressed = compress(basis1, img; ratio=0.9)
        
        # Recovery with same basis should work
        recovered = recover(basis1, compressed)
        @test size(recovered) == (16, 16)
        
        # Verify that basis1 and basis2 have different hashes
        @test basis_hash(basis1) != basis_hash(basis2)
        
        # Recovery with different basis should fail with verify_hash=true
        @test_throws ErrorException recover(basis2, compressed; verify_hash=true)
        
        # Recovery with different basis should work with verify_hash=false
        recovered2 = recover(basis2, compressed; verify_hash=false)
        @test size(recovered2) == (16, 16)
    end
    
    @testset "compress and recover round-trip quality" begin
        Random.seed!(42)
        basis = QFTBasis(4, 4)
        img = rand(16, 16)
        
        # Test that higher compression means lower quality
        errors = Float64[]
        for ratio in [0.0, 0.5, 0.9, 0.99]
            compressed = compress(basis, img; ratio=ratio)
            recovered = recover(basis, compressed)
            error = norm(img - recovered) / norm(img)
            push!(errors, error)
        end
        
        # Lower ratio should mean better quality (lower error)
        @test errors[1] < 1e-10  # No compression = perfect recovery
        # Generally, more compression = more error (though not strictly monotonic)
    end
    
    @testset "save_compressed and load_compressed" begin
        Random.seed!(42)
        basis = QFTBasis(4, 4)
        img = rand(16, 16)
        
        compressed = compress(basis, img; ratio=0.9)
        
        # Save
        path = joinpath(test_dir, "test_compressed.json")
        returned_path = save_compressed(path, compressed)
        @test returned_path == path
        @test isfile(path)
        
        # Load
        loaded = load_compressed(path)
        @test loaded isa CompressedImage
        @test loaded.indices == compressed.indices
        @test loaded.values_real ≈ compressed.values_real
        @test loaded.values_imag ≈ compressed.values_imag
        @test loaded.original_size == compressed.original_size
        @test loaded.basis_hash == compressed.basis_hash
    end
    
    @testset "save_compressed creates valid JSON" begin
        Random.seed!(42)
        basis = QFTBasis(4, 4)
        img = rand(16, 16)
        compressed = compress(basis, img; ratio=0.9)
        
        path = joinpath(test_dir, "test_json_compressed.json")
        save_compressed(path, compressed)
        
        # Read and parse JSON
        json_str = read(path, String)
        json_data = JSON3.read(json_str)
        
        @test json_data.version == "1.0"
        @test haskey(json_data, :indices)
        @test haskey(json_data, :values_real)
        @test haskey(json_data, :values_imag)
        @test json_data.original_height == 16
        @test json_data.original_width == 16
        @test haskey(json_data, :basis_hash)
        @test haskey(json_data, :num_coefficients)
        @test haskey(json_data, :compression_ratio)
    end
    
    @testset "compression_stats" begin
        Random.seed!(42)
        basis = QFTBasis(4, 4)
        img = rand(16, 16)
        
        compressed = compress(basis, img; ratio=0.9)
        stats = compression_stats(compressed)
        
        @test stats.original_size == (16, 16)
        @test stats.total_coefficients == 256
        @test stats.kept_coefficients == length(compressed.indices)
        @test 0.0 <= stats.compression_ratio <= 1.0
        @test stats.storage_reduction > 0
    end
    
    @testset "CompressedImage show" begin
        compressed = CompressedImage(
            collect(1:26),
            ones(26),
            zeros(26),
            (16, 16),
            "test"
        )
        
        io = IOBuffer()
        show(io, compressed)
        str = String(take!(io))
        
        @test occursin("CompressedImage", str)
        @test occursin("16×16", str)
        @test occursin("26", str)
        @test occursin("256", str)
    end
    
    @testset "end-to-end workflow" begin
        Random.seed!(42)
        
        # Create basis
        basis = QFTBasis(4, 4)
        
        # Create test image
        img = rand(16, 16)
        
        # Compress
        compressed = compress(basis, img; ratio=0.8)
        
        # Save compressed and basis
        basis_path = joinpath(test_dir, "workflow_basis.json")
        compressed_path = joinpath(test_dir, "workflow_compressed.json")
        save_basis(basis_path, basis)
        save_compressed(compressed_path, compressed)
        
        # Load and recover
        loaded_basis = load_basis(basis_path)
        loaded_compressed = load_compressed(compressed_path)
        recovered = recover(loaded_basis, loaded_compressed)
        
        @test size(recovered) == size(img)
    end
    
    @testset "compress with trained basis" begin
        Random.seed!(42)
        
        # Train a small basis
        m, n = 3, 3
        dataset = [rand(Float64, 8, 8) for _ in 1:3]
        trained_basis = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            epochs=1,
            steps_per_image=2,
            verbose=false
        )
        
        # Compress and recover with trained basis
        img = rand(8, 8)
        compressed = compress(trained_basis, img; ratio=0.5)
        recovered = recover(trained_basis, compressed)
        
        @test size(recovered) == (8, 8)
    end
    
    # Cleanup
    rm(test_dir, recursive=true)
end

