# ============================================================================
# Tests for Sparse Basis Training (training.jl)
# ============================================================================

@testset "train_basis" begin
    
    @testset "basic training" begin
        Random.seed!(42)
        
        # Create small dataset for quick testing
        m, n = 3, 3  # 8×8 images
        num_images = 5
        dataset = [rand(Float64, 8, 8) for _ in 1:num_images]
        
        # Train with minimal steps for testing
        basis = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            loss=ParametricDFT.L1Norm(),
            epochs=1,
            steps_per_image=5,
            validation_split=0.2,
            verbose=false
        )
        
        @test basis isa QFTBasis
        @test basis.m == m
        @test basis.n == n
        @test image_size(basis) == (8, 8)
    end
    
    @testset "training with MSELoss" begin
        Random.seed!(42)
        
        m, n = 3, 3
        dataset = [rand(Float64, 8, 8) for _ in 1:4]
        
        # Keep 50% of coefficients
        k = round(Int, 2^(m+n) * 0.5)
        
        basis = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            loss=ParametricDFT.MSELoss(k),
            epochs=1,
            steps_per_image=3,
            validation_split=0.25,
            verbose=false
        )
        
        @test basis isa QFTBasis
    end
    
    @testset "training with L2Norm" begin
        Random.seed!(42)
        
        m, n = 3, 3
        dataset = [rand(Float64, 8, 8) for _ in 1:4]
        
        basis = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            loss=ParametricDFT.L2Norm(),
            epochs=1,
            steps_per_image=3,
            validation_split=0.25,
            verbose=false
        )
        
        @test basis isa QFTBasis
    end
    
    @testset "training without shuffle" begin
        Random.seed!(42)
        
        m, n = 3, 3
        dataset = [rand(Float64, 8, 8) for _ in 1:4]
        
        basis = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            epochs=1,
            steps_per_image=2,
            shuffle=false,
            verbose=false
        )
        
        @test basis isa QFTBasis
    end
    
    @testset "validation split edge cases" begin
        Random.seed!(42)
        
        m, n = 3, 3
        dataset = [rand(Float64, 8, 8) for _ in 1:5]
        
        # Test with no validation (0.0 split)
        # Note: validation_split must be < 1.0, but 0.0 will still create at least 1 validation sample
        basis = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            epochs=1,
            steps_per_image=2,
            validation_split=0.1,
            verbose=false
        )
        @test basis isa QFTBasis
    end
    
    @testset "input validation" begin
        m, n = 3, 3
        
        # Test empty dataset
        @test_throws AssertionError train_basis(
            QFTBasis, Matrix{Float64}[];
            m=m, n=n,
            verbose=false
        )
        
        # Test wrong image size
        wrong_dataset = [rand(Float64, 16, 16)]  # 16×16 instead of 8×8
        @test_throws AssertionError train_basis(
            QFTBasis, wrong_dataset;
            m=m, n=n,
            verbose=false
        )
        
        # Test invalid validation_split
        dataset = [rand(Float64, 8, 8) for _ in 1:3]
        @test_throws AssertionError train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            validation_split=1.5,
            verbose=false
        )
    end
    
    @testset "early stopping" begin
        Random.seed!(42)
        
        m, n = 3, 3
        dataset = [rand(Float64, 8, 8) for _ in 1:6]
        
        # Train with early stopping enabled
        basis = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            epochs=10,  # Many epochs to trigger early stopping
            steps_per_image=2,
            early_stopping_patience=1,
            verbose=false
        )
        
        @test basis isa QFTBasis
    end
    
    @testset "trained basis produces valid transforms" begin
        Random.seed!(42)
        
        m, n = 3, 3
        dataset = [rand(Float64, 8, 8) for _ in 1:4]
        
        basis = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            epochs=1,
            steps_per_image=3,
            verbose=false
        )
        
        # Test that trained basis can transform images
        test_img = rand(8, 8)
        freq = forward_transform(basis, test_img)
        @test size(freq) == (8, 8)
        
        # Test round-trip
        recovered = inverse_transform(basis, freq)
        @test isapprox(real.(recovered), test_img, rtol=1e-10)
    end
    
    @testset "complex input images" begin
        Random.seed!(42)
        
        m, n = 3, 3
        # Dataset with complex images
        dataset = [rand(ComplexF64, 8, 8) for _ in 1:4]
        
        basis = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            epochs=1,
            steps_per_image=2,
            verbose=false
        )
        
        @test basis isa QFTBasis
    end
end

# ============================================================================
# Tests for EntangledQFTBasis Training
# ============================================================================

@testset "train_basis EntangledQFTBasis" begin
    
    @testset "basic entangled training" begin
        Random.seed!(42)
        
        m, n = 3, 3  # 8×8 images
        num_images = 5
        dataset = [rand(Float64, 8, 8) for _ in 1:num_images]
        
        # Train with minimal steps for testing
        basis = train_basis(
            EntangledQFTBasis, dataset;
            m=m, n=n,
            loss=ParametricDFT.MSELoss(10),
            epochs=1,
            steps_per_image=5,
            validation_split=0.2,
            verbose=false
        )
        
        @test basis isa EntangledQFTBasis
        @test basis.m == m
        @test basis.n == n
        @test basis.n_entangle == min(m, n)
        @test image_size(basis) == (8, 8)
    end
    
    @testset "entangled training with custom initial phases" begin
        Random.seed!(42)
        
        m, n = 3, 3
        initial_phases = [0.1, 0.2, 0.3]
        dataset = [rand(Float64, 8, 8) for _ in 1:4]
        
        basis = train_basis(
            EntangledQFTBasis, dataset;
            m=m, n=n,
            entangle_phases=initial_phases,
            epochs=1,
            steps_per_image=3,
            verbose=false
        )
        
        @test basis isa EntangledQFTBasis
        @test basis.n_entangle == 3
    end
    
    @testset "entangled training produces valid transforms" begin
        Random.seed!(42)
        
        m, n = 3, 3
        dataset = [rand(Float64, 8, 8) for _ in 1:4]
        
        basis = train_basis(
            EntangledQFTBasis, dataset;
            m=m, n=n,
            epochs=1,
            steps_per_image=3,
            verbose=false
        )
        
        # Test that trained basis can transform images
        test_img = rand(8, 8)
        freq = forward_transform(basis, test_img)
        @test size(freq) == (8, 8)
        
        # Test round-trip
        recovered = inverse_transform(basis, freq)
        @test isapprox(real.(recovered), test_img, rtol=1e-10)
    end
    
    @testset "entangled training input validation" begin
        m, n = 3, 3
        
        # Test empty dataset
        @test_throws AssertionError train_basis(
            EntangledQFTBasis, Matrix{Float64}[];
            m=m, n=n,
            verbose=false
        )
        
        # Test wrong image size
        wrong_dataset = [rand(Float64, 16, 16)]  # 16×16 instead of 8×8
        @test_throws AssertionError train_basis(
            EntangledQFTBasis, wrong_dataset;
            m=m, n=n,
            verbose=false
        )
    end
    
    @testset "entangled trained basis can be saved and loaded" begin
        Random.seed!(42)
        
        m, n = 3, 3
        dataset = [rand(Float64, 8, 8) for _ in 1:4]
        
        trained_basis = train_basis(
            EntangledQFTBasis, dataset;
            m=m, n=n,
            epochs=1,
            steps_per_image=2,
            verbose=false
        )
        
        # Save and load
        path = tempname() * ".json"
        save_basis(path, trained_basis)
        loaded = load_basis(path)
        
        @test loaded isa EntangledQFTBasis
        @test loaded.m == trained_basis.m
        @test loaded.n == trained_basis.n
        @test loaded.n_entangle == trained_basis.n_entangle
        
        # Cleanup
        rm(path)
    end
end

