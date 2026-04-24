# ============================================================================
# Tests for Sparse Basis Training (training.jl)
# ============================================================================

# ============================================================================
# Dispatch interface: _init_circuit, _build_basis, _basis_name
# ============================================================================

@testset "_init_circuit and _build_basis dispatch" begin
    for (BasisType, m, n, kwargs) in [
        (QFTBasis, 3, 3, NamedTuple()),
        (EntangledQFTBasis, 3, 3, NamedTuple()),
        (TEBDBasis, 2, 2, (phases=randn(4) * 0.1,)),
        (MERABasis, 2, 2, (phases=randn(4) * 0.1,)),
    ]
        @testset "$BasisType" begin
            optcode, inverse_code, tensors = ParametricDFT._init_circuit(BasisType, m, n; kwargs...)
            @test optcode isa OMEinsum.AbstractEinsum
            @test inverse_code isa OMEinsum.AbstractEinsum
            @test !isempty(tensors)

            basis = ParametricDFT._build_basis(BasisType, m, n, tensors, optcode, inverse_code; kwargs...)
            @test basis isa BasisType
            @test basis.m == m
            @test basis.n == n

            @test ParametricDFT._basis_name(BasisType) isa String
        end
    end
end

# ============================================================================
# Common training: basic smoke test for all basis types
# ============================================================================

TRAINING_CONFIGS = [
    (
        type = QFTBasis,
        m = 3, n = 3,
        extra_kwargs = (loss=ParametricDFT.L1Norm(),),
        extra_checks = (basis, m, n) -> begin
            @test image_size(basis) == (2^m, 2^n)
        end,
    ),
    (
        type = EntangledQFTBasis,
        m = 3, n = 3,
        extra_kwargs = (loss=ParametricDFT.MSELoss(10),),
        extra_checks = (basis, m, n) -> begin
            @test basis.n_entangle == min(m, n)
            @test image_size(basis) == (2^m, 2^n)
        end,
    ),
    (
        type = TEBDBasis,
        m = 2, n = 2,
        extra_kwargs = (loss=ParametricDFT.L1Norm(),),
        extra_checks = (basis, m, n) -> begin
            @test length(basis.phases) == m + n
        end,
    ),
]

for cfg in TRAINING_CONFIGS
    type_name = string(cfg.type)
    m, n = cfg.m, cfg.n
    img_size = (2^m, 2^n)

    @testset "train_basis $type_name" begin
        @testset "basic training" begin
            Random.seed!(42)

            dataset = [rand(Float64, img_size...) for _ in 1:5]

            basis, _ = train_basis(
                cfg.type, dataset;
                m=m, n=n,
                cfg.extra_kwargs...,
                epochs=1,
                steps_per_image=5,
                validation_split=0.2,
            )

            @test basis isa cfg.type
            @test basis.m == m
            @test basis.n == n
            cfg.extra_checks(basis, m, n)
        end
    end
end

# ============================================================================
# QFTBasis training pipeline features (shared _train_basis_core code path)
# ============================================================================

@testset "train_basis pipeline" begin

    @testset "training with MSELoss" begin
        Random.seed!(42)

        m, n = 3, 3
        dataset = [rand(Float64, 8, 8) for _ in 1:4]
        k = round(Int, 2^(m+n) * 0.5)

        basis, _ = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            loss=ParametricDFT.MSELoss(k),
            epochs=1,
            steps_per_image=3,
            validation_split=0.25,
        )

        @test basis isa QFTBasis
        @test basis.m == m
        @test basis.n == n
    end

    @testset "training without shuffle" begin
        Random.seed!(42)

        m, n = 3, 3
        dataset = [rand(Float64, 8, 8) for _ in 1:4]

        basis, _ = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            epochs=1,
            steps_per_image=2,
            shuffle=false,
        )

        @test basis isa QFTBasis
        @test basis.m == m
        @test basis.n == n
    end

    @testset "validation split edge cases" begin
        Random.seed!(42)

        m, n = 3, 3
        dataset = [rand(Float64, 8, 8) for _ in 1:5]

        basis, _ = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            epochs=1,
            steps_per_image=2,
            validation_split=0.1,
        )
        @test basis isa QFTBasis
        @test basis.m == m
        @test basis.n == n
    end

    @testset "single image dataset" begin
        Random.seed!(42)

        m, n = 2, 2
        dataset = [rand(Float64, 4, 4)]

        basis, _ = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            epochs=1,
            steps_per_image=2,
        )

        @test basis isa QFTBasis
        @test basis.m == m
        @test basis.n == n
    end

    @testset "input validation" begin
        m, n = 3, 3

        @test_throws AssertionError train_basis(
            QFTBasis, Matrix{Float64}[];
            m=m, n=n,
        )

        wrong_dataset = [rand(Float64, 16, 16)]
        @test_throws AssertionError train_basis(
            QFTBasis, wrong_dataset;
            m=m, n=n,
        )

        dataset = [rand(Float64, 8, 8) for _ in 1:3]
        @test_throws AssertionError train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            validation_split=1.5,
        )
    end

    @testset "early stopping" begin
        Random.seed!(42)

        m, n = 3, 3
        dataset = [rand(Float64, 8, 8) for _ in 1:6]

        basis, _ = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            epochs=10,
            steps_per_image=2,
            early_stopping_patience=1,
        )

        @test basis isa QFTBasis
        @test basis.m == m
        @test basis.n == n
    end

    @testset "trained basis produces valid transforms" begin
        Random.seed!(42)

        m, n = 3, 3
        dataset = [rand(Float64, 8, 8) for _ in 1:4]

        basis, _ = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            epochs=1,
            steps_per_image=3,
        )

        test_img = rand(8, 8)
        freq = forward_transform(basis, test_img)
        @test size(freq) == (8, 8)

        recovered = inverse_transform(basis, freq)
        @test isapprox(real.(recovered), test_img, rtol=1e-10)
    end

    @testset "complex input images" begin
        Random.seed!(42)

        m, n = 3, 3
        dataset = [rand(ComplexF64, 8, 8) for _ in 1:4]

        basis, _ = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            epochs=1,
            steps_per_image=2,
        )

        @test basis isa QFTBasis
        @test basis.m == m
        @test basis.n == n
    end
end

# ============================================================================
# Type-specific training kwargs
# ============================================================================

@testset "type-specific training" begin

    @testset "EntangledQFTBasis custom initial phases" begin
        Random.seed!(42)

        m, n = 3, 3
        initial_phases = [0.1, 0.2, 0.3]
        dataset = [rand(Float64, 8, 8) for _ in 1:4]

        basis, _ = train_basis(
            EntangledQFTBasis, dataset;
            m=m, n=n,
            entangle_phases=initial_phases,
            epochs=1,
            steps_per_image=3,
        )

        @test basis isa EntangledQFTBasis
        @test basis.n_entangle == 3
    end

    @testset "TEBDBasis custom phases" begin
        Random.seed!(42)

        m, n = 2, 2
        initial_phases = [0.1, 0.2, 0.3, 0.4]
        dataset = [rand(Float64, 4, 4) for _ in 1:4]

        basis, _ = train_basis(
            TEBDBasis, dataset;
            m=m, n=n,
            phases=initial_phases,
            epochs=1,
            steps_per_image=3,
        )

        @test basis isa TEBDBasis
        @test length(basis.phases) == 4
    end
end

# ============================================================================
# Batched training
# ============================================================================

@testset "batched training" begin

    @testset "adam with batch_size > 1 uses batched einsum" begin
        Random.seed!(42)
        m, n = 2, 2
        dataset = [rand(Float64, 4, 4) for _ in 1:6]

        basis, history = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            loss=ParametricDFT.L1Norm(),
            epochs=3,
            steps_per_image=5,
            batch_size=3,
            optimizer=:adam,
        )

        @test basis isa QFTBasis
        @test basis.m == m
        @test basis.n == n
        initial_loss = history.step_train_losses[1]
        final_loss = history.step_train_losses[end]
        @test final_loss < initial_loss
    end

    @testset "gradient_descent with batch_size > 1" begin
        Random.seed!(42)
        m, n = 2, 2
        dataset = [rand(Float64, 4, 4) for _ in 1:6]

        basis, history = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            loss=ParametricDFT.L1Norm(),
            epochs=3,
            steps_per_image=5,
            batch_size=3,
            optimizer=:gradient_descent,
        )

        @test basis isa QFTBasis
        initial_loss = history.step_train_losses[1]
        final_loss = history.step_train_losses[end]
        @test final_loss < initial_loss
    end

    @testset "post-training roundtrip (unitarity preserved)" begin
        Random.seed!(42)
        m, n = 2, 2
        dataset = [rand(Float64, 4, 4) for _ in 1:6]

        for (BasisType, loss) in [
            (EntangledQFTBasis, ParametricDFT.L1Norm()),
            (TEBDBasis, ParametricDFT.L1Norm()),
            (MERABasis, ParametricDFT.L1Norm()),
        ]
            basis, _ = train_basis(
                BasisType, dataset;
                m=m, n=n,
                loss=loss,
                epochs=3,
                steps_per_image=3,
                batch_size=2,
                optimizer=:adam,
            )

            # Circuit must remain unitary after training
            x = randn(ComplexF64, 2^m, 2^n)
            fwd = forward_transform(basis, x)
            @test norm(fwd) ≈ norm(x) rtol=1e-8

            # Roundtrip must recover original input
            roundtrip = inverse_transform(basis, fwd)
            @test roundtrip ≈ x rtol=1e-8
        end
    end

    @testset "batch_size=1 fallback still works" begin
        Random.seed!(42)
        m, n = 2, 2
        dataset = [rand(Float64, 4, 4) for _ in 1:4]

        basis, history = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            loss=ParametricDFT.L1Norm(),
            epochs=5,
            steps_per_image=5,
            batch_size=1,
            optimizer=:adam,
        )

        @test basis isa QFTBasis
        # With batch_size=1, step losses are on different images so compare epoch averages
        @test length(history.step_train_losses) > 1
    end

    @testset "_cosine_with_warmup schedule" begin
        total = 1000
        warmup_frac = 0.1
        lr_peak  = 0.01
        lr_final = 0.001
        f(step) = ParametricDFT._cosine_with_warmup(step, total;
                      warmup_frac=warmup_frac, lr_peak=lr_peak, lr_final=lr_final)

        # Step 0 during warmup → near 0 (at most lr_peak / warmup_steps)
        @test f(0) < lr_peak * 0.01

        # End of warmup (step = warmup_steps = 100) → approximately lr_peak
        @test isapprox(f(100), lr_peak; rtol=1e-10)

        # Last step → approximately lr_final
        @test isapprox(f(total), lr_final; rtol=1e-10)

        # Midway between warmup end and total → between lr_peak and lr_final, strictly
        mid = f(round(Int, (100 + total) / 2))
        @test lr_final < mid < lr_peak
    end

    @testset "train_basis one-step-per-batch descent" begin
        Random.seed!(4444)
        images = [rand(Float64, 4, 4) for _ in 1:8]
        basis, history = ParametricDFT.train_basis(QFTBasis, images;
            m = 2, n = 2,
            loss = ParametricDFT.MSELoss(4),
            epochs = 2,
            batch_size = 4,
            optimizer = :adam,
            validation_split = 0.25,
            early_stopping_patience = 10,
            warmup_frac = 0.1,
            lr_peak  = 0.01,
            lr_final = 0.001,
            max_grad_norm = nothing,
            shuffle = false,
        )
        # Loss should decrease from first to last epoch on average
        @test last(history.train_losses) <= first(history.train_losses)
        # Unitarity preserved for the Hadamard-role tensors (those classified
        # as UnitaryManifold by the library's runtime check). Phase tensors
        # (CPHASE factored as 2x2 with row-norm √2) land on the other
        # manifold and are not expected to satisfy UU† = I.
        for t in basis.tensors
            m = ParametricDFT.classify_manifold(t)
            if m isa ParametricDFT.UnitaryManifold
                d = size(t, 1)
                @test isapprox(t * t', Matrix{ComplexF64}(I, d, d); atol=1e-8)
            end
        end
    end

    @testset "train_basis deprecation warning for steps_per_image" begin
        Random.seed!(6666)
        images = [rand(Float64, 4, 4) for _ in 1:4]
        @test_logs (:warn, r"steps_per_image") match_mode=:any begin
            ParametricDFT.train_basis(QFTBasis, images;
                m = 2, n = 2,
                loss = ParametricDFT.MSELoss(4),
                epochs = 1, batch_size = 4,
                optimizer = :adam,
                validation_split = 0.25,
                early_stopping_patience = 10,
                steps_per_image = 5,
                warmup_frac = 0.1, lr_peak = 0.01, lr_final = 0.001,
                shuffle = false,
            )
        end
    end

end
