@testset "Materialized Unitary" begin

    @testset "build_circuit_unitary correctness" begin
        Random.seed!(60)
        m, n = 3, 3  # 8×8 images for fast testing
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)

        # Build batched einsum for D basis vectors
        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)

        U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors), m, n)
        @test size(U) == (D, D)

        # Verify U is unitary
        @test U * U' ≈ Matrix{ComplexF64}(I, D, D) atol=1e-8

        # Verify each column matches per-image einsum
        for j in 1:min(8, D)  # check first 8 columns for speed
            e_j = zeros(ComplexF64, D)
            e_j[j] = 1.0
            expected = vec(optcode(tensors..., reshape(e_j, fill(2, m + n)...)))
            @test U[:, j] ≈ expected atol=1e-10
        end
    end

    @testset "build_circuit_unitary entangled QFT" begin
        Random.seed!(61)
        m, n = 3, 3
        optcode, tensors_raw, _ = ParametricDFT.entangled_qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)

        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)

        U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors), m, n)
        @test size(U) == (D, D)
        @test U * U' ≈ Matrix{ComplexF64}(I, D, D) atol=1e-8
    end

    @testset "materialized_forward matches einsum forward" begin
        Random.seed!(62)
        m, n = 3, 3
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)

        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)
        U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors), m, n)

        # Compare for random images
        for _ in 1:5
            img = randn(ComplexF64, 2^m, 2^n)
            einsum_result = reshape(optcode(tensors..., reshape(img, fill(2, m + n)...)), 2^m, 2^n)
            matmul_result = reshape(U * vec(img), 2^m, 2^n)
            @test matmul_result ≈ einsum_result atol=1e-10
        end
    end

    @testset "materialized_forward batched" begin
        Random.seed!(63)
        m, n = 3, 3
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)

        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)
        U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors), m, n)

        # Batched forward: U * [img1 | img2 | ... | imgB]
        B = 4
        images = [randn(ComplexF64, 2^m, 2^n) for _ in 1:B]
        X = hcat([vec(img) for img in images]...)
        result = U * X  # D × B

        for i in 1:B
            einsum_result = vec(optcode(tensors..., reshape(images[i], fill(2, m + n)...)))
            @test result[:, i] ≈ einsum_result atol=1e-10
        end
    end

end

@testset "Materialized Loss Functions" begin

    @testset "materialized_loss L1 matches einsum" begin
        Random.seed!(64)
        m, n = 3, 3
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)
        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)
        U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors), m, n)

        images = [randn(ComplexF64, 2^m, 2^n) for _ in 1:4]
        loss_mat = ParametricDFT.materialized_loss_l1(U, images, m, n)
        batch_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, 4)
        loss_ein = ParametricDFT.batched_loss_l1(batch_optcode, Tuple(tensors), images, m, n)
        @test loss_mat ≈ loss_ein atol=1e-8
    end

    @testset "materialized_loss L2 matches einsum" begin
        Random.seed!(65)
        m, n = 3, 3
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)
        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)
        U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors), m, n)

        images = [randn(ComplexF64, 2^m, 2^n) for _ in 1:4]
        loss_mat = ParametricDFT.materialized_loss_l2(U, images, m, n)
        batch_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, 4)
        loss_ein = ParametricDFT.batched_loss_l2(batch_optcode, Tuple(tensors), images, m, n)
        @test loss_mat ≈ loss_ein atol=1e-8
    end

    @testset "materialized_loss MSE matches einsum" begin
        Random.seed!(66)
        m, n = 3, 3
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        inverse_code, _ = ParametricDFT.qft_code(m, n; inverse=true)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)
        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)
        U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(tensors), m, n)

        images = [randn(ComplexF64, 2^m, 2^n) for _ in 1:4]
        k = 10
        loss_mat = ParametricDFT.materialized_loss_mse(U, images, m, n, k)
        batch_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, 4)
        loss_ein = ParametricDFT.batched_loss_mse(batch_optcode, Tuple(tensors), images, m, n, k, inverse_code)
        @test loss_mat ≈ loss_ein atol=1e-8
    end

end

@testset "Materialized AD Gradients" begin

    @testset "gradient through materialized L1" begin
        Random.seed!(67)
        m, n = 2, 2  # Small for fast finite diff
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)
        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)

        images = [randn(ComplexF64, 2^m, 2^n) for _ in 1:2]

        loss_fn = ts -> begin
            U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(ts), m, n)
            ParametricDFT.materialized_loss_l1(U, images, m, n)
        end

        grads = Zygote.gradient(loss_fn, tensors)[1]
        @test grads !== nothing
        @test length(grads) == length(tensors)

        # Finite difference check for first two gates
        eps = 1e-6
        for idx in 1:min(2, length(tensors))
            for i in 1:2, j in 1:2
                ts_plus = deepcopy(tensors)
                ts_minus = deepcopy(tensors)
                ts_plus[idx][i, j] += eps
                ts_minus[idx][i, j] -= eps
                fd_grad = (loss_fn(ts_plus) - loss_fn(ts_minus)) / (2 * eps)
                @test real(grads[idx][i, j]) ≈ real(fd_grad) atol=1e-4
            end
        end
    end

    @testset "materialized gradient matches einsum gradient" begin
        Random.seed!(68)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)
        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)
        batch_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, 2)

        images = [randn(ComplexF64, 2^m, 2^n) for _ in 1:2]

        # Materialized gradient
        mat_loss_fn = ts -> begin
            U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(ts), m, n)
            ParametricDFT.materialized_loss_l1(U, images, m, n)
        end
        grads_mat = Zygote.gradient(mat_loss_fn, tensors)[1]

        # Einsum gradient
        ein_loss_fn = ts -> ParametricDFT.batched_loss_l1(batch_optcode, Tuple(ts), images, m, n)
        grads_ein = Zygote.gradient(ein_loss_fn, tensors)[1]

        # Gradients should match
        for idx in 1:length(tensors)
            @test grads_mat[idx] ≈ grads_ein[idx] atol=1e-6
        end
    end

end

@testset "Device Strategy" begin

    @testset "select_device_strategy" begin
        @test ParametricDFT.select_device_strategy(3, 3, 4, :cpu) == :einsum_cpu
        @test ParametricDFT.select_device_strategy(6, 6, 4, :gpu) == :materialized_gpu
        @test ParametricDFT.select_device_strategy(3, 3, 4, :gpu) == :einsum_gpu
        @test ParametricDFT.select_device_strategy(7, 7, 1, :gpu) == :materialized_gpu
        @test ParametricDFT.select_device_strategy(2, 2, 8, :cpu) == :einsum_cpu
    end

    @testset "materialized path produces valid training" begin
        Random.seed!(69)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [Matrix{ComplexF64}(t) for t in tensors_raw]
        D = 2^(m + n)
        n_gates = length(tensors)
        flat_batched, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        unitary_optcode = ParametricDFT.optimize_batched_code(flat_batched, blabel, D)

        images = [Complex{Float64}.(randn(Float64, 2^m, 2^n)) for _ in 1:4]

        # Build materialized loss
        loss_fn = ts -> begin
            U = ParametricDFT.build_circuit_unitary(unitary_optcode, Tuple(ts), m, n)
            ParametricDFT.materialized_loss_l1(U, images, m, n)
        end

        grad_fn = ts -> begin
            _, back = Zygote.pullback(loss_fn, ts)
            back(1.0)[1]
        end

        opt = ParametricDFT.RiemannianAdam(lr=0.01)
        initial_loss = loss_fn(tensors)
        result = ParametricDFT.optimize!(opt, tensors, loss_fn, grad_fn; max_iter=10)
        final_loss = loss_fn(result)

        # Loss should decrease
        @test final_loss < initial_loss
    end

end
