# ============================================================================
# Tests for Loss Functions and Batched Einsum (loss.jl)
# ============================================================================

@testset "topk_truncate" begin
    Random.seed!(42)

    @testset "keeps exactly k elements" begin
        x = rand(ComplexF64, 4, 4)
        for k in [1, 5, 10, 16]
            y = ParametricDFT.topk_truncate(x, k)
            @test size(y) == size(x)
            @test count(!iszero, y) == k
        end
    end

    @testset "k larger than matrix size" begin
        x = rand(ComplexF64, 3, 3)
        y = ParametricDFT.topk_truncate(x, 100)
        @test count(!iszero, y) == 9
        @test y ≈ x
    end

    @testset "selects by magnitude only" begin
        # With uniform magnitudes, any k positions are valid (no center bias)
        x = ones(ComplexF64, 8, 8)
        y = ParametricDFT.topk_truncate(x, 4)
        @test count(!iszero, y) == 4

        # With varying magnitudes, highest-magnitude entries are selected
        x2 = zeros(ComplexF64, 4, 4)
        x2[1, 1] = 10.0
        x2[2, 3] = 8.0
        x2[4, 4] = 6.0
        x2[3, 1] = 1.0
        y2 = ParametricDFT.topk_truncate(x2, 3)
        @test y2[1, 1] == x2[1, 1]
        @test y2[2, 3] == x2[2, 3]
        @test y2[4, 4] == x2[4, 4]
        @test iszero(y2[3, 1])
    end

    @testset "gradient passes through selected indices" begin
        x = rand(ComplexF64, 4, 4)
        k = 5
        y, pullback = Zygote.pullback(z -> ParametricDFT.topk_truncate(z, k), x)
        @test count(!iszero, y) == k

        ȳ = ones(ComplexF64, 4, 4)
        x̄ = pullback(ȳ)[1]
        for i in 1:4, j in 1:4
            if iszero(y[i, j])
                @test iszero(x̄[i, j])
            else
                @test x̄[i, j] ≈ ȳ[i, j]
            end
        end
    end

    @testset "deterministic output" begin
        x = rand(ComplexF64, 4, 4)
        @test ParametricDFT.topk_truncate(x, 5) == ParametricDFT.topk_truncate(x, 5)
    end

    @testset "values preserved exactly" begin
        x = rand(ComplexF64, 4, 4)
        y = ParametricDFT.topk_truncate(x, 8)
        for i in 1:4, j in 1:4
            if !iszero(y[i, j])
                @test y[i, j] == x[i, j]
            end
        end
    end
end

@testset "loss functions" begin
    Random.seed!(1234)
    m, n = 3, 3
    optcode, tensors = ParametricDFT.qft_code(m, n)
    optcode_inv, _ = ParametricDFT.qft_code(m, n; inverse=true)
    pic = rand(ComplexF64, 2^m, 2^n)

    # Test L1Norm
    loss_l1 = ParametricDFT.L1Norm()
    l1_value = ParametricDFT.loss_function(tensors, m, n, optcode, pic, loss_l1)
    @test l1_value isa Float64
    @test l1_value > 0.0

    # Test L2Norm
    loss_l2 = ParametricDFT.L2Norm()
    l2_value = ParametricDFT.loss_function(tensors, m, n, optcode, pic, loss_l2)
    @test l2_value isa Float64
    @test l2_value > 0.0

    # Test MSELoss
    for k in [1, 5, 20, 2^(m+n)]
        loss_mse = ParametricDFT.MSELoss(k)
        mse_value = ParametricDFT.loss_function(tensors, m, n, optcode, pic, loss_mse; inverse_code=optcode_inv)
        @test mse_value isa Float64
        @test mse_value >= 0.0
    end
end

@testset "Batched Einsum" begin

    @testset "make_batched_code" begin
        m, n = 3, 3
        optcode, tensors = ParametricDFT.qft_code(m, n)
        n_gates = length(tensors)

        batched_flat, batch_label = ParametricDFT.make_batched_code(optcode, n_gates)

        flat_orig = OMEinsum.flatten(optcode)
        max_orig = maximum(OMEinsum.uniquelabels(flat_orig))
        @test batch_label == max_orig + 1

        # Gate indices unchanged, image input and output have batch label
        orig_ixs = OMEinsum.getixsv(flat_orig)
        new_ixs = OMEinsum.getixsv(batched_flat)
        for i in 1:n_gates
            @test new_ixs[i] == orig_ixs[i]
        end
        @test new_ixs[n_gates + 1] == vcat(orig_ixs[n_gates + 1], [batch_label])
        @test OMEinsum.getiyv(batched_flat) == vcat(OMEinsum.getiyv(flat_orig), [batch_label])
    end

    @testset "batched L1 loss matches per-image L1" begin
        Random.seed!(42)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [ComplexF64.(t) for t in tensors_raw]
        n_gates = length(tensors)
        B = 4
        batch = [rand(ComplexF64, 2^m, 2^n) for _ in 1:B]

        per_image = sum(ParametricDFT.loss_function(tensors, m, n, optcode, img, ParametricDFT.L1Norm()) for img in batch) / B
        batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, B)

        @test isapprox(ParametricDFT.batched_loss_l1(batched_opt, tensors, batch, m, n), per_image, rtol=1e-10)
    end

    @testset "batched L2 loss matches per-image L2" begin
        Random.seed!(42)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [ComplexF64.(t) for t in tensors_raw]
        n_gates = length(tensors)
        B = 4
        batch = [rand(ComplexF64, 2^m, 2^n) for _ in 1:B]

        per_image = sum(ParametricDFT.loss_function(tensors, m, n, optcode, img, ParametricDFT.L2Norm()) for img in batch) / B
        batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, B)

        @test isapprox(ParametricDFT.batched_loss_l2(batched_opt, tensors, batch, m, n), per_image, rtol=1e-10)
    end

    @testset "batched MSE loss matches per-image MSE" begin
        Random.seed!(42)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        optcode_inv, _ = ParametricDFT.qft_code(m, n; inverse=true)
        tensors = [ComplexF64.(t) for t in tensors_raw]
        n_gates = length(tensors)
        B, k = 3, 5
        batch = [rand(ComplexF64, 2^m, 2^n) for _ in 1:B]

        per_image = sum(ParametricDFT.loss_function(tensors, m, n, optcode, img, ParametricDFT.MSELoss(k); inverse_code=optcode_inv) for img in batch) / B
        batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, B)

        @test isapprox(ParametricDFT.batched_loss_mse(batched_opt, tensors, batch, m, n, k, optcode_inv), per_image, rtol=1e-10)
    end

    @testset "Zygote gradients through batched losses" begin
        Random.seed!(42)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [ComplexF64.(t) for t in tensors_raw]
        n_gates = length(tensors)
        B = 3
        batch = [rand(ComplexF64, 2^m, 2^n) for _ in 1:B]

        batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, B)

        for (loss_type, batched_fn) in [
            (ParametricDFT.L1Norm(), (opt, ts, b, m, n) -> ParametricDFT.batched_loss_l1(opt, ts, b, m, n)),
            (ParametricDFT.L2Norm(), (opt, ts, b, m, n) -> ParametricDFT.batched_loss_l2(opt, ts, b, m, n)),
        ]
            batched_grad = Zygote.gradient(ts -> batched_fn(batched_opt, ts, batch, m, n), tensors)[1]
            per_image_grad = Zygote.gradient(tensors) do ts
                sum(ParametricDFT.loss_function(ts, m, n, optcode, img, loss_type) for img in batch) / B
            end[1]

            for i in 1:n_gates
                @test isapprox(batched_grad[i], per_image_grad[i], rtol=1e-6)
            end
        end
    end

    @testset "works with all circuit types" begin
        Random.seed!(42)
        m, n = 2, 2
        B = 3
        batch = [rand(ComplexF64, 2^m, 2^n) for _ in 1:B]

        for code_fn in [ParametricDFT.qft_code, ParametricDFT.entangled_qft_code, ParametricDFT.tebd_code]
            optcode, tensors_raw = code_fn(m, n)
            tensors = [ComplexF64.(t) for t in tensors_raw]
            n_gates = length(tensors)

            per_image = sum(ParametricDFT.loss_function(tensors, m, n, optcode, img, ParametricDFT.L1Norm()) for img in batch) / B
            batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
            batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, B)

            @test isapprox(ParametricDFT.batched_loss_l1(batched_opt, tensors, batch, m, n), per_image, rtol=1e-10)
        end
    end

    @testset "different runtime batch size than optimization batch size" begin
        Random.seed!(42)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [ComplexF64.(t) for t in tensors_raw]
        n_gates = length(tensors)

        # Optimize for batch_size=8, but run with batch_size=3
        batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, 8)
        B = 3
        batch = [rand(ComplexF64, 2^m, 2^n) for _ in 1:B]

        per_image = sum(ParametricDFT.loss_function(tensors, m, n, optcode, img, ParametricDFT.L2Norm()) for img in batch) / B
        @test isapprox(ParametricDFT.batched_loss_l2(batched_opt, tensors, batch, m, n), per_image, rtol=1e-10)
    end

    @testset "Zygote gradient through batched_loss_mse" begin
        Random.seed!(42)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        optcode_inv, _ = ParametricDFT.qft_code(m, n; inverse=true)
        tensors = [ComplexF64.(t) for t in tensors_raw]
        n_gates = length(tensors)
        B, k = 3, 5
        batch = [rand(ComplexF64, 2^m, 2^n) for _ in 1:B]
        batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, B)
        zg = Zygote.gradient(
            ts -> ParametricDFT.batched_loss_mse(batched_opt, ts, batch, m, n, k, optcode_inv),
            tensors)[1]
        @test zg isa Vector
        @test length(zg) == n_gates
        @test all(g isa AbstractMatrix for g in zg)
        # Finite-difference check for tensors[1][1,1]
        ε = 1e-5
        ts_p = deepcopy(tensors); ts_m = deepcopy(tensors)
        ts_p[1][1, 1] += ε; ts_m[1][1, 1] -= ε
        fd = (ParametricDFT.batched_loss_mse(batched_opt, ts_p, batch, m, n, k, optcode_inv) -
              ParametricDFT.batched_loss_mse(batched_opt, ts_m, batch, m, n, k, optcode_inv)) / (2*ε)
        @test isapprox(real(zg[1][1, 1]), fd, rtol=1e-3)
    end

end
