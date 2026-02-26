# ============================================================================
# Tests for Batched Einsum Operations (batched_einsum.jl)
# ============================================================================

@testset "Batched Einsum" begin

    @testset "make_batched_code" begin
        m, n = 3, 3
        optcode, tensors = ParametricDFT.qft_code(m, n)
        n_gates = length(tensors)

        batched_flat, batch_label = ParametricDFT.make_batched_code(optcode, n_gates)

        # Batch label should be one beyond max existing label
        flat_orig = OMEinsum.flatten(optcode)
        max_orig = maximum(OMEinsum.uniquelabels(flat_orig))
        @test batch_label == max_orig + 1

        # Gate indices should be unchanged
        orig_ixs = OMEinsum.getixsv(flat_orig)
        new_ixs = OMEinsum.getixsv(batched_flat)
        for i in 1:n_gates
            @test new_ixs[i] == orig_ixs[i]
        end

        # Image input should have batch label appended
        @test new_ixs[n_gates + 1] == vcat(orig_ixs[n_gates + 1], [batch_label])

        # Output should have batch label appended
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

        # Per-image L1 loss
        per_image_loss = sum(
            ParametricDFT.loss_function(tensors, m, n, optcode, img, ParametricDFT.L1Norm())
            for img in batch
        ) / B

        # Batched L1 loss
        batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, B)
        batched_loss = ParametricDFT.batched_loss_l1(batched_opt, tensors, batch, m, n)

        @test isapprox(batched_loss, per_image_loss, rtol=1e-10)
    end

    @testset "batched L2 loss matches per-image L2" begin
        Random.seed!(42)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [ComplexF64.(t) for t in tensors_raw]
        n_gates = length(tensors)
        B = 4
        batch = [rand(ComplexF64, 2^m, 2^n) for _ in 1:B]

        per_image_loss = sum(
            ParametricDFT.loss_function(tensors, m, n, optcode, img, ParametricDFT.L2Norm())
            for img in batch
        ) / B

        batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, B)
        batched_loss = ParametricDFT.batched_loss_l2(batched_opt, tensors, batch, m, n)

        @test isapprox(batched_loss, per_image_loss, rtol=1e-10)
    end

    @testset "batched MSE loss matches per-image MSE" begin
        Random.seed!(42)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        optcode_inv, _ = ParametricDFT.qft_code(m, n; inverse=true)
        tensors = [ComplexF64.(t) for t in tensors_raw]
        n_gates = length(tensors)
        B = 3
        k = 5
        batch = [rand(ComplexF64, 2^m, 2^n) for _ in 1:B]

        per_image_loss = sum(
            ParametricDFT.loss_function(tensors, m, n, optcode, img, ParametricDFT.MSELoss(k); inverse_code=optcode_inv)
            for img in batch
        ) / B

        batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, B)
        batched_loss = ParametricDFT.batched_loss_mse(batched_opt, tensors, batch, m, n, k, optcode_inv)

        @test isapprox(batched_loss, per_image_loss, rtol=1e-10)
    end

    @testset "Zygote gradients through batched L1" begin
        Random.seed!(42)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [ComplexF64.(t) for t in tensors_raw]
        n_gates = length(tensors)
        B = 3
        batch = [rand(ComplexF64, 2^m, 2^n) for _ in 1:B]

        batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, B)

        # Batched gradient
        batched_grad = Zygote.gradient(
            ts -> ParametricDFT.batched_loss_l1(batched_opt, ts, batch, m, n),
            tensors
        )[1]

        # Per-image gradient sum
        per_image_grad = Zygote.gradient(tensors) do ts
            sum(
                ParametricDFT.loss_function(ts, m, n, optcode, img, ParametricDFT.L1Norm())
                for img in batch
            ) / B
        end[1]

        for i in 1:n_gates
            @test isapprox(batched_grad[i], per_image_grad[i], rtol=1e-6)
        end
    end

    @testset "Zygote gradients through batched L2" begin
        Random.seed!(42)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [ComplexF64.(t) for t in tensors_raw]
        n_gates = length(tensors)
        B = 3
        batch = [rand(ComplexF64, 2^m, 2^n) for _ in 1:B]

        batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, B)

        batched_grad = Zygote.gradient(
            ts -> ParametricDFT.batched_loss_l2(batched_opt, ts, batch, m, n),
            tensors
        )[1]

        per_image_grad = Zygote.gradient(tensors) do ts
            sum(
                ParametricDFT.loss_function(ts, m, n, optcode, img, ParametricDFT.L2Norm())
                for img in batch
            ) / B
        end[1]

        for i in 1:n_gates
            @test isapprox(batched_grad[i], per_image_grad[i], rtol=1e-6)
        end
    end

    @testset "works with EntangledQFT circuit" begin
        Random.seed!(42)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.entangled_qft_code(m, n)
        tensors = [ComplexF64.(t) for t in tensors_raw]
        n_gates = length(tensors)
        B = 3
        batch = [rand(ComplexF64, 2^m, 2^n) for _ in 1:B]

        per_image_loss = sum(
            ParametricDFT.loss_function(tensors, m, n, optcode, img, ParametricDFT.L1Norm())
            for img in batch
        ) / B

        batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, B)
        batched_loss = ParametricDFT.batched_loss_l1(batched_opt, tensors, batch, m, n)

        @test isapprox(batched_loss, per_image_loss, rtol=1e-10)
    end

    @testset "works with TEBD circuit" begin
        Random.seed!(42)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.tebd_code(m, n)
        tensors = [ComplexF64.(t) for t in tensors_raw]
        n_gates = length(tensors)
        B = 3
        batch = [rand(ComplexF64, 2^m, 2^n) for _ in 1:B]

        per_image_loss = sum(
            ParametricDFT.loss_function(tensors, m, n, optcode, img, ParametricDFT.L1Norm())
            for img in batch
        ) / B

        batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, B)
        batched_loss = ParametricDFT.batched_loss_l1(batched_opt, tensors, batch, m, n)

        @test isapprox(batched_loss, per_image_loss, rtol=1e-10)
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

        per_image_loss = sum(
            ParametricDFT.loss_function(tensors, m, n, optcode, img, ParametricDFT.L2Norm())
            for img in batch
        ) / B

        batched_loss = ParametricDFT.batched_loss_l2(batched_opt, tensors, batch, m, n)

        @test isapprox(batched_loss, per_image_loss, rtol=1e-10)
    end

end
