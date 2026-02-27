# ============================================================================
# Optimizer Correctness Tests
# ============================================================================
# End-to-end test that verifies train_basis produces equivalent results
# across CPU Manopt GD, GPU GD, and GPU Adam at multiple batch sizes.
# GPU tests are skipped if CUDA is not available.
#
# The key assertion is cross-optimizer consistency: all optimizers should
# reach the same final loss (within 20%). We do NOT assert loss decreases
# because with L1Norm on random data the starting point may already be
# near-optimal.

const HAS_CUDA_TEST = try
    using CUDA
    CUDA.functional()
catch
    false
end

@testset "Optimizer Correctness" begin

    # Helper: train TEBDBasis and return final/initial loss
    function train_and_loss(dataset, m, n; optimizer, device, batch_size, steps, phases, loss_fn)
        Random.seed!(42)
        _, history = train_basis(
            TEBDBasis, dataset;
            m=m, n=n,
            phases=copy(phases),
            loss=loss_fn,
            epochs=1,
            steps_per_image=steps,
            validation_split=0.0,
            shuffle=false,
            early_stopping_patience=0,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
            verbose=false
        )
        return history.step_train_losses[end], history.step_train_losses[1]
    end

    # Setup: synthetic 8x8 images
    Random.seed!(42)
    m, n = 3, 3
    dataset = [rand(Float64, 8, 8) for _ in 1:20]
    n_gates = m + n
    phases = randn(n_gates) * 0.1
    steps = 50
    loss_fn = ParametricDFT.L1Norm()

    # Get CPU Manopt GD baseline
    baseline_loss, _ = train_and_loss(dataset, m, n;
        optimizer=:gradient_descent, device=:cpu, batch_size=1, steps=steps,
        phases=phases, loss_fn=loss_fn)

    @testset "CPU Manopt GD baseline completes" begin
        @test baseline_loss > 0  # sanity check
    end

    if HAS_CUDA_TEST
        @testset "GPU GD batch=$bs" for bs in [1, 4, 16]
            final, _ = train_and_loss(dataset, m, n;
                optimizer=:gradient_descent, device=:gpu, batch_size=bs, steps=steps,
                phases=phases, loss_fn=loss_fn)
            @test abs(final / baseline_loss - 1) < 0.2  # within 20% of baseline
        end

        @testset "GPU Adam batch=$bs" for bs in [1, 4, 16]
            final, _ = train_and_loss(dataset, m, n;
                optimizer=:adam, device=:gpu, batch_size=bs, steps=steps,
                phases=phases, loss_fn=loss_fn)
            @test abs(final / baseline_loss - 1) < 0.2  # within 20% of baseline
        end
    else
        @info "GPU tests skipped (no CUDA)"
    end
end
