# ============================================================================
# Optimizer Correctness Cross-Check
# ============================================================================
# Compares CPU Manopt GD, GPU GD, and GPU Adam at multiple batch sizes
# to verify all optimization paths in train_basis produce equivalent results.
#
# CPU only:  julia --project examples/optimizer_crosscheck.jl
# With GPU:  CUDA_VISIBLE_DEVICES=1 julia --project -e 'using CUDA; include("examples/optimizer_crosscheck.jl")'
# With MNIST: CUDA_VISIBLE_DEVICES=1 julia --project -e 'using CUDA; ARGS=["mnist"]; include("examples/optimizer_crosscheck.jl")'
# ============================================================================

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using ParametricDFT
using Random
using Printf
using Statistics

const HAS_CUDA = @static if isdefined(Main, :CUDA)
    Main.CUDA.functional()
else
    false
end

const RUN_MNIST = length(ARGS) > 0 && lowercase(ARGS[1]) == "mnist"

# ============================================================================
# Helper: run one optimizer configuration and return final loss
# ============================================================================

function run_config(dataset, m, n, optimizer, device, batch_size, steps, seed, phases;
                    loss_fn=ParametricDFT.L1Norm())
    Random.seed!(seed)
    basis, history = train_basis(
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
    final_loss = history.step_train_losses[end]
    initial_loss = history.step_train_losses[1]
    return (; basis, final_loss, initial_loss, history)
end

# ============================================================================
# Run a phase and print results table
# ============================================================================

function run_phase(label, dataset, m, n, steps, seed)
    println("\n" * "="^70)
    println("  $label: $(length(dataset)) images, $(2^m)x$(2^n), $steps steps")
    println("="^70)

    # Fixed initial phases for reproducibility
    Random.seed!(seed)
    n_gates = m + n
    phases = randn(n_gates) * 0.1

    configs = [
        ("Manopt GD", :gradient_descent, :cpu, 1),
    ]

    # GPU configs only if CUDA available
    if HAS_CUDA
        for bs in [1, 4, 16]
            push!(configs, ("GPU GD", :gradient_descent, :gpu, bs))
        end
        for bs in [1, 4, 16]
            push!(configs, ("GPU Adam", :adam, :gpu, bs))
        end
    else
        println("\n  GPU: skipped (no CUDA)\n")
    end

    # Run all configurations
    results = []
    for (name, opt, dev, bs) in configs
        bs_clamped = min(bs, length(dataset))
        print("  Running $name (batch=$bs_clamped)... ")
        flush(stdout)
        t = @elapsed r = run_config(dataset, m, n, opt, dev, bs_clamped, steps, seed, phases)
        println("$(round(t, digits=1))s")
        flush(stdout)
        push!(results, (; name, optimizer=opt, device=dev, batch_size=bs_clamped, r.final_loss, r.initial_loss))
    end

    # Print results table
    baseline = results[1].final_loss
    println()
    @printf("  %-12s  %-6s  %5s  %12s  %10s  %10s  %8s\n",
            "Optimizer", "Device", "Batch", "Final Loss", "vs Baseline", "Reduction", "Status")
    println("  " * "-"^68)

    # Track batch_size=1 loss per optimizer for vs-BS=1 comparison
    bs1_loss = Dict{Symbol, Float64}()

    all_pass = true
    for r in results
        reduction = (1 - r.final_loss / r.initial_loss) * 100
        if r === results[1]
            status = "baseline"
            vs_base = "--"
        else
            pct = (r.final_loss / baseline - 1) * 100
            vs_base = @sprintf("%+.1f%%", pct)
            pass = abs(pct) < 20
            # Also check vs batch_size=1 for same optimizer
            if r.batch_size == 1
                bs1_loss[r.optimizer] = r.final_loss
            end
            if haskey(bs1_loss, r.optimizer) && r.batch_size > 1
                pct_bs1 = (r.final_loss / bs1_loss[r.optimizer] - 1) * 100
                pass = pass && abs(pct_bs1) < 20
            end
            status = pass ? "PASS" : "FAIL"
            if !pass
                all_pass = false
            end
        end
        @printf("  %-12s  %-6s  %5d  %12.6f  %10s  %9.1f%%  %8s\n",
                r.name, r.device, r.batch_size, r.final_loss, vs_base, reduction, status)
    end

    println()
    if all_pass
        println("  All configurations PASSED (within 20% tolerance)")
    else
        println("  WARNING: Some configurations FAILED")
    end

    return results
end

# ============================================================================
# Phase 1: Synthetic images
# ============================================================================

Random.seed!(42)
m1, n1 = 3, 3
dataset_synth = [rand(Float64, 2^m1, 2^n1) for _ in 1:20]
results_synth = run_phase("Phase 1: Synthetic 8x8", dataset_synth, m1, n1, 50, 42)

# ============================================================================
# Phase 2: MNIST (optional)
# ============================================================================

if RUN_MNIST
    try
        using MLDatasets
        using Images

        function pad_image(raw_img::AbstractMatrix, target_size)
            padded = zeros(Float64, target_size, target_size)
            r, c = size(raw_img)
            r_off = (target_size - r) ÷ 2 + 1
            c_off = (target_size - c) ÷ 2 + 1
            padded[r_off:r_off+r-1, c_off:c_off+c-1] = Float64.(raw_img)
            return padded
        end

        m2, n2 = 5, 5
        mnist = MLDatasets.MNIST(:train)
        raw_imgs = [Float64.(mnist.features[:, :, i]') for i in 1:20]
        dataset_mnist = [pad_image(img, 2^m2) for img in raw_imgs]
        results_mnist = run_phase("Phase 2: MNIST 32x32", dataset_mnist, m2, n2, 50, 42)
    catch e
        println("\nSkipping MNIST phase: $e")
    end
end

println("\nDone!")
