# ============================================================================
# CUDA/GPU Tests (manual-run on GPU machines)
# ============================================================================
# Run with: julia --project -e 'using CUDA; include("test/cuda_tests.jl")'
# Requires a CUDA-capable GPU and CUDA.jl installed.

using ParametricDFT
using Test
using CUDA
using LinearAlgebra
using Random
using OMEinsum

@testset "CUDA GPU Tests" begin

    @testset "to_device round-trip" begin
        x = rand(ComplexF64, 4, 4)
        x_gpu = ParametricDFT.to_device(x, :gpu)
        @test x_gpu isa CuArray
        x_cpu = Array(x_gpu)
        @test x_cpu ≈ x
    end

    @testset "topk_truncate on CuArray" begin
        x = rand(ComplexF64, 8, 8)
        x_gpu = CuArray(x)
        k = 10

        y_cpu = ParametricDFT.topk_truncate(x, k)
        y_gpu = ParametricDFT.topk_truncate(x_gpu, k)

        @test y_gpu isa CuArray
        @test Array(y_gpu) ≈ y_cpu
    end

    @testset "batched_forward with CuArray" begin
        Random.seed!(42)
        m, n = 2, 2
        optcode, tensors_raw = ParametricDFT.qft_code(m, n)
        tensors = [ComplexF64.(t) for t in tensors_raw]
        tensors_gpu = [CuArray(t) for t in tensors]
        n_gates = length(tensors)
        B = 4
        batch_cpu = [rand(ComplexF64, 2^m, 2^n) for _ in 1:B]
        batch_gpu = [CuArray(img) for img in batch_cpu]

        batched_flat, blabel = ParametricDFT.make_batched_code(optcode, n_gates)
        batched_opt = ParametricDFT.optimize_batched_code(batched_flat, blabel, B)

        result_cpu = ParametricDFT.batched_forward(batched_opt, tensors, batch_cpu, m, n)
        result_gpu = ParametricDFT.batched_forward(batched_opt, tensors_gpu, batch_gpu, m, n)

        @test result_gpu isa CuArray
        @test Array(result_gpu) ≈ result_cpu atol=1e-10
    end

    @testset "end-to-end train_basis with device=:gpu" begin
        Random.seed!(42)
        m, n = 2, 2
        dataset = [rand(Float64, 4, 4) for _ in 1:4]

        basis, _ = train_basis(
            QFTBasis, dataset;
            m=m, n=n,
            loss=ParametricDFT.L1Norm(),
            epochs=1,
            steps_per_image=3,
            batch_size=2,
            optimizer=:adam,
            device=:gpu,
            verbose=false
        )

        @test basis isa QFTBasis
        @test basis.m == m
        @test basis.n == n
    end

end
