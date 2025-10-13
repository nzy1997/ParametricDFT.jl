using ParametricDFT
using Test
using FFTW
using OMEinsum
using LinearAlgebra
using Manifolds
using RecursiveArrayTools

@testset "qft" begin
    qubit_num = 3
    optcode, tensors = ParametricDFT.qft_code(qubit_num)
    pic = rand(2^(qubit_num))
    mat2 = ParametricDFT.ft_mat(tensors,optcode,qubit_num)
    @test LinearAlgebra.norm(fft(pic[[1, 8, 4, 6, 2, 7, 3, 5]])/sqrt(2^qubit_num)- mat2*pic) < 1e-6
end

@testset "fft with training" begin
    qubit_num = 3
    pic = rand(2^(qubit_num))
    theta = ParametricDFT.fft_with_training(qubit_num, pic, ParametricDFT.L1Norm())
    @show theta
    # @test theta isa Vector
end

@testset "generate manifold" begin
    n = 3
    M = ParametricDFT.generate_manifold(n)
    @test M isa ProductManifold

    optcode, tensors = ParametricDFT.qft_code(n)

    p = ParametricDFT.tensors2point(tensors, n)
    @test p isa ArrayPartition
    @test is_point(M, p)

    tensors2 = ParametricDFT.point2tensors(p, n)
    @test tensors2 == tensors
end

@testset "sort order" begin
    n = 4
    optcode, tensors = ParametricDFT.qft_code(n)
    H = [1 1; 1 -1]./sqrt(2)
    @test tensors[1:n] == fill(H, n)
    @test length(tensors) == n*(n+1)/2
end