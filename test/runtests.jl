using QDFT
using Test
using FFTW
using OMEinsum
using LinearAlgebra

@testset "qft" begin
    qubit_num = 3
    optcode, tensors = QDFT.qft_code(qubit_num)

    tn2 = QDFT.qft_tensors(qubit_num,[1.0],QDFT.SingleParameter())
    @test isapprox(tn2, tensors;atol = 1e-10)

    pic = rand(2^(qubit_num ))

    reshape(pic, (fill(2, qubit_num)...,))

    push!(tensors,reshape(pic, (fill(2, qubit_num)...,)))

    fftpic_tn = reshape(optcode(tensors...),2^(qubit_num))

    @test LinearAlgebra.norm(fft(pic[[1, 8, 4, 6, 2, 7, 3, 5]])/sqrt(2^qubit_num)- fftpic_tn) < 1e-6
end

@testset "L1Norm loss" begin
    qubit_num = 3
    optcode, tensors = QDFT.qft_code(qubit_num)
    pic = rand(2^(qubit_num))
    loss = QDFT.loss_function(qubit_num,optcode,randn(1),pic,QDFT.L1Norm(),QDFT.SingleParameter())
    @show loss
    @test loss > 0
end

@testset "fft with training" begin
    qubit_num = 3
    pic = rand(2^(qubit_num))
    theta = QDFT.fft_with_training(qubit_num, pic, QDFT.L1Norm())
    @show theta
    # @test theta isa Vector
end

@testset "generate manifold" begin
    n = 3
    M = QDFT.generate_manifold(n)
    @test M isa ProductManifold

    optcode, tensors = QDFT.qft_code(n)

    p = QDFT.tensors2point(tensors, n)
    @test p isa ArrayPartition
    @test is_point(M, p)

    tensors2 = QDFT.point2tensors(p, n)
    @test tensors2 == tensors
end

@testset "sort order" begin
    n = 4
    optcode, tensors = QDFT.qft_code(n)
    H = [1 1; 1 -1]./sqrt(2)
    @test tensors[1:n] == fill(H, n)
    @test length(tensors) == n*(n+1)/2
end