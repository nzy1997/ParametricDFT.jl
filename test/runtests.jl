using ParametricDFT
using Test
# using FFTW
using OMEinsum
using LinearAlgebra
using Manifolds, Random
using RecursiveArrayTools
using Yao
using JSON3

@testset "ParametricDFT.jl" begin

@testset "qft" begin
    Random.seed!(1234)
    m, n = 3, 3
    optcode, tensors = ParametricDFT.qft_code(m, n)
    pic = rand(ComplexF64, 2^m, 2^n)
    result = reshape(optcode(tensors..., reshape(pic, fill(2, m+n)...)), 2^m, 2^n)
    # Test that result has expected dimensions
    @test size(result) == (2^m, 2^n)
end

@testset "fft with training" begin
    Random.seed!(1234)
    m, n = 2, 2
    pic = rand(ComplexF64, 2^m, 2^n)
    theta = ParametricDFT.fft_with_training(m, n, pic, ParametricDFT.L1Norm(); steps=10)
    @test theta isa ArrayPartition
end

@testset "generate manifold" begin
    m, n = 3, 3
    optcode, tensors = ParametricDFT.qft_code(m, n)
    M = ParametricDFT.generate_manifold(tensors)
    @test M isa ProductManifold

    p = ParametricDFT.tensors2point(tensors, M)
    @test p isa ArrayPartition
    @test is_point(M, p)

    tensors2 = ParametricDFT.point2tensors(p, M)
    @test tensors2 == tensors
end

@testset "fft and ifft are inverses" begin
    Random.seed!(1234)
    m, n = 3, 3
    optcode, tensors = ParametricDFT.qft_code(m, n)
    tensors[1] = rand_unitary(2)
    tensors[end][1,2] = exp(im*pi/4)
    tensors[end][2,1] = exp(-im*pi/5)
    optcode_inv, tensors_inv = ParametricDFT.qft_code(m, n; inverse=true)
    # Create a random input image
    pic = rand(ComplexF64, 2^m, 2^n)

    # Apply forward FFT
    fft_result = ParametricDFT.ft_mat(tensors, optcode, m, n, pic)

    # Apply inverse FFT
    reconstructed = ParametricDFT.ift_mat(conj.(tensors), optcode_inv, m, n, fft_result)

    # Check that we recover the original image (up to numerical precision)
    @test isapprox(reconstructed, pic, rtol=1e-10)

    # Also verify the norm is preserved (unitary property)
    @test isapprox(norm(fft_result), norm(pic), rtol=1e-10)
end

# Include additional test files for new features
include("basis_tests.jl")
include("training_tests.jl")
include("serialization_tests.jl")
include("compression_tests.jl")

end  # @testset "ParametricDFT.jl"