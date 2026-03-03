using ParametricDFT
using Test
using OMEinsum
using LinearAlgebra
using Random
using Yao
using JSON3
using Statistics
using Zygote

@testset "ParametricDFT.jl" begin

@testset "qft" begin
    Random.seed!(1234)
    m, n = 3, 3
    optcode, tensors = ParametricDFT.qft_code(m, n)
    pic = rand(ComplexF64, 2^m, 2^n)
    result = reshape(optcode(tensors..., reshape(pic, fill(2, m+n)...)), 2^m, 2^n)
    @test size(result) == (2^m, 2^n)
end

@testset "fft and ifft are inverses" begin
    Random.seed!(1234)
    m, n = 3, 3
    optcode, tensors = ParametricDFT.qft_code(m, n)
    tensors[1] = rand_unitary(2)
    tensors[end][1,2] = exp(im*pi/4)
    tensors[end][2,1] = exp(-im*pi/5)
    optcode_inv, tensors_inv = ParametricDFT.qft_code(m, n; inverse=true)
    pic = rand(ComplexF64, 2^m, 2^n)

    fft_result = ParametricDFT.ft_mat(tensors, optcode, m, n, pic)
    reconstructed = ParametricDFT.ift_mat(conj.(tensors), optcode_inv, m, n, fft_result)

    @test isapprox(reconstructed, pic, rtol=1e-10)
    @test isapprox(norm(fft_result), norm(pic), rtol=1e-10)
end

# Include test files
include("basis_tests.jl")
include("entangled_qft_tests.jl")
include("tebd_tests.jl")
include("training_tests.jl")
include("serialization_tests.jl")
include("compression_tests.jl")
include("loss_tests.jl")
include("manifold_tests.jl")
include("optimizer_tests.jl")

end  # @testset "ParametricDFT.jl"
