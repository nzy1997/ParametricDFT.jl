using QDFT
using Test
using FFTW
using OMEinsum

@testset "qft" begin
    qubit_num = 3
    optcode, tensors = QDFT.qft_code(qubit_num)

    tn2 = QDFT.qft_tensors(qubit_num,[1.0],QDFT.SingleParameter())
    @test isapprox(tn2[1:end-1], tensors;atol = 1e-10)

    pic = rand(2^(qubit_num ))

    reshape(pic, (fill(2, qubit_num)...,))

    push!(tensors,reshape(pic, (fill(2, qubit_num)...,)))

    fftpic_tn = reshape(optcode(tensors...),2^(qubit_num))

    @test norm(fft(pic[[1, 8, 4, 6, 2, 7, 3, 5]])/sqrt(2^qubit_num)- fftpic_tn) < 1e-6
end
