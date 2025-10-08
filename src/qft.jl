_A(i, j) = control(i, j=>shift(2π/(1<<(i-j+1))))
_B(n, k) = chain(n, j==k ? put(k=>H) : _A(j, k) for j in k:n)
_qft(n) = chain(_B(n, k) for k in 1:n)

function qft_code(qubit_num::Int)
    qc = _qft(qubit_num)
    tn = yao2einsum(qc)
    code = OMEinsum.flatten(tn.code)
    push!(code.ixs, collect(1:qubit_num))
    code = DynamicEinCode(code.ixs, collect(qubit_num+1:2*qubit_num))

    optcode = optimize_code(code,uniformsize(code, 2), TreeSA())
    return optcode, tn.tensors
end

abstract type ParameterizationMethod end
struct SingleParameter <: ParameterizationMethod end

function qft_tensors(n::Int, theta::Vector{Float64}, method::SingleParameter)
    @assert length(theta) == 1
    tns = Vector{AbstractArray{ComplexF64}}(undef, n*(n-1)+1)
    H = [1.0+0.0im 1.0+0.0im; 1.0+0.0im -1.0+0.0im]/sqrt(2)
    count_tensor = 0
    for j in 1:n-1
        count_tensor += 1
        tns[count_tensor] = H
        for k in j+1:n
            count_tensor += 1
            tns[count_tensor] = [1.0+0.0im 1.0+0.0im; 1.0+0.0im exp(im*π*theta[1]/2^(k-j))]
        end
    end
    count_tensor += 1
    tns[count_tensor] = H
    return tns
end

abstract type AbstractLoss end
struct L1Norm <: AbstractLoss end


function loss_function(n::Int,optcode::OMEinsum.AbstractEinsum,theta::Vector{Float64},pics::Vector,loss::L1Norm,method::SingleParameter)
    @assert length(pics) == 2^n
    tensors = qft_tensors(n,theta,method)
    tensors[end] = reshape(pics, (fill(2, n)...,))
    fft_res = reshape(optcode(tensors...), 2^n)
    return _loss_function(fft_res,pics,loss)
end

_loss_function(fft_res,pics,loss::L1Norm) = sum(abs.(fft_res))