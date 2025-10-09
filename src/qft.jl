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

function qft_tensors(n::Int, theta::Vector, method::SingleParameter)
    @show typeof(theta)
    @assert length(theta) == 1
    tns = Vector{AbstractArray{ComplexF64}}(undef, n*(n-1)+1)
    H = Matrix{ComplexF64}(undef, 2, 2)
    H[1,1] = 1.0+0.0im
    H[1,2] = 1.0+0.0im
    H[2,1] = 1.0+0.0im
    H[2,2] = -1.0+0.0im
    H ./= sqrt(2)
    
    count_tensor = 0
    for j in 1:n-1
        count_tensor += 1
        tns[count_tensor] = H
        for k in j+1:n
            count_tensor += 1
            # 避免使用矩阵字面量，显式构造矩阵
            ctrl_mat = Matrix{ComplexF64}(undef, 2, 2)
            ctrl_mat[1,1] = 1.0+0.0im
            ctrl_mat[1,2] = 1.0+0.0im
            ctrl_mat[2,1] = 1.0+0.0im
            ctrl_mat[2,2] = exp(im*π*theta[1]/2^(k-j))
            tns[count_tensor] = ctrl_mat
        end
    end
    count_tensor += 1
    tns[count_tensor] = H
    return tns
end

abstract type AbstractLoss end
struct L1Norm <: AbstractLoss end


function loss_function(n::Int,optcode::OMEinsum.AbstractEinsum,theta::Vector,pics::Vector,loss::L1Norm,method::SingleParameter)
    @assert length(pics) == 2^n
    tensors = qft_tensors(n,theta,method)
    @show typeof(tensors)
    tensors[end] = reshape(pics, (fill(2, n)...,))
    fft_res = reshape(optcode(tensors...), 2^n)
    return _loss_function(fft_res,pics,loss)
end

_loss_function(fft_res,pics,loss::L1Norm) = sum(abs.(fft_res))

