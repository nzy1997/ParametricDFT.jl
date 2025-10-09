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
initial_theta(n::Int, method::SingleParameter) = [1.0]

function qft_tensors(n::Int, theta::Vector, method::SingleParameter)
    @assert length(theta) == 1
    tns = AbstractArray{ComplexF64}[]
    H = ComplexF64[1.0 1.0; 1.0 -1.0] ./ sqrt(2)
    
    for j in 1:n-1
        push!(tns, H)
        for k in j+1:n
            ctrl_mat = ComplexF64[1.0 1.0; 1.0 exp(im*π*theta[1]/2^(k-j))]
            push!(tns, ctrl_mat)
        end
    end
    push!(tns, H)
    return tns
end

abstract type AbstractLoss end
struct L1Norm <: AbstractLoss end


function loss_function(n::Int,optcode::OMEinsum.AbstractEinsum,theta::Vector,pic::Vector,loss::AbstractLoss,method::ParameterizationMethod)
    @assert length(pic) == 2^n
    tensors = qft_tensors(n,theta,method)
    push!(tensors, reshape(pic, (fill(2, n)...,)))
    return _loss_function(optcode(tensors...),pic,loss)
end

_loss_function(fft_res,pic,loss::L1Norm) = sum(abs.(fft_res))

function fft_with_training(n::Int, pic::Vector,loss::AbstractLoss,method::ParameterizationMethod)
    optcode, _ = qft_code(n)
    f(x) = loss_function(n,optcode,x,pic,loss,method)

    rule = Optimisers.Adam(0.1)
    theta = initial_theta(n, method)
    state = Optimisers.setup(rule, theta)
    grad = zero(theta)

    max_epochs = 100
    for epoch in 1:max_epochs
        println("Epoch $epoch: Loss = $(f(theta))")
        # grad = Zygote.gradient(f, theta)
        grad = 0.1
        Optimisers.update!(state, theta, grad)
    end
    tensors = qft_tensors(n,theta,method)
    push!(tensors, reshape(pic, (fill(2, n)...,)))
    return reshape(optcode(tensors...),2^n)
end