using Yao
using FFTW
using OMEinsum
using Combinatorics

qubit_num = 4
A(i, j) = control(i, j=>shift(2π/(1<<(i-j+1))))
B(n, k) = chain(n, j==k ? put(k=>H) : A(j, k) for j in k:n)
qft(n) = chain(B(n, k) for k in 1:n)
qc = qft(qubit_num)

tn = yao2einsum(qc)

code = OMEinsum.flatten(tn.code)
push!(code.ixs, collect(1:qubit_num))
code = DynamicEinCode(code.ixs, collect(qubit_num+1:2*qubit_num))

optcode = optimize_code(code,uniformsize(code, 2), TreeSA())

pic = rand(2^(qubit_num ))

reshape(pic, (fill(2, qubit_num)...,))

push!(tn.tensors,reshape(pic, (fill(2, qubit_num)...,)))

fftpic_tn = reshape(code(tn.tensors...),2^(qubit_num))


for perm in permutations(1:2^qubit_num)
    # @show perm
    if norm(fft(pic[perm])/sqrt(2^qubit_num)- fftpic_tn) < 1e-6
        @show perm
    end
end

function qft_tensor(n,theta)
    tns = Vector{AbstractArray{ComplexF64}}()
    H = [1.0+0.0im 1.0+0.0im; 1.0+0.0im -1.0+0.0im]/sqrt(2)
    for j in 1:n-1
        push!(tns,H)
        for k in j+1:n
            push!(tns, [1.0+0.0im 1.0+0.0im; 1.0+0.0im exp(im*π*theta/2^(k-j))])
        end
    end
    push!(tns,H)
    return tns
end

tn2 = qft_tensor(qubit_num,0.8)
isapprox(tn2, tn.tensors;atol = 1e-10)

