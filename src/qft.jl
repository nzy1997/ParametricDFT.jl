_A(i, j) = control(i, j=>shift(2π/(1<<(i-j+1))))
# _A(n, i, j) = put(n,(i,j)=>matblock(rand_unitary(4)))
_B(n, k) = chain(n, j==k ? put(k=>H) : _A( j, k) for j in k:n)
_qft(n) = chain(_B(n, k) for k in 1:n)

function qft_code(qubit_num::Int)
    qc = _qft(qubit_num)
    tn = yao2einsum(qc)
    code = OMEinsum.flatten(tn.code)
    perm_vec = sort_order(qubit_num)

    @assert length(perm_vec) == length(code.ixs)

    ixs = code.ixs[perm_vec]
    tensors = tn.tensors[perm_vec]
    code_reorder = DynamicEinCode(ixs, code.iy)

    optcode = optimize_code(code_reorder,uniformsize(code, 2), TreeSA())
    return optcode, tensors
end

abstract type AbstractLoss end
struct L1Norm <: AbstractLoss end


function loss_function(tensors,n::Int,optcode::OMEinsum.AbstractEinsum, pic::Vector,loss::AbstractLoss)
    @assert length(pic) == 2^n
    fft_mat = optcode(tensors...)
    fft_mat = reshape(fft_mat, 2^(n),2^(n))
    fft_pic = fft_mat * pic
    return _loss_function(fft_pic,pic,loss)
end

_loss_function(fft_res,pic,loss::L1Norm) = sum(abs.(fft_res))

function fft_with_training(n::Int, pic::Vector,loss::AbstractLoss)
    optcode, tensors = qft_code(n)
    M = generate_manifold(n)
    f(M,p) = loss_function(point2tensors(p,n),n,optcode,pic,loss)
    grad_f2(M,p) = ManifoldDiff.gradient(M, x->f(M,x), p, RiemannianProjectionBackend(AutoZygote()))
    
    m = gradient_descent(M, f, grad_f2, tensors2point(tensors,n);
        debug=[:Iteration,(:Change, "|Δp|: %1.9f |"),
            (:Cost, " F(x): %1.11f | "), "\n", :Stop]
      )
      return m
end

function generate_manifold(n::Int)
    M2 = UnitaryMatrices(2)
    M1 = PowerManifold(UnitaryMatrices(1),4)
    return ProductManifold(fill(M2,n)...,fill(M1,n*(n+1) ÷ 2-n)...)
end

function tensors2point(tensors,n::Int)
    return ArrayPartition(tensors[1:n]...,[[tensors[count_num][1,1];;;tensors[count_num][1,2];;;tensors[count_num][2,1];;;tensors[count_num][2,2]] for count_num in n+1:n*(n+1) ÷ 2]...)
end

function point2tensors(p,n::Int)
    # tensors = collect(p.x)
    # count_num = 0
    # for j in 1:n
    #     for i in j:n
    #         if i == j
    #             count_num += 1
    #         else
    #             count_num += 1
    #             tensors[count_num] = reshape(tensors[count_num],2,2)
    #         end
    #     end
    # end
    return [j< n+1 ? p.x[j] : reshape(p.x[j],2,2) for j in 1:n*(n+1) ÷ 2]
end

function sort_order(n::Int)
    hcount = 0
    totalcount = 0
    perm_vec = Vector{Int64}()
    for j in 1:n
        for i in j:n
            totalcount += 1
            if i == j
                hcount += 1
                push!(perm_vec, totalcount)
            end
        end
    end
    totalcount = 0
    for j in 1:n
        for i in j:n
            totalcount += 1
            if i != j
                hcount += 1
                push!(perm_vec, totalcount)
            end
        end
    end
    return perm_vec
end