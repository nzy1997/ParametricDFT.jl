# ============================================================================
# Quantum Fourier Transform Circuit Construction
# ============================================================================

# Controlled phase shift gate: phase angle is 2π/2^(i-j+1)
_A(i, j) = control(i, j=>shift(2π/(1<<(i-j+1))))

# k-th stage of QFT: Hadamard on qubit k, then controlled phases from qubits j>k
_B(n, k) = chain(n, j==k ? put(k=>H) : _A(j, k) for j in k:n)

# Construct full n-qubit QFT circuit
_qft(n) = chain(_B(n, k) for k in 1:n)

"""
    qft_code(qubit_num::Int)

Generate an optimized tensor network representation of the QFT circuit.

# Arguments
- `qubit_num::Int`: Number of qubits

# Returns
- `optcode::AbstractEinsum`: Optimized einsum contraction code
- `tensors::Vector`: Initial circuit parameters (unitary matrices)
"""
function qft_code(qubit_num::Int)
    qc = _qft(qubit_num)
    tn = yao2einsum(qc)
    code = OMEinsum.flatten(tn.code)
    perm_vec = sort_order(qubit_num)
    
    @assert length(perm_vec) == length(code.ixs)
    
    ixs = code.ixs[perm_vec]
    tensors = tn.tensors[perm_vec]
    code_reorder = DynamicEinCode(ixs, code.iy)
    
    optcode = optimize_code(code_reorder, uniformsize(code, 2), TreeSA())
    return optcode, tensors
end

# ============================================================================
# Loss Functions
# ============================================================================

"""
    AbstractLoss

Abstract base type for loss functions. Custom loss functions should inherit from 
this type and implement `_loss_function(fft_result, input, loss)`.
"""
abstract type AbstractLoss end

"""
    L1Norm <: AbstractLoss

L1 norm loss: minimizes sum of absolute values in the transformed domain.
This encourages sparsity in the frequency representation.
"""
struct L1Norm <: AbstractLoss end

"""
    loss_function(tensors, n::Int, optcode::AbstractEinsum, pic::Vector, loss::AbstractLoss)

Compute the loss for current circuit parameters.

# Arguments
- `tensors::Vector`: Circuit parameters (unitary matrices)
- `n::Int`: Number of qubits
- `optcode::AbstractEinsum`: Optimized einsum code
- `pic::Vector`: Input signal (length must be 2^n)
- `loss::AbstractLoss`: Loss function type

# Returns
- Loss value
"""
function loss_function(tensors, n::Int, optcode::OMEinsum.AbstractEinsum, pic::Vector, loss::AbstractLoss)
    @assert length(pic) == 2^n "Input vector length must be 2^n"
    mat1 = ft_mat(tensors, optcode, n)
    fft_pic = mat1 * pic
    return _loss_function(fft_pic, pic, loss)
end

# Compute L1 norm: sum of absolute values
_loss_function(fft_res, pic, loss::L1Norm) = sum(abs.(fft_res))

# ============================================================================
# Training
# ============================================================================

"""
    fft_with_training(n::Int, pic::Vector, loss::AbstractLoss; steps::Int=1000)

Train a parametric quantum DFT circuit using Riemannian gradient descent.

# Arguments
- `n::Int`: Number of qubits (input length must be 2^n)
- `pic::Vector`: Input signal
- `loss::AbstractLoss`: Loss function (e.g., `L1Norm()`)
- `steps::Int=1000`: Maximum optimization iterations

# Returns
- Optimized parameters on the manifold (use `point2tensors` to convert to tensors)

# Example
```julia
theta = fft_with_training(8, rand(256), L1Norm(); steps=200)
```
"""
function fft_with_training(n::Int, pic::Vector, loss::AbstractLoss; steps::Int=1000)
    optcode, tensors = qft_code(n)
    M = generate_manifold(n)
    
    f(M, p) = loss_function(point2tensors(p, n), n, optcode, pic, loss)
    grad_f2(M, p) = ManifoldDiff.gradient(M, x->f(M, x), p, RiemannianProjectionBackend(AutoZygote()))
    
    m = gradient_descent(
        M, f, grad_f2, tensors2point(tensors, n);
        debug = [:Iteration, (:Change, "|Δp|: %1.9f |"),
                (:Cost, " F(x): %1.11f | "), "\n", :Stop],
        stopping_criterion = StopAfterIteration(steps) | StopWhenGradientNormLess(1e-5)
    )
    
    return m
end

# ============================================================================
# Manifold Structure and Conversions
# ============================================================================

"""
    generate_manifold(n::Int)

Generate product manifold for n-qubit QFT parameters.
Returns a product of U(2) manifolds for Hadamard gates and U(1)^4 for controlled gates.
"""
function generate_manifold(n::Int)
    M2 = UnitaryMatrices(2)
    M1 = PowerManifold(UnitaryMatrices(1), 4)
    return ProductManifold(fill(M2, n)..., fill(M1, n*(n+1) ÷ 2 - n)...)
end

"""
    tensors2point(tensors, n::Int)

Convert circuit tensors (unitary matrices) to a point on the product manifold.
"""
function tensors2point(tensors, n::Int)
    return ArrayPartition(
        tensors[1:n]...,
        [[tensors[count_num][1, 1];;; tensors[count_num][1, 2];;; 
          tensors[count_num][2, 1];;; tensors[count_num][2, 2]] 
         for count_num in n+1:n*(n+1) ÷ 2]...
    )
end

"""
    point2tensors(p, n::Int)

Convert a manifold point back to circuit tensors (unitary matrices).
"""
function point2tensors(p, n::Int)
    return [j < n+1 ? p.x[j] : reshape(p.x[j], 2, 2) for j in 1:n*(n+1) ÷ 2]
end

# ============================================================================
# Helper Functions
# ============================================================================

# Generate permutation vector to reorder tensors: Hadamard gates first, then controlled gates
function sort_order(n::Int)
    hcount = 0
    totalcount = 0
    perm_vec = Vector{Int64}()
    
    # First pass: collect Hadamard gate indices (diagonal elements)
    for j in 1:n
        for i in j:n
            totalcount += 1
            if i == j
                hcount += 1
                push!(perm_vec, totalcount)
            end
        end
    end
    
    # Second pass: collect controlled gate indices (off-diagonal elements)
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

"""
    ft_mat(theta::Vector, code::AbstractEinsum, n::Int)

Construct DFT matrix from circuit parameters by contracting the tensor network.
"""
function ft_mat(theta::Vector, code::OMEinsum.AbstractEinsum, n::Int)
    return reshape(code(theta...), 2^n, 2^n)
end

"""
    ift_mat(tensors::Vector, code::AbstractEinsum, n::Int)

Construct inverse DFT matrix from circuit parameters.
"""
function ift_mat(tensors::Vector, code::OMEinsum.AbstractEinsum, n::Int)
    return reshape(code(tensors...), 2^n, 2^n)
end
