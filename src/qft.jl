# ============================================================================
# Quantum Fourier Transform Circuit Construction
# ============================================================================
# This file provides the QFT circuit construction and tensor network representation.

"""
    qft_code(m::Int, n::Int; inverse=false)

Generate an optimized tensor network representation of the QFT circuit.

# Arguments
- `m::Int`: Number of qubits for row indices
- `n::Int`: Number of qubits for column indices
- `inverse::Bool`: If true, generate inverse transform code

# Returns
- `optcode::AbstractEinsum`: Optimized einsum contraction code
- `tensors::Vector`: Initial circuit parameters (unitary matrices)
"""
function qft_code(m::Int, n::Int; inverse=false)
    qc1 = Yao.EasyBuild.qft_circuit(m)
    qc2 = Yao.EasyBuild.qft_circuit(n)
    qc = chain(subroutine(m + n, qc1, 1:m), subroutine(m + n, qc2, m+1:m+n))
    tn = yao2einsum(qc; optimizer=nothing)
    perm_vec = sortperm(tn.tensors, by= x-> !(x ≈ mat(H)))
    ixs = tn.code.ixs[perm_vec]
    tensors = tn.tensors[perm_vec]
    if inverse
        code_reorder = DynamicEinCode([ixs..., tn.code.iy[m+n+1:end]], tn.code.iy[1:m+n])
    else
        code_reorder = DynamicEinCode([ixs..., tn.code.iy[1:m+n]], tn.code.iy[m+n+1:end])
    end
    optcode = optimize_code(code_reorder, uniformsize(tn.code, 2), TreeSA())
    return optcode, tensors
end

# ============================================================================
# Manifold Structure and Conversions
# ============================================================================

"""
    generate_manifold(tensors)

Generate product manifold for m+n-qubit QFT parameters.
Returns a product of U(2) manifolds for Hadamard gates and U(1)^4 for controlled gates.
"""
function generate_manifold(tensors)
    M2 = UnitaryMatrices(2)
    M1 = PowerManifold(UnitaryMatrices(1), 4)
    return ProductManifold(map(x -> x ≈ mat(H) ? M2 : M1, tensors)...)
end

"""
    tensors2point(tensors, M::ProductManifold)

Convert circuit tensors (unitary matrices) to a point on the product manifold.
"""
function tensors2point(tensors, M::ProductManifold)
    return ArrayPartition(
        [mi isa UnitaryMatrices ? tensors[j] : 
         [tensors[j][1, 1];;; tensors[j][1, 2];;; 
          tensors[j][2, 1];;; tensors[j][2, 2]] 
         for (j, mi) in enumerate(M.manifolds)]...
    )
end

"""
    point2tensors(p, M)

Convert a manifold point back to circuit tensors (unitary matrices).
"""
function point2tensors(p, M)
    return [mi isa UnitaryMatrices ? p.x[j] : reshape(p.x[j], 2, 2) for (j, mi) in enumerate(M.manifolds)]
end

# ============================================================================
# Training with Single Image
# ============================================================================

"""
    fft_with_training(m::Int, n::Int, pic::Matrix, loss::AbstractLoss; steps::Int=1000, use_cuda::Bool=false)

Train a parametric 2D quantum DFT circuit using Riemannian gradient descent.

# Arguments
- `m::Int`: Number of qubits for row dimension (image height = 2^m)
- `n::Int`: Number of qubits for column dimension (image width = 2^n)
- `pic::Matrix`: Input signal (size must be 2^m × 2^n)
- `loss::AbstractLoss`: Loss function (e.g., `L1Norm()`)
- `steps::Int=1000`: Maximum optimization iterations
- `use_cuda::Bool=false`: Whether to use CUDA acceleration (not yet implemented)

# Returns
- Optimized parameters on the manifold (use `point2tensors` to convert to tensors)

# Example
```julia
m, n = 6, 6  # For 64×64 image
pic = rand(ComplexF64, 2^m, 2^n)
# L1 norm loss
theta = fft_with_training(m, n, pic, L1Norm(); steps=200)
# MSE loss with truncation (keep top 100 elements)
theta = fft_with_training(m, n, pic, MSELoss(100); steps=200)
```
"""
function fft_with_training(m::Int, n::Int, pic::Matrix, loss::AbstractLoss; steps::Int=1000, use_cuda::Bool=false)
    optcode, tensors = qft_code(m, n)
    M = generate_manifold(tensors)
    
    # Generate inverse code if needed for MSE loss
    inverse_code = nothing
    if loss isa MSELoss
        inverse_code, _ = qft_code(m, n; inverse=true)
    end
    
    f(M, p) = loss_function(point2tensors(p, M), m, n, optcode, pic, loss; inverse_code=inverse_code)
    grad_f2(M, p) = ManifoldDiff.gradient(M, x->f(M, x), p, RiemannianProjectionBackend(AutoZygote()))
    
    result = gradient_descent(
        M, f, grad_f2, tensors2point(tensors, M);
        debug = [:Iteration, (:Change, "|Δp|: %1.9f |"),
                (:Cost, " F(x): %1.11f | "), "\n", :Stop],
        stopping_criterion = StopAfterIteration(steps) | StopWhenGradientNormLess(1e-5)
    )
    
    return result
end

# ============================================================================
# Helper Functions
# ============================================================================

"""
    ft_mat(tensors::Vector, code::AbstractEinsum, m::Int, n::Int, pic::Matrix)

Apply 2D DFT to an image using the trained circuit parameters.

# Arguments
- `tensors::Vector`: Circuit tensors (unitary matrices)
- `code::AbstractEinsum`: Optimized einsum code
- `m::Int`: Number of qubits for row dimension
- `n::Int`: Number of qubits for column dimension
- `pic::Matrix`: Input image (size 2^m × 2^n)

# Returns
- Transformed image in frequency domain (size 2^m × 2^n)
"""
function ft_mat(tensors::Vector, code::OMEinsum.AbstractEinsum, m::Int, n::Int, pic::Matrix)
    @assert size(pic) == (2^m, 2^n) "Input size must be 2^m × 2^n"
    return reshape(code(tensors..., reshape(pic, fill(2, m+n)...)), 2^m, 2^n)
end

"""
    ift_mat(tensors::Vector, code::AbstractEinsum, m::Int, n::Int, pic::Matrix)

Apply inverse 2D DFT using the inverse QFT circuit with trained parameters.

# Arguments
- `tensors::Vector`: Circuit tensors (unitary matrices) from inverse QFT (use qft_code(m, n; inverse=true))
- `code::AbstractEinsum`: Optimized einsum code from inverse QFT
- `m::Int`: Number of qubits for row dimension
- `n::Int`: Number of qubits for column dimension
- `pic::Matrix`: Input in frequency domain (size 2^m × 2^n)

# Returns
- Transformed image in spatial domain (size 2^m × 2^n)
"""
function ift_mat(tensors::Vector, code::OMEinsum.AbstractEinsum, m::Int, n::Int, pic::Matrix)
    @assert size(pic) == (2^m, 2^n) "Input size must be 2^m × 2^n"
    return reshape(code(tensors..., reshape(pic, fill(2, m+n)...)), 2^m, 2^n)
end
