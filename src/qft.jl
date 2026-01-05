# ============================================================================
# Quantum Fourier Transform Circuit Construction
# ============================================================================

"""
    qft_code(m::Int, n::Int)

Generate an optimized tensor network representation of the QFT circuit.

# Arguments
- `m::Int`: Number of qubits for row indices
- `n::Int`: Number of qubits for column indices

# Returns
- `optcode::AbstractEinsum`: Optimized einsum contraction code
- `tensors::Vector`: Initial circuit parameters (unitary matrices)
"""
function qft_code(m::Int, n::Int; inverse=false)
    qc1 = Yao.EasyBuild.qft_circuit(m)
    qc2 = Yao.EasyBuild.qft_circuit(n)
    qc = chain(subroutine(m + n, qc1, 1:m), subroutine(m + n, qc2, m+1:m+n))
    # if inverse
    #     qc = qc'
    # end
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
# Loss Functions
# ============================================================================

"""
    AbstractLoss

Abstract base type for loss functions. Custom loss functions should inherit from 
this type and implement `_loss_function(fft_result, input, loss)`.
"""
abstract type AbstractLoss end



"""
    topk_truncate(x::AbstractMatrix, k::Integer)

Return a matrix where frequency-dependent truncation is applied for image compression.
Low-frequency components (near center) are kept with higher priority, while high-frequency
components (away from center) are kept with lower priority.

This is more appropriate for image compression than global top-k selection, as it
preserves more low-frequency information which contains most of the image structure.
"""
function topk_truncate(x::AbstractMatrix{T}, k::Integer) where {T}
    m, n = size(x)
    k2 = min(Int(k), length(x))
    
    # Calculate frequency distances from center (DC component)
    center_i, center_j = (m + 1) ÷ 2, (n + 1) ÷ 2
    max_dist = sqrt((m/2)^2 + (n/2)^2)
    
    # Create frequency-weighted scores: combine magnitude with frequency position
    # Lower frequency (closer to center) gets higher weight
    scores = zeros(Float64, m, n)
    mags = abs.(x)
    
    @inbounds for j in 1:n, i in 1:m
        freq_dist = sqrt((i - center_i)^2 + (j - center_j)^2)
        # Weight: higher for low frequencies (smaller distance)
        # Use inverse distance weighting, normalized by max distance
        freq_weight = 1.0 - (freq_dist / max_dist) * 0.5  # Scale factor to balance magnitude vs frequency
        scores[i, j] = mags[i, j] * (1.0 + freq_weight)
    end
    
    # Select top k based on weighted scores
    scores_flat = vec(scores)
    idx = partialsortperm(scores_flat, k2, rev=true)
    
    y = zeros(T, m, n)
    @inbounds for flat_idx in idx
        # Julia arrays are column-major: flat_idx = (j-1)*m + i
        i = ((flat_idx - 1) % m) + 1
        j = ((flat_idx - 1) ÷ m) + 1
        y[i, j] = x[i, j]
    end
    return y
end

function ChainRulesCore.rrule(::typeof(topk_truncate), x::AbstractMatrix{T}, k::Integer) where {T}
    m, n = size(x)
    k2 = min(Int(k), length(x))
    
    # Calculate frequency distances from center
    center_i, center_j = (m + 1) ÷ 2, (n + 1) ÷ 2
    max_dist = sqrt((m/2)^2 + (n/2)^2)
    
    # Create frequency-weighted scores
    scores = zeros(Float64, m, n)
    mags = abs.(x)
    
    @inbounds for j in 1:n, i in 1:m
        freq_dist = sqrt((i - center_i)^2 + (j - center_j)^2)
        freq_weight = 1.0 - (freq_dist / max_dist) * 0.5
        scores[i, j] = mags[i, j] * (1.0 + freq_weight)
    end
    
    # Select top k based on weighted scores
    scores_flat = vec(scores)
    idx = partialsortperm(scores_flat, k2, rev=true)
    
    y = zeros(T, m, n)
    @inbounds for flat_idx in idx
        # Julia arrays are column-major: flat_idx = (j-1)*m + i
        i = ((flat_idx - 1) % m) + 1
        j = ((flat_idx - 1) ÷ m) + 1
        y[i, j] = x[i, j]
    end
    
    function pullback(ȳ)
        x̄ = zeros(T, m, n)
        @inbounds for flat_idx in idx
            # Julia arrays are column-major: flat_idx = (j-1)*m + i
            i = ((flat_idx - 1) % m) + 1
            j = ((flat_idx - 1) ÷ m) + 1
            x̄[i, j] = ȳ[i, j]
        end
        return (ChainRulesCore.NoTangent(), x̄, ChainRulesCore.NoTangent())
    end
    return y, pullback
end

"""
    L1Norm <: AbstractLoss

L1 norm loss: minimizes sum of absolute values in the transformed domain.
This encourages sparsity in the frequency representation.
"""
struct L1Norm <: AbstractLoss end

"""
    L2Norm <: AbstractLoss

L2 norm loss: minimizes sum of squared magnitudes in the transformed domain.
This encourages energy concentration (squared magnitude) with smoother gradients
compared to L1 norm, and less aggressive sparsity promotion than L1.
"""
struct L2Norm <: AbstractLoss end

"""
    MSELoss <: AbstractLoss

Mean Squared Error loss with truncation: minimizes reconstruction error after
forward transform, truncation (keeping top k elements), and inverse transform.

# Fields
- `k::Int`: Number of top elements to keep after truncation (by magnitude)

# Equation
L(θ) = Σᵢ ||xᵢ - T(θ)⁻¹(truncate(T(θ)(xᵢ), k))||²₂

This loss encourages the circuit to learn a representation where the top k
frequency components capture most of the signal information.
"""
struct MSELoss <: AbstractLoss
    k::Int
    MSELoss(k::Int) = k > 0 ? new(k) : error("k must be positive")
end

"""
    loss_function(tensors, m::Int, n::Int, optcode::AbstractEinsum, pic::Matrix, loss::AbstractLoss; inverse_code=nothing)

Compute the loss for current circuit parameters.

# Arguments
- `tensors::Vector`: Circuit parameters (unitary matrices)
- `m::Int`: Number of qubits for row dimension
- `n::Int`: Number of qubits for column dimension
- `optcode::AbstractEinsum`: Optimized einsum code (forward transform)
- `pic::Matrix`: Input signal (size must be 2^m × 2^n)
- `loss::AbstractLoss`: Loss function type
- `inverse_code::AbstractEinsum`: Optional inverse einsum code (required for MSELoss)

# Returns
- Loss value
"""
function loss_function(tensors::AbstractVector, m::Int, n::Int, optcode::OMEinsum.AbstractEinsum, pic::Matrix, loss::AbstractLoss; inverse_code=nothing)
    # Avoid splatting an AbstractVector during AD; Zygote may produce tuple tangents for varargs.
    # We delegate to the Tuple method for a stable tangent type.
    return loss_function(Tuple(tensors), m, n, optcode, pic, loss; inverse_code=inverse_code)
end

function loss_function(tensors::Tuple, m::Int, n::Int, optcode::OMEinsum.AbstractEinsum, pic::Matrix, loss::AbstractLoss; inverse_code=nothing)
    @assert (size(pic) == (2^m, 2^n)) "Input matrix size must be 2^m × 2^n"
    fft_pic = reshape(optcode(tensors..., reshape(pic, fill(2, m+n)...)), 2^m, 2^n)
    return _loss_function(fft_pic, pic, loss, tensors, m, n, inverse_code)
end

# Compute L1 norm: sum of absolute values
function _loss_function(fft_res, pic, loss::L1Norm, tensors, m, n, inverse_code)
    return sum(abs.(fft_res))
end

# Compute L2 norm: sum of squared magnitudes
function _loss_function(fft_res, pic, loss::L2Norm, tensors, m, n, inverse_code)
    return sum(abs2.(fft_res))
end

# Compute MSE with truncation: ||x - T⁻¹(truncate(T(x), k))||²₂
function _loss_function(fft_res, pic, loss::MSELoss, tensors, m, n, inverse_code)
    if inverse_code === nothing
        error("MSELoss requires inverse_code to be provided")
    end
    
    # Truncate: use frequency-dependent truncation for image compression
    # This keeps more low-frequency components and fewer high-frequency components
    k = min(loss.k, length(fft_res))
    fft_truncated = topk_truncate(fft_res, k)
    
    # Apply inverse transform
    reconstructed = reshape(inverse_code(conj.(tensors)..., reshape(fft_truncated, fill(2, m+n)...)), 2^m, 2^n)
    
    # Compute MSE: sum of squared differences
    return sum(abs2.(pic .- reconstructed))
end



# ============================================================================
# Training
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
