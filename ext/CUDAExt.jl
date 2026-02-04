module CUDAExt

using CUDA
using ParametricDFT

"""
    to_device(x, ::Val{:gpu})

Move array `x` to GPU using CUDA.jl. Converts to CuArray for GPU-accelerated computation.
"""
function ParametricDFT.to_device(x::AbstractArray, ::Val{:gpu})
    return CUDA.cu(x)
end

# Scalars and non-array types pass through unchanged
function ParametricDFT.to_device(x, ::Val{:gpu})
    return x
end

end # module
