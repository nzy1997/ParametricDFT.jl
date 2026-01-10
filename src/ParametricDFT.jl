module ParametricDFT

using Yao
using OMEinsum
using Manifolds
using ManifoldDiff
using Zygote
using ADTypes
using Manopt
using RecursiveArrayTools
using ChainRulesCore
using SHA
using JSON3
using StructTypes
using Random

export fft_with_training, qft_code, ft_mat, ift_mat
export point2tensors, tensors2point, generate_manifold
export AbstractLoss, L1Norm, L2Norm, MSELoss

# Sparse basis exports
export AbstractSparseBasis, QFTBasis
export forward_transform, inverse_transform, image_size, num_parameters, basis_hash
export get_manifold

# Entangled QFT exports
export entangled_qft_code, entanglement_gate
export get_entangle_tensor_indices, extract_entangle_phases
export EntangledQFTBasis
export num_entangle_parameters, get_entangle_phases

# Training exports
export train_basis

# Serialization exports
export save_basis, load_basis, basis_to_dict, dict_to_basis

# Compression exports
export CompressedImage, compress, compress_with_k
export recover, save_compressed, load_compressed, compression_stats

include("qft.jl")
include("basis.jl")
include("entangled_qft.jl")
include("training.jl")
include("serialization.jl")
include("compression.jl")

end # module
