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

# Loss function exports
export AbstractLoss, L1Norm, L2Norm, MSELoss
export topk_truncate, loss_function

# QFT circuit exports
export fft_with_training, qft_code, ft_mat, ift_mat
export point2tensors, tensors2point, generate_manifold

# Entangled QFT circuit exports
export entangled_qft_code, entanglement_gate
export get_entangle_tensor_indices, extract_entangle_phases

# Sparse basis exports
export AbstractSparseBasis, QFTBasis, EntangledQFTBasis
export forward_transform, inverse_transform, image_size, num_parameters, basis_hash
export get_manifold
export num_entangle_parameters, get_entangle_phases

# Training exports
export train_basis

# Serialization exports
export save_basis, load_basis, basis_to_dict, dict_to_basis

# Compression exports
export CompressedImage, compress, compress_with_k
export recover, save_compressed, load_compressed, compression_stats

# Include files in dependency order:
# 1. Loss functions (no internal dependencies)
include("loss.jl")
# 2. QFT circuit (uses loss_function from loss.jl for fft_with_training)
include("qft.jl")
# 3. Entangled QFT circuit (standalone circuit code)
include("entangled_qft.jl")
# 4. Basis implementations (uses qft_code and entangled_qft_code)
include("basis.jl")
# 5. Training (uses basis types and loss functions)
include("training.jl")
# 6. Serialization (uses basis types)
include("serialization.jl")
# 7. Compression (uses basis types)
include("compression.jl")

end # module
