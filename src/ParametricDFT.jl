module ParametricDFT

using Yao
using OMEinsum
using Zygote
using ChainRulesCore
using SHA
using JSON3
using StructTypes
using Random
using CairoMakie
using LinearAlgebra

# Loss function exports
export AbstractLoss, L1Norm, MSELoss
export topk_truncate, loss_function

# QFT circuit exports
export qft_code, ft_mat, ift_mat

# Entangled QFT circuit exports
export entangled_qft_code, entanglement_gate
export get_entangle_tensor_indices, extract_entangle_phases

# TEBD circuit exports
export tebd_code
export get_tebd_gate_indices, extract_tebd_phases

# MERA circuit exports
export mera_code
export get_mera_gate_indices, extract_mera_phases

# Sparse basis exports
export AbstractSparseBasis, QFTBasis, EntangledQFTBasis, TEBDBasis, MERABasis
export forward_transform, inverse_transform, image_size, num_parameters, basis_hash
export num_entangle_parameters, get_entangle_phases
export num_gates, get_phases

# Manifold exports
export AbstractRiemannianManifold, UnitaryManifold, PhaseManifold
export classify_manifold, group_by_manifold

# Optimizer exports
export AbstractRiemannianOptimizer, RiemannianGD, RiemannianAdam
export optimize!

# Training exports
export train_basis, save_loss_history, load_loss_history
export to_device

# Visualization exports
export TrainingHistory, plot_training_loss, plot_training_loss_steps,
       plot_training_comparison, plot_training_comparison_steps,
       plot_training_grid, save_training_plots, ema_smooth

# Serialization exports
export save_basis, load_basis, basis_to_dict, dict_to_basis

# Compression exports
export CompressedImage, compress, compress_with_k
export recover, save_compressed, load_compressed, compression_stats

# Circuit visualization exports
export plot_circuit

# Include files in dependency order:
# 0. Einsum code cache (used by circuit code generators)
include("einsum_cache.jl")
# 1. Loss functions (no internal dependencies)
include("loss.jl")
# 2. QFT circuit
include("qft.jl")
# 3. Entangled QFT circuit (standalone circuit code)
include("entangled_qft.jl")
# 3b. TEBD circuit (standalone circuit code)
include("tebd.jl")
# 3c. MERA circuit (standalone circuit code)
include("mera.jl")
# 4. Basis implementations (uses qft_code and entangled_qft_code)
include("basis.jl")
# 5. Manifold abstraction (abstract types, batched ops, stack/unstack)
include("manifolds.jl")
# 6. Riemannian optimizers (uses manifold API)
include("optimizers.jl")
# 7. Training (uses basis types, loss functions, and optimizers)
include("training.jl")
# 8. Serialization (uses basis types)
include("serialization.jl")
# 9. Compression (uses basis types)
include("compression.jl")
# 10. Visualization (uses CairoMakie for training loss visualization)
include("visualization.jl")
# 11. Circuit visualization (uses CairoMakie for circuit diagrams)
include("circuit_visualization.jl")

end # module
