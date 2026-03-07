# ================================================================================
# Circuit Visualization Examples
# ================================================================================
# Generates circuit diagrams for all basis types using the plot_circuit API.
#
# Run with: julia --project=examples examples/circuit_visualization.jl
# ================================================================================

using ParametricDFT
using CairoMakie

CairoMakie.activate!(type = "png", px_per_unit = 2)

# ================================================================================
# Main
# ================================================================================

function main()
    output_dir = joinpath(@__DIR__, "CircuitDiagrams")
    mkpath(output_dir)

    println("="^60)
    println("Generating Circuit Diagrams")
    println("="^60)
    println("\nOutput: $output_dir\n")

    # QFT circuits
    println("1. 2D QFT (3×3)")
    plot_circuit(QFTBasis(3, 3); output_path=joinpath(output_dir, "qft_2d_3x3.png"))

    # Entangled QFT circuits with different positions
    for (i, pos) in enumerate((:front, :middle, :back))
        println("$(i+1). Entangled QFT (3×3, $pos)")
        basis = EntangledQFTBasis(3, 3;
            entangle_phases=[π/4, π/3, π/6], entangle_position=pos)
        plot_circuit(basis; output_path=joinpath(output_dir, "entangled_qft_3x3_$pos.png"))
    end

    # TEBD circuits
    println("5. TEBD Circuit (3×3)")
    plot_circuit(TEBDBasis(3, 3); output_path=joinpath(output_dir, "tebd_3x3.png"))

    println("6. TEBD Circuit (4×4, trained)")
    basis_tebd = TEBDBasis(4, 4; phases=[π/4, π/3, π/2, π/6, π/5, π/7, π/8, π/9])
    plot_circuit(basis_tebd; output_path=joinpath(output_dir, "tebd_4x4_trained.png"))

    # MERA circuits
    println("7. MERA Circuit (2×2)")
    plot_circuit(MERABasis(2, 2); output_path=joinpath(output_dir, "mera_2x2.png"))

    println("8. MERA Circuit (4×4)")
    plot_circuit(MERABasis(4, 4); output_path=joinpath(output_dir, "mera_4x4.png"))

    println("\n" * "="^60)
    println("Done! Generated 8 circuit diagrams.")
    println("="^60)
end

main()
