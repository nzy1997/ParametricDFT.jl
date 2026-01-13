# ================================================================================
# Tensor Network Circuit Visualization
# ================================================================================
# Generates professional circuit diagrams for QFT and Entangled QFT circuits.
#
# Run with: julia --project=examples examples/tensor_network_visualization.jl
# ================================================================================

using ParametricDFT
using CairoMakie
using OMEinsum

CairoMakie.activate!(type = "png", px_per_unit = 2)

# ================================================================================
# Configuration
# ================================================================================

const COLORS = (
    wire = "#333333",
    hadamard = "#4285F4",      # Blue
    phase_gate = "#34A853",    # Green  
    entangle_gate = "#EA4335", # Red
    text = "#333333",
    label = "#666666"
)

const GATE_SIZE = 0.6
const WIRE_WIDTH = 2
const QUBIT_SPACING = 1.0
const GATE_SPACING = 1.2

# ================================================================================
# Drawing Primitives
# ================================================================================

"""Draw a qubit wire with labels."""
function draw_qubit!(ax, y, x_start, x_end, label_in, label_out)
    lines!(ax, [x_start, x_end], [y, y], color=COLORS.wire, linewidth=WIRE_WIDTH)
    text!(ax, x_start - 0.5, y, text=label_in, align=(:right, :center), 
          fontsize=14, font=:bold, color=COLORS.text)
    text!(ax, x_end + 0.5, y, text=label_out, align=(:left, :center), 
          fontsize=14, color=COLORS.label)
end

"""Draw a gate box with label."""
function draw_gate!(ax, x, y, label; color=COLORS.hadamard, size=GATE_SIZE)
    half = size / 2
    poly!(ax, Point2f[(x-half, y-half), (x+half, y-half), (x+half, y+half), (x-half, y+half)],
          color=color, strokewidth=2, strokecolor=:white)
    text!(ax, x, y, text=label, align=(:center, :center), 
          fontsize=14, font=:bold, color=:white)
end

"""Draw a controlled gate between two qubits."""
function draw_controlled_gate!(ax, x, y_ctrl, y_target, label; color=COLORS.phase_gate)
    scatter!(ax, [x], [y_ctrl], markersize=12, color=COLORS.wire)
    lines!(ax, [x, x], [y_ctrl, y_target], color=COLORS.wire, linewidth=WIRE_WIDTH)
    draw_gate!(ax, x, y_target, label; color=color)
end

"""Draw an entanglement gate connecting two qubits with phase annotation."""
function draw_entangle_gate!(ax, x, y1, y2, label, phase)
    scatter!(ax, [x, x], [y1, y2], markersize=12, color=COLORS.entangle_gate)
    lines!(ax, [x, x], [y1, y2], color=COLORS.entangle_gate, linewidth=WIRE_WIDTH+1)
    
    y_mid = (y1 + y2) / 2
    draw_gate!(ax, x, y_mid, label; color=COLORS.entangle_gate, size=GATE_SIZE*0.8)
    
    phase_str = "φ=$(round(phase, digits=2))"
    text!(ax, x + GATE_SIZE, y_mid, text=phase_str, align=(:left, :center), 
          fontsize=10, color=COLORS.label)
end

# ================================================================================
# Circuit Structure Definition
# ================================================================================

"""
    CircuitSpec

Specification for a quantum circuit to be drawn.
"""
struct CircuitSpec
    n_row_qubits::Int           # Number of row qubits (x)
    n_col_qubits::Int           # Number of column qubits (y), 0 for 1D QFT
    entangle_phases::Vector{Float64}  # Entanglement phases (empty for standard QFT)
    title::String
end

# Convenience constructors
CircuitSpec(m::Int; title="QFT Circuit") = CircuitSpec(m, 0, Float64[], title)
CircuitSpec(m::Int, n::Int; title="2D QFT Circuit") = CircuitSpec(m, n, Float64[], title)
function CircuitSpec(m::Int, n::Int, phases::Vector{<:Real}; title="Entangled QFT Circuit")
    CircuitSpec(m, n, Float64.(phases), title)
end

is_1d(spec::CircuitSpec) = spec.n_col_qubits == 0
is_entangled(spec::CircuitSpec) = !isempty(spec.entangle_phases)
total_qubits(spec::CircuitSpec) = spec.n_row_qubits + spec.n_col_qubits
n_entangle(spec::CircuitSpec) = length(spec.entangle_phases)

# ================================================================================
# Unified Circuit Drawing
# ================================================================================

"""
    draw_qft_layer!(ax, x_start, qubits, y_offset)

Draw QFT gates (H + M gates) for a group of qubits.
Returns the x position after all gates.
"""
function draw_qft_layer!(ax, x_start, n_qubits, y_offset)
    x_pos = x_start
    
    for q in 1:n_qubits
        y_q = y_offset - q * QUBIT_SPACING
        
        # Hadamard gate
        draw_gate!(ax, x_pos, y_q, "H"; color=COLORS.hadamard)
        x_pos += GATE_SPACING
        
        # Controlled phase gates
        for target in (q+1):n_qubits
            y_target = y_offset - target * QUBIT_SPACING
            k = target - q + 1
            draw_controlled_gate!(ax, x_pos, y_q, y_target, "M$k"; color=COLORS.phase_gate)
            x_pos += GATE_SPACING
        end
    end
    
    return x_pos
end

"""
    draw_entangle_layer!(ax, x_start, m, n, phases)

Draw entanglement gates between corresponding row and column qubits.
Returns the x position after all gates.
"""
function draw_entangle_layer!(ax, x_start, m, n, phases)
    x_pos = x_start
    n_ent = min(m, n)
    
    for k in 1:n_ent
        x_idx = m - k + 1
        y_idx = n - k + 1
        
        y_x = -x_idx * QUBIT_SPACING
        y_y = -(m + y_idx) * QUBIT_SPACING
        
        phase = k <= length(phases) ? phases[k] : 0.0
        draw_entangle_gate!(ax, x_pos, y_x, y_y, "E$k", phase)
        x_pos += GATE_SPACING * 1.2
    end
    
    return x_pos
end

"""
    plot_circuit(spec::CircuitSpec; output_path=nothing)

Generate a circuit diagram from a CircuitSpec.
"""
function plot_circuit(spec::CircuitSpec; output_path=nothing)
    m = spec.n_row_qubits
    n = spec.n_col_qubits
    total = is_1d(spec) ? m : m + n
    
    # Calculate dimensions
    n_m_gates = div(m * (m - 1), 2)
    n_n_gates = is_1d(spec) ? 0 : div(n * (n - 1), 2)
    n_ent_gates = is_entangled(spec) ? n_entangle(spec) : 0
    total_gates = m + n_m_gates + (is_1d(spec) ? 0 : n + n_n_gates) + n_ent_gates
    
    width = max(700, 150 + total_gates * 55)
    height = 120 + total * 70
    
    fig = Figure(size=(width, height), backgroundcolor=:white)
    ax = Axis(fig[1, 1], title=spec.title, titlesize=20, aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    
    # Calculate x positions
    x_start = 1.5
    x_qft_end = x_start + (m + n_m_gates + 1) * GATE_SPACING
    x_end = is_entangled(spec) ? x_qft_end + (n_ent_gates + 1) * GATE_SPACING : x_qft_end
    
    # Draw qubits
    if is_1d(spec)
        for q in 1:m
            y = -q * QUBIT_SPACING
            draw_qubit!(ax, y, x_start, x_end, "|q$q⟩", "k$q")
        end
    else
        # Row qubits
        for q in 1:m
            y = -q * QUBIT_SPACING
            draw_qubit!(ax, y, x_start, x_end, "|x$q⟩", "kₓ$q")
        end
        # Column qubits
        for q in 1:n
            y = -(m + q) * QUBIT_SPACING
            draw_qubit!(ax, y, x_start, x_end, "|y$q⟩", "kᵧ$q")
        end
        # Separator
        y_sep = -(m + 0.5) * QUBIT_SPACING
        lines!(ax, [x_start, x_end], [y_sep, y_sep], 
               color=COLORS.label, linewidth=1, linestyle=:dash)
    end
    
    # Draw QFT gates
    x_pos = x_start + GATE_SPACING
    x_pos = draw_qft_layer!(ax, x_pos, m, 0)
    
    if !is_1d(spec)
        # Reset x for parallel column QFT (or use staggered position)
        x_pos_col = x_start + GATE_SPACING
        draw_qft_layer!(ax, x_pos_col, n, -m * QUBIT_SPACING)
    end
    
    # Draw entanglement gates
    if is_entangled(spec)
        # Separator line
        x_sep = x_qft_end - GATE_SPACING/2
        lines!(ax, [x_sep, x_sep], [0, -(total + 0.3) * QUBIT_SPACING], 
               color=COLORS.label, linewidth=1, linestyle=:dot)
        
        draw_entangle_layer!(ax, x_qft_end, m, n, spec.entangle_phases)
        
        # Legend
        draw_legend!(ax, x_end + 1.0, -0.3)
    end
    
    # Set limits
    xlims!(ax, x_start - 1.5, x_end + (is_entangled(spec) ? 3.5 : 1.5))
    ylims!(ax, -(total + 0.5) * QUBIT_SPACING, 0.7 * QUBIT_SPACING)
    
    if output_path !== nothing
        save(output_path, fig)
        println("  Saved: $output_path")
    end
    
    return fig
end

"""Draw a gate legend for entangled circuits."""
function draw_legend!(ax, x, y)
    text!(ax, x, y, text="Gates:", align=(:left, :top), fontsize=11, font=:bold)
    
    items = [
        (COLORS.hadamard, "H", "Hadamard"),
        (COLORS.phase_gate, "M", "Phase (2π/2ᵏ)"),
        (COLORS.entangle_gate, "E", "Entangle (φ)")
    ]
    
    for (i, (color, sym, label)) in enumerate(items)
        yi = y - i * 0.55
        poly!(ax, Point2f[(x, yi), (x+0.35, yi), (x+0.35, yi-0.35), (x, yi-0.35)],
              color=color, strokewidth=1, strokecolor=:white)
        text!(ax, x+0.175, yi-0.175, text=sym, align=(:center, :center), fontsize=9, color=:white)
        text!(ax, x+0.5, yi-0.175, text=label, align=(:left, :center), fontsize=9)
    end
end

# ================================================================================
# Comparison Plot
# ================================================================================

"""
    plot_comparison(m::Int, n::Int; output_path=nothing)

Generate a side-by-side comparison of standard vs entangled QFT.
"""
function plot_comparison(m::Int, n::Int; output_path=nothing)
    # Get statistics
    _, tensors_std = qft_code(m, n)
    _, tensors_ent, n_ent = entangled_qft_code(m, n)
    params_std = sum(prod(size(t)) for t in tensors_std)
    params_ent = sum(prod(size(t)) for t in tensors_ent)
    
    fig = Figure(size=(900, 450), backgroundcolor=:white)
    Label(fig[0, 1:2], "Standard vs Entangled QFT ($m×$n)", fontsize=22, font=:bold)
    
    total = m + n
    
    for (col, (title, has_entangle, color_ent)) in enumerate([
        ("Standard 2D QFT", false, nothing),
        ("Entangled QFT", true, COLORS.entangle_gate)
    ])
        ax = Axis(fig[1, col], title=title, titlesize=16)
        hidedecorations!(ax)
        hidespines!(ax)
        
        x_end = has_entangle ? 7.0 : 5.0
        
        # Draw qubits
        for q in 1:total
            y = -q * 0.7
            label = q <= m ? "x$q" : "y$(q-m)"
            lines!(ax, [0, x_end], [y, y], color=COLORS.wire, linewidth=2)
            text!(ax, -0.2, y, text=label, align=(:right, :center), fontsize=11)
        end
        
        # H gates
        for q in 1:total
            y = -q * 0.7
            poly!(ax, Point2f[(0.7, y-0.2), (1.1, y-0.2), (1.1, y+0.2), (0.7, y+0.2)],
                  color=COLORS.hadamard, strokewidth=1, strokecolor=:white)
            text!(ax, 0.9, y, text="H", align=(:center, :center), fontsize=9, color=:white)
        end
        
        # M gates annotation
        text!(ax, 2.5, -(total+0.8)*0.7, text="M gates", align=(:center, :top), 
              fontsize=11, color=COLORS.phase_gate)
        
        # E gates for entangled
        if has_entangle
            for k in 1:min(m, n)
                x_idx = m - k + 1
                y_idx = n - k + 1
                y1 = -x_idx * 0.7
                y2 = -(m + y_idx) * 0.7
                x = 4.8 + k * 0.4
                
                scatter!(ax, [x, x], [y1, y2], markersize=7, color=color_ent)
                lines!(ax, [x, x], [y1, y2], color=color_ent, linewidth=2)
            end
            text!(ax, 5.5, -(total+0.8)*0.7, text="E gates", align=(:center, :top), 
                  fontsize=11, color=color_ent)
        end
        
        xlims!(ax, -0.8, x_end + 0.5)
        ylims!(ax, -(total+1)*0.7, 0.4)
    end
    
    # Statistics
    Label(fig[2, 1:2], 
          "Standard: $(length(tensors_std)) tensors, $params_std params  |  " *
          "Entangled: $(length(tensors_ent)) tensors, $params_ent params (+$n_ent E gates)",
          fontsize=13)
    
    rowgap!(fig.layout, 1, 5)
    
    if output_path !== nothing
        save(output_path, fig)
        println("  Saved: $output_path")
    end
    
    return fig
end

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
    
    # 1. Single QFT
    println("1. Single QFT (3 qubits)")
    plot_circuit(CircuitSpec(3; title="QFT Circuit (3 qubits)");
        output_path=joinpath(output_dir, "qft_3qubit.png"))
    
    # 2. 2D QFT
    println("2. 2D QFT (3×3)")
    plot_circuit(CircuitSpec(3, 3; title="2D QFT Circuit (3×3)");
        output_path=joinpath(output_dir, "qft_2d_3x3.png"))
    
    # 3. Entangled QFT (zero phases)
    println("3. Entangled QFT (3×3, φ=0)")
    plot_circuit(CircuitSpec(3, 3, zeros(3); title="Entangled QFT (3×3, φ=0)");
        output_path=joinpath(output_dir, "entangled_qft_3x3_zero.png"))
    
    # 4. Entangled QFT (trained phases)
    println("4. Entangled QFT (3×3, trained)")
    plot_circuit(CircuitSpec(3, 3, [π/4, π/3, π/6]; title="Entangled QFT (3×3, trained)");
        output_path=joinpath(output_dir, "entangled_qft_3x3_trained.png"))
    
    # 5. Large circuit
    println("5. Entangled QFT (5×5)")
    plot_circuit(CircuitSpec(5, 5, zeros(5); title="Entangled QFT (5×5 → 32×32 images)");
        output_path=joinpath(output_dir, "entangled_qft_5x5.png"))
    
    # 6. Comparison
    println("6. Comparison diagram")
    plot_comparison(3, 3; output_path=joinpath(output_dir, "comparison_3x3.png"))
    
    println("\n" * "="^60)
    println("Done! Generated 6 circuit diagrams.")
    println("="^60)
end

main()
