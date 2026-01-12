# ================================================================================
# Tensor Network Circuit Visualization
# ================================================================================
# This script generates professional circuit diagrams for QFT and Entangled QFT
# circuits as PNG images, similar to PennyLane's circuit visualization.
#
# Run with: julia --project=examples examples/tensor_network_visualization.jl
# ================================================================================

using ParametricDFT
using CairoMakie
using Yao
using OMEinsum

# Set up nice defaults for Makie
CairoMakie.activate!(type = "png", px_per_unit = 2)

# ================================================================================
# Color Scheme and Constants
# ================================================================================

const COLORS = (
    background = :white,
    wire = "#333333",
    hadamard = "#4285F4",      # Google Blue
    phase_gate = "#34A853",    # Google Green  
    entangle_gate = "#EA4335", # Google Red
    control = "#333333",
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

"""
    draw_wire!(ax, y, x_start, x_end)

Draw a horizontal qubit wire.
"""
function draw_wire!(ax, y, x_start, x_end)
    lines!(ax, [x_start, x_end], [y, y], color=COLORS.wire, linewidth=WIRE_WIDTH)
end

"""
    draw_hadamard!(ax, x, y)

Draw a Hadamard gate at position (x, y).
"""
function draw_hadamard!(ax, x, y)
    # Gate box
    poly!(ax, 
        Point2f[(x - GATE_SIZE/2, y - GATE_SIZE/2), 
                (x + GATE_SIZE/2, y - GATE_SIZE/2),
                (x + GATE_SIZE/2, y + GATE_SIZE/2),
                (x - GATE_SIZE/2, y + GATE_SIZE/2)],
        color=COLORS.hadamard, strokewidth=2, strokecolor=:white)
    # Label
    text!(ax, x, y, text="H", align=(:center, :center), 
          fontsize=18, font=:bold, color=:white)
end

"""
    draw_phase_gate!(ax, x, y_control, y_target, label)

Draw a controlled phase gate (M gate) from control to target qubit.
"""
function draw_phase_gate!(ax, x, y_control, y_target, label)
    # Control dot
    scatter!(ax, [x], [y_control], markersize=12, color=COLORS.control)
    
    # Vertical connection line
    lines!(ax, [x, x], [y_control, y_target], color=COLORS.control, linewidth=WIRE_WIDTH)
    
    # Target gate box
    poly!(ax, 
        Point2f[(x - GATE_SIZE/2, y_target - GATE_SIZE/2), 
                (x + GATE_SIZE/2, y_target - GATE_SIZE/2),
                (x + GATE_SIZE/2, y_target + GATE_SIZE/2),
                (x - GATE_SIZE/2, y_target + GATE_SIZE/2)],
        color=COLORS.phase_gate, strokewidth=2, strokecolor=:white)
    
    # Label
    text!(ax, x, y_target, text=label, align=(:center, :center), 
          fontsize=14, font=:bold, color=:white)
end

"""
    draw_entangle_gate!(ax, x, y1, y2, label, phase_str)

Draw an entanglement gate connecting two qubits.
"""
function draw_entangle_gate!(ax, x, y1, y2, label, phase_str)
    # Control dots on both qubits
    scatter!(ax, [x, x], [y1, y2], markersize=12, color=COLORS.entangle_gate)
    
    # Vertical connection line
    lines!(ax, [x, x], [y1, y2], color=COLORS.entangle_gate, linewidth=WIRE_WIDTH+1)
    
    # Gate label box in the middle
    y_mid = (y1 + y2) / 2
    poly!(ax, 
        Point2f[(x - GATE_SIZE*0.8, y_mid - GATE_SIZE/3), 
                (x + GATE_SIZE*0.8, y_mid - GATE_SIZE/3),
                (x + GATE_SIZE*0.8, y_mid + GATE_SIZE/3),
                (x - GATE_SIZE*0.8, y_mid + GATE_SIZE/3)],
        color=COLORS.entangle_gate, strokewidth=2, strokecolor=:white)
    
    # Label
    text!(ax, x, y_mid, text=label, align=(:center, :center), 
          fontsize=12, font=:bold, color=:white)
    
    # Phase annotation
    text!(ax, x + GATE_SIZE, y_mid, text=phase_str, align=(:left, :center), 
          fontsize=10, color=COLORS.label)
end

"""
    draw_qubit_label!(ax, x, y, label)

Draw a qubit label at the start of a wire.
"""
function draw_qubit_label!(ax, x, y, label)
    text!(ax, x, y, text=label, align=(:right, :center), 
          fontsize=14, font=:bold, color=COLORS.text)
end

"""
    draw_output_label!(ax, x, y, label)

Draw an output label at the end of a wire.
"""
function draw_output_label!(ax, x, y, label)
    text!(ax, x, y, text=label, align=(:left, :center), 
          fontsize=14, color=COLORS.label)
end

# ================================================================================
# Circuit Diagram Generation
# ================================================================================

"""
    plot_qft_circuit(m::Int; title="QFT Circuit", output_path=nothing)

Generate a circuit diagram for the QFT on m qubits.
Returns the Figure object.
"""
function plot_qft_circuit(m::Int; title="QFT Circuit", output_path=nothing)
    # Calculate figure dimensions
    n_gates = m + div(m * (m - 1), 2)  # H gates + M gates
    width = max(800, 150 + n_gates * GATE_SPACING * 80)
    height = 100 + m * QUBIT_SPACING * 80
    
    fig = Figure(size=(width, height), backgroundcolor=COLORS.background)
    
    ax = Axis(fig[1, 1], 
              title=title,
              titlesize=20,
              titlegap=10,
              aspect=DataAspect())
    
    hidedecorations!(ax)
    hidespines!(ax)
    
    # Calculate positions
    x_start = 1.0
    x_end = x_start + (m + div(m * (m - 1), 2) + 1) * GATE_SPACING
    
    # Draw qubit wires and labels
    for q in 1:m
        y = -q * QUBIT_SPACING
        draw_wire!(ax, y, x_start - 0.5, x_end + 0.5)
        draw_qubit_label!(ax, x_start - 0.7, y, "|q$(q)⟩")
        draw_output_label!(ax, x_end + 0.7, y, "k$(q)")
    end
    
    # Draw gates
    x_pos = x_start + GATE_SPACING
    
    for q in 1:m
        y_q = -q * QUBIT_SPACING
        
        # Hadamard gate
        draw_hadamard!(ax, x_pos, y_q)
        x_pos += GATE_SPACING
        
        # Controlled phase gates
        for target in (q+1):m
            y_target = -target * QUBIT_SPACING
            k = target - q + 1
            draw_phase_gate!(ax, x_pos, y_q, y_target, "M$(k)")
            x_pos += GATE_SPACING
        end
    end
    
    # Set limits
    xlims!(ax, x_start - 1.5, x_end + 1.5)
    ylims!(ax, -(m + 0.5) * QUBIT_SPACING, 0.5 * QUBIT_SPACING)
    
    if output_path !== nothing
        save(output_path, fig)
        println("  Saved: $output_path")
    end
    
    return fig
end

"""
    plot_2d_qft_circuit(m::Int, n::Int; title="2D QFT Circuit", output_path=nothing)

Generate a circuit diagram for the 2D QFT (separate QFT on row and column qubits).
"""
function plot_2d_qft_circuit(m::Int, n::Int; title="2D QFT Circuit", output_path=nothing)
    total_qubits = m + n
    
    # Calculate figure dimensions  
    max_gates = max(m + div(m * (m - 1), 2), n + div(n * (n - 1), 2))
    width = max(900, 200 + max_gates * GATE_SPACING * 70)
    height = 150 + total_qubits * QUBIT_SPACING * 70
    
    fig = Figure(size=(width, height), backgroundcolor=COLORS.background)
    
    ax = Axis(fig[1, 1], 
              title=title,
              titlesize=22,
              titlegap=15,
              aspect=DataAspect())
    
    hidedecorations!(ax)
    hidespines!(ax)
    
    # Calculate positions
    x_start = 1.5
    x_end = x_start + (max_gates + 2) * GATE_SPACING
    
    # Section separator position
    y_separator = -(m + 0.5) * QUBIT_SPACING
    
    # Draw row qubits (x)
    for q in 1:m
        y = -q * QUBIT_SPACING
        draw_wire!(ax, y, x_start - 0.5, x_end + 0.5)
        draw_qubit_label!(ax, x_start - 0.7, y, "|x$(q)⟩")
        draw_output_label!(ax, x_end + 0.7, y, "kₓ$(q)")
    end
    
    # Draw column qubits (y)
    for q in 1:n
        y = -(m + q) * QUBIT_SPACING
        draw_wire!(ax, y, x_start - 0.5, x_end + 0.5)
        draw_qubit_label!(ax, x_start - 0.7, y, "|y$(q)⟩")
        draw_output_label!(ax, x_end + 0.7, y, "kᵧ$(q)")
    end
    
    # Draw separator line
    lines!(ax, [x_start - 0.3, x_end + 0.3], [y_separator, y_separator], 
           color=COLORS.label, linewidth=1, linestyle=:dash)
    text!(ax, x_start - 0.5, y_separator, text="row/col", align=(:right, :center),
          fontsize=10, color=COLORS.label)
    
    # Draw QFT_x gates (row qubits)
    x_pos = x_start + GATE_SPACING
    for q in 1:m
        y_q = -q * QUBIT_SPACING
        draw_hadamard!(ax, x_pos, y_q)
        x_pos += GATE_SPACING
        
        for target in (q+1):m
            y_target = -target * QUBIT_SPACING
            k = target - q + 1
            draw_phase_gate!(ax, x_pos, y_q, y_target, "M$(k)")
            x_pos += GATE_SPACING
        end
    end
    
    # Draw QFT_y gates (column qubits) - start from same x position
    x_pos = x_start + GATE_SPACING
    for q in 1:n
        y_q = -(m + q) * QUBIT_SPACING
        draw_hadamard!(ax, x_pos, y_q)
        x_pos += GATE_SPACING
        
        for target in (q+1):n
            y_target = -(m + target) * QUBIT_SPACING
            k = target - q + 1
            draw_phase_gate!(ax, x_pos, y_q, y_target, "M$(k)")
            x_pos += GATE_SPACING
        end
    end
    
    # Set limits
    xlims!(ax, x_start - 2, x_end + 2)
    ylims!(ax, -(total_qubits + 0.5) * QUBIT_SPACING, 0.5 * QUBIT_SPACING)
    
    if output_path !== nothing
        save(output_path, fig)
        println("  Saved: $output_path")
    end
    
    return fig
end

"""
    plot_entangled_qft_circuit(m::Int, n::Int; phases=nothing, title="Entangled QFT Circuit", output_path=nothing)

Generate a circuit diagram for the entangled QFT with XY correlation gates.
"""
function plot_entangled_qft_circuit(m::Int, n::Int; phases=nothing, title="Entangled QFT Circuit", output_path=nothing)
    total_qubits = m + n
    n_entangle = min(m, n)
    
    if phases === nothing
        phases = zeros(n_entangle)
    end
    
    # Calculate figure dimensions
    max_gates = max(m + div(m * (m - 1), 2), n + div(n * (n - 1), 2))
    width = max(1000, 250 + (max_gates + n_entangle + 2) * GATE_SPACING * 60)
    height = 180 + total_qubits * QUBIT_SPACING * 65
    
    fig = Figure(size=(width, height), backgroundcolor=COLORS.background)
    
    ax = Axis(fig[1, 1], 
              title=title,
              titlesize=22,
              titlegap=15,
              aspect=DataAspect())
    
    hidedecorations!(ax)
    hidespines!(ax)
    
    # Calculate positions
    x_start = 2.0
    x_entangle_start = x_start + (max_gates + 2) * GATE_SPACING
    x_end = x_entangle_start + (n_entangle + 1) * GATE_SPACING
    
    # Section separator position
    y_separator = -(m + 0.5) * QUBIT_SPACING
    
    # Draw row qubits (x)
    for q in 1:m
        y = -q * QUBIT_SPACING
        draw_wire!(ax, y, x_start - 0.5, x_end + 0.5)
        draw_qubit_label!(ax, x_start - 0.7, y, "|x$(q)⟩")
        draw_output_label!(ax, x_end + 0.7, y, "kₓ$(q)")
    end
    
    # Draw column qubits (y)
    for q in 1:n
        y = -(m + q) * QUBIT_SPACING
        draw_wire!(ax, y, x_start - 0.5, x_end + 0.5)
        draw_qubit_label!(ax, x_start - 0.7, y, "|y$(q)⟩")
        draw_output_label!(ax, x_end + 0.7, y, "kᵧ$(q)")
    end
    
    # Draw separator line
    lines!(ax, [x_start - 0.3, x_end + 0.3], [y_separator, y_separator], 
           color=COLORS.label, linewidth=1, linestyle=:dash)
    
    # Draw QFT_x gates (row qubits)
    x_pos = x_start + GATE_SPACING
    for q in 1:m
        y_q = -q * QUBIT_SPACING
        draw_hadamard!(ax, x_pos, y_q)
        x_pos += GATE_SPACING
        
        for target in (q+1):m
            y_target = -target * QUBIT_SPACING
            k = target - q + 1
            draw_phase_gate!(ax, x_pos, y_q, y_target, "M$(k)")
            x_pos += GATE_SPACING
        end
    end
    
    # Draw QFT_y gates (column qubits)
    x_pos = x_start + GATE_SPACING
    for q in 1:n
        y_q = -(m + q) * QUBIT_SPACING
        draw_hadamard!(ax, x_pos, y_q)
        x_pos += GATE_SPACING
        
        for target in (q+1):n
            y_target = -(m + target) * QUBIT_SPACING
            k = target - q + 1
            draw_phase_gate!(ax, x_pos, y_q, y_target, "M$(k)")
            x_pos += GATE_SPACING
        end
    end
    
    # Draw vertical separator before entanglement layer
    x_sep = x_entangle_start - GATE_SPACING/2
    lines!(ax, [x_sep, x_sep], [0, -(total_qubits + 0.3) * QUBIT_SPACING], 
           color=COLORS.label, linewidth=1, linestyle=:dot)
    
    # Draw entanglement gates
    x_pos = x_entangle_start
    for k in 1:n_entangle
        x_idx = m - k + 1
        y_idx = n - k + 1
        
        y_x = -x_idx * QUBIT_SPACING
        y_y = -(m + y_idx) * QUBIT_SPACING
        
        phase_val = round(phases[k], digits=2)
        phase_deg = round(rad2deg(phases[k]), digits=0)
        phase_str = "φ=$(phase_val)"
        
        draw_entangle_gate!(ax, x_pos, y_x, y_y, "E$(k)", phase_str)
        x_pos += GATE_SPACING * 1.2
    end
    
    # Add legend
    legend_x = x_end + 1.5
    legend_y = -0.5 * QUBIT_SPACING
    
    # Legend title
    text!(ax, legend_x, legend_y, text="Gates:", align=(:left, :top),
          fontsize=12, font=:bold, color=COLORS.text)
    
    # H gate legend
    poly!(ax, Point2f[(legend_x, legend_y - 0.5), (legend_x + 0.4, legend_y - 0.5),
                      (legend_x + 0.4, legend_y - 0.9), (legend_x, legend_y - 0.9)],
          color=COLORS.hadamard, strokewidth=1, strokecolor=:white)
    text!(ax, legend_x + 0.2, legend_y - 0.7, text="H", align=(:center, :center),
          fontsize=10, color=:white)
    text!(ax, legend_x + 0.6, legend_y - 0.7, text="Hadamard", align=(:left, :center),
          fontsize=10, color=COLORS.text)
    
    # M gate legend
    poly!(ax, Point2f[(legend_x, legend_y - 1.2), (legend_x + 0.4, legend_y - 1.2),
                      (legend_x + 0.4, legend_y - 1.6), (legend_x, legend_y - 1.6)],
          color=COLORS.phase_gate, strokewidth=1, strokecolor=:white)
    text!(ax, legend_x + 0.2, legend_y - 1.4, text="M", align=(:center, :center),
          fontsize=10, color=:white)
    text!(ax, legend_x + 0.6, legend_y - 1.4, text="Phase (2π/2ᵏ)", align=(:left, :center),
          fontsize=10, color=COLORS.text)
    
    # E gate legend
    poly!(ax, Point2f[(legend_x, legend_y - 1.9), (legend_x + 0.4, legend_y - 1.9),
                      (legend_x + 0.4, legend_y - 2.3), (legend_x, legend_y - 2.3)],
          color=COLORS.entangle_gate, strokewidth=1, strokecolor=:white)
    text!(ax, legend_x + 0.2, legend_y - 2.1, text="E", align=(:center, :center),
          fontsize=10, color=:white)
    text!(ax, legend_x + 0.6, legend_y - 2.1, text="Entangle (φ)", align=(:left, :center),
          fontsize=10, color=COLORS.text)
    
    # Set limits
    xlims!(ax, x_start - 2.5, x_end + 4)
    ylims!(ax, -(total_qubits + 0.5) * QUBIT_SPACING, 0.8 * QUBIT_SPACING)
    
    if output_path !== nothing
        save(output_path, fig)
        println("  Saved: $output_path")
    end
    
    return fig
end

"""
    plot_tensor_network_comparison(m::Int, n::Int; output_path=nothing)

Generate a side-by-side comparison of standard QFT vs entangled QFT.
"""
function plot_tensor_network_comparison(m::Int, n::Int; output_path=nothing)
    # Get tensor counts
    _, tensors_std = qft_code(m, n)
    _, tensors_ent, n_entangle = entangled_qft_code(m, n)
    
    params_std = sum(prod(size(t)) for t in tensors_std)
    params_ent = sum(prod(size(t)) for t in tensors_ent)
    
    fig = Figure(size=(900, 500), backgroundcolor=COLORS.background)
    
    # Title
    Label(fig[0, 1:2], "Tensor Network Comparison: Standard vs Entangled QFT",
          fontsize=24, font=:bold, halign=:center)
    
    # Standard QFT panel
    ax1 = Axis(fig[1, 1], title="Standard 2D QFT", titlesize=18)
    hidedecorations!(ax1)
    hidespines!(ax1)
    
    # Draw simplified standard QFT
    for q in 1:(m+n)
        y = -q * 0.8
        lines!(ax1, [0, 5], [y, y], color=COLORS.wire, linewidth=2)
        text!(ax1, -0.3, y, text=q <= m ? "x$(q)" : "y$(q-m)", 
              align=(:right, :center), fontsize=12)
    end
    
    # H gates
    for q in 1:(m+n)
        y = -q * 0.8
        poly!(ax1, Point2f[(0.8, y-0.25), (1.2, y-0.25), (1.2, y+0.25), (0.8, y+0.25)],
              color=COLORS.hadamard, strokewidth=1, strokecolor=:white)
        text!(ax1, 1.0, y, text="H", align=(:center, :center), fontsize=10, color=:white)
    end
    
    # M gates (simplified)
    text!(ax1, 2.5, -(m+n+1)*0.4, text="M gates", align=(:center, :top), 
          fontsize=12, color=COLORS.phase_gate)
    
    xlims!(ax1, -1, 6)
    ylims!(ax1, -(m+n+1)*0.8, 0.5)
    
    # Entangled QFT panel
    ax2 = Axis(fig[1, 2], title="Entangled QFT", titlesize=18)
    hidedecorations!(ax2)
    hidespines!(ax2)
    
    # Draw simplified entangled QFT
    for q in 1:(m+n)
        y = -q * 0.8
        lines!(ax2, [0, 7], [y, y], color=COLORS.wire, linewidth=2)
        text!(ax2, -0.3, y, text=q <= m ? "x$(q)" : "y$(q-m)", 
              align=(:right, :center), fontsize=12)
    end
    
    # H gates
    for q in 1:(m+n)
        y = -q * 0.8
        poly!(ax2, Point2f[(0.8, y-0.25), (1.2, y-0.25), (1.2, y+0.25), (0.8, y+0.25)],
              color=COLORS.hadamard, strokewidth=1, strokecolor=:white)
        text!(ax2, 1.0, y, text="H", align=(:center, :center), fontsize=10, color=:white)
    end
    
    # E gates
    for k in 1:min(m, n)
        x_idx = m - k + 1
        y_idx = n - k + 1
        y1 = -x_idx * 0.8
        y2 = -(m + y_idx) * 0.8
        x = 5.0 + k * 0.5
        
        scatter!(ax2, [x, x], [y1, y2], markersize=8, color=COLORS.entangle_gate)
        lines!(ax2, [x, x], [y1, y2], color=COLORS.entangle_gate, linewidth=2)
    end
    
    text!(ax2, 2.5, -(m+n+1)*0.4, text="M gates", align=(:center, :top), 
          fontsize=12, color=COLORS.phase_gate)
    text!(ax2, 5.5, -(m+n+1)*0.4, text="E gates", align=(:center, :top), 
          fontsize=12, color=COLORS.entangle_gate)
    
    xlims!(ax2, -1, 8)
    ylims!(ax2, -(m+n+1)*0.8, 0.5)
    
    # Statistics table
    Label(fig[2, 1:2], 
          "Standard QFT: $(length(tensors_std)) tensors, $params_std params  |  " *
          "Entangled QFT: $(length(tensors_ent)) tensors, $params_ent params (+$n_entangle E gates)",
          fontsize=14, halign=:center)
    
    rowgap!(fig.layout, 1, 10)
    
    if output_path !== nothing
        save(output_path, fig)
        println("  Saved: $output_path")
    end
    
    return fig
end

# ================================================================================
# Main Function
# ================================================================================

function main()
    # Create output directory
    output_dir = joinpath(@__DIR__, "CircuitDiagrams")
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    
    println("="^70)
    println("Generating Circuit Diagrams")
    println("="^70)
    println("\nOutput directory: $output_dir\n")
    
    # 1. Single QFT circuit (3 qubits)
    println("1. Generating single QFT circuit (3 qubits)...")
    plot_qft_circuit(3; 
        title="Quantum Fourier Transform (3 qubits)",
        output_path=joinpath(output_dir, "qft_3qubit.png"))
    
    # 2. 2D QFT circuit (3x3)
    println("2. Generating 2D QFT circuit (3×3)...")
    plot_2d_qft_circuit(3, 3;
        title="2D QFT Circuit (3×3 qubits → 8×8 images)",
        output_path=joinpath(output_dir, "qft_2d_3x3.png"))
    
    # 3. Entangled QFT with zero phases
    println("3. Generating Entangled QFT circuit (3×3, zero phases)...")
    plot_entangled_qft_circuit(3, 3;
        phases=zeros(3),
        title="Entangled QFT Circuit (3×3, φ=0)",
        output_path=joinpath(output_dir, "entangled_qft_3x3_zero.png"))
    
    # 4. Entangled QFT with example phases
    println("4. Generating Entangled QFT circuit (3×3, example phases)...")
    example_phases = [π/4, π/3, π/6]
    plot_entangled_qft_circuit(3, 3;
        phases=example_phases,
        title="Entangled QFT Circuit (3×3, trained phases)",
        output_path=joinpath(output_dir, "entangled_qft_3x3_trained.png"))
    
    # 5. Larger circuit (5x5 for 32x32 images)
    println("5. Generating large Entangled QFT circuit (5×5)...")
    plot_entangled_qft_circuit(5, 5;
        phases=zeros(5),
        title="Entangled QFT Circuit (5×5 qubits → 32×32 images)",
        output_path=joinpath(output_dir, "entangled_qft_5x5.png"))
    
    # 6. Comparison diagram
    println("6. Generating comparison diagram...")
    plot_tensor_network_comparison(3, 3;
        output_path=joinpath(output_dir, "comparison_3x3.png"))
    
    println("\n" * "="^70)
    println("Circuit diagrams generated successfully!")
    println("="^70)
    println("""
    
    Generated files:
    ├── qft_3qubit.png           - Single QFT circuit
    ├── qft_2d_3x3.png           - 2D QFT (standard)
    ├── entangled_qft_3x3_zero.png   - Entangled QFT (φ=0)
    ├── entangled_qft_3x3_trained.png - Entangled QFT (trained phases)
    ├── entangled_qft_5x5.png    - Large entangled circuit
    └── comparison_3x3.png       - Side-by-side comparison
    
    """)
end

# Run the main function
main()
