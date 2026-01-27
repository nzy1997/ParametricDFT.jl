# ================================================================================
# TEBD Circuit Visualization
# ================================================================================
# Generates circuit diagrams for TEBD circuits with ring topology.
#
# Run with: julia --project=examples examples/tebd_visualization.jl
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
    tebd_gate = "#9C27B0",    # Purple for TEBD gates
    wrap_gate = "#FF5722",    # Deep Orange for wrap-around gates
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

"""Draw a TEBD gate connecting two qubits with phase annotation."""
function draw_tebd_gate!(ax, x, y1, y2, label, phase; is_wrap=false)
    color = is_wrap ? COLORS.wrap_gate : COLORS.tebd_gate
    scatter!(ax, [x, x], [y1, y2], markersize=12, color=color)
    lines!(ax, [x, x], [y1, y2], color=color, linewidth=WIRE_WIDTH+1)
    
    y_mid = (y1 + y2) / 2
    half = GATE_SIZE * 0.4
    poly!(ax, Point2f[(x-half, y_mid-half), (x+half, y_mid-half), (x+half, y_mid+half), (x-half, y_mid+half)],
          color=color, strokewidth=2, strokecolor=:white)
    text!(ax, x, y_mid, text=label, align=(:center, :center), 
          fontsize=12, font=:bold, color=:white)
    
    phase_str = "θ=$(round(phase, digits=2))"
    text!(ax, x + GATE_SIZE*0.6, y_mid, text=phase_str, align=(:left, :center), 
          fontsize=9, color=COLORS.label)
end

"""Draw a curved wrap-around connection for periodic boundary."""
function draw_wrap_connection!(ax, x, y_top, y_bottom; color=COLORS.wrap_gate)
    # Draw a curved line on the left side to show the wrap-around
    offset = 0.3
    n_points = 20
    xs = [x - offset * sin(π * t) for t in range(0, 1, length=n_points)]
    ys = range(y_top, y_bottom, length=n_points)
    lines!(ax, xs, collect(ys), color=color, linewidth=WIRE_WIDTH, linestyle=:dash)
end

# ================================================================================
# TEBD Circuit Drawing
# ================================================================================

"""
    plot_tebd_circuit(n::Int; phases=nothing, title="TEBD Circuit", output_path=nothing)

Generate a circuit diagram for TEBD with ring topology.
"""
function plot_tebd_circuit(n::Int; phases=nothing, title="TEBD Circuit", output_path=nothing)
    if phases === nothing
        phases = zeros(n)
    end
    n_gates = n
    
    # Calculate dimensions
    width = max(600, 150 + n_gates * 80)
    height = 120 + n * 70
    
    fig = Figure(size=(width, height), backgroundcolor=:white)
    ax = Axis(fig[1, 1], title=title, titlesize=20, aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    
    x_start = 2.0
    x_end = x_start + (n_gates + 2) * GATE_SPACING
    
    # Draw qubits
    for q in 1:n
        y = -q * QUBIT_SPACING
        draw_qubit!(ax, y, x_start, x_end, "|q$q⟩", "q'$q")
    end
    
    # Draw TEBD gates
    x_pos = x_start + GATE_SPACING
    
    # Gates 1 to n-1: Sequential nearest-neighbor (1-2, 2-3, ..., n-1-n)
    for i in 1:(n-1)
        y_i = -i * QUBIT_SPACING
        y_j = -(i+1) * QUBIT_SPACING
        gate_label = "T$i"
        draw_tebd_gate!(ax, x_pos, y_i, y_j, gate_label, phases[i]; is_wrap=false)
        x_pos += GATE_SPACING
    end
    
    # Gate n: Last-First wrap-around (n-1) to close the ring
    x_pos += GATE_SPACING * 0.3
    y1 = -1 * QUBIT_SPACING
    yn = -n * QUBIT_SPACING
    draw_tebd_gate!(ax, x_pos, yn, y1, "T$n", phases[n]; is_wrap=true)
    
    # Legend
    draw_tebd_legend!(ax, x_end + 0.5, -0.3)
    
    # Set limits
    xlims!(ax, x_start - 1.5, x_end + 3.0)
    ylims!(ax, -(n + 0.5) * QUBIT_SPACING, 0.7 * QUBIT_SPACING)
    
    if output_path !== nothing
        save(output_path, fig)
        println("  Saved: $output_path")
    end
    
    return fig
end

"""Draw a legend for TEBD circuits."""
function draw_tebd_legend!(ax, x, y)
    text!(ax, x, y, text="Gates:", align=(:left, :top), fontsize=11, font=:bold)
    
    items = [
        (COLORS.tebd_gate, "T", "TEBD (nearest)"),
        (COLORS.wrap_gate, "T", "TEBD (wrap)")
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
# Ring Topology Diagram
# ================================================================================

"""
    plot_tebd_ring(n::Int; output_path=nothing)

Generate a ring topology diagram showing TEBD connectivity.
"""
function plot_tebd_ring(n::Int; output_path=nothing)
    fig = Figure(size=(500, 500), backgroundcolor=:white)
    ax = Axis(fig[1, 1], title="TEBD Ring Topology ($n qubits)", titlesize=18, aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    
    # Calculate positions in a circle
    radius = 2.0
    angles = [2π * (i-1) / n - π/2 for i in 1:n]
    xs = radius * cos.(angles)
    ys = radius * sin.(angles)
    
    # Draw edges (ring connections)
    for i in 1:n
        j = i == n ? 1 : i + 1
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[j], ys[j]
        color = (i == 1 || i == n) ? COLORS.wrap_gate : COLORS.tebd_gate
        lines!(ax, [x1, x2], [y1, y2], color=color, linewidth=3)
    end
    
    # Draw first-last wrap (special connection shown across center)
    lines!(ax, [xs[1], xs[n]], [ys[1], ys[n]], color=COLORS.wrap_gate, 
           linewidth=2, linestyle=:dash)
    
    # Draw qubit nodes
    scatter!(ax, xs, ys, markersize=35, color=:white, strokewidth=2, strokecolor=COLORS.wire)
    for i in 1:n
        text!(ax, xs[i], ys[i], text="q$i", align=(:center, :center), 
              fontsize=14, font=:bold, color=COLORS.text)
    end
    
    # Connection labels
    text!(ax, 0, -radius - 0.5, text="Ring: T₁(1,2) → T₂(2,3) → ... → Tₙ₋₁(n-1,n) → Tₙ(n,1)", 
          align=(:center, :top), fontsize=11, color=COLORS.label)
    
    xlims!(ax, -radius - 1, radius + 1)
    ylims!(ax, -radius - 1.2, radius + 0.8)
    
    if output_path !== nothing
        save(output_path, fig)
        println("  Saved: $output_path")
    end
    
    return fig
end

# ================================================================================
# Tensor Network Statistics
# ================================================================================

"""Print tensor network statistics for TEBD circuit."""
function print_tebd_stats(n::Int)
    optcode, tensors, n_gates = tebd_code(n)
    total_params = sum(prod(size(t)) for t in tensors)
    
    println("\nTEBD Circuit Statistics ($n qubits):")
    println("  - Number of gates: $n_gates")
    println("  - Number of tensors: $(length(tensors))")
    println("  - Total parameters: $total_params")
    println("  - Einsum code: $(optcode)")
    
    return optcode, tensors, n_gates
end

# ================================================================================
# Main
# ================================================================================

function main()
    output_dir = joinpath(@__DIR__, "TEBDDiagrams")
    mkpath(output_dir)
    
    println("="^60)
    println("Generating TEBD Circuit Diagrams")
    println("="^60)
    println("\nOutput: $output_dir\n")
    
    # 1. Small TEBD (4 qubits)
    println("1. TEBD Circuit (4 qubits)")
    plot_tebd_circuit(4; title="TEBD Circuit (4 qubits, ring topology)",
        output_path=joinpath(output_dir, "tebd_4qubit.png"))
    print_tebd_stats(4)
    
    # 2. Medium TEBD (6 qubits)
    println("\n2. TEBD Circuit (6 qubits)")
    plot_tebd_circuit(6; title="TEBD Circuit (6 qubits, ring topology)",
        output_path=joinpath(output_dir, "tebd_6qubit.png"))
    print_tebd_stats(6)
    
    # 3. TEBD with custom phases
    println("\n3. TEBD Circuit (5 qubits, custom phases)")
    custom_phases = [π/4, π/3, π/2, π/6, π/5]  # 5 phases for 5 qubits
    plot_tebd_circuit(5; phases=custom_phases, 
        title="TEBD Circuit (5 qubits, trained phases)",
        output_path=joinpath(output_dir, "tebd_5qubit_trained.png"))
    
    # 4. Ring topology diagram
    println("\n4. Ring Topology Diagram (6 qubits)")
    plot_tebd_ring(6; output_path=joinpath(output_dir, "tebd_ring_6qubit.png"))
    
    # 5. Large TEBD (8 qubits)
    println("\n5. TEBD Circuit (8 qubits)")
    plot_tebd_circuit(8; title="TEBD Circuit (8 qubits, ring topology)",
        output_path=joinpath(output_dir, "tebd_8qubit.png"))
    print_tebd_stats(8)
    
    println("\n" * "="^60)
    println("Done! Generated 5 TEBD circuit diagrams.")
    println("="^60)
end

main()
