# ================================================================================
# Unified Circuit Visualization
# ================================================================================
# Generates professional circuit diagrams for various circuit types:
# - QFT (1D and 2D)
# - Entangled QFT
# - TEBD (2D chain topology: row chain + column chain)
#
# Run with: julia --project=examples examples/circuit_visualization.jl
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
    tebd_gate = "#9C27B0",     # Purple
    wrap_gate = "#FF5722",     # Deep Orange
    text = "#333333",
    label = "#666666"
)

const GATE_SIZE = 0.6
const WIRE_WIDTH = 2
const QUBIT_SPACING = 1.0
const GATE_SPACING = 1.2

# ================================================================================
# Abstract Circuit Specification
# ================================================================================

abstract type AbstractCircuitSpec end

"""
    QFTCircuitSpec - Specification for QFT circuits
"""
struct QFTCircuitSpec <: AbstractCircuitSpec
    n_row_qubits::Int
    n_col_qubits::Int  # 0 for 1D QFT
    title::String
end

QFTCircuitSpec(m::Int; title="QFT Circuit") = QFTCircuitSpec(m, 0, title)
QFTCircuitSpec(m::Int, n::Int; title="2D QFT Circuit") = QFTCircuitSpec(m, n, title)

is_1d(spec::QFTCircuitSpec) = spec.n_col_qubits == 0
total_qubits(spec::QFTCircuitSpec) = spec.n_row_qubits + spec.n_col_qubits

"""
    EntangledQFTCircuitSpec - Specification for Entangled QFT circuits
"""
struct EntangledQFTCircuitSpec <: AbstractCircuitSpec
    n_row_qubits::Int
    n_col_qubits::Int
    entangle_phases::Vector{Float64}
    title::String
end

function EntangledQFTCircuitSpec(m::Int, n::Int; phases=nothing, title="Entangled QFT Circuit")
    n_ent = min(m, n)
    if phases === nothing
        phases = zeros(n_ent)
    end
    EntangledQFTCircuitSpec(m, n, Float64.(phases), title)
end

total_qubits(spec::EntangledQFTCircuitSpec) = spec.n_row_qubits + spec.n_col_qubits
n_entangle(spec::EntangledQFTCircuitSpec) = length(spec.entangle_phases)

"""
    TEBDCircuitSpec - Specification for 2D TEBD circuits (already defined in tebd.jl)
    
The TEBD circuit has m row qubits and n column qubits with:
- Row chain: (x1,x2), (x2,x3), ..., (x_{m-1},x_m) for m-1 gates
- Column chain: (y1,y2), (y2,y3), ..., (y_{n-1},y_n) for n-1 gates
"""
# TEBDCircuitSpec is already defined in tebd.jl with:
#   n_row_qubits, n_col_qubits, phases, title

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

"""Draw an entanglement/TEBD gate connecting two qubits with phase annotation."""
function draw_two_qubit_gate!(ax, x, y1, y2, label, phase; color=COLORS.entangle_gate, show_phase=true)
    scatter!(ax, [x, x], [y1, y2], markersize=12, color=color)
    lines!(ax, [x, x], [y1, y2], color=color, linewidth=WIRE_WIDTH+1)
    
    y_mid = (y1 + y2) / 2
    draw_gate!(ax, x, y_mid, label; color=color, size=GATE_SIZE*0.8)
    
    if show_phase
        phase_str = "φ=$(round(phase, digits=2))"
        text!(ax, x + GATE_SIZE, y_mid, text=phase_str, align=(:left, :center), 
              fontsize=10, color=COLORS.label)
    end
end

# ================================================================================
# QFT Circuit Drawing
# ================================================================================

"""Draw QFT gates (H + M gates) for a group of qubits."""
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

"""Draw entanglement gates between corresponding row and column qubits."""
function draw_entangle_layer!(ax, x_start, m, n, phases)
    x_pos = x_start
    n_ent = min(m, n)
    
    for k in 1:n_ent
        x_idx = m - k + 1
        y_idx = n - k + 1
        
        y_x = -x_idx * QUBIT_SPACING
        y_y = -(m + y_idx) * QUBIT_SPACING
        
        phase = k <= length(phases) ? phases[k] : 0.0
        draw_two_qubit_gate!(ax, x_pos, y_x, y_y, "E$k", phase; color=COLORS.entangle_gate)
        x_pos += GATE_SPACING * 1.2
    end
    
    return x_pos
end

# ================================================================================
# Plot Functions for Each Circuit Type
# ================================================================================

"""Plot a QFT circuit."""
function plot_circuit(spec::QFTCircuitSpec; output_path=nothing)
    m = spec.n_row_qubits
    n = spec.n_col_qubits
    total = is_1d(spec) ? m : m + n
    
    # Calculate dimensions
    n_m_gates = div(m * (m - 1), 2)
    n_n_gates = is_1d(spec) ? 0 : div(n * (n - 1), 2)
    total_gates = m + n_m_gates + (is_1d(spec) ? 0 : n + n_n_gates)
    
    width = max(700, 150 + total_gates * 55)
    height = 120 + total * 70
    
    fig = Figure(size=(width, height), backgroundcolor=:white)
    ax = Axis(fig[1, 1], title=spec.title, titlesize=20, aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    
    x_start = 1.5
    x_qft_end = x_start + (m + n_m_gates + 1) * GATE_SPACING
    x_end = x_qft_end
    
    # Draw qubits
    if is_1d(spec)
        for q in 1:m
            y = -q * QUBIT_SPACING
            draw_qubit!(ax, y, x_start, x_end, "|q$q⟩", "k$q")
        end
    else
        for q in 1:m
            y = -q * QUBIT_SPACING
            draw_qubit!(ax, y, x_start, x_end, "|x$q⟩", "kₓ$q")
        end
        for q in 1:n
            y = -(m + q) * QUBIT_SPACING
            draw_qubit!(ax, y, x_start, x_end, "|y$q⟩", "kᵧ$q")
        end
        y_sep = -(m + 0.5) * QUBIT_SPACING
        lines!(ax, [x_start, x_end], [y_sep, y_sep], 
               color=COLORS.label, linewidth=1, linestyle=:dash)
    end
    
    # Draw QFT gates
    x_pos = x_start + GATE_SPACING
    x_pos = draw_qft_layer!(ax, x_pos, m, 0)
    
    if !is_1d(spec)
        x_pos_col = x_start + GATE_SPACING
        draw_qft_layer!(ax, x_pos_col, n, -m * QUBIT_SPACING)
    end
    
    xlims!(ax, x_start - 1.5, x_end + 1.5)
    ylims!(ax, -(total + 0.5) * QUBIT_SPACING, 0.7 * QUBIT_SPACING)
    
    if output_path !== nothing
        save(output_path, fig)
        println("  Saved: $output_path")
    end
    
    return fig
end

"""Plot an Entangled QFT circuit."""
function plot_circuit(spec::EntangledQFTCircuitSpec; output_path=nothing)
    m = spec.n_row_qubits
    n = spec.n_col_qubits
    total = m + n
    n_ent_gates = n_entangle(spec)
    
    n_m_gates = div(m * (m - 1), 2)
    n_n_gates = div(n * (n - 1), 2)
    total_gates = m + n_m_gates + n + n_n_gates + n_ent_gates
    
    width = max(700, 150 + total_gates * 55)
    height = 120 + total * 70
    
    fig = Figure(size=(width, height), backgroundcolor=:white)
    ax = Axis(fig[1, 1], title=spec.title, titlesize=20, aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    
    x_start = 1.5
    x_qft_end = x_start + (m + n_m_gates + 1) * GATE_SPACING
    x_end = x_qft_end + (n_ent_gates + 1) * GATE_SPACING
    
    # Draw qubits
    for q in 1:m
        y = -q * QUBIT_SPACING
        draw_qubit!(ax, y, x_start, x_end, "|x$q⟩", "kₓ$q")
    end
    for q in 1:n
        y = -(m + q) * QUBIT_SPACING
        draw_qubit!(ax, y, x_start, x_end, "|y$q⟩", "kᵧ$q")
    end
    y_sep = -(m + 0.5) * QUBIT_SPACING
    lines!(ax, [x_start, x_end], [y_sep, y_sep], 
           color=COLORS.label, linewidth=1, linestyle=:dash)
    
    # Draw QFT gates
    x_pos = x_start + GATE_SPACING
    draw_qft_layer!(ax, x_pos, m, 0)
    draw_qft_layer!(ax, x_pos, n, -m * QUBIT_SPACING)
    
    # Separator line before entanglement
    x_sep = x_qft_end - GATE_SPACING/2
    lines!(ax, [x_sep, x_sep], [0, -(total + 0.3) * QUBIT_SPACING], 
           color=COLORS.label, linewidth=1, linestyle=:dot)
    
    # Draw entanglement gates
    draw_entangle_layer!(ax, x_qft_end, m, n, spec.entangle_phases)
    
    # Legend
    draw_legend!(ax, x_end + 1.0, -0.3, [
        (COLORS.hadamard, "H", "Hadamard"),
        (COLORS.phase_gate, "M", "Phase (2π/2ᵏ)"),
        (COLORS.entangle_gate, "E", "Entangle (φ)")
    ])
    
    xlims!(ax, x_start - 1.5, x_end + 3.5)
    ylims!(ax, -(total + 0.5) * QUBIT_SPACING, 0.7 * QUBIT_SPACING)
    
    if output_path !== nothing
        save(output_path, fig)
        println("  Saved: $output_path")
    end
    
    return fig
end

"""Draw a gate legend."""
function draw_legend!(ax, x, y, items)
    text!(ax, x, y, text="Gates:", align=(:left, :top), fontsize=11, font=:bold)
    
    for (i, (color, sym, label)) in enumerate(items)
        yi = y - i * 0.55
        poly!(ax, Point2f[(x, yi), (x+0.35, yi), (x+0.35, yi-0.35), (x, yi-0.35)],
              color=color, strokewidth=1, strokecolor=:white)
        text!(ax, x+0.175, yi-0.175, text=sym, align=(:center, :center), fontsize=9, color=:white)
        text!(ax, x+0.5, yi-0.175, text=label, align=(:left, :center), fontsize=9)
    end
end

"""Plot a 2D TEBD circuit with two separate ring topologies.

The TEBD circuit consists of:
1. Hadamard layer: H gates on all m+n qubits (creates frequency basis)
2. Two separate rings of controlled-phase gates:
   - Row ring: (x1,x2), (x2,x3), ..., (x_{m-1},x_m), (x_m,x1) for m gates
   - Column ring: (y1,y2), (y2,y3), ..., (y_{n-1},y_n), (y_n,y1) for n gates
"""
function plot_circuit(spec::TEBDCircuitSpec; output_path=nothing)
    m = spec.n_row_qubits
    n = spec.n_col_qubits
    total = m + n
    phases = spec.phases
    n_row = m  # Row ring gates
    n_col = n  # Column ring gates
    n_gates = n_row + n_col
    
    # Account for Hadamard layer + TEBD gates + wrap-around gates
    width = max(700, 150 + (n_gates + 5) * 80)
    height = 120 + total * 70
    
    fig = Figure(size=(width, height), backgroundcolor=:white)
    ax = Axis(fig[1, 1], title=spec.title, titlesize=20, aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    
    x_start = 2.0
    x_end = x_start + (n_gates + 6) * GATE_SPACING
    
    # Draw row qubits (x₁ to xₘ)
    for q in 1:m
        y = -q * QUBIT_SPACING
        draw_qubit!(ax, y, x_start, x_end, "|x$q⟩", "x'$q")
    end
    
    # Separator line between row and column qubits
    y_sep = -(m + 0.5) * QUBIT_SPACING
    lines!(ax, [x_start, x_end], [y_sep, y_sep], 
           color=COLORS.label, linewidth=1, linestyle=:dash)
    
    # Draw column qubits (y₁ to yₙ)
    for q in 1:n
        y = -(m + q) * QUBIT_SPACING
        draw_qubit!(ax, y, x_start, x_end, "|y$q⟩", "y'$q")
    end
    
    x_pos = x_start + GATE_SPACING
    
    # Layer 1: Hadamard gates on all qubits
    for q in 1:total
        y_q = -q * QUBIT_SPACING
        draw_gate!(ax, x_pos, y_q, "H"; color=COLORS.hadamard)
    end
    x_pos += GATE_SPACING
    
    # Separator line between Hadamard layer and phase layers
    x_sep = x_pos + GATE_SPACING * 0.3
    lines!(ax, [x_sep, x_sep], [-0.3 * QUBIT_SPACING, -(total + 0.3) * QUBIT_SPACING], 
           color=COLORS.label, linewidth=1, linestyle=:dot)
    x_pos += GATE_SPACING * 0.8
    
    gate_idx = 1
    
    # Layer 2a: Row ring - controlled-phase gates on row qubits (x1 to xm)
    # Sequential gates: (1,2), (2,3), ..., (m-1,m)
    for i in 1:(m-1)
        y_i = -i * QUBIT_SPACING
        y_j = -(i+1) * QUBIT_SPACING
        phase = gate_idx <= length(phases) ? phases[gate_idx] : 0.0
        draw_two_qubit_gate!(ax, x_pos, y_i, y_j, "Tₓ$i", phase; 
                            color=COLORS.tebd_gate, show_phase=true)
        x_pos += GATE_SPACING
        gate_idx += 1
    end
    
    # Row ring wrap-around gate: (m,1)
    x_pos += GATE_SPACING * 0.3
    y1_row = -1 * QUBIT_SPACING
    ym_row = -m * QUBIT_SPACING
    phase = gate_idx <= length(phases) ? phases[gate_idx] : 0.0
    draw_two_qubit_gate!(ax, x_pos, ym_row, y1_row, "Tₓ$m", phase; 
                        color=COLORS.wrap_gate, show_phase=true)
    x_pos += GATE_SPACING
    gate_idx += 1
    
    # Small gap between row and column rings
    x_pos += GATE_SPACING * 0.3
    
    # Layer 2b: Column ring - controlled-phase gates on column qubits (y1 to yn)
    # Sequential gates: (m+1,m+2), (m+2,m+3), ..., (m+n-1,m+n)
    for i in 1:(n-1)
        y_i = -(m + i) * QUBIT_SPACING
        y_j = -(m + i + 1) * QUBIT_SPACING
        col_idx = i
        phase = gate_idx <= length(phases) ? phases[gate_idx] : 0.0
        draw_two_qubit_gate!(ax, x_pos, y_i, y_j, "Tᵧ$col_idx", phase; 
                            color=COLORS.entangle_gate, show_phase=true)
        x_pos += GATE_SPACING
        gate_idx += 1
    end
    
    # Column ring wrap-around gate: (m+n,m+1)
    x_pos += GATE_SPACING * 0.3
    y1_col = -(m + 1) * QUBIT_SPACING
    yn_col = -(m + n) * QUBIT_SPACING
    phase = gate_idx <= length(phases) ? phases[gate_idx] : 0.0
    draw_two_qubit_gate!(ax, x_pos, yn_col, y1_col, "Tᵧ$n", phase; 
                        color=COLORS.wrap_gate, show_phase=true)
    
    # Legend
    draw_legend!(ax, x_end + 0.5, -0.3, [
        (COLORS.hadamard, "H", "Hadamard"),
        (COLORS.tebd_gate, "Tₓ", "Row phase"),
        (COLORS.entangle_gate, "Tᵧ", "Col phase"),
        (COLORS.wrap_gate, "T", "Wrap-around")
    ])
    
    xlims!(ax, x_start - 1.5, x_end + 3.5)
    ylims!(ax, -(total + 0.5) * QUBIT_SPACING, 0.7 * QUBIT_SPACING)
    
    if output_path !== nothing
        save(output_path, fig)
        println("  Saved: $output_path")
    end
    
    return fig
end

# ================================================================================
# 2D Ring Topology Diagram
# ================================================================================

"""Generate a 2D ring topology diagram showing TEBD connectivity."""
function plot_ring_topology(m::Int, n::Int; title="TEBD 2D Ring Topology", output_path=nothing)
    fig = Figure(size=(700, 550), backgroundcolor=:white)
    ax = Axis(fig[1, 1], title="$title ($m row + $n col qubits)", titlesize=18, aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    
    # Row qubits in a ring (top)
    row_radius = 1.2
    row_center = (0.0, 1.5)
    row_angles = [2π * (i-1) / m - π/2 for i in 1:m]
    row_xs = row_center[1] .+ row_radius * cos.(row_angles)
    row_ys = row_center[2] .+ row_radius * sin.(row_angles)
    
    # Column qubits in a ring (bottom)
    col_radius = 1.2
    col_center = (0.0, -1.5)
    col_angles = [2π * (i-1) / n - π/2 for i in 1:n]
    col_xs = col_center[1] .+ col_radius * cos.(col_angles)
    col_ys = col_center[2] .+ col_radius * sin.(col_angles)
    
    # Draw row ring connections
    for i in 1:m
        j = i == m ? 1 : i + 1
        color = (i == m) ? COLORS.wrap_gate : COLORS.tebd_gate
        lines!(ax, [row_xs[i], row_xs[j]], [row_ys[i], row_ys[j]], 
               color=color, linewidth=3)
    end
    
    # Draw column ring connections
    for i in 1:n
        j = i == n ? 1 : i + 1
        color = (i == n) ? COLORS.wrap_gate : COLORS.entangle_gate
        lines!(ax, [col_xs[i], col_xs[j]], [col_ys[i], col_ys[j]], 
               color=color, linewidth=3)
    end
    
    # Draw row qubit nodes
    scatter!(ax, row_xs, row_ys, markersize=40, 
             color=COLORS.hadamard, strokewidth=2, strokecolor=:white)
    for i in 1:m
        text!(ax, row_xs[i], row_ys[i], text="H", align=(:center, :center), 
              fontsize=11, font=:bold, color=:white)
        # Label outside
        lx = row_center[1] + (row_radius + 0.5) * cos(row_angles[i])
        ly = row_center[2] + (row_radius + 0.5) * sin(row_angles[i])
        text!(ax, lx, ly, text="x$i", align=(:center, :center), 
              fontsize=11, color=COLORS.text)
    end
    
    # Draw column qubit nodes
    scatter!(ax, col_xs, col_ys, markersize=40, 
             color=COLORS.hadamard, strokewidth=2, strokecolor=:white)
    for i in 1:n
        text!(ax, col_xs[i], col_ys[i], text="H", align=(:center, :center), 
              fontsize=11, font=:bold, color=:white)
        # Label outside
        lx = col_center[1] + (col_radius + 0.5) * cos(col_angles[i])
        ly = col_center[2] + (col_radius + 0.5) * sin(col_angles[i])
        text!(ax, lx, ly, text="y$i", align=(:center, :center), 
              fontsize=11, color=COLORS.text)
    end
    
    # Labels
    text!(ax, 0, row_center[2] + row_radius + 0.9, text="Row ring (x qubits)", 
          align=(:center, :center), fontsize=13, font=:bold, color=COLORS.tebd_gate)
    text!(ax, 0, col_center[2] - col_radius - 0.9, text="Column ring (y qubits)", 
          align=(:center, :center), fontsize=13, font=:bold, color=COLORS.entangle_gate)
    
    # Description
    text!(ax, 0, 0, text="Layer 1: Hadamard on all qubits\nLayer 2a: Row ring Tₓ₁→...→Tₓ$(m) (wrap)\nLayer 2b: Col ring Tᵧ₁→...→Tᵧ$(n) (wrap)", 
          align=(:center, :center), fontsize=10, color=COLORS.label)
    
    xlims!(ax, -3.5, 3.5)
    ylims!(ax, -3.5, 3.5)
    
    if output_path !== nothing
        save(output_path, fig)
        println("  Saved: $output_path")
    end
    
    return fig
end

# ================================================================================
# Comparison Diagram
# ================================================================================

"""Generate a side-by-side comparison of standard vs entangled QFT."""
function plot_comparison(m::Int, n::Int; output_path=nothing)
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
        
        for q in 1:total
            y = -q * 0.7
            label = q <= m ? "x$q" : "y$(q-m)"
            lines!(ax, [0, x_end], [y, y], color=COLORS.wire, linewidth=2)
            text!(ax, -0.2, y, text=label, align=(:right, :center), fontsize=11)
        end
        
        for q in 1:total
            y = -q * 0.7
            poly!(ax, Point2f[(0.7, y-0.2), (1.1, y-0.2), (1.1, y+0.2), (0.7, y+0.2)],
                  color=COLORS.hadamard, strokewidth=1, strokecolor=:white)
            text!(ax, 0.9, y, text="H", align=(:center, :center), fontsize=9, color=:white)
        end
        
        text!(ax, 2.5, -(total+0.8)*0.7, text="M gates", align=(:center, :top), 
              fontsize=11, color=COLORS.phase_gate)
        
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
# Print Statistics
# ================================================================================

"""Print tensor network statistics for any circuit type."""
function print_circuit_stats(spec::QFTCircuitSpec)
    m, n = spec.n_row_qubits, spec.n_col_qubits
    if is_1d(spec)
        qc = Yao.EasyBuild.qft_circuit(m)
        println("\nQFT Circuit Statistics ($m qubits):")
    else
        optcode, tensors = qft_code(m, n)
        println("\nQFT Circuit Statistics ($(m)×$(n)):")
        println("  - Number of tensors: $(length(tensors))")
        println("  - Total parameters: $(sum(prod(size(t)) for t in tensors))")
    end
end

function print_circuit_stats(spec::EntangledQFTCircuitSpec)
    m, n = spec.n_row_qubits, spec.n_col_qubits
    optcode, tensors, n_ent = entangled_qft_code(m, n)
    println("\nEntangled QFT Statistics ($(m)×$(n)):")
    println("  - Number of tensors: $(length(tensors))")
    println("  - Entanglement gates: $n_ent")
    println("  - Total parameters: $(sum(prod(size(t)) for t in tensors))")
end

function print_circuit_stats(spec::TEBDCircuitSpec)
    m, n = spec.n_row_qubits, spec.n_col_qubits
    total = m + n
    optcode, tensors, n_row, n_col = tebd_code(m, n)
    n_hadamards = total
    n_gates = n_row + n_col
    println("\nTEBD Circuit Statistics ($(m)×$(n) = $total qubits):")
    println("  - Row qubits: $m")
    println("  - Column qubits: $n")
    println("  - Hadamard gates: $n_hadamards (Layer 1)")
    println("  - Row ring gates: $n_row (Layer 2a)")
    println("  - Column ring gates: $n_col (Layer 2b)")
    println("  - Row ring: (x₁,x₂)→...→(xₘ,x₁)")
    println("  - Col ring: (y₁,y₂)→...→(yₙ,y₁)")
    println("  - Total tensors: $(length(tensors)) ($n_hadamards H + $n_gates phase)")
    println("  - Total parameters: $(sum(prod(size(t)) for t in tensors))")
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
    
    # QFT circuits
    println("1. Single QFT (3 qubits)")
    plot_circuit(QFTCircuitSpec(3; title="QFT Circuit (3 qubits)");
        output_path=joinpath(output_dir, "qft_3qubit.png"))
    
    println("2. 2D QFT (3×3)")
    plot_circuit(QFTCircuitSpec(3, 3; title="2D QFT Circuit (3×3)");
        output_path=joinpath(output_dir, "qft_2d_3x3.png"))
    
    # Entangled QFT circuits
    println("3. Entangled QFT (3×3, φ=0)")
    plot_circuit(EntangledQFTCircuitSpec(3, 3; title="Entangled QFT (3×3, φ=0)");
        output_path=joinpath(output_dir, "entangled_qft_3x3_zero.png"))
    
    println("4. Entangled QFT (3×3, trained)")
    plot_circuit(EntangledQFTCircuitSpec(3, 3; phases=[π/4, π/3, π/6], 
        title="Entangled QFT (3×3, trained)");
        output_path=joinpath(output_dir, "entangled_qft_3x3_trained.png"))
    
    # TEBD circuits (2D ring topology)
    println("5. TEBD Circuit (3×3 qubits)")
    plot_circuit(TEBDCircuitSpec(3, 3; title="TEBD Circuit (3×3 qubits, ring topology)");
        output_path=joinpath(output_dir, "tebd_3x3.png"))
    print_circuit_stats(TEBDCircuitSpec(3, 3))
    
    println("6. TEBD Circuit (4×4 qubits)")
    plot_circuit(TEBDCircuitSpec(4, 4; title="TEBD Circuit (4×4 qubits, ring topology)");
        output_path=joinpath(output_dir, "tebd_4x4.png"))
    print_circuit_stats(TEBDCircuitSpec(4, 4))
    
    println("7. TEBD with trained phases (3×3)")
    # 3×3: 3 row ring gates + 3 col ring gates = 6 total
    plot_circuit(TEBDCircuitSpec(3, 3; phases=[π/4, π/3, π/2, π/6, π/5, π/7],
        title="TEBD Circuit (3×3 qubits, trained)");
        output_path=joinpath(output_dir, "tebd_3x3_trained.png"))
    
    # Ring topology diagram
    println("8. Ring Topology Diagram")
    plot_ring_topology(4, 4; output_path=joinpath(output_dir, "tebd_ring_4x4.png"))
    
    # TEBD asymmetric
    println("9. TEBD Circuit (3×4 qubits)")
    plot_circuit(TEBDCircuitSpec(3, 4; title="TEBD Circuit (3×4 qubits, asymmetric)");
        output_path=joinpath(output_dir, "tebd_3x4.png"))
    print_circuit_stats(TEBDCircuitSpec(3, 4))
    
    println("10. TEBD Circuit (5×5, trained)")
    # 5×5: 5 row ring gates + 5 col ring gates = 10 total
    plot_circuit(TEBDCircuitSpec(5, 5; phases=[π/6, π/4, π/3, π/2, π/5, π/7, π/8, π/9, π/10, π/11],
        title="TEBD Circuit (5×5 qubits, trained)");
        output_path=joinpath(output_dir, "tebd_5x5_trained.png"))
    
    # Comparison
    println("11. QFT vs Entangled Comparison")
    plot_comparison(3, 3; output_path=joinpath(output_dir, "comparison_3x3.png"))
    
    println("\n" * "="^60)
    println("Done! Generated 11 circuit diagrams.")
    println("="^60)
end

main()
