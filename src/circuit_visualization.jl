# ============================================================================
# Circuit Visualization
# ============================================================================
# Professional circuit diagram generation for quantum circuit types.
# Provides drawing primitives and per-basis `plot_circuit` methods.

# ============================================================================
# Constants
# ============================================================================

"""Color palette for circuit diagrams."""
const CIRCUIT_COLORS = (
    wire = "#333333",
    hadamard = "#4285F4",       # Blue
    phase_gate = "#34A853",     # Green
    entangle_gate = "#EA4335",  # Red
    tebd_gate = "#9C27B0",      # Purple
    mera_gate = "#FF9800",      # Orange
    wrap_gate = "#FF5722",      # Deep Orange
    text = "#333333",
    label = "#666666"
)

"""Default gate box size (side length)."""
const CIRCUIT_GATE_SIZE = 0.6

"""Default wire stroke width."""
const CIRCUIT_WIRE_WIDTH = 2

"""Default vertical spacing between qubit wires."""
const CIRCUIT_QUBIT_SPACING = 1.0

"""Default horizontal spacing between gates."""
const CIRCUIT_GATE_SPACING = 1.2

# ============================================================================
# Internal Spec Types
# ============================================================================

"""Abstract base type for internal circuit layout specifications."""
abstract type _AbstractCircuitSpec end

"""QFT circuit specification (1D when `n_col_qubits == 0`, 2D otherwise)."""
struct _QFTCircuitSpec <: _AbstractCircuitSpec
    n_row_qubits::Int
    n_col_qubits::Int
    title::String
end

_QFTCircuitSpec(m::Int; title="QFT Circuit") = _QFTCircuitSpec(m, 0, title)
_QFTCircuitSpec(m::Int, n::Int; title="2D QFT Circuit") = _QFTCircuitSpec(m, n, title)

_is_1d(spec::_QFTCircuitSpec) = spec.n_col_qubits == 0
_total_qubits(spec::_QFTCircuitSpec) = spec.n_row_qubits + spec.n_col_qubits

"""Entangled QFT circuit specification."""
struct _EntangledQFTCircuitSpec <: _AbstractCircuitSpec
    n_row_qubits::Int
    n_col_qubits::Int
    entangle_phases::Vector{Float64}
    entangle_position::Symbol
    title::String
end

"""TEBD circuit specification."""
struct _TEBDCircuitSpec <: _AbstractCircuitSpec
    n_row_qubits::Int
    n_col_qubits::Int
    phases::Vector{Float64}
    title::String
end

"""MERA circuit specification."""
struct _MERACircuitSpec <: _AbstractCircuitSpec
    n_row_qubits::Int
    n_col_qubits::Int
    phases::Vector{Float64}
    title::String
end

# ============================================================================
# Drawing Primitives
# ============================================================================

"""
    _draw_qubit!(ax, y, x_start, x_end, label_in, label_out)

Draw a horizontal qubit wire from `x_start` to `x_end` at vertical position `y`,
with input label on the left and output label on the right.
"""
function _draw_qubit!(ax, y, x_start, x_end, label_in, label_out)
    lines!(ax, [x_start, x_end], [y, y],
           color=CIRCUIT_COLORS.wire, linewidth=CIRCUIT_WIRE_WIDTH)
    text!(ax, x_start - 0.5, y, text=label_in, align=(:right, :center),
          fontsize=14, font=:bold, color=CIRCUIT_COLORS.text)
    text!(ax, x_end + 0.5, y, text=label_out, align=(:left, :center),
          fontsize=14, color=CIRCUIT_COLORS.label)
end

"""
    _draw_gate!(ax, x, y, label; color=CIRCUIT_COLORS.hadamard, size=CIRCUIT_GATE_SIZE)

Draw a colored gate box centred at `(x, y)` with a text label.
"""
function _draw_gate!(ax, x, y, label;
                     color=CIRCUIT_COLORS.hadamard,
                     size=CIRCUIT_GATE_SIZE)
    half = size / 2
    poly!(ax,
          Point2f[(x-half, y-half), (x+half, y-half),
                  (x+half, y+half), (x-half, y+half)],
          color=color, strokewidth=2, strokecolor=:white)
    text!(ax, x, y, text=label, align=(:center, :center),
          fontsize=14, font=:bold, color=:white)
end

"""
    _draw_controlled_gate!(ax, x, y_ctrl, y_target, label; color=CIRCUIT_COLORS.phase_gate)

Draw a controlled gate: a control dot at `y_ctrl`, a vertical wire, and a gate box
at `y_target`.
"""
function _draw_controlled_gate!(ax, x, y_ctrl, y_target, label;
                                color=CIRCUIT_COLORS.phase_gate)
    scatter!(ax, [x], [y_ctrl], markersize=12, color=CIRCUIT_COLORS.wire)
    lines!(ax, [x, x], [y_ctrl, y_target],
           color=CIRCUIT_COLORS.wire, linewidth=CIRCUIT_WIRE_WIDTH)
    _draw_gate!(ax, x, y_target, label; color=color)
end

"""
    _draw_two_qubit_gate!(ax, x, y1, y2, label, phase; color=CIRCUIT_COLORS.entangle_gate, show_phase=true)

Draw a two-qubit gate connecting wires at `y1` and `y2`: two dots, a connecting
wire, a gate box at the midpoint, and an optional phase annotation.
"""
function _draw_two_qubit_gate!(ax, x, y1, y2, label, phase;
                               color=CIRCUIT_COLORS.entangle_gate,
                               show_phase=true)
    scatter!(ax, [x, x], [y1, y2], markersize=12, color=color)
    lines!(ax, [x, x], [y1, y2], color=color, linewidth=CIRCUIT_WIRE_WIDTH + 1)

    y_mid = (y1 + y2) / 2
    _draw_gate!(ax, x, y_mid, label; color=color, size=CIRCUIT_GATE_SIZE * 0.8)

    if show_phase
        phase_str = "phi=$(round(phase, digits=2))"
        text!(ax, x + CIRCUIT_GATE_SIZE, y_mid, text=phase_str,
              align=(:left, :center), fontsize=10, color=CIRCUIT_COLORS.label)
    end
end

"""
    _draw_legend!(ax, x, y, items)

Draw a gate colour legend. `items` is a vector of `(color, symbol, description)`
tuples.
"""
function _draw_legend!(ax, x, y, items)
    text!(ax, x, y, text="Gates:", align=(:left, :top), fontsize=11, font=:bold)

    for (i, (color, sym, label)) in enumerate(items)
        yi = y - i * 0.55
        poly!(ax, Point2f[(x, yi), (x+0.35, yi),
                          (x+0.35, yi-0.35), (x, yi-0.35)],
              color=color, strokewidth=1, strokecolor=:white)
        text!(ax, x + 0.175, yi - 0.175, text=sym,
              align=(:center, :center), fontsize=9, color=:white)
        text!(ax, x + 0.5, yi - 0.175, text=label,
              align=(:left, :center), fontsize=9)
    end
end

# ============================================================================
# QFT Circuit Drawing
# ============================================================================

"""
    _draw_qft_layer!(ax, x_start, n_qubits, y_offset)

Draw QFT gates (Hadamard + controlled M_k phase gates) for a group of qubits.

# Arguments
- `ax`: CairoMakie axis
- `x_start`: horizontal position for the first gate
- `n_qubits`: number of qubits in this QFT layer
- `y_offset`: vertical offset applied to all qubit positions

# Returns
- `x_pos`: the horizontal position after the last gate drawn
"""
function _draw_qft_layer!(ax, x_start, n_qubits, y_offset)
    x_pos = x_start

    for q in 1:n_qubits
        y_q = y_offset - q * CIRCUIT_QUBIT_SPACING

        # Hadamard gate
        _draw_gate!(ax, x_pos, y_q, "H"; color=CIRCUIT_COLORS.hadamard)
        x_pos += CIRCUIT_GATE_SPACING

        # Controlled phase gates
        for target in (q+1):n_qubits
            y_target = y_offset - target * CIRCUIT_QUBIT_SPACING
            k = target - q + 1
            _draw_controlled_gate!(ax, x_pos, y_q, y_target, "M$k";
                                   color=CIRCUIT_COLORS.phase_gate)
            x_pos += CIRCUIT_GATE_SPACING
        end
    end

    return x_pos
end

"""
    _plot_circuit(spec::_QFTCircuitSpec; output_path=nothing)

Render a full QFT circuit diagram. Produces a 1D layout when
`spec.n_col_qubits == 0` and a 2D layout (row + column QFT blocks separated
by a dashed line) otherwise.

# Arguments
- `spec::_QFTCircuitSpec`: circuit layout specification
- `output_path`: optional file path; when provided the figure is saved as PNG

# Returns
- `Figure`: the CairoMakie figure object
"""
function _plot_circuit(spec::_QFTCircuitSpec; output_path=nothing)
    m = spec.n_row_qubits
    n = spec.n_col_qubits
    total = _is_1d(spec) ? m : m + n

    # Calculate gate counts for sizing
    n_m_gates = div(m * (m - 1), 2)
    n_n_gates = _is_1d(spec) ? 0 : div(n * (n - 1), 2)
    total_gates = m + n_m_gates + (_is_1d(spec) ? 0 : n + n_n_gates)

    width = max(700, 150 + total_gates * 55)
    height = 120 + total * 70

    fig = Figure(size=(width, height), backgroundcolor=:white)
    ax = Axis(fig[1, 1], title=spec.title, titlesize=20, aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)

    x_start = 1.5
    x_qft_end = x_start + (m + n_m_gates + 1) * CIRCUIT_GATE_SPACING
    x_end = x_qft_end

    # Draw qubit wires
    if _is_1d(spec)
        for q in 1:m
            y = -q * CIRCUIT_QUBIT_SPACING
            _draw_qubit!(ax, y, x_start, x_end, "|q$q>", "k$q")
        end
    else
        for q in 1:m
            y = -q * CIRCUIT_QUBIT_SPACING
            _draw_qubit!(ax, y, x_start, x_end, "|x$q>", "kx$q")
        end
        for q in 1:n
            y = -(m + q) * CIRCUIT_QUBIT_SPACING
            _draw_qubit!(ax, y, x_start, x_end, "|y$q>", "ky$q")
        end
        # Separator between row and column qubits
        y_sep = -(m + 0.5) * CIRCUIT_QUBIT_SPACING
        lines!(ax, [x_start, x_end], [y_sep, y_sep],
               color=CIRCUIT_COLORS.label, linewidth=1, linestyle=:dash)
    end

    # Draw QFT gates for row qubits
    x_pos = x_start + CIRCUIT_GATE_SPACING
    x_pos = _draw_qft_layer!(ax, x_pos, m, 0)

    # Draw QFT gates for column qubits (2D only)
    if !_is_1d(spec)
        x_pos_col = x_start + CIRCUIT_GATE_SPACING
        _draw_qft_layer!(ax, x_pos_col, n, -m * CIRCUIT_QUBIT_SPACING)
    end

    xlims!(ax, x_start - 1.5, x_end + 1.5)
    ylims!(ax, -(total + 0.5) * CIRCUIT_QUBIT_SPACING, 0.7 * CIRCUIT_QUBIT_SPACING)

    if output_path !== nothing
        save(output_path, fig)
    end

    return fig
end

# ============================================================================
# Public API
# ============================================================================

"""
    plot_circuit(basis::AbstractSparseBasis; output_path=nothing)

Generate a professional circuit diagram for a sparse basis.

# Arguments
- `basis::AbstractSparseBasis`: the basis whose circuit to visualize
- `output_path`: optional file path to save the figure (PNG recommended)

# Returns
- `Figure`: the CairoMakie figure object

# Example
```julia
basis = QFTBasis(3, 3)
fig = plot_circuit(basis)

# Save to file
plot_circuit(basis; output_path="qft_circuit.png")
```
"""
function plot_circuit end

"""
    plot_circuit(basis::QFTBasis; output_path=nothing)

Draw a QFT circuit diagram for the given `QFTBasis`. Produces a 1D layout when
the basis has zero column qubits, and a 2D layout otherwise.

# Arguments
- `basis::QFTBasis`: the QFT basis to visualize
- `output_path`: optional file path to save the figure

# Returns
- `Figure`: the CairoMakie figure object
"""
function plot_circuit(basis::QFTBasis; output_path=nothing)
    m = basis.m
    n = basis.n
    title = n == 0 ? "QFT Circuit ($m qubits)" : "2D QFT Circuit ($(m)x$(n))"
    spec = _QFTCircuitSpec(m, n; title=title)
    return _plot_circuit(spec; output_path=output_path)
end

# ============================================================================
# Entangled QFT Circuit Drawing
# ============================================================================

"""
    _draw_entangle_layer!(ax, x_start, m, n, phases, y_offset_row, y_offset_col)

Draw entanglement gates connecting row and column qubits.

# Arguments
- `ax`: CairoMakie axis
- `x_start`: horizontal position for the first gate
- `m`: number of row qubits
- `n`: number of column qubits
- `phases`: entanglement phase values
- `y_offset_row`: vertical offset for row qubits
- `y_offset_col`: vertical offset for column qubits

# Returns
- `x_pos`: the horizontal position after the last gate drawn
"""
function _draw_entangle_layer!(ax, x_start, m, n, phases, y_offset_row, y_offset_col)
    x_pos = x_start
    n_entangle = min(m, n)

    for i in 1:n_entangle
        y_row = y_offset_row - i * CIRCUIT_QUBIT_SPACING
        y_col = y_offset_col - i * CIRCUIT_QUBIT_SPACING
        phase = i <= length(phases) ? phases[i] : 0.0
        _draw_two_qubit_gate!(ax, x_pos, y_row, y_col, "E", phase;
                              color=CIRCUIT_COLORS.entangle_gate)
        x_pos += CIRCUIT_GATE_SPACING
    end

    return x_pos
end

"""
    _plot_circuit(spec::_EntangledQFTCircuitSpec; output_path=nothing)

Render a full Entangled QFT circuit diagram. Shows QFT blocks for row and column
qubits with entanglement gates placed according to `spec.entangle_position`.

# Arguments
- `spec::_EntangledQFTCircuitSpec`: circuit layout specification
- `output_path`: optional file path; when provided the figure is saved as PNG

# Returns
- `Figure`: the CairoMakie figure object
"""
function _plot_circuit(spec::_EntangledQFTCircuitSpec; output_path=nothing)
    m = spec.n_row_qubits
    n = spec.n_col_qubits
    total = m + n

    n_m_gates = div(m * (m - 1), 2)
    n_n_gates = div(n * (n - 1), 2)
    n_entangle = min(m, n)
    total_gates = m + n_m_gates + n + n_n_gates + n_entangle

    width = max(700, 150 + total_gates * 55)
    height = 120 + total * 70

    fig = Figure(size=(width, height), backgroundcolor=:white)
    ax = Axis(fig[1, 1], title=spec.title, titlesize=20, aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)

    x_start = 1.5
    x_end = x_start + (total_gates + 2) * CIRCUIT_GATE_SPACING

    # Draw qubit wires
    for q in 1:m
        y = -q * CIRCUIT_QUBIT_SPACING
        _draw_qubit!(ax, y, x_start, x_end, "|x$q⟩", "kx$q")
    end
    for q in 1:n
        y = -(m + q) * CIRCUIT_QUBIT_SPACING
        _draw_qubit!(ax, y, x_start, x_end, "|y$q⟩", "ky$q")
    end

    # Separator between row and column qubits
    y_sep = -(m + 0.5) * CIRCUIT_QUBIT_SPACING
    lines!(ax, [x_start, x_end], [y_sep, y_sep],
           color=CIRCUIT_COLORS.label, linewidth=1, linestyle=:dash)

    x_pos = x_start + CIRCUIT_GATE_SPACING

    if spec.entangle_position == :front
        x_pos = _draw_entangle_layer!(ax, x_pos, m, n, spec.entangle_phases, 0, -m * CIRCUIT_QUBIT_SPACING)
        x_pos += CIRCUIT_GATE_SPACING * 0.5
    end

    # QFT for row qubits
    x_row = _draw_qft_layer!(ax, x_pos, m, 0)

    # QFT for column qubits (same x start)
    _draw_qft_layer!(ax, x_pos, n, -m * CIRCUIT_QUBIT_SPACING)

    x_pos = x_row

    if spec.entangle_position == :middle
        x_pos += CIRCUIT_GATE_SPACING * 0.5
        x_pos = _draw_entangle_layer!(ax, x_pos, m, n, spec.entangle_phases, 0, -m * CIRCUIT_QUBIT_SPACING)
    end

    if spec.entangle_position == :back
        x_pos += CIRCUIT_GATE_SPACING * 0.5
        x_pos = _draw_entangle_layer!(ax, x_pos, m, n, spec.entangle_phases, 0, -m * CIRCUIT_QUBIT_SPACING)
    end

    xlims!(ax, x_start - 1.5, x_end + 1.5)
    ylims!(ax, -(total + 0.5) * CIRCUIT_QUBIT_SPACING, 0.7 * CIRCUIT_QUBIT_SPACING)

    if output_path !== nothing
        save(output_path, fig)
    end

    return fig
end

"""
    plot_circuit(basis::EntangledQFTBasis; output_path=nothing)

Draw an Entangled QFT circuit diagram showing QFT blocks and entanglement gates.

# Arguments
- `basis::EntangledQFTBasis`: the entangled QFT basis to visualize
- `output_path`: optional file path to save the figure

# Returns
- `Figure`: the CairoMakie figure object
"""
function plot_circuit(basis::EntangledQFTBasis; output_path=nothing)
    title = "Entangled QFT Circuit ($(basis.m)x$(basis.n), $(basis.entangle_position))"
    spec = _EntangledQFTCircuitSpec(basis.m, basis.n,
                                     Float64.(basis.entangle_phases),
                                     basis.entangle_position, title)
    return _plot_circuit(spec; output_path=output_path)
end

# ============================================================================
# TEBD Circuit Drawing
# ============================================================================

"""
    _draw_tebd_layer!(ax, x_start, n_qubits, phases, y_offset)

Draw TEBD nearest-neighbour two-qubit gates in a ring topology.

# Arguments
- `ax`: CairoMakie axis
- `x_start`: horizontal position for the first gate
- `n_qubits`: number of qubits in this TEBD layer
- `phases`: phase values for each gate
- `y_offset`: vertical offset applied to all qubit positions

# Returns
- `x_pos`: the horizontal position after the last gate drawn
"""
function _draw_tebd_layer!(ax, x_start, n_qubits, phases, y_offset)
    x_pos = x_start

    for i in 1:n_qubits
        q1 = i
        q2 = mod1(i + 1, n_qubits)
        y1 = y_offset - q1 * CIRCUIT_QUBIT_SPACING
        y2 = y_offset - q2 * CIRCUIT_QUBIT_SPACING
        phase = i <= length(phases) ? phases[i] : 0.0

        if q2 == 1 && n_qubits > 2
            # Wrap-around gate (last connects to first)
            _draw_two_qubit_gate!(ax, x_pos, y1, y2, "T", phase;
                                  color=CIRCUIT_COLORS.wrap_gate, show_phase=true)
        else
            _draw_two_qubit_gate!(ax, x_pos, y1, y2, "T", phase;
                                  color=CIRCUIT_COLORS.tebd_gate, show_phase=true)
        end
        x_pos += CIRCUIT_GATE_SPACING
    end

    return x_pos
end

"""
    _plot_circuit(spec::_TEBDCircuitSpec; output_path=nothing)

Render a full TEBD circuit diagram with nearest-neighbour gates in ring topology.

# Arguments
- `spec::_TEBDCircuitSpec`: circuit layout specification
- `output_path`: optional file path; when provided the figure is saved as PNG

# Returns
- `Figure`: the CairoMakie figure object
"""
function _plot_circuit(spec::_TEBDCircuitSpec; output_path=nothing)
    m = spec.n_row_qubits
    n = spec.n_col_qubits
    total = m + n
    n_gates = m + n

    width = max(700, 150 + n_gates * 55)
    height = 120 + total * 70

    fig = Figure(size=(width, height), backgroundcolor=:white)
    ax = Axis(fig[1, 1], title=spec.title, titlesize=20, aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)

    x_start = 1.5
    x_end = x_start + (n_gates + 2) * CIRCUIT_GATE_SPACING

    # Draw qubit wires
    for q in 1:m
        y = -q * CIRCUIT_QUBIT_SPACING
        _draw_qubit!(ax, y, x_start, x_end, "|x$q⟩", "kx$q")
    end
    for q in 1:n
        y = -(m + q) * CIRCUIT_QUBIT_SPACING
        _draw_qubit!(ax, y, x_start, x_end, "|y$q⟩", "ky$q")
    end

    # Separator
    y_sep = -(m + 0.5) * CIRCUIT_QUBIT_SPACING
    lines!(ax, [x_start, x_end], [y_sep, y_sep],
           color=CIRCUIT_COLORS.label, linewidth=1, linestyle=:dash)

    x_pos = x_start + CIRCUIT_GATE_SPACING

    # Row TEBD gates
    row_phases = spec.phases[1:min(m, length(spec.phases))]
    x_pos = _draw_tebd_layer!(ax, x_pos, m, row_phases, 0)

    # Column TEBD gates
    col_phases = length(spec.phases) > m ? spec.phases[m+1:end] : Float64[]
    _draw_tebd_layer!(ax, x_pos, n, col_phases, -m * CIRCUIT_QUBIT_SPACING)

    xlims!(ax, x_start - 1.5, x_end + 1.5)
    ylims!(ax, -(total + 0.5) * CIRCUIT_QUBIT_SPACING, 0.7 * CIRCUIT_QUBIT_SPACING)

    if output_path !== nothing
        save(output_path, fig)
    end

    return fig
end

"""
    plot_circuit(basis::TEBDBasis; output_path=nothing)

Draw a TEBD circuit diagram showing nearest-neighbour two-qubit gates in ring topology.

# Arguments
- `basis::TEBDBasis`: the TEBD basis to visualize
- `output_path`: optional file path to save the figure

# Returns
- `Figure`: the CairoMakie figure object
"""
function plot_circuit(basis::TEBDBasis; output_path=nothing)
    title = "TEBD Circuit ($(basis.m)x$(basis.n))"
    spec = _TEBDCircuitSpec(basis.m, basis.n, Float64.(basis.phases), title)
    return _plot_circuit(spec; output_path=output_path)
end

# ============================================================================
# MERA Circuit Drawing
# ============================================================================

"""
    _draw_mera_layer!(ax, x_start, n_qubits, phases, y_offset)

Draw MERA disentangler and isometry gates for a group of qubits.
MERA uses stride-doubling layers: disentanglers on pairs (1,2), (3,4), ...
followed by isometries on pairs (2,3), (4,5), ...

# Arguments
- `ax`: CairoMakie axis
- `x_start`: horizontal position for the first gate
- `n_qubits`: number of qubits in this MERA layer
- `phases`: phase values for each gate
- `y_offset`: vertical offset applied to all qubit positions

# Returns
- `x_pos`: the horizontal position after the last gate drawn
"""
function _draw_mera_layer!(ax, x_start, n_qubits, phases, y_offset)
    x_pos = x_start
    phase_idx = 0

    if n_qubits < 2
        return x_pos
    end

    # Disentanglers: pairs (1,2), (3,4), ...
    for i in 1:2:(n_qubits - 1)
        y1 = y_offset - i * CIRCUIT_QUBIT_SPACING
        y2 = y_offset - (i + 1) * CIRCUIT_QUBIT_SPACING
        phase_idx += 1
        phase = phase_idx <= length(phases) ? phases[phase_idx] : 0.0
        _draw_two_qubit_gate!(ax, x_pos, y1, y2, "D", phase;
                              color=CIRCUIT_COLORS.mera_gate, show_phase=true)
        x_pos += CIRCUIT_GATE_SPACING
    end

    x_pos += CIRCUIT_GATE_SPACING * 0.3

    # Isometries: pairs (2,3), (4,5), ...
    for i in 2:2:(n_qubits - 1)
        y1 = y_offset - i * CIRCUIT_QUBIT_SPACING
        y2 = y_offset - (i + 1) * CIRCUIT_QUBIT_SPACING
        phase_idx += 1
        phase = phase_idx <= length(phases) ? phases[phase_idx] : 0.0
        _draw_two_qubit_gate!(ax, x_pos, y1, y2, "I", phase;
                              color=CIRCUIT_COLORS.mera_gate, show_phase=true)
        x_pos += CIRCUIT_GATE_SPACING
    end

    return x_pos
end

"""
    _plot_circuit(spec::_MERACircuitSpec; output_path=nothing)

Render a full MERA circuit diagram with disentangler and isometry layers.

# Arguments
- `spec::_MERACircuitSpec`: circuit layout specification
- `output_path`: optional file path; when provided the figure is saved as PNG

# Returns
- `Figure`: the CairoMakie figure object
"""
function _plot_circuit(spec::_MERACircuitSpec; output_path=nothing)
    m = spec.n_row_qubits
    n = spec.n_col_qubits
    total = m + n

    n_row_gates = m >= 2 ? 2 * (m - 1) : 0
    n_col_gates = n >= 2 ? 2 * (n - 1) : 0
    total_gates = n_row_gates + n_col_gates

    width = max(700, 150 + max(total_gates, 4) * 55)
    height = 120 + total * 70

    fig = Figure(size=(width, height), backgroundcolor=:white)
    ax = Axis(fig[1, 1], title=spec.title, titlesize=20, aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)

    x_start = 1.5
    x_end = x_start + (max(total_gates, 4) + 2) * CIRCUIT_GATE_SPACING

    # Draw qubit wires
    for q in 1:m
        y = -q * CIRCUIT_QUBIT_SPACING
        _draw_qubit!(ax, y, x_start, x_end, "|x$q⟩", "kx$q")
    end
    for q in 1:n
        y = -(m + q) * CIRCUIT_QUBIT_SPACING
        _draw_qubit!(ax, y, x_start, x_end, "|y$q⟩", "ky$q")
    end

    # Separator
    if n > 0
        y_sep = -(m + 0.5) * CIRCUIT_QUBIT_SPACING
        lines!(ax, [x_start, x_end], [y_sep, y_sep],
               color=CIRCUIT_COLORS.label, linewidth=1, linestyle=:dash)
    end

    x_pos = x_start + CIRCUIT_GATE_SPACING

    # Row MERA gates
    row_phases = spec.phases[1:min(n_row_gates, length(spec.phases))]
    x_pos = _draw_mera_layer!(ax, x_pos, m, row_phases, 0)

    # Column MERA gates
    col_phases = length(spec.phases) > n_row_gates ? spec.phases[n_row_gates+1:end] : Float64[]
    _draw_mera_layer!(ax, x_pos, n, col_phases, -m * CIRCUIT_QUBIT_SPACING)

    xlims!(ax, x_start - 1.5, x_end + 1.5)
    ylims!(ax, -(total + 0.5) * CIRCUIT_QUBIT_SPACING, 0.7 * CIRCUIT_QUBIT_SPACING)

    if output_path !== nothing
        save(output_path, fig)
    end

    return fig
end

"""
    plot_circuit(basis::MERABasis; output_path=nothing)

Draw a MERA circuit diagram showing disentangler and isometry layers.

# Arguments
- `basis::MERABasis`: the MERA basis to visualize
- `output_path`: optional file path to save the figure

# Returns
- `Figure`: the CairoMakie figure object
"""
function plot_circuit(basis::MERABasis; output_path=nothing)
    title = "MERA Circuit ($(basis.m)x$(basis.n))"
    spec = _MERACircuitSpec(basis.m, basis.n, Float64.(basis.phases), title)
    return _plot_circuit(spec; output_path=output_path)
end
