# ================================================================================
# MERA Circuit Visualization
# ================================================================================
# Generates circuit diagrams for the Multi-scale Entanglement Renormalization
# Ansatz (MERA), a hierarchical tensor network with alternating layers of:
#   - Disentanglers D_k ∈ U(4): two-qubit unitaries removing short-range entanglement
#   - Isometries   W_k ∈ St(2,4): 2-to-1 coarse-graining maps
#
# For n = 2^k qubits, MERA has k layers. Each layer halves the effective qubits.
# Total gates: (n-1) disentanglers + (n-1) isometries.
#
# Run with: julia --project=examples examples/mera_circuit_visualization.jl
# ================================================================================

using CairoMakie

CairoMakie.activate!(type = "png", px_per_unit = 2)

# ================================================================================
# Configuration
# ================================================================================

const COLORS = (
    wire = "#333333",
    wire_inactive = "#CCCCCC",
    disentangler = "#E91E63",     # Pink/Magenta
    isometry = "#FF9800",          # Orange
    text = "#333333",
    label = "#666666",
    layer_sep = "#BBBBBB",
    layer_bg_even = ("#E91E63", 0.06),
    layer_bg_odd = ("#FF9800", 0.06),
)

const GATE_SIZE = 0.6
const WIRE_WIDTH = 2
const QUBIT_SPACING = 1.0
const GATE_SPACING = 1.2

# ================================================================================
# Drawing Primitives
# ================================================================================

"""Draw a gate box with label."""
function draw_gate!(ax, x, y, label; color=COLORS.disentangler, size=GATE_SIZE)
    half = size / 2
    poly!(ax, Point2f[(x-half, y-half), (x+half, y-half), (x+half, y+half), (x-half, y+half)],
          color=color, strokewidth=2, strokecolor=:white)
    text!(ax, x, y, text=label, align=(:center, :center),
          fontsize=14, font=:bold, color=:white)
end

"""Draw a disentangler gate spanning two qubits (rectangular box)."""
function draw_disentangler!(ax, x, y1, y2, label)
    color = COLORS.disentangler
    scatter!(ax, [x, x], [y1, y2], markersize=12, color=color)
    lines!(ax, [x, x], [y1, y2], color=color, linewidth=WIRE_WIDTH+1)
    y_mid = (y1 + y2) / 2
    draw_gate!(ax, x, y_mid, label; color=color, size=GATE_SIZE * 0.9)
end

"""Draw an isometry gate spanning two qubits (trapezoid: 2 inputs → 1 output).

The trapezoid is wider on the left (input side) and narrower on the right (output).
`y_keep` is the surviving qubit wire, `y_discard` is the coarse-grained one.
"""
function draw_isometry!(ax, x, y_keep, y_discard, label)
    color = COLORS.isometry

    # Dots on both wires
    scatter!(ax, [x, x], [y_keep, y_discard], markersize=12, color=color)
    lines!(ax, [x, x], [y_keep, y_discard], color=color, linewidth=WIRE_WIDTH+1)

    # Trapezoid: wider on left (2 inputs), narrower on right (1 output)
    y_mid = (y_keep + y_discard) / 2
    y_span = abs(y_discard - y_keep)
    w = GATE_SIZE * 0.5
    h_in = min(y_span * 0.4, GATE_SIZE * 0.55)
    h_out = h_in * 0.35

    poly!(ax, Point2f[
        (x - w, y_mid - h_in),
        (x + w, y_mid - h_out),
        (x + w, y_mid + h_out),
        (x - w, y_mid + h_in)
    ], color=color, strokewidth=2, strokecolor=:white)
    text!(ax, x, y_mid, text=label, align=(:center, :center),
          fontsize=12, font=:bold, color=:white)
end

"""Draw a wire termination marker (×)."""
function draw_termination!(ax, x, y)
    s = 0.12
    lines!(ax, [x-s, x+s], [y-s, y+s], color=COLORS.wire_inactive, linewidth=2.5)
    lines!(ax, [x-s, x+s], [y+s, y-s], color=COLORS.wire_inactive, linewidth=2.5)
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

# ================================================================================
# MERA Circuit Layout
# ================================================================================

"""
    compute_mera_layout(n_qubits)

Compute positions of all gates and wire terminations for a binary MERA circuit.
Returns `(disentanglers, isometries, terminations, layer_x_ranges, x_end)` where
each disentangler/isometry entry is `(x, qubit_i, qubit_j, label, layer)`.
"""
function compute_mera_layout(n_qubits::Int)
    k = Int(log2(n_qubits))
    @assert 2^k == n_qubits "n_qubits must be a power of 2"

    active = collect(1:n_qubits)
    x_pos = 3.5

    disentanglers = Tuple{Float64,Int,Int,String,Int}[]
    isometries    = Tuple{Float64,Int,Int,String,Int}[]
    terminations  = Dict{Int,Float64}()       # qubit => x where terminated
    layer_x_ranges = Tuple{Float64,Float64}[] # (x_start, x_end) for each layer

    d_count = 0
    w_count = 0

    for layer in 1:k
        n_pairs = length(active) ÷ 2
        layer_x_start = x_pos - GATE_SPACING * 0.4

        # Disentanglers (all at the same x — they act on disjoint pairs)
        x_d = x_pos
        for p in 1:n_pairs
            d_count += 1
            i = active[2p - 1]
            j = active[2p]
            push!(disentanglers, (x_d, i, j, "D$d_count", layer))
        end
        x_pos += GATE_SPACING * 1.6

        # Isometries (same x for all pairs in this layer)
        x_w = x_pos
        for p in 1:n_pairs
            w_count += 1
            i = active[2p - 1]   # survives
            j = active[2p]       # coarse-grained
            push!(isometries, (x_w, i, j, "W$w_count", layer))
            terminations[j] = x_w + GATE_SIZE * 0.6
        end
        x_pos += GATE_SPACING * 1.6

        layer_x_end = x_pos - GATE_SPACING * 0.4
        push!(layer_x_ranges, (layer_x_start, layer_x_end))

        # Update active qubits (keep first of each pair)
        active = active[1:2:end]

        x_pos += GATE_SPACING * 0.4
    end

    x_end = x_pos + GATE_SPACING * 0.5
    return disentanglers, isometries, terminations, layer_x_ranges, x_end
end

# ================================================================================
# MERA Circuit Diagram
# ================================================================================

"""
    plot_mera_circuit(n_qubits; title, output_path)

Draw a horizontal MERA circuit diagram for `n_qubits` (must be a power of 2).

Layout:
- Horizontal qubit wires run left → right.
- Each layer contains disentangler gates (D) followed by isometry gates (W).
- After an isometry, the coarse-grained wire terminates (×) and becomes dashed.
- Surviving wires continue to the next layer.
"""
function plot_mera_circuit(n_qubits::Int; title="MERA Circuit", output_path=nothing)
    k = Int(log2(n_qubits))

    disentanglers, isometries, terminations, layer_x_ranges, x_end =
        compute_mera_layout(n_qubits)

    qubit_y = [-q * QUBIT_SPACING for q in 1:n_qubits]
    x_start = 2.0

    # Figure dimensions
    width  = max(700, Int(ceil(150 + (x_end - x_start) * 75)))
    height = 120 + n_qubits * 70

    fig = Figure(size=(width, height), backgroundcolor=:white)
    ax = Axis(fig[1, 1], title=title, titlesize=20, aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)

    y_top = 0.5 * QUBIT_SPACING
    y_bot = -(n_qubits + 0.5) * QUBIT_SPACING

    # --- Layer background shading and labels ---
    for (l, (lx_start, lx_end)) in enumerate(layer_x_ranges)
        bg_color = iseven(l) ? COLORS.layer_bg_even : COLORS.layer_bg_odd
        poly!(ax,
            Point2f[(lx_start, y_bot), (lx_end, y_bot), (lx_end, y_top), (lx_start, y_top)],
            color=(bg_color[1], bg_color[2]))
        text!(ax, (lx_start + lx_end) / 2, y_top + 0.25,
              text="Layer $l", align=(:center, :bottom),
              fontsize=12, font=:bold, color=COLORS.label)
    end

    # --- Layer separator lines ---
    for (lx_start, _) in layer_x_ranges
        lines!(ax, [lx_start, lx_start], [y_top, y_bot],
               color=COLORS.layer_sep, linewidth=1, linestyle=:dot)
    end
    # closing separator after last layer
    let (_, lx_end) = layer_x_ranges[end]
        lines!(ax, [lx_end, lx_end], [y_top, y_bot],
               color=COLORS.layer_sep, linewidth=1, linestyle=:dot)
    end

    # --- Qubit wires ---
    for q in 1:n_qubits
        y = qubit_y[q]
        if haskey(terminations, q)
            x_term = terminations[q]
            # active portion
            lines!(ax, [x_start, x_term], [y, y],
                   color=COLORS.wire, linewidth=WIRE_WIDTH)
            draw_termination!(ax, x_term, y)
            # inactive portion
            lines!(ax, [x_term + 0.15, x_end], [y, y],
                   color=COLORS.wire_inactive, linewidth=1, linestyle=:dash)
        else
            # survives to final output
            lines!(ax, [x_start, x_end], [y, y],
                   color=COLORS.wire, linewidth=WIRE_WIDTH)
        end

        # input label
        text!(ax, x_start - 0.5, y, text="|q$q⟩", align=(:right, :center),
              fontsize=14, font=:bold, color=COLORS.text)
        # output label (only for surviving wire)
        if !haskey(terminations, q)
            text!(ax, x_end + 0.5, y, text="out", align=(:left, :center),
                  fontsize=14, color=COLORS.label)
        end
    end

    # --- Disentangler gates ---
    for (x, i, j, label, _layer) in disentanglers
        draw_disentangler!(ax, x, qubit_y[i], qubit_y[j], label)
    end

    # --- Isometry gates ---
    for (x, i, j, label, _layer) in isometries
        draw_isometry!(ax, x, qubit_y[i], qubit_y[j], label)
    end

    # --- Legend ---
    draw_legend!(ax, x_end + 1.0, -0.3, [
        (COLORS.disentangler, "D", "Disentangler U(4)"),
        (COLORS.isometry,     "W", "Isometry St(2,4)"),
    ])

    xlims!(ax, x_start - 1.5, x_end + 4.0)
    ylims!(ax, y_bot - 0.3, y_top + 0.7)

    if output_path !== nothing
        save(output_path, fig)
        println("  Saved: $output_path")
    end

    return fig
end

# ================================================================================
# MERA Hierarchy Tree Diagram
# ================================================================================

"""
    plot_mera_tree(n_qubits; title, output_path)

Draw a top-down tree diagram of the MERA coarse-graining hierarchy.
Level 0 (top) has `n_qubits` input nodes; each subsequent level halves the count
by pairing nodes through a disentangler + isometry block.
"""
function plot_mera_tree(n_qubits::Int; title="MERA Hierarchy", output_path=nothing)
    k = Int(log2(n_qubits))

    fig = Figure(size=(max(600, n_qubits * 90), 200 + k * 180), backgroundcolor=:white)
    ax = Axis(fig[1, 1], title="$title ($n_qubits qubits, $k layers)",
              titlesize=18, aspect=DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)

    level_y(l) = -l * 2.0
    node_radius = 0.3

    # Compute node positions at each level
    # Level l has n_qubits / 2^l nodes, evenly spaced
    positions = Vector{Vector{Float64}}(undef, k + 1)   # x-coords per level
    for l in 0:k
        n_nodes = n_qubits ÷ (2^l)
        if n_nodes == 0
            n_nodes = 1
        end
        spacing = (n_qubits - 1) * 1.0 / max(n_nodes - 1, 1)
        if n_nodes == 1
            positions[l+1] = [0.0]
        else
            positions[l+1] = collect(range(-(n_qubits-1)/2, (n_qubits-1)/2, length=n_nodes))
        end
    end

    # Draw edges and gate blocks between levels
    d_count = 0
    w_count = 0
    for l in 0:k-1
        parent_xs = positions[l+1]
        child_xs  = positions[l+2]

        for c in eachindex(child_xs)
            # Each child comes from a pair of parents: (2c-1, 2c)
            p1 = 2c - 1
            p2 = 2c
            x_p1 = parent_xs[p1]
            x_p2 = parent_xs[p2]
            x_c  = child_xs[c]
            y_p  = level_y(l) - node_radius
            y_c  = level_y(l + 1) + node_radius

            # Mid-point for the gate block
            y_mid = (level_y(l) + level_y(l + 1)) / 2

            # Lines from parents down to gate block
            lines!(ax, [x_p1, x_p1], [y_p, y_mid + 0.25],
                   color=COLORS.disentangler, linewidth=2)
            lines!(ax, [x_p2, x_p2], [y_p, y_mid + 0.25],
                   color=COLORS.disentangler, linewidth=2)

            # Disentangler box between the two parent lines
            d_count += 1
            x_mid_d = (x_p1 + x_p2) / 2
            d_w = abs(x_p2 - x_p1) * 0.6 + 0.3
            d_h = 0.35
            poly!(ax, Point2f[
                (x_mid_d - d_w/2, y_mid + d_h + 0.15),
                (x_mid_d + d_w/2, y_mid + d_h + 0.15),
                (x_mid_d + d_w/2, y_mid + 0.15),
                (x_mid_d - d_w/2, y_mid + 0.15)],
                color=COLORS.disentangler, strokewidth=1.5, strokecolor=:white)
            text!(ax, x_mid_d, y_mid + d_h/2 + 0.15,
                  text="D$d_count", align=(:center, :center),
                  fontsize=10, font=:bold, color=:white)

            # Lines from disentangler down to isometry
            lines!(ax, [x_p1, x_c], [y_mid + 0.15, y_mid - 0.15],
                   color=COLORS.isometry, linewidth=2)
            lines!(ax, [x_p2, x_c], [y_mid + 0.15, y_mid - 0.15],
                   color=COLORS.isometry, linewidth=2)

            # Isometry trapezoid
            w_count += 1
            iso_w_top = abs(x_p2 - x_p1) * 0.5 + 0.2
            iso_w_bot = 0.3
            iso_h = 0.35
            poly!(ax, Point2f[
                (x_mid_d - iso_w_top/2, y_mid - 0.15),
                (x_mid_d + iso_w_top/2, y_mid - 0.15),
                (x_c + iso_w_bot/2, y_mid - 0.15 - iso_h),
                (x_c - iso_w_bot/2, y_mid - 0.15 - iso_h)],
                color=COLORS.isometry, strokewidth=1.5, strokecolor=:white)
            text!(ax, (x_mid_d + x_c) / 2, y_mid - 0.15 - iso_h/2,
                  text="W$w_count", align=(:center, :center),
                  fontsize=10, font=:bold, color=:white)

            # Line from isometry down to child node
            lines!(ax, [x_c, x_c], [y_mid - 0.15 - iso_h, y_c],
                   color=COLORS.wire, linewidth=2)
        end
    end

    # Draw nodes
    for l in 0:k
        y = level_y(l)
        xs = positions[l+1]
        n_nodes = length(xs)

        if l == 0
            # Input nodes
            node_color = "#4285F4"
            for (i, x) in enumerate(xs)
                scatter!(ax, [x], [y], markersize=30, color=node_color,
                         strokewidth=2, strokecolor=:white)
                text!(ax, x, y, text="q$i", align=(:center, :center),
                      fontsize=9, font=:bold, color=:white)
            end
        elseif l == k
            # Final output node
            node_color = "#34A853"
            for (_, x) in enumerate(xs)
                scatter!(ax, [x], [y], markersize=30, color=node_color,
                         strokewidth=2, strokecolor=:white)
                text!(ax, x, y, text="out", align=(:center, :center),
                      fontsize=9, font=:bold, color=:white)
            end
        else
            # Intermediate nodes
            node_color = "#9C27B0"
            for (i, x) in enumerate(xs)
                scatter!(ax, [x], [y], markersize=25, color=node_color,
                         strokewidth=2, strokecolor=:white)
                text!(ax, x, y, text="q'$i", align=(:center, :center),
                      fontsize=8, font=:bold, color=:white)
            end
        end

        # Level label
        if l == 0
            text!(ax, -(n_qubits-1)/2 - 1.5, y,
                  text="Input", align=(:right, :center),
                  fontsize=11, font=:bold, color=COLORS.label)
        elseif l == k
            text!(ax, -(n_qubits-1)/2 - 1.5, y,
                  text="Output", align=(:right, :center),
                  fontsize=11, font=:bold, color=COLORS.label)
        else
            text!(ax, -(n_qubits-1)/2 - 1.5, y,
                  text="Layer $l", align=(:right, :center),
                  fontsize=11, font=:bold, color=COLORS.label)
        end
    end

    x_span = (n_qubits - 1) / 2
    xlims!(ax, -x_span - 2.5, x_span + 2.5)
    ylims!(ax, level_y(k) - 1.0, 1.5)

    if output_path !== nothing
        save(output_path, fig)
        println("  Saved: $output_path")
    end

    return fig
end

# ================================================================================
# Print MERA Statistics
# ================================================================================

function print_mera_stats(n_qubits::Int)
    k = Int(log2(n_qubits))
    n_disentanglers = n_qubits - 1
    n_isometries = n_qubits - 1

    println("\nMERA Statistics ($n_qubits qubits = 2^$k):")
    println("  - Number of layers: $k")
    for l in 1:k
        n_pairs = n_qubits ÷ (2^l)
        println("  - Layer $l: $n_pairs disentanglers + $n_pairs isometries " *
                "(acting on $(n_qubits ÷ 2^(l-1)) → $(n_qubits ÷ 2^l) qubits)")
    end
    println("  - Total disentanglers: $n_disentanglers (each ∈ U(4), 4×4 unitary)")
    println("  - Total isometries:    $n_isometries (each ∈ St(2,4), 2×4 isometry)")
    println("  - Disentangler params: $(n_disentanglers * 16) (16 real per U(4))")
    println("  - Isometry params:     $(n_isometries * 8) (8 real per St(2,4))")
    println("  - Total parameters:    $(n_disentanglers * 16 + n_isometries * 8)")
    println("  - Circuit depth:       O(log₂ n) = $k")
end

# ================================================================================
# Main
# ================================================================================

function main()
    output_dir = joinpath(@__DIR__, "MERADiagrams")
    mkpath(output_dir)

    println("=" ^ 60)
    println("Generating MERA Circuit Diagrams")
    println("=" ^ 60)
    println("\nOutput: $output_dir\n")

    # --- Circuit diagrams ---
    println("1. MERA Circuit (4 qubits, 2 layers)")
    plot_mera_circuit(4;
        title="MERA Circuit (4 qubits, 2 layers)",
        output_path=joinpath(output_dir, "mera_4qubit.png"))
    print_mera_stats(4)

    println("\n2. MERA Circuit (8 qubits, 3 layers)")
    plot_mera_circuit(8;
        title="MERA Circuit (8 qubits, 3 layers)",
        output_path=joinpath(output_dir, "mera_8qubit.png"))
    print_mera_stats(8)

    println("\n3. MERA Circuit (16 qubits, 4 layers)")
    plot_mera_circuit(16;
        title="MERA Circuit (16 qubits, 4 layers)",
        output_path=joinpath(output_dir, "mera_16qubit.png"))
    print_mera_stats(16)

    # --- Hierarchy tree diagrams ---
    println("\n4. MERA Hierarchy Tree (4 qubits)")
    plot_mera_tree(4;
        title="MERA Hierarchy",
        output_path=joinpath(output_dir, "mera_tree_4qubit.png"))

    println("\n5. MERA Hierarchy Tree (8 qubits)")
    plot_mera_tree(8;
        title="MERA Hierarchy",
        output_path=joinpath(output_dir, "mera_tree_8qubit.png"))

    println("\n" * "=" ^ 60)
    println("Done! Generated 5 MERA diagrams.")
    println("=" ^ 60)
end

main()
