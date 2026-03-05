# ============================================================================
# Basis Demo — Minimal showcase of the three basis types in ParametricDFT.jl
# ============================================================================
#
# This script demonstrates QFTBasis, EntangledQFTBasis, and TEBDBasis by
# performing forward/inverse transforms and truncated reconstruction on a
# random 8x8 complex image.
#
# Run:
#   julia --project=. examples/basis_demo.jl
# ============================================================================

using ParametricDFT, Random, LinearAlgebra

Random.seed!(42)

# Small basis sizes: 8×8 images (2^3 × 2^3)
m, n = 3, 3
img = rand(ComplexF64, 2^m, 2^n)

println("=" ^ 60)
println("ParametricDFT.jl — Basis Type Demonstration")
println("=" ^ 60)
println("Image size: $(2^m)×$(2^n) (m=$m, n=$n)")
println()

# Store results for summary
results = []

for (name, basis) in [
    ("QFTBasis",          QFTBasis(m, n)),
    ("EntangledQFTBasis", EntangledQFTBasis(m, n)),
    ("TEBDBasis",         TEBDBasis(m, n)),
]
    println("-" ^ 60)
    println("Basis type:      $name")
    println("Parameters:      $(num_parameters(basis))")
    println("Image size:      $(image_size(basis))")

    # Forward transform
    coeffs = forward_transform(basis, img)
    println("Coefficients:    $(size(coeffs)) array, $(length(coeffs)) elements")

    # Full round-trip (no truncation)
    recovered = inverse_transform(basis, coeffs)
    full_err = norm(img - recovered) / norm(img)
    println("Round-trip error: $full_err")

    # Truncated reconstruction (keep 50% of coefficients)
    k = round(Int, length(coeffs) * 0.5)
    truncated = topk_truncate(coeffs, k)
    approx = inverse_transform(basis, truncated)
    trunc_err = norm(img - approx) / norm(img)
    println("Truncated error (k=$k): $trunc_err")
    println()

    push!(results, (name=name, full_err=full_err, trunc_err=trunc_err))
end

# Summary comparison
println("=" ^ 60)
println("Summary")
println("=" ^ 60)
println(rpad("Basis", 22), rpad("Round-trip", 18), "Truncated (50%)")
println("-" ^ 60)
for r in results
    println(rpad(r.name, 22), rpad(string(round(r.full_err; sigdigits=4)), 18),
            round(r.trunc_err; sigdigits=4))
end
println("=" ^ 60)
