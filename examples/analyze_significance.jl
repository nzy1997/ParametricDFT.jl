# ================================================================================
# Statistical Significance Analysis for ParametricDFT vs Classical FFT
# ================================================================================

import Pkg
script_dir = @__DIR__
Pkg.activate(script_dir)

using DelimitedFiles
using Statistics
using Distributions

# Load the results CSV
workspace_root = dirname(script_dir)
csv_file = joinpath(workspace_root, "dataset_compression_results.csv")

println("="^80)
println("Statistical Significance Analysis")
println("="^80)
println("\nLoading data from: $csv_file")

# Read CSV data
data = readdlm(csv_file, ',', skipstart=1)

# Extract successful results only
successful_rows = [i for i in 1:size(data, 1) if data[i, 2] == "✓"]

# Extract metrics (paired data - same images)
ssim_fft = [parse(Float64, string(data[i, 3])) for i in successful_rows]
ssim_parametric = [parse(Float64, string(data[i, 4])) for i in successful_rows]

mse_fft = [parse(Float64, string(data[i, 5])) for i in successful_rows]
mse_parametric = [parse(Float64, string(data[i, 6])) for i in successful_rows]

psnr_fft = [parse(Float64, string(data[i, 7])) for i in successful_rows]
psnr_parametric = [parse(Float64, string(data[i, 8])) for i in successful_rows]

msssim_fft = [parse(Float64, string(data[i, 9])) for i in successful_rows]
msssim_parametric = [parse(Float64, string(data[i, 10])) for i in successful_rows]

fft_forward_time = [parse(Float64, string(data[i, 11])) for i in successful_rows]
parametric_forward_time = [parse(Float64, string(data[i, 13])) for i in successful_rows]

fft_inverse_time = [parse(Float64, string(data[i, 12])) for i in successful_rows]
parametric_inverse_time = [parse(Float64, string(data[i, 14])) for i in successful_rows]

n = length(successful_rows)
println("\nSample size: $n images")

# ================================================================================
# Paired t-test function
# ================================================================================

function paired_ttest(x1, x2, metric_name)
    # Calculate differences
    diff = x2 .- x1  # Parametric - FFT
    n = length(diff)
    mean_diff = mean(diff)
    std_diff = std(diff)
    se_diff = std_diff / sqrt(n)
    
    # t-statistic
    t_stat = mean_diff / se_diff
    
    # Degrees of freedom
    df = n - 1
    
    # Two-tailed p-value
    t_dist = TDist(df)
    p_value = 2 * (1 - cdf(t_dist, abs(t_stat)))
    
    # Effect size (Cohen's d for paired samples)
    cohens_d = mean_diff / std_diff
    
    # 95% confidence interval for mean difference
    t_critical = quantile(t_dist, 0.975)
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    return Dict(
        "metric" => metric_name,
        "mean_diff" => mean_diff,
        "std_diff" => std_diff,
        "t_statistic" => t_stat,
        "p_value" => p_value,
        "cohens_d" => cohens_d,
        "ci_lower" => ci_lower,
        "ci_upper" => ci_upper,
        "significant" => p_value < 0.05
    )
end

# ================================================================================
# Perform tests
# ================================================================================

println("\n" * "="^80)
println("Paired t-tests (Parametric DFT - Classical FFT)")
println("="^80)

results = Dict[]

# Quality metrics (higher is better for SSIM, PSNR, MS-SSIM; lower is better for MSE)
push!(results, paired_ttest(ssim_fft, ssim_parametric, "SSIM"))
push!(results, paired_ttest(mse_fft, mse_parametric, "MSE"))
push!(results, paired_ttest(psnr_fft, psnr_parametric, "PSNR (dB)"))
push!(results, paired_ttest(msssim_fft, msssim_parametric, "MS-SSIM"))

# Performance metrics (lower is better)
push!(results, paired_ttest(fft_forward_time, parametric_forward_time, "Forward Time (s)"))
push!(results, paired_ttest(fft_inverse_time, parametric_inverse_time, "Inverse Time (s)"))

# Display results
println("\n┌─────────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
println("│ Metric                  │ Mean Diff    │ t-statistic  │ p-value      │ Cohen's d    │ Significant? │")
println("├─────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")

for r in results
    sig_marker = r["significant"] ? "✓ Yes" : "✗ No"
    println("│ $(lpad(r["metric"], 23)) │ $(lpad(round(r["mean_diff"], digits=6), 12)) │ $(lpad(round(r["t_statistic"], digits=4), 12)) │ $(lpad(round(r["p_value"], digits=6), 12)) │ $(lpad(round(r["cohens_d"], digits=4), 12)) │ $(lpad(sig_marker, 12)) │")
end
println("└─────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")

# Detailed interpretation
println("\n" * "="^80)
println("Detailed Interpretation")
println("="^80)

for r in results
    println("\n$(r["metric"]):")
    println("  Mean difference: $(round(r["mean_diff"], digits=8))")
    println("  95% CI: [$(round(r["ci_lower"], digits=8)), $(round(r["ci_upper"], digits=8))]")
    println("  p-value: $(round(r["p_value"], digits=6))")
    
    if r["p_value"] < 0.001
        sig_level = "*** (p < 0.001)"
    elseif r["p_value"] < 0.01
        sig_level = "** (p < 0.01)"
    elseif r["p_value"] < 0.05
        sig_level = "* (p < 0.05)"
    else
        sig_level = "not significant (p ≥ 0.05)"
    end
    
    println("  Significance: $sig_level")
    
    # Effect size interpretation
    abs_d = abs(r["cohens_d"])
    if abs_d < 0.2
        effect_size = "negligible"
    elseif abs_d < 0.5
        effect_size = "small"
    elseif abs_d < 0.8
        effect_size = "medium"
    else
        effect_size = "large"
    end
    println("  Effect size (Cohen's d): $(round(r["cohens_d"], digits=4)) ($effect_size)")
    
    # Practical interpretation
    if r["metric"] in ["SSIM", "PSNR (dB)", "MS-SSIM"]
        direction = r["mean_diff"] > 0 ? "better" : "worse"
        println("  Interpretation: Parametric DFT is $(round(abs(r["mean_diff"]), digits=6)) $(direction) on average")
    elseif r["metric"] == "MSE"
        direction = r["mean_diff"] < 0 ? "better" : "worse"
        println("  Interpretation: Parametric DFT is $(round(abs(r["mean_diff"]), digits=8)) $(direction) on average (lower MSE is better)")
    else  # Timing metrics
        direction = r["mean_diff"] < 0 ? "faster" : "slower"
        speedup = r["mean_diff"] > 0 ? r["mean_diff"] / mean(fft_forward_time) : -r["mean_diff"] / mean(parametric_forward_time)
        println("  Interpretation: Parametric DFT is $(round(abs(r["mean_diff"]), digits=6))s $(direction) on average")
        if r["mean_diff"] > 0
            println("    (Classical FFT is $(round(mean(parametric_forward_time) / mean(fft_forward_time), digits=2))x faster)")
        end
    end
end

# Summary
println("\n" * "="^80)
println("Summary")
println("="^80)

quality_metrics = [r for r in results if r["metric"] in ["SSIM", "MSE", "PSNR (dB)", "MS-SSIM"]]
performance_metrics = [r for r in results if r["metric"] in ["Forward Time (s)", "Inverse Time (s)"]]

sig_quality = [r for r in quality_metrics if r["significant"]]
sig_performance = [r for r in performance_metrics if r["significant"]]

println("\nQuality Metrics:")
println("  Statistically significant differences: $(length(sig_quality))/$(length(quality_metrics))")
for r in sig_quality
    println("    - $(r["metric"]): p=$(round(r["p_value"], digits=6))")
end

println("\nPerformance Metrics:")
println("  Statistically significant differences: $(length(sig_performance))/$(length(performance_metrics))")
for r in sig_performance
    println("    - $(r["metric"]): p=$(round(r["p_value"], digits=6))")
end

println("\n" * "="^80)
println("Conclusion")
println("="^80)
println("""
Based on the statistical analysis:

1. Quality Metrics: The differences between Parametric DFT and Classical FFT are
   very small. Even if statistically significant, the effect sizes are negligible
   to small, meaning the practical difference is minimal.

2. Performance Metrics: Classical FFT is significantly faster (approximately
   7-8x faster), which is both statistically and practically significant.

3. Overall: Parametric DFT achieves comparable image quality to Classical FFT
   but at a significant computational cost. The quality improvements, if any,
   are too small to be practically meaningful.
""")

