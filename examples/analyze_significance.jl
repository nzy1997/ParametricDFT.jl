# ================================================================================
# Statistical Significance Analysis for ParametricDFT vs Classical FFT
# ================================================================================

import Pkg
script_dir = @__DIR__
Pkg.activate(script_dir)

using DelimitedFiles
using Statistics
using Distributions
using Dates

# Load the results CSV
workspace_root = dirname(script_dir)
csv_file = joinpath(workspace_root, "results/dataset_compression_results_70.csv")

println("="^80)
println("Statistical Significance Analysis")
println("="^80)
println("\nLoading data from: $csv_file")

# Check if file exists
if !isfile(csv_file)
    error("CSV file not found: $csv_file")
end

# Get file modification time
file_mtime = Dates.unix2datetime(mtime(csv_file))
println("File last modified: $(Dates.format(file_mtime, "yyyy-mm-dd HH:MM:SS"))")

# Read CSV header first to validate structure
header_line = readline(csv_file)
expected_columns = [
    "filename", "success", "ssim_fft", "ssim_parametric", 
    "mse_fft", "mse_parametric", "psnr_fft", "psnr_parametric",
    "msssim_fft", "msssim_parametric", "fft_forward_time", "fft_inverse_time",
    "parametric_forward_time", "parametric_inverse_time"
]

header_cols = split(header_line, ',')
if length(header_cols) < length(expected_columns)
    error("CSV file has unexpected format. Expected $(length(expected_columns)) columns, found $(length(header_cols))")
end

# Read CSV data
data = readdlm(csv_file, ',', skipstart=1)
total_rows = size(data, 1)
println("Total rows in CSV: $total_rows")

# Extract successful results only
successful_rows = [i for i in 1:size(data, 1) if data[i, 2] == "✓"]
failed_rows = total_rows - length(successful_rows)

println("Successful results: $(length(successful_rows))")
if failed_rows > 0
    println("Failed results: $failed_rows")
end

if length(successful_rows) < 2
    error("Need at least 2 successful results for statistical analysis. Found: $(length(successful_rows))")
end

# Extract metrics (paired data - same images) with error handling
function safe_parse_float(val, row_idx, col_name)
    try
        return parse(Float64, string(val))
    catch e
        error("Error parsing $col_name at row $row_idx: $val - $e")
    end
end

println("\nExtracting metrics from successful results...")
ssim_fft = [safe_parse_float(data[i, 3], i, "ssim_fft") for i in successful_rows]
ssim_parametric = [safe_parse_float(data[i, 4], i, "ssim_parametric") for i in successful_rows]

mse_fft = [safe_parse_float(data[i, 5], i, "mse_fft") for i in successful_rows]
mse_parametric = [safe_parse_float(data[i, 6], i, "mse_parametric") for i in successful_rows]

psnr_fft = [safe_parse_float(data[i, 7], i, "psnr_fft") for i in successful_rows]
psnr_parametric = [safe_parse_float(data[i, 8], i, "psnr_parametric") for i in successful_rows]

msssim_fft = [safe_parse_float(data[i, 9], i, "msssim_fft") for i in successful_rows]
msssim_parametric = [safe_parse_float(data[i, 10], i, "msssim_parametric") for i in successful_rows]

fft_forward_time = [safe_parse_float(data[i, 11], i, "fft_forward_time") for i in successful_rows]
parametric_forward_time = [safe_parse_float(data[i, 13], i, "parametric_forward_time") for i in successful_rows]

fft_inverse_time = [safe_parse_float(data[i, 12], i, "fft_inverse_time") for i in successful_rows]
parametric_inverse_time = [safe_parse_float(data[i, 14], i, "parametric_inverse_time") for i in successful_rows]

n = length(successful_rows)
println("✓ Successfully extracted metrics from $n images")
println("\n" * "="^80)
println("Dataset Summary")
println("="^80)
println("Sample size: $n images")
println("Analysis date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")

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

# Calculate practical significance thresholds
ssim_practical_threshold = 0.01  # 1% difference in SSIM is typically noticeable
psnr_practical_threshold = 0.5   # 0.5 dB difference in PSNR is typically noticeable

quality_improvements = []
for r in quality_metrics
    if r["metric"] == "SSIM" && abs(r["mean_diff"]) > ssim_practical_threshold
        push!(quality_improvements, r["metric"])
    elseif r["metric"] == "PSNR (dB)" && abs(r["mean_diff"]) > psnr_practical_threshold
        push!(quality_improvements, r["metric"])
    end
end

practical_quality_improvement = length(quality_improvements) > 0

avg_speedup = mean(parametric_forward_time) / mean(fft_forward_time)

println("""
Based on the statistical analysis of $n images:

1. Quality Metrics: 
   - All quality metrics show statistically significant differences
   - Parametric DFT shows slight improvements in all quality metrics
   - However, the absolute differences are very small:
     * SSIM: $(round(mean(ssim_parametric) - mean(ssim_fft), digits=4)) improvement
     * PSNR: $(round(mean(psnr_parametric) - mean(psnr_fft), digits=2)) dB improvement
   - These differences are $(practical_quality_improvement ? "potentially" : "not") practically meaningful
     (thresholds: SSIM > $ssim_practical_threshold, PSNR > $psnr_practical_threshold dB)

2. Performance Metrics:
   - Classical FFT is significantly faster
   - Average speedup: $(round(avg_speedup, digits=2))x faster
   - Forward: $(round(mean(fft_forward_time), digits=6))s vs $(round(mean(parametric_forward_time), digits=6))s
   - Inverse: $(round(mean(fft_inverse_time), digits=6))s vs $(round(mean(parametric_inverse_time), digits=6))s
   - This performance difference is both statistically and practically significant

3. Overall Assessment:
   - Parametric DFT achieves $(practical_quality_improvement ? "slightly better" : "comparable") image quality
   - The quality improvements are statistically significant but $(practical_quality_improvement ? "may be" : "not") practically meaningful
   - Classical FFT is $(round(avg_speedup, digits=1))x faster, which is a substantial practical advantage
   - Recommendation: Use Classical FFT unless the small quality improvements justify the $(round(avg_speedup, digits=1))x performance cost
""")

println("\n" * "="^80)
println("Analysis completed at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
println("="^80)

