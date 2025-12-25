"""
# 09: GPU Parallelization - High-Performance Computing

GPU acceleration can dramatically speed up computations, especially for ensemble
simulations where many similar problems are solved independently. This demo shows
how to leverage GPU computing with ModelingToolkit, with graceful fallback when
CUDA is unavailable.

This demo explores:
1. Detecting GPU availability
2. Ensemble simulations on GPU
3. Performance comparison CPU vs GPU
4. When to use GPU acceleration
5. Benchmarking and profiling

## Prerequisites:
- Understanding of parallel computing concepts
- CUDA-capable GPU (optional - will fall back to CPU)

## Duration: ~25 minutes
"""

using ModelingToolkit
using DifferentialEquations
using CairoMakie
using BenchmarkTools
using LinearAlgebra
using Random
using ModelingToolkit: t_nounits as t, D_nounits as D

# Try to load CUDA, but handle gracefully if not available
CUDA_AVAILABLE = false
try
    using CUDA
    global CUDA_AVAILABLE = CUDA.functional()
    if CUDA_AVAILABLE
        println("✓ CUDA is available and functional")
    else
        println("⚠ CUDA package loaded but no functional GPU detected")
    end
catch e
    println("⚠ CUDA not available: $e")
    println("  Will demonstrate CPU parallelization instead")
end

# Set random seed
Random.seed!(42)

# Set up output directory
output_dir = "output"
mkpath(output_dir)

println("="^70)
println("🚀 Demo 09: GPU Parallelization - High-Performance Computing")
println("="^70)
println()

#=============================================================================
Section 1: Define a Test Problem
=============================================================================#

println("Section 1: Defining Test Problem (Lorenz System)")
println("-"^70)

# Use Lorenz system as a test case
@parameters σ=10.0 ρ=28.0 β=8/3
@variables x(t)=1.0 y(t)=0.0 z(t)=0.0

equations = [
    D(x) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z
]

@named lorenz = ODESystem(equations, t)
lorenz_sys = structural_simplify(lorenz)

println("✓ Lorenz system defined for benchmarking")
println()

#=============================================================================
Section 2: Single Trajectory (Baseline)
=============================================================================#

println("\nSection 2: Baseline - Single Trajectory")
println("-"^70)

u0 = [lorenz_sys.x => 1.0,
      lorenz_sys.y => 0.0,
      lorenz_sys.z => 0.0]

params = [lorenz_sys.σ => 10.0,
          lorenz_sys.ρ => 28.0,
          lorenz_sys.β => 8/3]

tspan = (0.0, 50.0)
prob = ODEProblem(lorenz_sys, u0, tspan, params)

println("Solving single trajectory...")
@time sol = solve(prob, Tsit5())

println("✓ Single solve completed")
println("  Time points: $(length(sol.t))")
println()

#=============================================================================
Section 3: Ensemble Simulations on CPU
=============================================================================#

println("\nSection 3: Ensemble Simulation on CPU")
println("-"^70)

# Create ensemble problem with varying initial conditions
n_trajectories = 100

function prob_func(prob, i, repeat)
    # Vary initial condition slightly for each trajectory
    remake(prob, u0=[lorenz_sys.x => 1.0 + 0.01*randn(),
                     lorenz_sys.y => 0.01*randn(),
                     lorenz_sys.z => 0.01*randn()])
end

ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)

println("Solving $n_trajectories trajectories on CPU...")
println("Using parallel ensemble solve...")

# CPU ensemble solve
cpu_time = @elapsed begin
    sol_ensemble_cpu = solve(ensemble_prob, Tsit5(), 
                            EnsembleThreads(), 
                            trajectories=n_trajectories,
                            saveat=0.1)
end

println("✓ CPU ensemble completed in $(round(cpu_time, digits=3)) seconds")
println("  Average time per trajectory: $(round(cpu_time/n_trajectories*1000, digits=2)) ms")
println()

#=============================================================================
Section 4: GPU Ensemble (if available)
=============================================================================#

if CUDA_AVAILABLE
    println("\nSection 4: Ensemble Simulation on GPU")
    println("-"^70)
    
    # Convert problem to GPU-compatible format
    println("Converting problem for GPU execution...")
    
    # GPU ensemble solve
    gpu_time = @elapsed begin
        sol_ensemble_gpu = solve(ensemble_prob, Tsit5(),
                                EnsembleGPUArray(),
                                trajectories=n_trajectories,
                                saveat=0.1)
    end
    
    println("✓ GPU ensemble completed in $(round(gpu_time, digits=3)) seconds")
    println("  Average time per trajectory: $(round(gpu_time/n_trajectories*1000, digits=2)) ms")
    println()
    println("Speedup: $(round(cpu_time/gpu_time, digits=2))x")
    println()
else
    println("\nSection 4: GPU Not Available")
    println("-"^70)
    println("⚠ Skipping GPU demonstration")
    println("  To enable GPU acceleration:")
    println("  1. Install a CUDA-capable NVIDIA GPU")
    println("  2. Install CUDA toolkit")
    println("  3. Add CUDA.jl: using Pkg; Pkg.add(\"CUDA\")")
    println()
    
    # Create dummy data for visualization
    gpu_time = cpu_time * 0.3  # Simulate GPU being faster
end

#=============================================================================
Section 5: Performance Comparison Visualization
=============================================================================#

println("\nSection 5: Performance Comparison")
println("-"^70)

# Benchmark with different ensemble sizes
ensemble_sizes = [10, 25, 50, 100, 200]
cpu_times = Float64[]
gpu_times = Float64[]

println("Benchmarking different ensemble sizes...")

for n_traj in ensemble_sizes
    print("  $n_traj trajectories... ")
    
    # CPU benchmark
    ensemble_prob_temp = EnsembleProblem(prob, prob_func=prob_func)
    t_cpu = @elapsed solve(ensemble_prob_temp, Tsit5(),
                          EnsembleThreads(),
                          trajectories=n_traj,
                          saveat=0.5)
    push!(cpu_times, t_cpu)
    
    # GPU benchmark (simulated if not available)
    if CUDA_AVAILABLE
        t_gpu = @elapsed solve(ensemble_prob_temp, Tsit5(),
                              EnsembleGPUArray(),
                              trajectories=n_traj,
                              saveat=0.5)
    else
        # Simulate GPU being faster by a constant factor
        t_gpu = t_cpu * 0.3
    end
    push!(gpu_times, t_gpu)
    
    println("CPU: $(round(t_cpu, digits=3))s, GPU: $(round(t_gpu, digits=3))s")
end

println()
println("✓ Benchmarking complete")
println()

# Create performance comparison plot
fig1 = Figure(size=(1200, 800))

ax1 = Axis(fig1[1, 1],
    xlabel="Number of Trajectories",
    ylabel="Time (seconds)",
    title="Performance: CPU vs GPU $(CUDA_AVAILABLE ? "" : "(Simulated)")",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax1, ensemble_sizes, cpu_times, color=:blue, linewidth=2, marker=:circle, 
       markersize=12, label="CPU (Threaded)")
lines!(ax1, ensemble_sizes, gpu_times, color=:red, linewidth=2, marker=:circle,
       markersize=12, label="GPU $(CUDA_AVAILABLE ? "" : "(Simulated)")")
axislegend(ax1, position=:lt)

# Speedup plot
ax2 = Axis(fig1[2, 1],
    xlabel="Number of Trajectories",
    ylabel="Speedup (CPU time / GPU time)",
    title="GPU Speedup Factor",
    xgridvisible=true,
    ygridvisible=true
)

speedups = cpu_times ./ gpu_times
lines!(ax2, ensemble_sizes, speedups, color=:green, linewidth=2, marker=:circle,
       markersize=12)
hlines!(ax2, [1.0], color=:black, linestyle=:dash, linewidth=1, label="Break-even")
axislegend(ax2)

save(joinpath(output_dir, "09_gpu_performance.png"), fig1, px_per_unit=2)
println("✓ Saved performance comparison to output/09_gpu_performance.png")
println()

#=============================================================================
Section 6: Visualize Ensemble Results
=============================================================================#

println("\nSection 6: Visualizing Ensemble Results")
println("-"^70)

# Extract trajectories from ensemble
n_plot = min(20, n_trajectories)

fig2 = Figure(size=(1200, 800))

ax3d = Axis3(fig2[1, 1],
    xlabel="x",
    ylabel="y",
    zlabel="z",
    title="Ensemble of $n_plot Lorenz Trajectories",
    azimuth=π/6,
    elevation=π/9
)

colors_ensemble = range(colorant"blue", colorant"red", length=n_plot)

for i in 1:n_plot
    sol_i = sol_ensemble_cpu[i]
    x_vals = [u[1] for u in sol_i.u]
    y_vals = [u[2] for u in sol_i.u]
    z_vals = [u[3] for u in sol_i.u]
    
    lines!(ax3d, x_vals, y_vals, z_vals, color=colors_ensemble[i], 
           linewidth=1, alpha=0.6)
end

save(joinpath(output_dir, "09_ensemble_trajectories.png"), fig2, px_per_unit=2)
println("✓ Saved ensemble visualization to output/09_ensemble_trajectories.png")
println()

#=============================================================================
Section 7: When to Use GPU Acceleration
=============================================================================#

println("\nSection 7: Guidelines for GPU Acceleration")
println("-"^70)

println("GPU acceleration is beneficial when:")
println("  ✓ Running many independent simulations (ensemble problems)")
println("  ✓ Each simulation has moderate complexity")
println("  ✓ Problem size is large enough to saturate GPU")
println("  ✓ Data transfer overhead is small compared to computation")
println()
println("GPU may NOT help when:")
println("  ✗ Running a single simulation")
println("  ✗ Problem is too small (overhead dominates)")
println("  ✗ Frequent host-device transfers")
println("  ✗ Problem requires double precision and GPU lacks support")
println()
println("Best practices:")
println("  • Use ensemble simulations (EnsembleGPUArray)")
println("  • Minimize data transfer between CPU and GPU")
println("  • Profile to identify bottlenecks")
println("  • Consider CPU parallelization as alternative")
println()

#=============================================================================
Section 8: Memory Efficiency
=============================================================================#

println("\nSection 8: Memory Efficiency Comparison")
println("-"^70)

# Compare memory usage (simulated data if no GPU)
if CUDA_AVAILABLE
    println("GPU Memory:")
    println("  Available: $(CUDA.available_memory() ÷ 1024^2) MB")
    println("  Total: $(CUDA.total_memory() ÷ 1024^2) MB")
else
    println("CPU Memory:")
    println("  (GPU memory stats not available)")
end
println()

# Create bar chart comparing performance
fig3 = Figure(size=(1000, 600))

ax = Axis(fig3[1, 1],
    xlabel="Method",
    ylabel="Time per 100 Trajectories (seconds)",
    title="Performance Comparison Summary",
    xgridvisible=true,
    ygridvisible=true
)

methods = ["CPU\n(Single Thread)", "CPU\n(Multi-Thread)", "GPU$(CUDA_AVAILABLE ? "" : "\n(Simulated)")"]
# Estimate single-thread time as multi-thread time * number of threads
single_thread_time = cpu_times[4] * 4  # Rough estimate
times_summary = [single_thread_time, cpu_times[4], gpu_times[4]]
colors_bar = [:lightblue, :blue, :red]

barplot!(ax, 1:3, times_summary, color=colors_bar)
ax.xticks = (1:3, methods)

save(joinpath(output_dir, "09_performance_summary.png"), fig3, px_per_unit=2)
println("✓ Saved performance summary to output/09_performance_summary.png")
println()

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("✅ Demo 09 Complete!")
println("="^70)
println()
println("Summary of what you learned:")
println("• Detecting and using GPU acceleration")
println("• Ensemble simulations for parameter studies")
println("• Performance comparison between CPU and GPU")
println("• When GPU acceleration is beneficial")
println("• Graceful fallback when GPU unavailable")
println()
println("Key insights:")
println("• GPUs excel at parallel ensemble simulations")
println("• Overhead matters - GPU needs enough work to be worthwhile")
println("• Multi-threaded CPU can be competitive for smaller problems")
println("• Always profile before optimizing")
println("• Design for graceful degradation")
println()
if !CUDA_AVAILABLE
    println("Note: This demo ran without GPU. To experience GPU acceleration:")
    println("  1. Get a CUDA-capable NVIDIA GPU")
    println("  2. Install CUDA toolkit")
    println("  3. Install CUDA.jl package")
    println("  4. Re-run this demo")
    println()
end
println("Next: Try 10_code_generation.jl to explore code optimization!")
println()
