"""
# 08: Optimization Problems - Parameter Estimation and Model Fitting

Parameter estimation is crucial for fitting mathematical models to experimental data.
This demo shows how to use ModelingToolkit with optimization libraries to estimate
unknown parameters from noisy observations.

This demo explores:
1. Parameter estimation from noisy data
2. Loss function formulation
3. Optimization with Optimization.jl
4. Model calibration
5. Uncertainty quantification

## Prerequisites:
- Understanding of optimization
- Basic statistics knowledge

## Duration: ~30 minutes
"""

using ModelingToolkit
using DifferentialEquations
using Optimization
using OptimizationOptimJL
using CairoMakie
using LinearAlgebra
using Random
using ModelingToolkit: t_nounits as t, D_nounits as D

# Set random seed for reproducibility
Random.seed!(42)

# Set up output directory
output_dir = "output"
mkpath(output_dir)

println("="^70)
println("📈 Demo 08: Optimization - Parameter Estimation from Data")
println("="^70)
println()

#=============================================================================
Section 1: Generate Synthetic Data
=============================================================================#

println("Section 1: Generating Synthetic Data")
println("-"^70)

# We'll fit parameters of an exponential decay model
# dx/dt = -k*x
# True solution: x(t) = x0 * exp(-k*t)

# True parameters (to be estimated)
k_true = 0.5
x0_true = 10.0

println("True model: dx/dt = -k*x")
println("True parameters:")
println("  k = $k_true")
println("  x0 = $x0_true")
println()

# Generate synthetic data with noise
t_data = range(0, 10, length=30)
x_true = x0_true .* exp.(-k_true .* t_data)
noise_level = 0.5
x_data = x_true .+ noise_level .* randn(length(t_data))

println("✓ Generated $(length(t_data)) data points with noise level $noise_level")
println()

#=============================================================================
Section 2: Define the Model
=============================================================================#

println("\nSection 2: Defining the Model for Fitting")
println("-"^70)

# Define the ODE model
@parameters k x0_param
@variables x(t)

eqs = [D(x) ~ -k * x]

@named decay_model = ODESystem(eqs, t)
decay_sys = structural_simplify(decay_model)

println("✓ Exponential decay model defined")
println()

#=============================================================================
Section 3: Define Loss Function
=============================================================================#

println("\nSection 3: Defining Loss Function")
println("-"^70)

"""
Loss function for parameter estimation.
Computes sum of squared errors between model predictions and data.
"""
function loss_function(params, prob_template, t_data, x_data)
    # Extract parameters
    k_est, x0_est = params
    
    # Check for physical validity
    if k_est <= 0 || x0_est <= 0
        return Inf
    end
    
    # Set up problem with current parameters
    u0 = [decay_sys.x => x0_est]
    p = [decay_sys.k => k_est]
    
    prob = remake(prob_template, u0=u0, p=p)
    
    # Solve ODE
    sol = solve(prob, Tsit5(), saveat=t_data, verbose=false)
    
    # Check if solve was successful
    if sol.retcode != :Success
        return Inf
    end
    
    # Extract predictions
    x_pred = [sol(t_val)[1] for t_val in t_data]
    
    # Compute sum of squared errors
    sse = sum((x_data .- x_pred).^2)
    
    return sse
end

println("✓ Loss function defined (sum of squared errors)")
println()

#=============================================================================
Section 4: Perform Optimization
=============================================================================#

println("\nSection 4: Performing Parameter Estimation")
println("-"^70)

# Initial guess (deliberately wrong)
k_init = 0.2
x0_init = 8.0

println("Initial guess:")
println("  k = $k_init")
println("  x0 = $x0_init")
println()

# Create template problem
u0_template = [decay_sys.x => x0_init]
p_template = [decay_sys.k => k_init]
tspan = (0.0, 10.0)
prob_template = ODEProblem(decay_sys, u0_template, tspan, p_template)

# Set up optimization problem
initial_params = [k_init, x0_init]

# Wrapper function for Optimization.jl
optf = OptimizationFunction(
    (params, p) -> loss_function(params, prob_template, t_data, x_data),
    Optimization.AutoForwardDiff()
)

optprob = OptimizationProblem(optf, initial_params, lb=[0.0, 0.0], ub=[2.0, 20.0])

println("Optimizing... (this may take a moment)")

# Solve optimization problem using Nelder-Mead (derivative-free)
result = solve(optprob, NelderMead(), maxiters=1000)

k_est, x0_est = result.u

println()
println("✓ Optimization complete!")
println()
println("Estimated parameters:")
println("  k = $(round(k_est, digits=4)) (true: $k_true)")
println("  x0 = $(round(x0_est, digits=4)) (true: $x0_true)")
println()
println("Relative errors:")
println("  k: $(round(abs(k_est - k_true)/k_true * 100, digits=2))%")
println("  x0: $(round(abs(x0_est - x0_true)/x0_true * 100, digits=2))%")
println()

#=============================================================================
Section 5: Visualize Results
=============================================================================#

println("\nSection 5: Visualizing Results")
println("-"^70)

# Generate predictions with estimated parameters
u0_est = [decay_sys.x => x0_est]
p_est = [decay_sys.k => k_est]
prob_est = ODEProblem(decay_sys, u0_est, tspan, p_est)
sol_est = solve(prob_est, Tsit5())

# Fine time grid for smooth plotting
t_fine = range(0, 10, length=200)
x_est = [sol_est(t_val)[1] for t_val in t_fine]
x_true_fine = x0_true .* exp.(-k_true .* t_fine)

# Create visualization
fig1 = Figure(size=(1200, 800))

ax1 = Axis(fig1[1, 1],
    xlabel="Time",
    ylabel="x(t)",
    title="Parameter Estimation Results",
    xgridvisible=true,
    ygridvisible=true
)

# Plot true curve
lines!(ax1, t_fine, x_true_fine, color=:black, linewidth=3, linestyle=:dash, label="True model")

# Plot noisy data
scatter!(ax1, t_data, x_data, color=:blue, markersize=10, label="Noisy data")

# Plot fitted curve
lines!(ax1, t_fine, x_est, color=:red, linewidth=2, label="Fitted model")

axislegend(ax1, position=:rt)

# Residuals plot
ax2 = Axis(fig1[2, 1],
    xlabel="Time",
    ylabel="Residual",
    title="Fit Residuals",
    xgridvisible=true,
    ygridvisible=true
)

x_est_at_data = [sol_est(t_val)[1] for t_val in t_data]
residuals = x_data .- x_est_at_data

scatter!(ax2, t_data, residuals, color=:purple, markersize=10)
hlines!(ax2, [0], color=:black, linestyle=:dash, linewidth=1)

save(joinpath(output_dir, "08_parameter_estimation.png"), fig1, px_per_unit=2)
println("✓ Saved results to output/08_parameter_estimation.png")
println()

#=============================================================================
Section 6: Multi-Parameter Fitting (Damped Oscillator)
=============================================================================#

println("\nSection 6: Multi-Parameter Fitting - Damped Oscillator")
println("-"^70)

# Fit a damped harmonic oscillator: x'' + 2ζω₀x' + ω₀²x = 0
# Parameters to estimate: ζ (damping ratio) and ω₀ (natural frequency)

# True parameters
ζ_true = 0.1
ω0_true = 2.0
x0_osc_true = 1.0
v0_true = 0.0

println("Damped oscillator: x'' + 2ζω₀x' + ω₀²x = 0")
println("True parameters:")
println("  ζ = $ζ_true (damping ratio)")
println("  ω₀ = $ω0_true (natural frequency)")
println()

# Generate synthetic data
@parameters ζ ω0
@variables x_osc(t) v_osc(t)

eqs_osc = [
    D(x_osc) ~ v_osc,
    D(v_osc) ~ -2*ζ*ω0*v_osc - ω0^2*x_osc
]

@named damped_osc = ODESystem(eqs_osc, t)
osc_sys = structural_simplify(damped_osc)

# Generate true data
u0_osc_true = [osc_sys.x_osc => x0_osc_true, osc_sys.v_osc => v0_true]
p_osc_true = [osc_sys.ζ => ζ_true, osc_sys.ω0 => ω0_true]
tspan_osc = (0.0, 10.0)
prob_osc_true = ODEProblem(osc_sys, u0_osc_true, tspan_osc, p_osc_true)
sol_osc_true = solve(prob_osc_true, Tsit5())

t_data_osc = range(0, 10, length=40)
x_data_osc = [sol_osc_true(t_val)[1] for t_val in t_data_osc] .+ 0.05 .* randn(length(t_data_osc))

println("✓ Generated oscillator data with $(length(t_data_osc)) points")
println()

# Define loss function for oscillator
function loss_osc(params, prob_template, t_data, x_data)
    ζ_est, ω0_est = params
    
    if ζ_est <= 0 || ω0_est <= 0 || ζ_est > 1
        return Inf
    end
    
    u0 = [osc_sys.x_osc => 1.0, osc_sys.v_osc => 0.0]
    p = [osc_sys.ζ => ζ_est, osc_sys.ω0 => ω0_est]
    
    prob = remake(prob_template, u0=u0, p=p)
    sol = solve(prob, Tsit5(), saveat=t_data, verbose=false)
    
    if sol.retcode != :Success
        return Inf
    end
    
    x_pred = [sol(t_val)[1] for t_val in t_data]
    return sum((x_data .- x_pred).^2)
end

# Optimize
initial_osc = [0.15, 1.5]
prob_osc_template = ODEProblem(osc_sys, u0_osc_true, tspan_osc, p_osc_true)

optf_osc = OptimizationFunction(
    (params, p) -> loss_osc(params, prob_osc_template, t_data_osc, x_data_osc),
    Optimization.AutoForwardDiff()
)

optprob_osc = OptimizationProblem(optf_osc, initial_osc, lb=[0.01, 0.5], ub=[0.99, 5.0])

println("Optimizing oscillator parameters...")
result_osc = solve(optprob_osc, NelderMead(), maxiters=1000)

ζ_est, ω0_est = result_osc.u

println()
println("✓ Optimization complete!")
println()
println("Estimated parameters:")
println("  ζ = $(round(ζ_est, digits=4)) (true: $ζ_true)")
println("  ω₀ = $(round(ω0_est, digits=4)) (true: $ω0_true)")
println()

# Visualize
u0_osc_est = [osc_sys.x_osc => 1.0, osc_sys.v_osc => 0.0]
p_osc_est = [osc_sys.ζ => ζ_est, osc_sys.ω0 => ω0_est]
prob_osc_est = ODEProblem(osc_sys, u0_osc_est, tspan_osc, p_osc_est)
sol_osc_est = solve(prob_osc_est, Tsit5())

t_fine_osc = range(0, 10, length=200)

fig2 = Figure(size=(1200, 600))

ax = Axis(fig2[1, 1],
    xlabel="Time",
    ylabel="Displacement",
    title="Damped Oscillator - Parameter Estimation",
    xgridvisible=true,
    ygridvisible=true
)

# True solution
x_true_osc = [sol_osc_true(t_val)[1] for t_val in t_fine_osc]
lines!(ax, t_fine_osc, x_true_osc, color=:black, linewidth=3, linestyle=:dash, label="True")

# Data
scatter!(ax, t_data_osc, x_data_osc, color=:blue, markersize=8, label="Data")

# Fitted solution
x_est_osc = [sol_osc_est(t_val)[1] for t_val in t_fine_osc]
lines!(ax, t_fine_osc, x_est_osc, color=:red, linewidth=2, label="Fitted")

axislegend(ax, position=:rt)

save(joinpath(output_dir, "08_damped_oscillator_fit.png"), fig2, px_per_unit=2)
println("✓ Saved oscillator fit to output/08_damped_oscillator_fit.png")
println()

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("✅ Demo 08 Complete!")
println("="^70)
println()
println("Summary of what you learned:")
println("• Formulating parameter estimation as optimization")
println("• Defining loss functions for model fitting")
println("• Using Optimization.jl for parameter estimation")
println("• Fitting single and multiple parameters")
println("• Visualizing fit quality and residuals")
println()
println("Key insights:")
println("• Good initial guesses help optimization converge faster")
println("• Loss function design affects estimation quality")
println("• Residual plots help diagnose fit quality")
println("• More parameters require more data for reliable estimation")
println("• Physical constraints (bounds) improve optimization")
println()
println("Next: Try 09_gpu_parallelization.jl to explore GPU acceleration!")
println()
