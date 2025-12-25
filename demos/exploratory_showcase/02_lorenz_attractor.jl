"""
# 02: Lorenz Attractor - Chaos and Sensitive Dependence on Initial Conditions

The Lorenz system is one of the most famous examples of a chaotic dynamical system.
Discovered by Edward Lorenz in 1963 while studying atmospheric convection, it demonstrates
how deterministic systems can exhibit unpredictable behavior - the so-called "butterfly effect."

This demo explores:
1. Defining ODEs with ModelingToolkit
2. Solving nonlinear differential equations
3. Visualizing 3D trajectories
4. Understanding chaos and sensitivity to initial conditions
5. Creating beautiful phase space plots

## The Lorenz Equations:
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz

## Parameters:
σ (sigma) = 10.0  - Prandtl number
ρ (rho)   = 28.0  - Rayleigh number
β (beta)  = 8/3   - Geometric factor

## Prerequisites:
- Basic understanding of differential equations
- Familiarity with phase space concepts (helpful)

## Duration: ~20 minutes
"""

using ModelingToolkit
using DifferentialEquations
using CairoMakie
using LinearAlgebra
using ModelingToolkit: t_nounits as t, D_nounits as D

# Set up output directory
output_dir = "output"
mkpath(output_dir)

println("="^70)
println("🦋 Demo 02: Lorenz Attractor - The Butterfly Effect")
println("="^70)
println()

#=============================================================================
Section 1: Define the Lorenz System
=============================================================================#

println("Section 1: Defining the Lorenz System")
println("-"^70)

# Define symbolic parameters
@parameters σ=10.0 ρ=28.0 β=8/3

# Define symbolic variables (state variables)
@variables x(t)=1.0 y(t)=0.0 z(t)=0.0

# Define the differential equations
# The Lorenz equations describe simplified atmospheric convection
equations = [
    D(x) ~ σ * (y - x),      # Rate of change of x
    D(y) ~ x * (ρ - z) - y,  # Rate of change of y
    D(z) ~ x * y - β * z     # Rate of change of z
]

println("The Lorenz Equations:")
println("dx/dt = σ(y - x)")
println("dy/dt = x(ρ - z) - y")
println("dz/dt = xy - βz")
println()
println("Where:")
println("  σ (sigma) = 10.0  - Prandtl number (ratio of momentum diffusivity to thermal diffusivity)")
println("  ρ (rho)   = 28.0  - Rayleigh number (ratio of buoyancy to viscous forces)")
println("  β (beta)  = 8/3   - Geometric factor")
println()

# Create the ODE system
@named lorenz = ODESystem(equations, t)

# Simplify the system (performs symbolic simplifications and transformations)
lorenz_sys = structural_simplify(lorenz)

println("✓ Lorenz system created and simplified")
println()

#=============================================================================
Section 2: Solve the System
=============================================================================#

println("\nSection 2: Solving the Lorenz System")
println("-"^70)

# Define initial conditions
# Small changes in initial conditions lead to vastly different trajectories!
u0 = [x => 1.0,
      y => 0.0,
      z => 0.0]

# Define parameters (already set in defaults, but shown for clarity)
params = [σ => 10.0,
          ρ => 28.0,
          β => 8/3]

# Time span for simulation
tspan = (0.0, 50.0)

# Create the ODE problem
prob = ODEProblem(lorenz_sys, u0, tspan, params)

println("Initial conditions: x=1.0, y=0.0, z=0.0")
println("Time span: 0 to 50 time units")
println("Solving...")

# Solve the system
# Tsit5() is a good general-purpose solver (Tsitouras 5/4 Runge-Kutta method)
sol = solve(prob, Tsit5(), saveat=0.01)

println("✓ Solution computed with $(length(sol.t)) time points")
println()

#=============================================================================
Section 3: Extract Solution Data
=============================================================================#

# Extract trajectories
x_vals = [u[1] for u in sol.u]
y_vals = [u[2] for u in sol.u]
z_vals = [u[3] for u in sol.u]
t_vals = sol.t

println("Trajectory statistics:")
println("  x range: [$(minimum(x_vals)), $(maximum(x_vals))]")
println("  y range: [$(minimum(y_vals)), $(maximum(y_vals))]")
println("  z range: [$(minimum(z_vals)), $(maximum(z_vals))]")
println()

#=============================================================================
Section 4: 3D Visualization - The Butterfly Attractor
=============================================================================#

println("\nSection 4: Creating 3D Visualization")
println("-"^70)

# Create a beautiful 3D plot of the Lorenz attractor
fig = Figure(size=(1200, 900), fontsize=16)

# Main 3D plot
ax3d = Axis3(fig[1, 1],
    xlabel="x",
    ylabel="y",
    zlabel="z",
    title="Lorenz Attractor - The Butterfly Effect",
    aspect=(1, 1, 1),
    azimuth=pi/6,
    elevation=pi/9
)

# Plot the trajectory with color gradient representing time
# This shows how the system evolves through phase space
lines!(ax3d, x_vals, y_vals, z_vals,
    color=t_vals,
    colormap=:plasma,
    linewidth=1.5,
    transparency=true
)

# Add starting point
scatter!(ax3d, [x_vals[1]], [y_vals[1]], [z_vals[1]],
    color=:green,
    markersize=20,
    label="Start"
)

# Add ending point
scatter!(ax3d, [x_vals[end]], [y_vals[end]], [z_vals[end]],
    color=:red,
    markersize=20,
    label="End"
)

save(joinpath(output_dir, "02_lorenz_3d_attractor.png"), fig, px_per_unit=2)
println("✓ Saved 3D attractor to output/02_lorenz_3d_attractor.png")
println()

#=============================================================================
Section 5: Phase Space Projections
=============================================================================#

println("\nSection 5: Phase Space Projections")
println("-"^70)

# Create 2D projections of the 3D attractor
fig2 = Figure(size=(1400, 450))

# X-Y plane
ax1 = Axis(fig2[1, 1],
    xlabel="x",
    ylabel="y",
    title="X-Y Projection",
    aspect=DataAspect()
)
lines!(ax1, x_vals, y_vals, color=t_vals, colormap=:viridis, linewidth=1)

# X-Z plane
ax2 = Axis(fig2[1, 2],
    xlabel="x",
    ylabel="z",
    title="X-Z Projection",
    aspect=DataAspect()
)
lines!(ax2, x_vals, z_vals, color=t_vals, colormap=:viridis, linewidth=1)

# Y-Z plane
ax3 = Axis(fig2[1, 3],
    xlabel="y",
    ylabel="z",
    title="Y-Z Projection",
    aspect=DataAspect()
)
lines!(ax3, y_vals, z_vals, color=t_vals, colormap=:viridis, linewidth=1)

save(joinpath(output_dir, "02_lorenz_projections.png"), fig2, px_per_unit=2)
println("✓ Saved phase space projections to output/02_lorenz_projections.png")
println()

#=============================================================================
Section 6: Time Series Analysis
=============================================================================#

println("\nSection 6: Time Series Analysis")
println("-"^70)

# Plot how x, y, z vary with time
fig3 = Figure(size=(1200, 900))

ax_x = Axis(fig3[1, 1],
    xlabel="Time",
    ylabel="x(t)",
    title="X Component vs Time",
    xgridvisible=true,
    ygridvisible=true
)
lines!(ax_x, t_vals, x_vals, color=:blue, linewidth=1.5)

ax_y = Axis(fig3[2, 1],
    xlabel="Time",
    ylabel="y(t)",
    title="Y Component vs Time",
    xgridvisible=true,
    ygridvisible=true
)
lines!(ax_y, t_vals, y_vals, color=:red, linewidth=1.5)

ax_z = Axis(fig3[3, 1],
    xlabel="Time",
    ylabel="z(t)",
    title="Z Component vs Time",
    xgridvisible=true,
    ygridvisible=true
)
lines!(ax_z, t_vals, z_vals, color=:green, linewidth=1.5)

save(joinpath(output_dir, "02_lorenz_timeseries.png"), fig3, px_per_unit=2)
println("✓ Saved time series to output/02_lorenz_timeseries.png")
println()

#=============================================================================
Section 7: Sensitivity to Initial Conditions
=============================================================================#

println("\nSection 7: Demonstrating Sensitive Dependence on Initial Conditions")
println("-"^70)

# Solve with slightly different initial conditions
# This demonstrates the "butterfly effect"
ε = 1e-5  # Tiny perturbation

println("Testing sensitivity with perturbation ε = $ε")
println()

# Original initial condition
u0_original = [x => 1.0, y => 0.0, z => 0.0]

# Perturbed initial condition (change x by ε)
u0_perturbed = [x => 1.0 + ε, y => 0.0, z => 0.0]

# Solve both systems
prob_original = ODEProblem(lorenz_sys, u0_original, tspan, params)
prob_perturbed = ODEProblem(lorenz_sys, u0_perturbed, tspan, params)

sol_original = solve(prob_original, Tsit5(), saveat=0.01)
sol_perturbed = solve(prob_perturbed, Tsit5(), saveat=0.01)

# Extract data
x_orig = [u[1] for u in sol_original.u]
x_pert = [u[1] for u in sol_perturbed.u]

# Compute difference (error growth)
error = abs.(x_orig .- x_pert)

println("Maximum separation between trajectories: $(maximum(error))")
println("Initial separation: $ε")
println("Amplification factor: $(maximum(error) / ε)")
println()

# Visualize the divergence
fig4 = Figure(size=(1200, 800))

# 3D plot comparing trajectories
ax3d_compare = Axis3(fig4[1, 1],
    xlabel="x",
    ylabel="y",
    zlabel="z",
    title="Comparing Nearly Identical Initial Conditions",
    azimuth=pi/6,
    elevation=pi/9
)

# Original trajectory
x_orig_full = [u[1] for u in sol_original.u]
y_orig_full = [u[2] for u in sol_original.u]
z_orig_full = [u[3] for u in sol_original.u]

# Perturbed trajectory
x_pert_full = [u[1] for u in sol_perturbed.u]
y_pert_full = [u[2] for u in sol_perturbed.u]
z_pert_full = [u[3] for u in sol_perturbed.u]

lines!(ax3d_compare, x_orig_full, y_orig_full, z_orig_full,
    color=:blue, linewidth=2, label="Original", alpha=0.7)
lines!(ax3d_compare, x_pert_full, y_pert_full, z_pert_full,
    color=:red, linewidth=2, label="Perturbed", alpha=0.7)

# Error growth plot
ax_error = Axis(fig4[2, 1],
    xlabel="Time",
    ylabel="|Δx|",
    title="Growth of Perturbation Over Time",
    yscale=log10,
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax_error, sol_original.t, error, color=:purple, linewidth=2)
hlines!(ax_error, [ε], color=:black, linestyle=:dash, linewidth=1, label="Initial ε")

save(joinpath(output_dir, "02_lorenz_sensitivity.png"), fig4, px_per_unit=2)
println("✓ Saved sensitivity analysis to output/02_lorenz_sensitivity.png")
println()

#=============================================================================
Section 8: Parameter Space Exploration
=============================================================================#

println("\nSection 8: Exploring Different Parameter Values")
println("-"^70)

# The behavior of the Lorenz system changes dramatically with parameter values
# Let's explore different ρ (rho) values

rho_values = [10.0, 20.0, 28.0, 40.0]
fig5 = Figure(size=(1400, 1400))

for (idx, rho_val) in enumerate(rho_values)
    row = div(idx - 1, 2) + 1
    col = mod(idx - 1, 2) + 1
    
    # Solve with different ρ value
    params_temp = [σ => 10.0, ρ => rho_val, β => 8/3]
    prob_temp = ODEProblem(lorenz_sys, u0, tspan, params_temp)
    sol_temp = solve(prob_temp, Tsit5(), saveat=0.01)
    
    # Extract data
    x_temp = [u[1] for u in sol_temp.u]
    y_temp = [u[2] for u in sol_temp.u]
    z_temp = [u[3] for u in sol_temp.u]
    
    # Create 3D plot
    ax = Axis3(fig5[row, col],
        xlabel="x",
        ylabel="y",
        zlabel="z",
        title="ρ = $rho_val",
        azimuth=pi/6,
        elevation=pi/9
    )
    
    lines!(ax, x_temp, y_temp, z_temp,
        color=sol_temp.t,
        colormap=:plasma,
        linewidth=1.5
    )
    
    println("  ρ = $rho_val: trajectory computed")
end

save(joinpath(output_dir, "02_lorenz_parameter_exploration.png"), fig5, px_per_unit=2)
println("✓ Saved parameter exploration to output/02_lorenz_parameter_exploration.png")
println()

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("✅ Demo 02 Complete!")
println("="^70)
println()
println("Summary of what you learned:")
println("• Defining ODE systems with ModelingToolkit")
println("• Solving nonlinear differential equations")
println("• Creating 3D visualizations of phase space trajectories")
println("• Understanding chaos and the butterfly effect")
println("• Analyzing sensitivity to initial conditions")
println("• Exploring parameter space")
println()
println("Key insights:")
println("• The Lorenz attractor is a strange attractor - trajectories are bounded")
println("  but never repeat exactly")
println("• Tiny changes in initial conditions lead to exponentially growing differences")
println("• The system is deterministic but unpredictable in the long term")
println("• Different parameter values lead to qualitatively different behaviors")
println()
println("Next: Try 03_nonlinear_pendulum.jl to explore classical mechanics!")
println()
