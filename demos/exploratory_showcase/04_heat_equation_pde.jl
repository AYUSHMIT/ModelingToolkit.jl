"""
# 04: Heat Equation - Parabolic PDEs and Diffusion

The heat equation is a fundamental parabolic partial differential equation that describes
how heat (or other diffusive quantities) spreads through space over time. It appears in
physics, finance, image processing, and many other fields.

This demo explores:
1. The Method of Lines for PDE discretization
2. 1D and 2D heat diffusion
3. Different boundary conditions (Dirichlet, Neumann)
4. Animated heatmap visualizations
5. Stability considerations

## The Heat Equation:
∂u/∂t = α∇²u

Where:
- u is temperature (or concentration)
- α is the thermal diffusivity
- ∇² is the Laplacian operator

## Prerequisites:
- Understanding of partial differential equations
- Basic knowledge of numerical methods

## Duration: ~30 minutes
"""

using ModelingToolkit
using DifferentialEquations
using CairoMakie
using LinearAlgebra
using SparseArrays
using ModelingToolkit: t_nounits as t, D_nounits as D

# Set up output directory
output_dir = "output"
mkpath(output_dir)

println("="^70)
println("🔥 Demo 04: Heat Equation - Modeling Diffusion")
println("="^70)
println()

#=============================================================================
Section 1: 1D Heat Equation with Dirichlet Boundary Conditions
=============================================================================#

println("Section 1: 1D Heat Equation")
println("-"^70)

# Method of Lines: discretize space, keep time continuous
# u_t = α * u_xx
# Discretize x: u_xx ≈ (u_{i-1} - 2u_i + u_{i+1}) / Δx²

# Parameters
α = 0.01  # Thermal diffusivity
L = 1.0   # Domain length
N = 50    # Number of spatial points
dx = L / (N - 1)
x_vals = range(0, L, length=N)

println("1D Heat Equation Setup:")
println("  Domain: [0, $L]")
println("  Grid points: $N")
println("  Spatial step: $dx")
println("  Thermal diffusivity: $α")
println()

# Create variables for each spatial point
@variables u[1:N](t)
@parameters α_param=α

# Build the equations using finite differences
eqs = Equation[]

# Boundary conditions: u(0,t) = 0, u(L,t) = 0 (Dirichlet)
push!(eqs, D(u[1]) ~ 0)  # Fixed at boundary
push!(eqs, D(u[N]) ~ 0)  # Fixed at boundary

# Interior points: central difference approximation
for i in 2:(N-1)
    # ∂u/∂t = α * (u_{i-1} - 2u_i + u_{i+1}) / Δx²
    laplacian = (u[i-1] - 2*u[i] + u[i+1]) / dx^2
    push!(eqs, D(u[i]) ~ α_param * laplacian)
end

# Create the system
@named heat_1d = ODESystem(eqs, t)
heat_1d_sys = structural_simplify(heat_1d)

# Initial condition: Gaussian pulse in the middle
u0_vals = [exp(-100 * (x - L/2)^2) for x in x_vals]
u0 = [u[i] => u0_vals[i] for i in 1:N]

# Solve
tspan = (0.0, 5.0)
prob_1d = ODEProblem(heat_1d_sys, u0, tspan, [α_param => α])
sol_1d = solve(prob_1d, Tsit5(), saveat=0.05)

println("✓ 1D heat equation solved")
println("  Time points: $(length(sol_1d.t))")
println()

# Visualize evolution
fig1 = Figure(size=(1200, 800))

ax1 = Axis(fig1[1, 1],
    xlabel="Position x",
    ylabel="Temperature u(x,t)",
    title="1D Heat Diffusion Over Time",
    xgridvisible=true,
    ygridvisible=true
)

# Plot multiple time snapshots
times_to_plot = [1, 10, 20, 40, 60, 80, length(sol_1d.t)]
colors = [:red, :orange, :yellow, :green, :blue, :purple, :black]

for (idx, t_idx) in enumerate(times_to_plot)
    u_snapshot = [sol_1d.u[t_idx][i] for i in 1:N]
    time_val = sol_1d.t[t_idx]
    lines!(ax1, x_vals, u_snapshot, color=colors[idx], linewidth=2, 
           label="t=$(round(time_val, digits=2))")
end

axislegend(ax1, position=:rt)

# Create heatmap showing evolution
ax2 = Axis(fig1[2, 1],
    xlabel="Position x",
    ylabel="Time t",
    title="Spatiotemporal Evolution (Heatmap)"
)

# Create matrix for heatmap
u_matrix = zeros(length(sol_1d.t), N)
for (t_idx, sol_t) in enumerate(sol_1d.u)
    for i in 1:N
        u_matrix[t_idx, i] = sol_t[i]
    end
end

hm = heatmap!(ax2, x_vals, sol_1d.t, u_matrix, colormap=:thermal)
Colorbar(fig1[2, 2], hm, label="Temperature")

save(joinpath(output_dir, "04_heat_equation_1d.png"), fig1, px_per_unit=2)
println("✓ Saved 1D visualization to output/04_heat_equation_1d.png")
println()

#=============================================================================
Section 2: 1D Heat Equation with Neumann Boundary Conditions
=============================================================================#

println("\nSection 2: Neumann Boundary Conditions (Insulated Ends)")
println("-"^70)

# Neumann BC: ∂u/∂x = 0 at boundaries (no heat flux)
# Approximate using one-sided differences

eqs_neumann = Equation[]

# Left boundary: ∂u/∂x = 0 → (u[2] - u[1])/dx = 0 → u[1] = u[2]
# But we still need the evolution equation
laplacian_left = 2 * (u[2] - u[1]) / dx^2
push!(eqs_neumann, D(u[1]) ~ α_param * laplacian_left)

# Interior points
for i in 2:(N-1)
    laplacian = (u[i-1] - 2*u[i] + u[i+1]) / dx^2
    push!(eqs_neumann, D(u[i]) ~ α_param * laplacian)
end

# Right boundary: ∂u/∂x = 0 → (u[N] - u[N-1])/dx = 0 → u[N] = u[N-1]
laplacian_right = 2 * (u[N-1] - u[N]) / dx^2
push!(eqs_neumann, D(u[N]) ~ α_param * laplacian_right)

@named heat_1d_neumann = ODESystem(eqs_neumann, t)
heat_1d_neumann_sys = structural_simplify(heat_1d_neumann)

# Solve with same initial condition
prob_1d_neumann = ODEProblem(heat_1d_neumann_sys, u0, tspan, [α_param => α])
sol_1d_neumann = solve(prob_1d_neumann, Tsit5(), saveat=0.05)

println("✓ 1D heat equation with Neumann BC solved")
println()

# Compare Dirichlet vs Neumann
fig2 = Figure(size=(1200, 600))

ax_dir = Axis(fig2[1, 1],
    xlabel="Position x",
    ylabel="Temperature",
    title="Dirichlet BC (Fixed at boundaries)"
)

ax_neu = Axis(fig2[1, 2],
    xlabel="Position x",
    ylabel="Temperature",
    title="Neumann BC (Insulated boundaries)"
)

# Plot final state
u_final_dir = [sol_1d.u[end][i] for i in 1:N]
u_final_neu = [sol_1d_neumann.u[end][i] for i in 1:N]

lines!(ax_dir, x_vals, u_final_dir, color=:blue, linewidth=2)
lines!(ax_neu, x_vals, u_final_neu, color=:red, linewidth=2)

save(joinpath(output_dir, "04_heat_equation_1d_comparison.png"), fig2, px_per_unit=2)
println("✓ Saved boundary condition comparison to output/04_heat_equation_1d_comparison.png")
println()

#=============================================================================
Section 3: 2D Heat Equation
=============================================================================#

println("\nSection 3: 2D Heat Equation")
println("-"^70)

# 2D heat equation: u_t = α(u_xx + u_yy)
# This creates a larger system, so we'll use a coarser grid

Nx = 30  # Grid points in x
Ny = 30  # Grid points in y
Lx = 1.0
Ly = 1.0
dx_2d = Lx / (Nx - 1)
dy_2d = Ly / (Ny - 1)

println("2D Heat Equation Setup:")
println("  Domain: [$Lx × $Ly]")
println("  Grid: $Nx × $Ny")
println("  Total variables: $(Nx * Ny)")
println()

# Helper function to convert 2D indices to 1D
idx(i, j) = (i - 1) * Ny + j

# Create variables
n_vars_2d = Nx * Ny
@variables u_2d[1:n_vars_2d](t)

# Build equations for 2D heat equation
eqs_2d = Equation[]

for i in 1:Nx
    for j in 1:Ny
        k = idx(i, j)
        
        # Boundary conditions: fixed at 0
        if i == 1 || i == Nx || j == 1 || j == Ny
            push!(eqs_2d, D(u_2d[k]) ~ 0)
        else
            # Interior point: 5-point stencil for Laplacian
            # ∇²u ≈ (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / h²
            # (assuming dx = dy = h)
            laplacian_2d = (u_2d[idx(i+1,j)] + u_2d[idx(i-1,j)] + 
                           u_2d[idx(i,j+1)] + u_2d[idx(i,j-1)] - 
                           4*u_2d[k]) / dx_2d^2
            push!(eqs_2d, D(u_2d[k]) ~ α_param * laplacian_2d)
        end
    end
end

@named heat_2d = ODESystem(eqs_2d, t)
println("Building 2D system (this may take a moment)...")
heat_2d_sys = structural_simplify(heat_2d)

# Initial condition: Gaussian blob in the center
u0_2d = Float64[]
for i in 1:Nx
    for j in 1:Ny
        x = (i - 1) * dx_2d
        y = (j - 1) * dy_2d
        # Gaussian centered at (0.5, 0.5)
        val = exp(-50 * ((x - Lx/2)^2 + (y - Ly/2)^2))
        push!(u0_2d, val)
    end
end

u0_2d_dict = [u_2d[i] => u0_2d[i] for i in 1:n_vars_2d]

# Solve (shorter time span for 2D)
tspan_2d = (0.0, 2.0)
prob_2d = ODEProblem(heat_2d_sys, u0_2d_dict, tspan_2d, [α_param => α])

println("Solving 2D system (this may take a while)...")
sol_2d = solve(prob_2d, Tsit5(), saveat=0.1)

println("✓ 2D heat equation solved")
println("  Time points: $(length(sol_2d.t))")
println()

#=============================================================================
Section 4: Visualize 2D Heat Diffusion
=============================================================================#

println("\nSection 4: Visualizing 2D Heat Diffusion")
println("-"^70)

# Create grid for visualization
x_grid = range(0, Lx, length=Nx)
y_grid = range(0, Ly, length=Ny)

# Function to reshape 1D solution to 2D grid
function sol_to_grid(sol_vec)
    grid = zeros(Nx, Ny)
    for i in 1:Nx
        for j in 1:Ny
            grid[i, j] = sol_vec[idx(i, j)]
        end
    end
    return grid
end

# Create snapshots at different times
fig3 = Figure(size=(1400, 800))

times_2d = [1, 5, 10, 15, 20]
titles_2d = ["t=0.0", "t=0.4", "t=0.9", "t=1.4", "t=1.9"]

for (plot_idx, t_idx) in enumerate(times_2d)
    row = div(plot_idx - 1, 3) + 1
    col = mod(plot_idx - 1, 3) + 1
    
    ax = Axis(fig3[row, col],
        xlabel="x",
        ylabel="y",
        title=titles_2d[plot_idx],
        aspect=DataAspect()
    )
    
    grid = sol_to_grid(sol_2d.u[t_idx])
    hm = heatmap!(ax, x_grid, y_grid, grid', colormap=:thermal, colorrange=(0, 1))
    
    if col == 3
        Colorbar(fig3[row, col+1], hm, label="Temperature")
    end
end

save(joinpath(output_dir, "04_heat_equation_2d_snapshots.png"), fig3, px_per_unit=2)
println("✓ Saved 2D snapshots to output/04_heat_equation_2d_snapshots.png")
println()

# Create 3D surface plot for one time point
fig4 = Figure(size=(1000, 800))

ax_3d = Axis3(fig4[1, 1],
    xlabel="x",
    ylabel="y",
    zlabel="Temperature",
    title="2D Heat Distribution (3D Surface)",
    azimuth=π/4,
    elevation=π/6
)

# Use middle time point
mid_idx = div(length(sol_2d.t), 2)
grid_3d = sol_to_grid(sol_2d.u[mid_idx])

surface!(ax_3d, x_grid, y_grid, grid_3d', colormap=:thermal)

save(joinpath(output_dir, "04_heat_equation_2d_surface.png"), fig4, px_per_unit=2)
println("✓ Saved 3D surface plot to output/04_heat_equation_2d_surface.png")
println()

#=============================================================================
Section 5: Stability Analysis
=============================================================================#

println("\nSection 5: Stability Considerations")
println("-"^70)

# The explicit finite difference scheme has a stability condition:
# Δt ≤ Δx² / (2α) in 1D
# Δt ≤ Δx² / (4α) in 2D

stable_dt_1d = dx^2 / (2 * α)
stable_dt_2d = dx_2d^2 / (4 * α)

println("Stability analysis:")
println("  1D: Δt should be ≤ $(round(stable_dt_1d, digits=4)) for explicit methods")
println("  2D: Δt should be ≤ $(round(stable_dt_2d, digits=4)) for explicit methods")
println()
println("Note: DifferentialEquations.jl uses adaptive timestepping")
println("      and implicit methods when needed, so stability is handled automatically.")
println()

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("✅ Demo 04 Complete!")
println("="^70)
println()
println("Summary of what you learned:")
println("• Method of Lines for PDE discretization")
println("• Implementing different boundary conditions (Dirichlet, Neumann)")
println("• Solving 1D and 2D heat equations")
println("• Creating heatmap and surface visualizations")
println("• Understanding stability considerations")
println()
println("Key insights:")
println("• The heat equation smooths out temperature differences over time")
println("• Boundary conditions significantly affect the solution")
println("• Dirichlet BC: Fixed values at boundaries (heat sink)")
println("• Neumann BC: Fixed flux at boundaries (insulation)")
println("• 2D diffusion spreads in all directions")
println("• Adaptive solvers handle stability automatically")
println()
println("Next: Try 05_wave_equation_pde.jl to explore hyperbolic PDEs!")
println()
