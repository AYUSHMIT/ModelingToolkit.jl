"""
# 05: Wave Equation - Hyperbolic PDEs and Wave Propagation

The wave equation describes the propagation of waves through a medium. It appears in
acoustics, electromagnetism, elasticity, and many other areas of physics.

This demo explores:
1. 1D and 2D wave equations
2. Method of Lines discretization
3. Different initial conditions (plucked string, drum)
4. Wave reflection at boundaries
5. Animated wave visualization

## The Wave Equation:
∂²u/∂t² = c²∇²u

Where:
- u is displacement
- c is wave speed
- ∇² is the Laplacian operator

## Prerequisites:
- Understanding of partial differential equations
- Basic wave mechanics

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
println("🌊 Demo 05: Wave Equation - Propagation and Interference")
println("="^70)
println()

#=============================================================================
Section 1: 1D Wave Equation - Vibrating String
=============================================================================#

println("Section 1: 1D Wave Equation - Vibrating String")
println("-"^70)

# Wave equation: u_tt = c² * u_xx
# Convert to first-order system:
# u_t = v
# v_t = c² * u_xx

# Parameters
c = 1.0   # Wave speed
L = 1.0   # String length
N = 100   # Number of spatial points
dx = L / (N - 1)
x_vals = range(0, L, length=N)

println("1D Wave Equation Setup:")
println("  Domain: [0, $L]")
println("  Grid points: $N")
println("  Wave speed: $c")
println()

# Create variables: u (displacement) and v (velocity)
@variables u[1:N](t) v[1:N](t)
@parameters c_param=c

# Build equations
eqs = Equation[]

# Boundary conditions: fixed ends (u = 0 at boundaries)
push!(eqs, D(u[1]) ~ 0)
push!(eqs, D(v[1]) ~ 0)
push!(eqs, D(u[N]) ~ 0)
push!(eqs, D(v[N]) ~ 0)

# Interior points
for i in 2:(N-1)
    # u_t = v
    push!(eqs, D(u[i]) ~ v[i])
    
    # v_t = c² * (u_{i-1} - 2u_i + u_{i+1}) / dx²
    laplacian = (u[i-1] - 2*u[i] + u[i+1]) / dx^2
    push!(eqs, D(v[i]) ~ c_param^2 * laplacian)
end

@named wave_1d = ODESystem(eqs, t)
wave_1d_sys = structural_simplify(wave_1d)

# Initial condition: plucked string (triangle wave)
pluck_position = L / 2
pluck_height = 0.1

u0_vals = [pluck_height * (1 - abs(x - pluck_position) / (L/2)) * 
           (abs(x - pluck_position) < L/2) for x in x_vals]
v0_vals = zeros(N)  # Initial velocity = 0

# Combine initial conditions
u0 = vcat([u[i] => u0_vals[i] for i in 1:N],
          [v[i] => v0_vals[i] for i in 1:N])

# Solve
tspan = (0.0, 5.0)
prob_1d = ODEProblem(wave_1d_sys, u0, tspan, [c_param => c])
sol_1d = solve(prob_1d, Tsit5(), saveat=0.01)

println("✓ 1D wave equation solved")
println("  Time points: $(length(sol_1d.t))")
println()

# Visualize wave propagation
fig1 = Figure(size=(1200, 800))

ax1 = Axis(fig1[1, 1],
    xlabel="Position x",
    ylabel="Displacement u(x,t)",
    title="1D Wave Propagation - Plucked String",
    xgridvisible=true,
    ygridvisible=true
)

# Plot multiple time snapshots
times_to_plot = [1, 25, 50, 75, 100, 125, 150]
colors = [:red, :orange, :yellow, :green, :blue, :purple, :black]

for (idx, t_idx) in enumerate(times_to_plot)
    u_snapshot = [sol_1d.u[t_idx][i] for i in 1:N]
    time_val = sol_1d.t[t_idx]
    lines!(ax1, x_vals, u_snapshot, color=colors[idx], linewidth=2,
           label="t=$(round(time_val, digits=2))")
end

axislegend(ax1, position=:rt)

# Spatiotemporal diagram
ax2 = Axis(fig1[2, 1],
    xlabel="Position x",
    ylabel="Time t",
    title="Spatiotemporal Evolution"
)

u_matrix = zeros(length(sol_1d.t), N)
for (t_idx, sol_t) in enumerate(sol_1d.u)
    for i in 1:N
        u_matrix[t_idx, i] = sol_t[i]
    end
end

hm = heatmap!(ax2, x_vals, sol_1d.t, u_matrix, colormap=:balance)
Colorbar(fig1[2, 2], hm, label="Displacement")

save(joinpath(output_dir, "05_wave_equation_1d.png"), fig1, px_per_unit=2)
println("✓ Saved 1D wave visualization to output/05_wave_equation_1d.png")
println()

#=============================================================================
Section 2: Energy Analysis
=============================================================================#

println("\nSection 2: Energy Conservation")
println("-"^70)

# Total energy = kinetic + potential
# E = ∫(v² + c²(u_x)²) dx

function compute_wave_energy(sol_t, N, dx, c)
    u_vals = [sol_t[i] for i in 1:N]
    v_vals = [sol_t[N+i] for i in 1:N]
    
    # Kinetic energy: ∫ v² dx
    kinetic = sum(v_vals.^2) * dx
    
    # Potential energy: ∫ c²(∂u/∂x)² dx
    u_x = diff(u_vals) / dx  # Spatial derivative
    potential = c^2 * sum(u_x.^2) * dx
    
    return kinetic, potential, kinetic + potential
end

energies = [compute_wave_energy(sol_t, N, dx, c) for sol_t in sol_1d.u]
kinetic_energy = [e[1] for e in energies]
potential_energy = [e[2] for e in energies]
total_energy = [e[3] for e in energies]

fig2 = Figure(size=(1200, 600))

ax = Axis(fig2[1, 1],
    xlabel="Time",
    ylabel="Energy",
    title="Energy Conservation in Wave Equation",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax, sol_1d.t, kinetic_energy, color=:blue, linewidth=2, label="Kinetic")
lines!(ax, sol_1d.t, potential_energy, color=:red, linewidth=2, label="Potential")
lines!(ax, sol_1d.t, total_energy, color=:black, linewidth=2, linestyle=:dash, label="Total")
axislegend(ax, position=:rt)

println("Energy statistics:")
println("  Initial total energy: $(total_energy[1])")
println("  Final total energy: $(total_energy[end])")
println("  Variation: $(maximum(total_energy) - minimum(total_energy))")
println()

save(joinpath(output_dir, "05_wave_equation_energy.png"), fig2, px_per_unit=2)
println("✓ Saved energy plot to output/05_wave_equation_energy.png")
println()

#=============================================================================
Section 3: 2D Wave Equation - Vibrating Membrane (Drum)
=============================================================================#

println("\nSection 3: 2D Wave Equation - Vibrating Drum")
println("-"^70)

# 2D wave equation: u_tt = c²(u_xx + u_yy)
# Use coarser grid for computational efficiency

Nx = 25
Ny = 25
Lx = 1.0
Ly = 1.0
dx_2d = Lx / (Nx - 1)
dy_2d = Ly / (Ny - 1)

println("2D Wave Equation Setup:")
println("  Domain: [$Lx × $Ly]")
println("  Grid: $Nx × $Ny")
println("  Total variables: $(2 * Nx * Ny)")
println()

# Helper function
idx(i, j) = (i - 1) * Ny + j

n_vars_2d = Nx * Ny
@variables u_2d[1:n_vars_2d](t) v_2d[1:n_vars_2d](t)

# Build equations
eqs_2d = Equation[]

for i in 1:Nx
    for j in 1:Ny
        k = idx(i, j)
        
        # Boundary conditions: fixed edges
        if i == 1 || i == Nx || j == 1 || j == Ny
            push!(eqs_2d, D(u_2d[k]) ~ 0)
            push!(eqs_2d, D(v_2d[k]) ~ 0)
        else
            # u_t = v
            push!(eqs_2d, D(u_2d[k]) ~ v_2d[k])
            
            # v_t = c² * ∇²u
            laplacian_2d = (u_2d[idx(i+1,j)] + u_2d[idx(i-1,j)] +
                           u_2d[idx(i,j+1)] + u_2d[idx(i,j-1)] -
                           4*u_2d[k]) / dx_2d^2
            push!(eqs_2d, D(v_2d[k]) ~ c_param^2 * laplacian_2d)
        end
    end
end

@named wave_2d = ODESystem(eqs_2d, t)
println("Building 2D wave system...")
wave_2d_sys = structural_simplify(wave_2d)

# Initial condition: Gaussian pulse in center
u0_2d = Float64[]
v0_2d = Float64[]

for i in 1:Nx
    for j in 1:Ny
        x = (i - 1) * dx_2d
        y = (j - 1) * dy_2d
        # Gaussian centered at (0.5, 0.5)
        displacement = 0.5 * exp(-50 * ((x - Lx/2)^2 + (y - Ly/2)^2))
        push!(u0_2d, displacement)
        push!(v0_2d, 0.0)
    end
end

u0_2d_dict = vcat([u_2d[i] => u0_2d[i] for i in 1:n_vars_2d],
                   [v_2d[i] => v0_2d[i] for i in 1:n_vars_2d])

# Solve
tspan_2d = (0.0, 2.0)
prob_2d = ODEProblem(wave_2d_sys, u0_2d_dict, tspan_2d, [c_param => c])

println("Solving 2D wave equation (this may take a while)...")
sol_2d = solve(prob_2d, Tsit5(), saveat=0.05)

println("✓ 2D wave equation solved")
println("  Time points: $(length(sol_2d.t))")
println()

#=============================================================================
Section 4: Visualize 2D Wave Propagation
=============================================================================#

println("\nSection 4: Visualizing 2D Wave Propagation")
println("-"^70)

x_grid = range(0, Lx, length=Nx)
y_grid = range(0, Ly, length=Ny)

function sol_to_grid_wave(sol_vec, n_vars)
    grid = zeros(Nx, Ny)
    for i in 1:Nx
        for j in 1:Ny
            grid[i, j] = sol_vec[idx(i, j)]
        end
    end
    return grid
end

# Create snapshots
fig3 = Figure(size=(1600, 900))

times_2d = [1, 8, 15, 22, 29, 36]
for (plot_idx, t_idx) in enumerate(times_2d)
    row = div(plot_idx - 1, 3) + 1
    col = mod(plot_idx - 1, 3) + 1
    
    time_val = sol_2d.t[t_idx]
    
    ax = Axis3(fig3[row, col],
        xlabel="x",
        ylabel="y",
        zlabel="u",
        title="t=$(round(time_val, digits=2))",
        azimuth=π/4,
        elevation=π/8
    )
    
    grid = sol_to_grid_wave(sol_2d.u[t_idx], n_vars_2d)
    surface!(ax, x_grid, y_grid, grid', colormap=:balance, colorrange=(-0.3, 0.3))
end

save(joinpath(output_dir, "05_wave_equation_2d_snapshots.png"), fig3, px_per_unit=2)
println("✓ Saved 2D wave snapshots to output/05_wave_equation_2d_snapshots.png")
println()

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("✅ Demo 05 Complete!")
println("="^70)
println()
println("Summary of what you learned:")
println("• Implementing the wave equation using Method of Lines")
println("• Simulating 1D vibrating string")
println("• Simulating 2D vibrating membrane (drum)")
println("• Analyzing energy conservation")
println("• Creating spatiotemporal visualizations")
println()
println("Key insights:")
println("• Waves propagate and reflect at boundaries")
println("• Energy oscillates between kinetic and potential forms")
println("• 2D waves create complex interference patterns")
println("• Fixed boundary conditions cause wave reflection")
println()
println("Next: Try 06_chemical_reactions.jl to explore reaction networks!")
println()
