"""
# 03: Nonlinear Pendulum - When Approximations Break Down

The simple pendulum is a classic problem in mechanics that demonstrates the difference
between linear and nonlinear dynamics. The small-angle approximation (sin(θ) ≈ θ)
works well for small oscillations, but breaks down for large angles.

This demo explores:
1. Deriving the pendulum equations of motion
2. Comparing linear vs. nonlinear solutions
3. Phase portraits and limit cycles
4. Energy conservation
5. The effect of damping

## The Pendulum Equations:
Full nonlinear: d²θ/dt² = -(g/L)sin(θ) - (c/m)dθ/dt
Linear approx:  d²θ/dt² = -(g/L)θ - (c/m)dθ/dt

## Parameters:
g = 9.81   - Gravitational acceleration (m/s²)
L = 1.0    - Pendulum length (m)
c = 0.5    - Damping coefficient (kg/s)
m = 1.0    - Mass (kg)

## Prerequisites:
- Classical mechanics basics
- Understanding of differential equations

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
println("⚖️  Demo 03: Nonlinear Pendulum - Beyond Small Angles")
println("="^70)
println()

#=============================================================================
Section 1: Define the Nonlinear Pendulum
=============================================================================#

println("Section 1: Defining the Pendulum Systems")
println("-"^70)

# Physical parameters
@parameters g=9.81 L=1.0 c=0.5 m=1.0

# State variables: θ (angle) and ω (angular velocity)
@variables θ(t)=π/3 ω(t)=0.0

# Nonlinear pendulum equation
# θ'' = -(g/L)sin(θ) - (c/m)θ'
# Rewrite as first-order system:
# θ' = ω
# ω' = -(g/L)sin(θ) - (c/m)ω

nonlinear_eqs = [
    D(θ) ~ ω,
    D(ω) ~ -(g/L) * sin(θ) - (c/m) * ω
]

@named pendulum_nonlinear = ODESystem(nonlinear_eqs, t)
pendulum_nonlinear_sys = structural_simplify(pendulum_nonlinear)

# Linear (small-angle) approximation: sin(θ) ≈ θ
linear_eqs = [
    D(θ) ~ ω,
    D(ω) ~ -(g/L) * θ - (c/m) * ω
]

@named pendulum_linear = ODESystem(linear_eqs, t)
pendulum_linear_sys = structural_simplify(pendulum_linear)

println("Nonlinear equation: θ'' = -(g/L)sin(θ) - (c/m)θ'")
println("Linear approximation: θ'' = -(g/L)θ - (c/m)θ'")
println()
println("Parameters:")
println("  g = 9.81 m/s²  (gravity)")
println("  L = 1.0 m      (length)")
println("  c = 0.5 kg/s   (damping)")
println("  m = 1.0 kg     (mass)")
println()

#=============================================================================
Section 2: Compare Small Angle Solutions
=============================================================================#

println("\nSection 2: Comparing Small Angle Oscillations")
println("-"^70)

# Initial conditions: small angle (10 degrees = π/18 rad)
θ0_small = π/18  # ~10 degrees
u0_small = [θ => θ0_small, ω => 0.0]
params = [g => 9.81, L => 1.0, c => 0.1, m => 1.0]  # Light damping

tspan = (0.0, 10.0)

# Solve both systems
prob_nl_small = ODEProblem(pendulum_nonlinear_sys, u0_small, tspan, params)
prob_l_small = ODEProblem(pendulum_linear_sys, u0_small, tspan, params)

sol_nl_small = solve(prob_nl_small, Tsit5(), saveat=0.01)
sol_l_small = solve(prob_l_small, Tsit5(), saveat=0.01)

println("Initial angle: $(rad2deg(θ0_small))° (small angle)")
println("✓ Both systems solved")
println()

# Extract data
θ_nl_small = [u[1] for u in sol_nl_small.u]
θ_l_small = [u[1] for u in sol_l_small.u]
t_vals = sol_nl_small.t

# Plot comparison
fig1 = Figure(size=(1200, 800))

ax1 = Axis(fig1[1, 1],
    xlabel="Time (s)",
    ylabel="Angle θ (rad)",
    title="Small Angle: Linear vs Nonlinear (θ₀ = 10°)",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax1, t_vals, θ_nl_small, color=:blue, linewidth=2, label="Nonlinear")
lines!(ax1, t_vals, θ_l_small, color=:red, linewidth=2, linestyle=:dash, label="Linear approximation")
axislegend(ax1, position=:rt)

# Error plot
ax2 = Axis(fig1[2, 1],
    xlabel="Time (s)",
    ylabel="|Error| (rad)",
    title="Approximation Error",
    xgridvisible=true,
    ygridvisible=true
)

error_small = abs.(θ_nl_small .- θ_l_small)
lines!(ax2, t_vals, error_small, color=:purple, linewidth=2)

save(joinpath(output_dir, "03_pendulum_small_angle.png"), fig1, px_per_unit=2)
println("✓ Saved small angle comparison to output/03_pendulum_small_angle.png")
println("  Maximum error: $(maximum(error_small)) rad ($(rad2deg(maximum(error_small)))°)")
println()

#=============================================================================
Section 3: Compare Large Angle Solutions
=============================================================================#

println("\nSection 3: Comparing Large Angle Oscillations")
println("-"^70)

# Initial conditions: large angle (120 degrees = 2π/3 rad)
θ0_large = 2π/3  # 120 degrees
u0_large = [θ => θ0_large, ω => 0.0]

# Solve both systems
prob_nl_large = ODEProblem(pendulum_nonlinear_sys, u0_large, tspan, params)
prob_l_large = ODEProblem(pendulum_linear_sys, u0_large, tspan, params)

sol_nl_large = solve(prob_nl_large, Tsit5(), saveat=0.01)
sol_l_large = solve(prob_l_large, Tsit5(), saveat=0.01)

println("Initial angle: $(rad2deg(θ0_large))° (large angle)")
println("✓ Both systems solved")
println()

# Extract data
θ_nl_large = [u[1] for u in sol_nl_large.u]
θ_l_large = [u[1] for u in sol_l_large.u]

# Plot comparison
fig2 = Figure(size=(1200, 800))

ax1 = Axis(fig2[1, 1],
    xlabel="Time (s)",
    ylabel="Angle θ (rad)",
    title="Large Angle: Linear vs Nonlinear (θ₀ = 120°)",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax1, t_vals, θ_nl_large, color=:blue, linewidth=2, label="Nonlinear (correct)")
lines!(ax1, t_vals, θ_l_large, color=:red, linewidth=2, linestyle=:dash, label="Linear (incorrect!)")
axislegend(ax1, position=:rt)

# Error plot
ax2 = Axis(fig2[2, 1],
    xlabel="Time (s)",
    ylabel="|Error| (rad)",
    title="Approximation Error (Note: Linear approximation breaks down!)",
    xgridvisible=true,
    ygridvisible=true
)

error_large = abs.(θ_nl_large .- θ_l_large)
lines!(ax2, t_vals, error_large, color=:purple, linewidth=2)

save(joinpath(output_dir, "03_pendulum_large_angle.png"), fig2, px_per_unit=2)
println("✓ Saved large angle comparison to output/03_pendulum_large_angle.png")
println("  Maximum error: $(maximum(error_large)) rad ($(rad2deg(maximum(error_large)))°)")
println("  Note: Linear approximation is very poor for large angles!")
println()

#=============================================================================
Section 4: Phase Portraits
=============================================================================#

println("\nSection 4: Phase Space Analysis")
println("-"^70)

# Create phase portraits for different initial conditions
fig3 = Figure(size=(1400, 600))

# Nonlinear phase portrait
ax_nl = Axis(fig3[1, 1],
    xlabel="Angle θ (rad)",
    ylabel="Angular velocity ω (rad/s)",
    title="Nonlinear Pendulum Phase Portrait",
    xgridvisible=true,
    ygridvisible=true
)

# Linear phase portrait
ax_l = Axis(fig3[1, 2],
    xlabel="Angle θ (rad)",
    ylabel="Angular velocity ω (rad/s)",
    title="Linear Pendulum Phase Portrait",
    xgridvisible=true,
    ygridvisible=true
)

# Solve for multiple initial angles
initial_angles = [π/6, π/3, π/2, 2π/3, 5π/6]
colors = [:blue, :green, :orange, :red, :purple]

for (θ0, color) in zip(initial_angles, colors)
    u0_temp = [θ => θ0, ω => 0.0]
    
    # Nonlinear
    prob_nl = ODEProblem(pendulum_nonlinear_sys, u0_temp, (0.0, 20.0), params)
    sol_nl = solve(prob_nl, Tsit5(), saveat=0.01)
    θ_nl = [u[1] for u in sol_nl.u]
    ω_nl = [u[2] for u in sol_nl.u]
    lines!(ax_nl, θ_nl, ω_nl, color=color, linewidth=2, label="$(rad2deg(θ0))°")
    
    # Linear
    prob_l = ODEProblem(pendulum_linear_sys, u0_temp, (0.0, 20.0), params)
    sol_l = solve(prob_l, Tsit5(), saveat=0.01)
    θ_l = [u[1] for u in sol_l.u]
    ω_l = [u[2] for u in sol_l.u]
    lines!(ax_l, θ_l, ω_l, color=color, linewidth=2, label="$(rad2deg(θ0))°")
end

# Mark equilibrium point
scatter!(ax_nl, [0], [0], color=:black, markersize=15, marker=:circle)
scatter!(ax_l, [0], [0], color=:black, markersize=15, marker=:circle)

axislegend(ax_nl, position=:lt)
axislegend(ax_l, position=:lt)

save(joinpath(output_dir, "03_pendulum_phase_portraits.png"), fig3, px_per_unit=2)
println("✓ Saved phase portraits to output/03_pendulum_phase_portraits.png")
println()

#=============================================================================
Section 5: Energy Analysis
=============================================================================#

println("\nSection 5: Energy Conservation Analysis")
println("-"^70)

# For undamped pendulum, total energy should be conserved
# E = (1/2)mL²ω² + mgL(1 - cos(θ))  (kinetic + potential)

params_undamped = [g => 9.81, L => 1.0, c => 0.0, m => 1.0]  # No damping
u0_energy = [θ => π/2, ω => 0.0]
tspan_energy = (0.0, 10.0)

prob_undamped = ODEProblem(pendulum_nonlinear_sys, u0_energy, tspan_energy, params_undamped)
sol_undamped = solve(prob_undamped, Tsit5(), saveat=0.01)

# Calculate energies
g_val, L_val, m_val = 9.81, 1.0, 1.0
θ_energy = [u[1] for u in sol_undamped.u]
ω_energy = [u[2] for u in sol_undamped.u]

kinetic_energy = 0.5 * m_val * L_val^2 * ω_energy.^2
potential_energy = m_val * g_val * L_val * (1 .- cos.(θ_energy))
total_energy = kinetic_energy .+ potential_energy

println("Energy analysis for undamped pendulum:")
println("  Initial total energy: $(total_energy[1]) J")
println("  Final total energy: $(total_energy[end]) J")
println("  Energy variation: $(maximum(total_energy) - minimum(total_energy)) J")
println()

# Plot energy
fig4 = Figure(size=(1200, 800))

ax_e = Axis(fig4[1, 1],
    xlabel="Time (s)",
    ylabel="Energy (J)",
    title="Energy Conservation in Undamped Pendulum",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax_e, sol_undamped.t, kinetic_energy, color=:blue, linewidth=2, label="Kinetic")
lines!(ax_e, sol_undamped.t, potential_energy, color=:red, linewidth=2, label="Potential")
lines!(ax_e, sol_undamped.t, total_energy, color=:black, linewidth=2, label="Total", linestyle=:dash)
axislegend(ax_e, position=:rt)

# Compare with damped case
prob_damped = ODEProblem(pendulum_nonlinear_sys, u0_energy, tspan_energy, params)
sol_damped = solve(prob_damped, Tsit5(), saveat=0.01)

θ_damped = [u[1] for u in sol_damped.u]
ω_damped = [u[2] for u in sol_damped.u]
total_energy_damped = 0.5 * m_val * L_val^2 * ω_damped.^2 + 
                      m_val * g_val * L_val * (1 .- cos.(θ_damped))

ax_e2 = Axis(fig4[2, 1],
    xlabel="Time (s)",
    ylabel="Total Energy (J)",
    title="Energy Dissipation with Damping",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax_e2, sol_undamped.t, total_energy, color=:blue, linewidth=2, label="Undamped")
lines!(ax_e2, sol_damped.t, total_energy_damped, color=:red, linewidth=2, label="Damped (c=0.1)")
axislegend(ax_e2, position=:rt)

save(joinpath(output_dir, "03_pendulum_energy.png"), fig4, px_per_unit=2)
println("✓ Saved energy analysis to output/03_pendulum_energy.png")
println()

#=============================================================================
Section 6: Period vs. Amplitude
=============================================================================#

println("\nSection 6: Nonlinear Effect on Period")
println("-"^70)

# The period of a nonlinear pendulum increases with amplitude
# For small angles: T ≈ 2π√(L/g)
# For large angles: T increases

function compute_period(sol)
    """Estimate period by finding zero crossings"""
    θ_vals = [u[1] for u in sol.u]
    t_vals = sol.t
    
    # Find zero crossings (from positive to negative)
    crossings = Float64[]
    for i in 1:(length(θ_vals)-1)
        if θ_vals[i] > 0 && θ_vals[i+1] <= 0
            push!(crossings, t_vals[i])
        end
    end
    
    if length(crossings) >= 2
        return 2 * (crossings[2] - crossings[1])  # Full period
    else
        return NaN
    end
end

amplitudes = range(0.1, π*0.9, length=15)
periods_nl = Float64[]
periods_l = Float64[]

println("Computing periods for different amplitudes...")
for amp in amplitudes
    u0_temp = [θ => amp, ω => 0.0]
    tspan_temp = (0.0, 30.0)
    
    # Nonlinear
    prob_nl = ODEProblem(pendulum_nonlinear_sys, u0_temp, tspan_temp, params_undamped)
    sol_nl = solve(prob_nl, Tsit5(), saveat=0.01)
    push!(periods_nl, compute_period(sol_nl))
    
    # Linear
    prob_l = ODEProblem(pendulum_linear_sys, u0_temp, tspan_temp, params_undamped)
    sol_l = solve(prob_l, Tsit5(), saveat=0.01)
    push!(periods_l, compute_period(sol_l))
end

# Theoretical small-angle period
T_small_angle = 2π * sqrt(L_val / g_val)

fig5 = Figure(size=(1000, 700))

ax = Axis(fig5[1, 1],
    xlabel="Initial Amplitude (degrees)",
    ylabel="Period (s)",
    title="Period vs Amplitude: Nonlinear Effect",
    xgridvisible=true,
    ygridvisible=true
)

scatter!(ax, rad2deg.(amplitudes), periods_nl, color=:blue, markersize=10, label="Nonlinear")
scatter!(ax, rad2deg.(amplitudes), periods_l, color=:red, markersize=10, label="Linear")
hlines!(ax, [T_small_angle], color=:black, linestyle=:dash, linewidth=2, label="Small-angle theory")
axislegend(ax, position=:lt)

save(joinpath(output_dir, "03_pendulum_period_vs_amplitude.png"), fig5, px_per_unit=2)
println("✓ Saved period analysis to output/03_pendulum_period_vs_amplitude.png")
println("  Theoretical small-angle period: $(T_small_angle) s")
println()

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("✅ Demo 03 Complete!")
println("="^70)
println()
println("Summary of what you learned:")
println("• Deriving equations of motion for a pendulum")
println("• Comparing linear approximations with exact nonlinear solutions")
println("• Creating phase portraits to visualize dynamics")
println("• Analyzing energy conservation and dissipation")
println("• Understanding how nonlinearity affects the period")
println()
println("Key insights:")
println("• Small-angle approximation (sin(θ) ≈ θ) is good for θ < ~15°")
println("• For large angles, the linear approximation fails dramatically")
println("• Phase portraits reveal qualitative differences between linear and nonlinear systems")
println("• The period increases with amplitude in the nonlinear case")
println("• Energy is conserved without damping, dissipated with damping")
println()
println("Next: Try 04_heat_equation_pde.jl to explore partial differential equations!")
println()
