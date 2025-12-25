"""
# 07: Mechanical Systems - Multi-Body Dynamics

Multi-body dynamics describes the motion of systems with multiple connected rigid bodies.
This is essential for robotics, vehicle dynamics, biomechanics, and more.

This demo explores:
1. Double pendulum - classic chaotic system
2. Spring-mass-damper systems
3. Energy conservation and dissipation
4. Lagrangian mechanics formulation
5. Animated mechanical motion

## Prerequisites:
- Classical mechanics
- Understanding of differential equations

## Duration: ~35 minutes
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
println("⚙️  Demo 07: Mechanical Systems - Multi-Body Dynamics")
println("="^70)
println()

#=============================================================================
Section 1: Double Pendulum - Chaos in Mechanics
=============================================================================#

println("Section 1: Double Pendulum - Deterministic Chaos")
println("-"^70)

# Double pendulum equations in Cartesian coordinates are complex
# We'll use angular coordinates (θ₁, θ₂)

# Physical parameters
@parameters g=9.81 L1=1.0 L2=1.0 m1=1.0 m2=1.0

# State variables: angles and angular velocities
@variables θ1(t)=π/2 θ2(t)=π/2 ω1(t)=0.0 ω2(t)=0.0

# Lagrangian formulation leads to coupled nonlinear equations
# These are the equations of motion for a double pendulum

# Helpers for cleaner equations
Δθ = θ1 - θ2
μ = 1 + m1/m2

# Angular accelerations (derived from Lagrangian mechanics)
α1_num = -g*(2*m1+m2)*sin(θ1) - m2*g*sin(θ1-2*θ2) - 2*sin(Δθ)*m2*(ω2^2*L2 + ω1^2*L1*cos(Δθ))
α1_den = L1*(2*m1 + m2 - m2*cos(2*Δθ))
α1 = α1_num / α1_den

α2_num = 2*sin(Δθ)*(ω1^2*L1*(m1+m2) + g*(m1+m2)*cos(θ1) + ω2^2*L2*m2*cos(Δθ))
α2_den = L2*(2*m1 + m2 - m2*cos(2*Δθ))
α2 = α2_num / α2_den

# First-order system
eqs_dp = [
    D(θ1) ~ ω1,
    D(θ2) ~ ω2,
    D(ω1) ~ α1,
    D(ω2) ~ α2
]

@named double_pendulum = ODESystem(eqs_dp, t)
dp_sys = structural_simplify(double_pendulum)

println("Double pendulum equations derived from Lagrangian mechanics")
println("Parameters:")
println("  g = 9.81 m/s²")
println("  L1 = L2 = 1.0 m")
println("  m1 = m2 = 1.0 kg")
println()

# Initial conditions
u0_dp = [θ1 => π/2, θ2 => π/2, ω1 => 0.0, ω2 => 0.0]
params_dp = [g => 9.81, L1 => 1.0, L2 => 1.0, m1 => 1.0, m2 => 1.0]

# Solve
tspan = (0.0, 20.0)
prob_dp = ODEProblem(dp_sys, u0_dp, tspan, params_dp)
sol_dp = solve(prob_dp, Tsit5(), saveat=0.02)

println("✓ Double pendulum solved")
println("  Time points: $(length(sol_dp.t))")
println()

# Extract solution
θ1_vals = [u[1] for u in sol_dp.u]
θ2_vals = [u[2] for u in sol_dp.u]
ω1_vals = [u[3] for u in sol_dp.u]
ω2_vals = [u[4] for u in sol_dp.u]

# Convert to Cartesian coordinates for visualization
L1_val, L2_val = 1.0, 1.0
x1 = L1_val * sin.(θ1_vals)
y1 = -L1_val * cos.(θ1_vals)
x2 = x1 + L2_val * sin.(θ2_vals)
y2 = y1 - L2_val * cos.(θ2_vals)

# Plot trajectories of both masses
fig1 = Figure(size=(1200, 600))

ax1 = Axis(fig1[1, 1],
    xlabel="x (m)",
    ylabel="y (m)",
    title="Trajectory of First Mass",
    aspect=DataAspect(),
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax1, x1, y1, color=sol_dp.t, colormap=:viridis, linewidth=1.5)
scatter!(ax1, [0], [0], color=:black, markersize=15, label="Pivot")

ax2 = Axis(fig1[1, 2],
    xlabel="x (m)",
    ylabel="y (m)",
    title="Trajectory of Second Mass (Chaotic!)",
    aspect=DataAspect(),
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax2, x2, y2, color=sol_dp.t, colormap=:plasma, linewidth=1.5)
scatter!(ax2, [0], [0], color=:black, markersize=15)

save(joinpath(output_dir, "07_double_pendulum_trajectories.png"), fig1, px_per_unit=2)
println("✓ Saved trajectories to output/07_double_pendulum_trajectories.png")
println()

# Energy analysis
function compute_dp_energy(θ1, θ2, ω1, ω2, g, L1, L2, m1, m2)
    # Kinetic energy
    T1 = 0.5 * m1 * (L1 * ω1)^2
    T2 = 0.5 * m2 * ((L1 * ω1)^2 + (L2 * ω2)^2 + 2*L1*L2*ω1*ω2*cos(θ1 - θ2))
    T = T1 + T2
    
    # Potential energy (taking y=0 at pivot)
    V1 = -m1 * g * L1 * cos(θ1)
    V2 = -m2 * g * (L1 * cos(θ1) + L2 * cos(θ2))
    V = V1 + V2
    
    return T, V, T + V
end

g_val, L1_val, L2_val, m1_val, m2_val = 9.81, 1.0, 1.0, 1.0, 1.0
energies_dp = [compute_dp_energy(θ1_vals[i], θ2_vals[i], ω1_vals[i], ω2_vals[i],
                                  g_val, L1_val, L2_val, m1_val, m2_val)
               for i in 1:length(sol_dp.t)]

T_dp = [e[1] for e in energies_dp]
V_dp = [e[2] for e in energies_dp]
E_dp = [e[3] for e in energies_dp]

fig2 = Figure(size=(1200, 600))

ax = Axis(fig2[1, 1],
    xlabel="Time (s)",
    ylabel="Energy (J)",
    title="Energy Conservation in Double Pendulum",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax, sol_dp.t, T_dp, color=:blue, linewidth=2, label="Kinetic")
lines!(ax, sol_dp.t, V_dp, color=:red, linewidth=2, label="Potential")
lines!(ax, sol_dp.t, E_dp, color=:black, linewidth=2, linestyle=:dash, label="Total")
axislegend(ax, position=:rt)

save(joinpath(output_dir, "07_double_pendulum_energy.png"), fig2, px_per_unit=2)
println("✓ Saved energy plot to output/07_double_pendulum_energy.png")
println("  Energy variation: $(maximum(E_dp) - minimum(E_dp)) J")
println()

#=============================================================================
Section 2: Spring-Mass-Damper System
=============================================================================#

println("\nSection 2: Spring-Mass-Damper System")
println("-"^70)

# Classic mechanical system: mx'' + cx' + kx = F(t)
# We'll simulate a mass-spring-damper with an external forcing

@parameters m=1.0 c=0.2 k=5.0 F0=1.0 ω_drive=2.0
@variables x(t)=0.0 v(t)=0.0

# Equations
# x' = v
# v' = (F(t) - c*v - k*x) / m
# F(t) = F0 * sin(ω_drive * t)

eqs_smd = [
    D(x) ~ v,
    D(v) ~ (F0 * sin(ω_drive * t) - c * v - k * x) / m
]

@named spring_mass = ODESystem(eqs_smd, t)
smd_sys = structural_simplify(spring_mass)

println("Spring-Mass-Damper System:")
println("  Mass: m = 1.0 kg")
println("  Damping: c = 0.2 kg/s")
println("  Spring constant: k = 5.0 N/m")
println("  Driving force: F(t) = 1.0 sin(2t) N")
println()

u0_smd = [x => 0.0, v => 0.0]
params_smd = [m => 1.0, c => 0.2, k => 5.0, F0 => 1.0, ω_drive => 2.0]

tspan_smd = (0.0, 30.0)
prob_smd = ODEProblem(smd_sys, u0_smd, tspan_smd, params_smd)
sol_smd = solve(prob_smd, Tsit5(), saveat=0.05)

println("✓ Spring-mass-damper solved")
println()

x_smd = [u[1] for u in sol_smd.u]
v_smd = [u[2] for u in sol_smd.u]

# Visualize
fig3 = Figure(size=(1200, 800))

ax1 = Axis(fig3[1, 1],
    xlabel="Time (s)",
    ylabel="Position (m)",
    title="Forced Oscillation with Damping",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax1, sol_smd.t, x_smd, color=:blue, linewidth=2)

# Phase space
ax2 = Axis(fig3[2, 1],
    xlabel="Position (m)",
    ylabel="Velocity (m/s)",
    title="Phase Space - Approach to Limit Cycle",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax2, x_smd, v_smd, color=sol_smd.t, colormap=:viridis, linewidth=2)

save(joinpath(output_dir, "07_spring_mass_damper.png"), fig3, px_per_unit=2)
println("✓ Saved spring-mass-damper to output/07_spring_mass_damper.png")
println()

#=============================================================================
Section 3: Coupled Oscillators
=============================================================================#

println("\nSection 3: Coupled Spring-Mass System")
println("-"^70)

# Two masses connected by springs
# m1*x1'' = -k1*x1 - k2*(x1 - x2)
# m2*x2'' = -k2*(x2 - x1) - k3*x2

@parameters m1_c=1.0 m2_c=1.0 k1=2.0 k2=1.0 k3=2.0
@variables x1(t)=1.0 x2(t)=0.5 v1(t)=0.0 v2(t)=0.0

eqs_coupled = [
    D(x1) ~ v1,
    D(x2) ~ v2,
    D(v1) ~ (-k1*x1 - k2*(x1 - x2)) / m1_c,
    D(v2) ~ (-k2*(x2 - x1) - k3*x2) / m2_c
]

@named coupled_osc = ODESystem(eqs_coupled, t)
coupled_sys = structural_simplify(coupled_osc)

println("Coupled oscillators:")
println("  Two masses connected by springs")
println("  k1 = k3 = 2.0 N/m (boundary springs)")
println("  k2 = 1.0 N/m (coupling spring)")
println()

u0_coupled = [x1 => 1.0, x2 => 0.5, v1 => 0.0, v2 => 0.0]
params_coupled = [m1_c => 1.0, m2_c => 1.0, k1 => 2.0, k2 => 1.0, k3 => 2.0]

tspan_coupled = (0.0, 20.0)
prob_coupled = ODEProblem(coupled_sys, u0_coupled, tspan_coupled, params_coupled)
sol_coupled = solve(prob_coupled, Tsit5(), saveat=0.05)

println("✓ Coupled oscillators solved")
println()

x1_coupled = [u[1] for u in sol_coupled.u]
x2_coupled = [u[2] for u in sol_coupled.u]

# Visualize
fig4 = Figure(size=(1200, 800))

ax1 = Axis(fig4[1, 1],
    xlabel="Time (s)",
    ylabel="Position (m)",
    title="Coupled Oscillators - Energy Transfer",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax1, sol_coupled.t, x1_coupled, color=:blue, linewidth=2, label="Mass 1")
lines!(ax1, sol_coupled.t, x2_coupled, color=:red, linewidth=2, label="Mass 2")
axislegend(ax1, position=:rt)

# Phase space for both masses
ax2 = Axis(fig4[2, 1],
    xlabel="x1 (m)",
    ylabel="x2 (m)",
    title="Configuration Space",
    aspect=DataAspect(),
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax2, x1_coupled, x2_coupled, color=sol_coupled.t, colormap=:plasma, linewidth=2)

save(joinpath(output_dir, "07_coupled_oscillators.png"), fig4, px_per_unit=2)
println("✓ Saved coupled oscillators to output/07_coupled_oscillators.png")
println()

#=============================================================================
Section 4: Normal Modes
=============================================================================#

println("\nSection 4: Normal Modes Analysis")
println("-"^70)

# For the coupled system, we can find normal modes
# These are special oscillation patterns where both masses move in sync

println("Normal modes:")
println("  Mode 1: Both masses oscillate in phase")
println("  Mode 2: Masses oscillate out of phase")
println()

# Simulate with initial conditions corresponding to each mode
# Mode 1: x1 = x2 (in-phase)
u0_mode1 = [x1 => 1.0, x2 => 1.0, v1 => 0.0, v2 => 0.0]
prob_mode1 = ODEProblem(coupled_sys, u0_mode1, tspan_coupled, params_coupled)
sol_mode1 = solve(prob_mode1, Tsit5(), saveat=0.05)

# Mode 2: x1 = -x2 (out-of-phase, approximately)
u0_mode2 = [x1 => 1.0, x2 => -1.0, v1 => 0.0, v2 => 0.0]
prob_mode2 = ODEProblem(coupled_sys, u0_mode2, tspan_coupled, params_coupled)
sol_mode2 = solve(prob_mode2, Tsit5(), saveat=0.05)

x1_mode1 = [u[1] for u in sol_mode1.u]
x2_mode1 = [u[2] for u in sol_mode1.u]
x1_mode2 = [u[1] for u in sol_mode2.u]
x2_mode2 = [u[2] for u in sol_mode2.u]

fig5 = Figure(size=(1200, 800))

ax1 = Axis(fig5[1, 1],
    xlabel="Time (s)",
    ylabel="Position (m)",
    title="Mode 1: In-Phase Oscillation",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax1, sol_mode1.t, x1_mode1, color=:blue, linewidth=2, label="Mass 1")
lines!(ax1, sol_mode1.t, x2_mode1, color=:red, linewidth=2, label="Mass 2")
axislegend(ax1)

ax2 = Axis(fig5[2, 1],
    xlabel="Time (s)",
    ylabel="Position (m)",
    title="Mode 2: Out-of-Phase Oscillation",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax2, sol_mode2.t, x1_mode2, color=:blue, linewidth=2, label="Mass 1")
lines!(ax2, sol_mode2.t, x2_mode2, color=:red, linewidth=2, label="Mass 2")
axislegend(ax2)

save(joinpath(output_dir, "07_normal_modes.png"), fig5, px_per_unit=2)
println("✓ Saved normal modes to output/07_normal_modes.png")
println()

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("✅ Demo 07 Complete!")
println("="^70)
println()
println("Summary of what you learned:")
println("• Double pendulum dynamics and chaos")
println("• Spring-mass-damper systems")
println("• Coupled oscillators and energy transfer")
println("• Normal modes of oscillation")
println("• Energy conservation in mechanical systems")
println()
println("Key insights:")
println("• Double pendulum exhibits deterministic chaos")
println("• Damping causes energy dissipation")
println("• Coupled systems can transfer energy between components")
println("• Normal modes are special oscillation patterns")
println("• Lagrangian mechanics provides elegant formulation")
println()
println("Next: Try 08_optimization_problems.jl to explore parameter estimation!")
println()
