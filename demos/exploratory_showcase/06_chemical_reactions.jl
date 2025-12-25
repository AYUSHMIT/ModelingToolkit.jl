"""
# 06: Chemical Reactions - Reaction Networks with Catalyst.jl

Chemical reaction networks describe how species interact and transform. Catalyst.jl
provides a domain-specific language for defining reaction networks, which can then
be simulated deterministically or stochastically.

This demo explores:
1. Defining reaction networks with Catalyst.jl
2. Classic models (Brusselator, Lotka-Volterra)
3. Oscillating reactions
4. Stochastic vs. deterministic simulations
5. Phase portraits of chemical kinetics

## Examples:
- Brusselator: Oscillating chemical reaction
- Lotka-Volterra: Predator-prey dynamics
- Michaelis-Menten: Enzyme kinetics

## Prerequisites:
- Basic chemical kinetics
- Understanding of ODEs

## Duration: ~25 minutes
"""

using ModelingToolkit
using Catalyst
using DifferentialEquations
using CairoMakie
using LinearAlgebra
using ModelingToolkit: t_nounits as t

# Set up output directory
output_dir = "output"
mkpath(output_dir)

println("="^70)
println("🧪 Demo 06: Chemical Reactions - Modeling Reaction Networks")
println("="^70)
println()

#=============================================================================
Section 1: Brusselator - Oscillating Chemical Reaction
=============================================================================#

println("Section 1: Brusselator - Autocatalytic Oscillations")
println("-"^70)

# The Brusselator is a theoretical model showing oscillating chemical reactions
# Reactions:
# A → X
# 2X + Y → 3X
# B + X → Y + D
# X → E

println("Brusselator reactions:")
println("  A → X")
println("  2X + Y → 3X   (autocatalytic)")
println("  B + X → Y + D")
println("  X → E")
println()

# Define the reaction network using Catalyst
brusselator = @reaction_network begin
    A, ∅ --> X         # Input of X from external source A
    1, 2X + Y --> 3X   # Autocatalytic step
    B, X --> Y         # Conversion of X to Y with B
    1, X --> ∅         # Removal of X
end

println("✓ Brusselator network defined")

# Convert to ODE system
@named brusselator_sys = convert(ODESystem, brusselator)
brusselator_ode = structural_simplify(brusselator_sys)

# Set parameters (chosen to produce oscillations)
A_val = 1.0
B_val = 3.0

# Initial conditions
u0_bruss = [brusselator_sys.X => 1.0,
            brusselator_sys.Y => 1.0]

p_bruss = [brusselator_sys.A => A_val,
           brusselator_sys.B => B_val]

# Solve
tspan = (0.0, 50.0)
prob_bruss = ODEProblem(brusselator_ode, u0_bruss, tspan, p_bruss)
sol_bruss = solve(prob_bruss, Tsit5(), saveat=0.1)

println("✓ Brusselator solved")
println("  Time points: $(length(sol_bruss.t))")
println()

# Extract concentrations
X_vals = [u[1] for u in sol_bruss.u]
Y_vals = [u[2] for u in sol_bruss.u]

# Plot time series
fig1 = Figure(size=(1200, 800))

ax1 = Axis(fig1[1, 1],
    xlabel="Time",
    ylabel="Concentration",
    title="Brusselator Time Series (A=$A_val, B=$B_val)",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax1, sol_bruss.t, X_vals, color=:blue, linewidth=2, label="[X]")
lines!(ax1, sol_bruss.t, Y_vals, color=:red, linewidth=2, label="[Y]")
axislegend(ax1, position=:rt)

# Phase portrait
ax2 = Axis(fig1[2, 1],
    xlabel="[X]",
    ylabel="[Y]",
    title="Phase Portrait - Limit Cycle",
    aspect=DataAspect(),
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax2, X_vals, Y_vals, color=sol_bruss.t, colormap=:viridis, linewidth=2)
scatter!(ax2, [X_vals[1]], [Y_vals[1]], color=:green, markersize=15, label="Start")
scatter!(ax2, [X_vals[end]], [Y_vals[end]], color=:red, markersize=15, label="End")

save(joinpath(output_dir, "06_brusselator.png"), fig1, px_per_unit=2)
println("✓ Saved Brusselator visualization to output/06_brusselator.png")
println()

#=============================================================================
Section 2: Lotka-Volterra - Predator-Prey Dynamics
=============================================================================#

println("\nSection 2: Lotka-Volterra - Predator-Prey Model")
println("-"^70)

# Classic predator-prey model
# Reactions representing:
# - Prey reproduction
# - Predator eating prey
# - Predator death

println("Lotka-Volterra reactions:")
println("  Prey → 2 Prey      (prey reproduction)")
println("  Prey + Predator → 2 Predator  (predation)")
println("  Predator → ∅       (predator death)")
println()

lotka_volterra = @reaction_network begin
    α, Prey --> 2Prey              # Prey reproduction
    β, Prey + Predator --> 2Predator  # Predation
    γ, Predator --> ∅              # Predator death
end

@named lv_sys = convert(ODESystem, lotka_volterra)
lv_ode = structural_simplify(lv_sys)

# Parameters (classic values)
u0_lv = [lv_sys.Prey => 10.0,
         lv_sys.Predator => 5.0]

p_lv = [lv_sys.α => 1.5,   # Prey growth rate
        lv_sys.β => 1.0,   # Predation rate
        lv_sys.γ => 3.0]   # Predator death rate

# Solve
tspan_lv = (0.0, 15.0)
prob_lv = ODEProblem(lv_ode, u0_lv, tspan_lv, p_lv)
sol_lv = solve(prob_lv, Tsit5(), saveat=0.05)

println("✓ Lotka-Volterra solved")
println()

# Extract populations
prey_vals = [u[1] for u in sol_lv.u]
pred_vals = [u[2] for u in sol_lv.u]

# Visualize
fig2 = Figure(size=(1200, 800))

ax1 = Axis(fig2[1, 1],
    xlabel="Time",
    ylabel="Population",
    title="Predator-Prey Dynamics",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax1, sol_lv.t, prey_vals, color=:green, linewidth=2, label="Prey")
lines!(ax1, sol_lv.t, pred_vals, color=:red, linewidth=2, label="Predator")
axislegend(ax1, position=:rt)

# Phase space
ax2 = Axis(fig2[2, 1],
    xlabel="Prey Population",
    ylabel="Predator Population",
    title="Phase Space - Periodic Cycles",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax2, prey_vals, pred_vals, color=sol_lv.t, colormap=:plasma, linewidth=2)
scatter!(ax2, [prey_vals[1]], [pred_vals[1]], color=:green, markersize=15)

save(joinpath(output_dir, "06_lotka_volterra.png"), fig2, px_per_unit=2)
println("✓ Saved Lotka-Volterra visualization to output/06_lotka_volterra.png")
println()

#=============================================================================
Section 3: Michaelis-Menten Enzyme Kinetics
=============================================================================#

println("\nSection 3: Michaelis-Menten Enzyme Kinetics")
println("-"^70)

# Enzyme-substrate reaction
# E + S ⇌ ES → E + P

println("Michaelis-Menten mechanism:")
println("  E + S ⇌ ES  (reversible binding)")
println("  ES → E + P  (catalysis)")
println()

michaelis_menten = @reaction_network begin
    k1, E + S --> ES     # Enzyme-substrate binding
    k_1, ES --> E + S    # Dissociation
    k2, ES --> E + P     # Product formation
end

@named mm_sys = convert(ODESystem, michaelis_menten)
mm_ode = structural_simplify(mm_sys)

# Initial conditions
u0_mm = [mm_sys.E => 1.0,    # Enzyme
         mm_sys.S => 10.0,   # Substrate
         mm_sys.ES => 0.0,   # Complex
         mm_sys.P => 0.0]    # Product

# Rate constants
p_mm = [mm_sys.k1 => 1.0,    # Association
        mm_sys.k_1 => 0.5,   # Dissociation
        mm_sys.k2 => 0.2]    # Catalysis

# Solve
tspan_mm = (0.0, 30.0)
prob_mm = ODEProblem(mm_ode, u0_mm, tspan_mm, p_mm)
sol_mm = solve(prob_mm, Tsit5(), saveat=0.1)

println("✓ Michaelis-Menten solved")
println()

# Extract concentrations
E_vals = [u[1] for u in sol_mm.u]
S_vals = [u[2] for u in sol_mm.u]
ES_vals = [u[3] for u in sol_mm.u]
P_vals = [u[4] for u in sol_mm.u]

# Visualize
fig3 = Figure(size=(1200, 800))

ax = Axis(fig3[1, 1],
    xlabel="Time",
    ylabel="Concentration",
    title="Enzyme Kinetics - Product Formation",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax, sol_mm.t, E_vals, color=:blue, linewidth=2, label="[E]")
lines!(ax, sol_mm.t, S_vals, color=:green, linewidth=2, label="[S]")
lines!(ax, sol_mm.t, ES_vals, color=:orange, linewidth=2, label="[ES]")
lines!(ax, sol_mm.t, P_vals, color=:red, linewidth=2, label="[P]")
axislegend(ax, position=:rt)

# Add second subplot showing conservation laws
ax2 = Axis(fig3[2, 1],
    xlabel="Time",
    ylabel="Total Concentration",
    title="Conservation: Total Enzyme = [E] + [ES]",
    xgridvisible=true,
    ygridvisible=true
)

total_enzyme = E_vals .+ ES_vals
lines!(ax2, sol_mm.t, total_enzyme, color=:purple, linewidth=2, label="[E] + [ES]")
hlines!(ax2, [1.0], color=:black, linestyle=:dash, linewidth=1, label="Expected")
axislegend(ax2)

save(joinpath(output_dir, "06_michaelis_menten.png"), fig3, px_per_unit=2)
println("✓ Saved Michaelis-Menten visualization to output/06_michaelis_menten.png")
println()

#=============================================================================
Section 4: Stochastic Simulation (if possible)
=============================================================================#

println("\nSection 4: Comparing Deterministic and Stochastic Simulations")
println("-"^70)

# For small molecule numbers, stochastic effects matter
# Use a simple birth-death process as example

birth_death = @reaction_network begin
    k_birth, ∅ --> A
    k_death, A --> ∅
end

@named bd_sys = convert(ODESystem, birth_death)
bd_ode = structural_simplify(bd_sys)

# Deterministic solution
u0_bd = [bd_sys.A => 50.0]
p_bd = [bd_sys.k_birth => 10.0,
        bd_sys.k_death => 0.2]

tspan_bd = (0.0, 10.0)
prob_bd_ode = ODEProblem(bd_ode, u0_bd, tspan_bd, p_bd)
sol_bd_ode = solve(prob_bd_ode, Tsit5(), saveat=0.1)

println("✓ Birth-death process simulated (deterministic)")

# Plot
fig4 = Figure(size=(1200, 600))

ax = Axis(fig4[1, 1],
    xlabel="Time",
    ylabel="Population",
    title="Birth-Death Process (Deterministic)",
    xgridvisible=true,
    ygridvisible=true
)

A_det = [u[1] for u in sol_bd_ode.u]
lines!(ax, sol_bd_ode.t, A_det, color=:blue, linewidth=2)

# Show equilibrium
equilibrium = 10.0 / 0.2  # k_birth / k_death = 50
hlines!(ax, [equilibrium], color=:red, linestyle=:dash, linewidth=2, label="Equilibrium")
axislegend(ax)

save(joinpath(output_dir, "06_birth_death.png"), fig4, px_per_unit=2)
println("✓ Saved birth-death visualization to output/06_birth_death.png")
println()

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("✅ Demo 06 Complete!")
println("="^70)
println()
println("Summary of what you learned:")
println("• Defining reaction networks with Catalyst.jl")
println("• Simulating oscillating reactions (Brusselator)")
println("• Modeling predator-prey dynamics (Lotka-Volterra)")
println("• Enzyme kinetics (Michaelis-Menten)")
println("• Understanding limit cycles in chemical systems")
println()
println("Key insights:")
println("• Chemical reactions can produce complex dynamics")
println("• Autocatalytic reactions can oscillate")
println("• Predator-prey systems show periodic cycles")
println("• Conservation laws constrain system behavior")
println("• Phase portraits reveal system structure")
println()
println("Next: Try 07_mechanical_systems.jl to explore multi-body dynamics!")
println()
