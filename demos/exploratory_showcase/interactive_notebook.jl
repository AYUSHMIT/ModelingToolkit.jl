### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ Cell order:
# ╟─intro
# ╟─setup
# ╠═imports
# ╟─controls
# ╠═lorenz_params
# ╠═simulate
# ╟─visualization
# ╠═plot_3d
# ╠═plot_timeseries
# ╟─comparison
# ╠═compare_ics

# ╔═╡ intro
md"""
# 🎮 Interactive ModelingToolkit Explorer

Welcome to the interactive exploration of dynamical systems with ModelingToolkit.jl!

This notebook allows you to:
- 🎚️ **Adjust parameters** with sliders in real-time
- 📊 **Visualize dynamics** with interactive plots
- 🔄 **Compare systems** with different configurations
- 🎓 **Learn** through experimentation

## Featured System: Lorenz Attractor

The Lorenz system is one of the most famous chaotic systems, discovered by Edward Lorenz
in 1963 while studying atmospheric convection. It demonstrates how simple nonlinear
equations can produce incredibly complex, unpredictable behavior.

### The Equations:
```math
\begin{aligned}
\frac{dx}{dt} &= \sigma(y - x) \\
\frac{dy}{dt} &= x(\rho - z) - y \\
\frac{dz}{dt} &= xy - \beta z
\end{aligned}
```

### Parameters:
- **σ (sigma)**: Prandtl number - ratio of momentum to thermal diffusivity
- **ρ (rho)**: Rayleigh number - ratio of buoyancy to viscous forces
- **β (beta)**: Geometric factor

Try changing the parameters below and see how the system behaves!
"""

# ╔═╡ setup
begin
    # This cell sets up the environment
    using Pkg
    # Pkg.activate(".") # Uncomment if you want to use the demo environment
end

# ╔═╡ imports
begin
    using PlutoUI
    using ModelingToolkit
    using DifferentialEquations
    using CairoMakie
    using LinearAlgebra
    using ModelingToolkit: t_nounits as t, D_nounits as D
    
    # Custom plotting theme
    set_theme!(
        backgroundcolor=:white,
        fontsize=14,
        linewidth=2,
        Axis=(xgridvisible=true, ygridvisible=true)
    )
end

# ╔═╡ controls
md"""
## 🎛️ Control Panel

Adjust the parameters and initial conditions below:

**Parameters:**

σ (Prandtl number): $(@bind σ_val Slider(5.0:0.5:20.0, default=10.0, show_value=true))

ρ (Rayleigh number): $(@bind ρ_val Slider(10.0:1.0:50.0, default=28.0, show_value=true))

β (Geometric factor): $(@bind β_val Slider(1.0:0.1:5.0, default=8/3, show_value=true))

**Initial Conditions:**

x₀: $(@bind x0_val Slider(-5.0:0.5:5.0, default=1.0, show_value=true))

y₀: $(@bind y0_val Slider(-5.0:0.5:5.0, default=0.0, show_value=true))

z₀: $(@bind z0_val Slider(-5.0:0.5:5.0, default=0.0, show_value=true))

**Simulation Time:**

Time span: $(@bind tmax_val Slider(10.0:5.0:100.0, default=50.0, show_value=true)) seconds
"""

# ╔═╡ lorenz_params
begin
    # Define the Lorenz system
    @parameters σ=σ_val ρ=ρ_val β=β_val
    @variables x(t)=x0_val y(t)=y0_val z(t)=z0_val
    
    equations = [
        D(x) ~ σ * (y - x),
        D(y) ~ x * (ρ - z) - y,
        D(z) ~ x * y - β * z
    ]
    
    @named lorenz_sys = ODESystem(equations, t)
    lorenz_simplified = structural_simplify(lorenz_sys)
    
    md"""
    ✅ **Lorenz system configured**
    - σ = $σ_val
    - ρ = $ρ_val  
    - β = $(round(β_val, digits=3))
    - Initial: ($x0_val, $y0_val, $z0_val)
    """
end

# ╔═╡ simulate
begin
    # Set up and solve the problem
    u0 = [lorenz_sys.x => x0_val,
          lorenz_sys.y => y0_val,
          lorenz_sys.z => z0_val]
    
    params = [lorenz_sys.σ => σ_val,
              lorenz_sys.ρ => ρ_val,
              lorenz_sys.β => β_val]
    
    tspan = (0.0, tmax_val)
    
    prob = ODEProblem(lorenz_simplified, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=0.01)
    
    # Extract solution
    x_vals = [u[1] for u in sol.u]
    y_vals = [u[2] for u in sol.u]
    z_vals = [u[3] for u in sol.u]
    t_vals = sol.t
    
    md"""
    ✅ **Simulation complete**
    - Time points: $(length(sol.t))
    - Duration: $(round(tmax_val, digits=1)) seconds
    """
end

# ╔═╡ visualization
md"""
## 📊 Visualizations

### 3D Trajectory
The famous "butterfly" shape of the Lorenz attractor:
"""

# ╔═╡ plot_3d
begin
    fig_3d = Figure(size=(800, 600))
    
    ax3d = Axis3(fig_3d[1, 1],
        xlabel="x",
        ylabel="y",
        zlabel="z",
        title="Lorenz Attractor (σ=$σ_val, ρ=$ρ_val, β=$(round(β_val, digits=2)))",
        azimuth=π/6,
        elevation=π/9
    )
    
    # Plot trajectory with time-based coloring
    lines!(ax3d, x_vals, y_vals, z_vals,
        color=t_vals,
        colormap=:plasma,
        linewidth=2
    )
    
    # Mark start and end points
    scatter!(ax3d, [x_vals[1]], [y_vals[1]], [z_vals[1]],
        color=:green,
        markersize=15,
        label="Start"
    )
    
    scatter!(ax3d, [x_vals[end]], [y_vals[end]], [z_vals[end]],
        color=:red,
        markersize=15,
        label="End"
    )
    
    fig_3d
end

# ╔═╡ plot_timeseries
begin
    fig_ts = Figure(size=(800, 600))
    
    ax_x = Axis(fig_ts[1, 1],
        ylabel="x(t)",
        title="State Variables Over Time",
        xgridvisible=true,
        ygridvisible=true
    )
    
    ax_y = Axis(fig_ts[2, 1],
        ylabel="y(t)",
        xgridvisible=true,
        ygridvisible=true
    )
    
    ax_z = Axis(fig_ts[3, 1],
        xlabel="Time",
        ylabel="z(t)",
        xgridvisible=true,
        ygridvisible=true
    )
    
    lines!(ax_x, t_vals, x_vals, color=:blue, linewidth=2)
    lines!(ax_y, t_vals, y_vals, color=:red, linewidth=2)
    lines!(ax_z, t_vals, z_vals, color=:green, linewidth=2)
    
    fig_ts
end

# ╔═╡ comparison
md"""
## 🔄 Sensitivity to Initial Conditions

The "butterfly effect": tiny changes in initial conditions lead to vastly different trajectories.

Compare with slightly perturbed initial condition: $(@bind show_comparison CheckBox(default=false))
"""

# ╔═╡ compare_ics
begin
    if show_comparison
        # Solve with slightly perturbed initial condition
        ε = 1e-5
        u0_perturbed = [lorenz_sys.x => x0_val + ε,
                       lorenz_sys.y => y0_val,
                       lorenz_sys.z => z0_val]
        
        prob_pert = ODEProblem(lorenz_simplified, u0_perturbed, tspan, params)
        sol_pert = solve(prob_pert, Tsit5(), saveat=0.01)
        
        x_pert = [u[1] for u in sol_pert.u]
        y_pert = [u[2] for u in sol_pert.u]
        z_pert = [u[3] for u in sol_pert.u]
        
        # Create comparison plot
        fig_comp = Figure(size=(1000, 700))
        
        ax3d_comp = Axis3(fig_comp[1, 1],
            xlabel="x",
            ylabel="y",
            zlabel="z",
            title="Original vs Perturbed (ε = $ε)",
            azimuth=π/6,
            elevation=π/9
        )
        
        lines!(ax3d_comp, x_vals, y_vals, z_vals,
            color=:blue, linewidth=2, label="Original", alpha=0.7)
        
        lines!(ax3d_comp, x_pert, y_pert, z_pert,
            color=:red, linewidth=2, label="Perturbed", alpha=0.7)
        
        # Plot divergence
        ax_div = Axis(fig_comp[2, 1],
            xlabel="Time",
            ylabel="|Δx|",
            title="Growth of Perturbation (log scale)",
            yscale=log10,
            xgridvisible=true,
            ygridvisible=true
        )
        
        error = abs.(x_vals .- x_pert)
        lines!(ax_div, t_vals, error, color=:purple, linewidth=2)
        hlines!(ax_div, [ε], color=:black, linestyle=:dash, linewidth=1)
        
        fig_comp
    else
        md"*Check the box above to see comparison with perturbed initial condition*"
    end
end

# ╔═╡ epilogue
md"""
---

## 🎓 Learning Exercises

Try these experiments to deepen your understanding:

1. **Finding Equilibria**: Set ρ < 1 and observe the system settle to the origin

2. **Transition to Chaos**: Slowly increase ρ from 10 to 30 and watch the transition from periodic to chaotic behavior

3. **Different Attractors**: Try σ=10, ρ=20, β=8/3 versus σ=10, ρ=40, β=8/3

4. **Sensitivity**: Enable the comparison and watch how quickly trajectories diverge

## 📚 Further Reading

- [ModelingToolkit.jl Documentation](https://docs.sciml.ai/ModelingToolkit/stable/)
- [Lorenz System on Wikipedia](https://en.wikipedia.org/wiki/Lorenz_system)
- *Chaos: Making a New Science* by James Gleick
- *Nonlinear Dynamics and Chaos* by Steven Strogatz

## 🚀 Next Steps

Explore the other demos in this showcase:
- `01_symbolic_algebra.jl` - Symbolic manipulation basics
- `02_lorenz_attractor.jl` - Detailed Lorenz analysis
- `03_nonlinear_pendulum.jl` - Classical mechanics
- And more!

---

*Created with ❤️ using ModelingToolkit.jl and Pluto.jl*
"""

# ╔═╡ Package Cell
begin
    import Pkg
    Pkg.activate(temp=true)
    Pkg.add([
        Pkg.PackageSpec(name="PlutoUI"),
        Pkg.PackageSpec(name="ModelingToolkit"),
        Pkg.PackageSpec(name="DifferentialEquations"),
        Pkg.PackageSpec(name="CairoMakie"),
    ])
end
