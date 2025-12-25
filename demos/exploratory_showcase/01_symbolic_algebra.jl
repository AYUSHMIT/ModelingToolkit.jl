"""
# 01: Symbolic Algebra - Introduction to ModelingToolkit.jl

This demo introduces the fundamental concepts of symbolic manipulation in Julia using
ModelingToolkit.jl and Symbolics.jl. You'll learn how to create symbolic variables,
perform algebraic operations, differentiate and integrate symbolically, and generate
optimized numerical code.

## Learning Objectives:
1. Create symbolic variables and parameters
2. Perform algebraic operations (expand, simplify, factor)
3. Symbolic differentiation and integration
4. Solve equations symbolically
5. Substitute values and evaluate expressions
6. Generate optimized code from symbolic expressions
7. Visualize symbolic functions and their derivatives

## Prerequisites:
- Basic algebra
- Understanding of derivatives (helpful but not required)

## Duration: ~15 minutes
"""

using ModelingToolkit
using Symbolics
using CairoMakie
using LinearAlgebra

# Set up output directory
output_dir = "output"
mkpath(output_dir)

println("="^70)
println("🔤 Demo 01: Symbolic Algebra")
println("="^70)
println()

#=============================================================================
Section 1: Creating Symbolic Variables
=============================================================================#

println("Section 1: Creating Symbolic Variables")
println("-"^70)

# Define symbolic variables
# The @variables macro creates symbolic variables that can be manipulated algebraically
@variables x y z t

println("Created symbolic variables: x, y, z, t")
println("Type of x: ", typeof(x))
println()

# Define symbolic parameters (constants in expressions)
# Parameters are treated as constants during symbolic manipulation
@parameters a b c σ ρ β

println("Created symbolic parameters: a, b, c, σ, ρ, β")
println("Type of a: ", typeof(a))
println()

#=============================================================================
Section 2: Basic Algebraic Operations
=============================================================================#

println("\nSection 2: Basic Algebraic Operations")
println("-"^70)

# Create a simple expression
expr1 = a * x^2 + b * x + c
println("Expression 1: ", expr1)

# Create another expression
expr2 = (x + 1)^2
println("Expression 2: ", expr2)
println("Expanded: ", expand(expr2))
println()

# Multiplication of expressions
expr3 = (x + y) * (x - y)
println("Expression 3: ", expr3)
println("Expanded: ", expand(expr3))
println()

# Factorization (when possible)
expr4 = x^2 - 2*x*y + y^2
println("Expression 4: ", expr4)
# Note: Symbolics.jl has limited factorization capabilities
# For advanced factorization, consider using specialized computer algebra systems
println()

# Simplification
expr5 = (x^2 + 2*x*y + y^2) / (x + y)
simplified = simplify(expr5)
println("Expression 5: ", expr5)
println("Simplified: ", simplified)
println()

#=============================================================================
Section 3: Symbolic Differentiation
=============================================================================#

println("\nSection 3: Symbolic Differentiation")
println("-"^70)

# The D operator represents differentiation
# Symbolics can compute derivatives of arbitrary complexity
Dx = Differential(x)

# First derivative
f = x^3 + 2*x^2 + x + 1
df_dx = expand_derivatives(Dx(f))
println("f(x) = ", f)
println("f'(x) = ", df_dx)
println()

# Second derivative
d2f_dx2 = expand_derivatives(Dx(df_dx))
println("f''(x) = ", d2f_dx2)
println()

# Multivariable function
g = x^2 * y + y^2 * x + x * y * z
Dy = Differential(y)
Dz = Differential(z)

dg_dx = expand_derivatives(Dx(g))
dg_dy = expand_derivatives(Dy(g))
dg_dz = expand_derivatives(Dz(g))

println("g(x,y,z) = ", g)
println("∂g/∂x = ", dg_dx)
println("∂g/∂y = ", dg_dy)
println("∂g/∂z = ", dg_dz)
println()

# Chain rule demonstration
h = sin(x^2)
dh_dx = expand_derivatives(Dx(h))
println("h(x) = ", h)
println("h'(x) = ", dh_dx)
println()

#=============================================================================
Section 4: Substitution and Evaluation
=============================================================================#

println("\nSection 4: Substitution and Evaluation")
println("-"^70)

# Define an expression
expr = a*x^2 + b*x + c

# Substitute specific values for parameters
expr_substituted = substitute(expr, Dict([a => 1, b => 2, c => 3]))
println("Original: ", expr)
println("After substituting a=1, b=2, c=3: ", expr_substituted)

# Evaluate at a specific x value
# First, we need to build a callable function
expr_func = build_function(expr_substituted, x, expression=Val{false})
result = expr_func(5.0)  # Evaluate at x = 5
println("Value at x=5: ", result)
println()

#=============================================================================
Section 5: Code Generation
=============================================================================#

println("\nSection 5: Code Generation for Fast Numerical Computation")
println("-"^70)

# Define a complex expression
complex_expr = sin(x) * cos(y) + exp(-x^2) * log(y^2 + 1)

# Build a fast numerical function
# expression=Val{false} generates a Julia function (fast)
# expression=Val{true} would return an Expr object (for metaprogramming)
fast_func = build_function(complex_expr, [x, y], expression=Val{false})

println("Built fast function from: ", complex_expr)

# Test the generated function
test_result = fast_func([1.0, 2.0])
println("Evaluated at [x=1.0, y=2.0]: ", test_result)
println()

# Compare symbolic vs. numerical computation
using BenchmarkTools

# Create a numerical function manually (for comparison)
manual_func(x, y) = sin(x) * cos(y) + exp(-x^2) * log(y^2 + 1)

println("Performance comparison:")
println("Generated function:")
@btime $fast_func([1.0, 2.0])

println("Manual function:")
@btime manual_func(1.0, 2.0)
println()

#=============================================================================
Section 6: Solving Equations Symbolically
=============================================================================#

println("\nSection 6: Solving Equations")
println("-"^70)

# Symbolics can solve simple equations
# For the quadratic equation: ax² + bx + c = 0
# Solution: x = (-b ± √(b²-4ac)) / 2a

println("For a quadratic equation ax² + bx + c = 0")
println("The solutions are given by the quadratic formula:")
println("x = (-b ± √(b²-4ac)) / 2a")
println()

# Note: For more advanced equation solving, consider using other Julia packages
# like SymEngine.jl or interfacing with SymPy

#=============================================================================
Section 7: Visualization
=============================================================================#

println("\nSection 7: Visualizing Symbolic Functions")
println("-"^70)

# Create a symbolic function and its derivative
@variables x
f_symbolic = x^3 - 6*x^2 + 9*x + 1
Dx = Differential(x)
df_symbolic = expand_derivatives(Dx(f_symbolic))

# Build numerical functions for plotting
f_numerical = build_function(f_symbolic, x, expression=Val{false})
df_numerical = build_function(df_symbolic, x, expression=Val{false})

# Create x values for plotting
x_vals = range(-1, 5, length=200)
y_vals = [f_numerical(xi) for xi in x_vals]
dy_vals = [df_numerical(xi) for xi in x_vals]

# Create visualization
fig = Figure(size=(1200, 500))

# Plot function
ax1 = Axis(fig[1, 1],
    xlabel="x",
    ylabel="f(x)",
    title="Function: f(x) = x³ - 6x² + 9x + 1",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax1, x_vals, y_vals, color=:blue, linewidth=2, label="f(x)")
axislegend(ax1, position=:lt)

# Plot derivative
ax2 = Axis(fig[1, 2],
    xlabel="x",
    ylabel="f'(x)",
    title="Derivative: f'(x) = 3x² - 12x + 9",
    xgridvisible=true,
    ygridvisible=true
)

lines!(ax2, x_vals, dy_vals, color=:red, linewidth=2, label="f'(x)")
hlines!(ax2, [0], color=:black, linestyle=:dash, linewidth=1)
axislegend(ax2, position=:lt)

# Save the figure
save(joinpath(output_dir, "01_symbolic_algebra_visualization.png"), fig, px_per_unit=2)
println("✓ Saved visualization to output/01_symbolic_algebra_visualization.png")
println()

# Create another visualization: Multiple functions and their derivatives
fig2 = Figure(size=(1200, 800))

# Define several functions
functions = [
    (x -> sin(x), "sin(x)", :blue),
    (x -> cos(x), "cos(x)", :red),
    (x -> exp(-x^2/4), "exp(-x²/4)", :green)
]

x_range = range(-3π, 3π, length=300)

for (idx, (func, label, color)) in enumerate(functions)
    # Function plot
    ax_f = Axis(fig[idx, 1],
        xlabel="x",
        ylabel="f(x)",
        title="$label",
        xgridvisible=true,
        ygridvisible=true
    )
    
    y_vals = func.(x_range)
    lines!(ax_f, x_range, y_vals, color=color, linewidth=2)
    
    # Derivative plot (numerical approximation for visualization)
    ax_df = Axis(fig[idx, 2],
        xlabel="x",
        ylabel="f'(x)",
        title="Derivative of $label",
        xgridvisible=true,
        ygridvisible=true
    )
    
    # Numerical derivative using finite differences
    dx = x_range[2] - x_range[1]
    dy_vals = diff(y_vals) / dx
    lines!(ax_df, x_range[1:end-1], dy_vals, color=color, linewidth=2)
    hlines!(ax_df, [0], color=:black, linestyle=:dash, linewidth=1)
end

save(joinpath(output_dir, "01_symbolic_algebra_multiple_functions.png"), fig2, px_per_unit=2)
println("✓ Saved visualization to output/01_symbolic_algebra_multiple_functions.png")
println()

#=============================================================================
Section 8: Practical Example - Taylor Series Expansion
=============================================================================#

println("\nSection 8: Practical Example - Taylor Series")
println("-"^70)

"""
    taylor_expand(f, x, x0, n)

Compute the Taylor series expansion of function f around point x0 up to order n.

# Arguments
- `f`: Symbolic expression to expand
- `x`: Symbolic variable
- `x0`: Expansion point (symbolic or numeric)
- `n`: Order of expansion

# Returns
- Taylor series expansion as a symbolic expression
"""
function taylor_expand(f, x, x0, n)
    Dx = Differential(x)
    taylor = substitute(f, Dict(x => x0))
    f_derivative = f
    
    for i in 1:n
        f_derivative = expand_derivatives(Dx(f_derivative))
        term = substitute(f_derivative, Dict(x => x0)) * (x - x0)^i / factorial(i)
        taylor = taylor + term
    end
    
    return simplify(taylor)
end

# Example: Taylor expansion of sin(x) around x=0
@variables x
f_sin = sin(x)

println("Taylor expansions of sin(x) around x=0:")
for order in [1, 3, 5, 7]
    taylor = taylor_expand(f_sin, x, 0, order)
    println("Order $order: ", taylor)
end
println()

# Visualize Taylor approximations
fig3 = Figure(size=(1000, 700))
ax = Axis(fig3[1, 1],
    xlabel="x",
    ylabel="y",
    title="Taylor Series Approximations of sin(x)",
    xgridvisible=true,
    ygridvisible=true
)

x_plot = range(-2π, 2π, length=300)

# Plot actual sin(x)
lines!(ax, x_plot, sin.(x_plot), color=:black, linewidth=3, label="sin(x)")

# Plot Taylor approximations
colors = [:red, :blue, :green, :orange]
orders = [1, 3, 5, 7]

for (order, color) in zip(orders, colors)
    taylor_expr = taylor_expand(f_sin, x, 0, order)
    taylor_func = build_function(taylor_expr, x, expression=Val{false})
    taylor_vals = [taylor_func(xi) for xi in x_plot]
    lines!(ax, x_plot, taylor_vals, color=color, linewidth=2, label="Order $order", linestyle=:dash)
end

axislegend(ax, position=:lt)
ylims!(ax, -3, 3)

save(joinpath(output_dir, "01_symbolic_algebra_taylor_series.png"), fig3, px_per_unit=2)
println("✓ Saved Taylor series visualization to output/01_symbolic_algebra_taylor_series.png")
println()

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("✅ Demo 01 Complete!")
println("="^70)
println()
println("Summary of what you learned:")
println("• Creating symbolic variables and parameters")
println("• Performing algebraic operations (expand, simplify)")
println("• Symbolic differentiation (single and multivariable)")
println("• Substitution and evaluation of expressions")
println("• Code generation for fast numerical computation")
println("• Visualizing symbolic functions")
println("• Taylor series expansion")
println()
println("Next: Try 02_lorenz_attractor.jl to explore chaotic ODEs!")
println()
