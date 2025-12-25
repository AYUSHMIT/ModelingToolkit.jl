# 🎨 ModelingToolkit.jl Exploratory Showcase

<div align="center">

**A Comprehensive Educational Journey Through Scientific Computing with Julia**

*From Symbolic Algebra to GPU-Accelerated Simulations*

[Quick Start](#-quick-start) • [Gallery](#-gallery) • [Learning Path](#-learning-path) • [Demos](#-demo-descriptions) • [Interactive](#-interactive-notebook)

---

</div>

## 🌟 Overview

Welcome to the **ModelingToolkit.jl Exploratory Showcase** – a carefully curated collection of progressively complex examples demonstrating the power and versatility of ModelingToolkit.jl for scientific computing. This showcase is designed for graduate students, researchers, and engineers who want to master symbolic-numeric computation in Julia.

Each demo includes:
- ✨ Beautiful, publication-quality visualizations
- 📚 Extensive documentation and comments
- 🎓 Educational explanations of concepts
- 🔬 Real-world applications
- 🎯 Progressive difficulty levels

## 🖼️ Gallery

> *Coming soon: Visualization outputs from each demo will be showcased here*

<table>
  <tr>
    <td align="center">
      <b>Lorenz Attractor</b><br/>
      <i>Chaotic dynamics in 3D</i>
    </td>
    <td align="center">
      <b>Heat Equation</b><br/>
      <i>Temperature diffusion</i>
    </td>
    <td align="center">
      <b>Wave Propagation</b><br/>
      <i>2D wave equation</i>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Chemical Reactions</b><br/>
      <i>Brusselator oscillations</i>
    </td>
    <td align="center">
      <b>Mechanical Systems</b><br/>
      <i>Double pendulum chaos</i>
    </td>
    <td align="center">
      <b>GPU Acceleration</b><br/>
      <i>Performance comparison</i>
    </td>
  </tr>
</table>

## 🚀 Quick Start

### Prerequisites

- **Julia 1.9+** (Download from [julialang.org](https://julialang.org/downloads/))
- **Basic knowledge** of differential equations and scientific computing
- **8GB RAM** minimum (16GB recommended for GPU demos)
- **CUDA-capable GPU** (optional, for GPU acceleration demos)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SciML/ModelingToolkit.jl.git
   cd ModelingToolkit.jl/demos/exploratory_showcase
   ```

2. **Activate the demo environment:**
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

3. **Run your first demo:**
   ```julia
   include("01_symbolic_algebra.jl")
   ```

## 📊 Learning Path

### 🟢 Beginner: Fundamentals

Perfect for those new to ModelingToolkit.jl and symbolic computing.

| Demo | File | Duration | Key Concepts |
|------|------|----------|--------------|
| **Symbolic Algebra** | `01_symbolic_algebra.jl` | 15 min | Variables, differentiation, simplification |
| **Lorenz Attractor** | `02_lorenz_attractor.jl` | 20 min | ODEs, chaotic systems, 3D visualization |
| **Nonlinear Pendulum** | `03_nonlinear_pendulum.jl` | 20 min | Classical mechanics, phase portraits |

### 🟡 Intermediate: Applied Physics

Explore real-world physics and chemistry applications.

| Demo | File | Duration | Key Concepts |
|------|------|----------|--------------|
| **Heat Equation** | `04_heat_equation_pde.jl` | 30 min | PDEs, Method of Lines, diffusion |
| **Wave Equation** | `05_wave_equation_pde.jl` | 30 min | Hyperbolic PDEs, wave propagation |
| **Chemical Reactions** | `06_chemical_reactions.jl` | 25 min | Reaction networks, Catalyst.jl |
| **Mechanical Systems** | `07_mechanical_systems.jl` | 35 min | Multi-body dynamics, Lagrangian mechanics |

### 🔴 Advanced: Performance & Optimization

Master optimization techniques and high-performance computing.

| Demo | File | Duration | Key Concepts |
|------|------|----------|--------------|
| **Optimization** | `08_optimization_problems.jl` | 30 min | Parameter estimation, model fitting |
| **GPU Parallelization** | `09_gpu_parallelization.jl` | 25 min | CUDA, ensemble simulations |
| **Code Generation** | `10_code_generation.jl` | 20 min | JIT compilation, C code generation |

### 🎮 Interactive: Hands-On Exploration

| Demo | File | Duration | Key Concepts |
|------|------|----------|--------------|
| **Interactive Notebook** | `interactive_notebook.jl` | ∞ | Pluto.jl, interactive widgets |

## 📚 Demo Descriptions

### 01: Symbolic Algebra 🔤

**Objective:** Master the fundamentals of symbolic manipulation in Julia.

Learn how to create symbolic variables, perform algebraic operations, differentiate and integrate symbolically, and generate optimized code from symbolic expressions.

**Highlights:**
- Symbolic differentiation and integration
- Expression simplification and factorization
- Substitution and evaluation
- Code generation for fast numerical computation

**Prerequisites:** Basic algebra

---

### 02: Lorenz Attractor 🦋

**Objective:** Explore chaotic dynamics through the classic Lorenz system.

Discover the butterfly effect and sensitive dependence on initial conditions through beautiful 3D visualizations of this iconic chaotic system.

**Highlights:**
- 3D trajectory visualization
- Phase space projections
- Sensitivity to initial conditions
- Animated attractor formation

**Prerequisites:** Basic differential equations

---

### 03: Nonlinear Pendulum ⚖️

**Objective:** Compare linear approximations with full nonlinear dynamics.

Understand when the small-angle approximation breaks down and explore the rich dynamics of a nonlinear pendulum with damping.

**Highlights:**
- Comparison of linear vs. nonlinear solutions
- Phase portraits and limit cycles
- Energy conservation analysis
- Animated pendulum motion

**Prerequisites:** Classical mechanics fundamentals

---

### 04: Heat Equation 🔥

**Objective:** Solve parabolic PDEs with the Method of Lines.

Model heat diffusion in 1D and 2D with various boundary conditions, visualizing temperature evolution over time.

**Highlights:**
- Method of Lines discretization
- Dirichlet and Neumann boundary conditions
- Animated heatmaps
- Stability analysis

**Prerequisites:** Partial differential equations

---

### 05: Wave Equation 🌊

**Objective:** Simulate wave propagation in 1D and 2D.

Model vibrating strings and drums, exploring wave speed, reflection, and interference patterns.

**Highlights:**
- 1D and 2D wave propagation
- Various initial conditions
- Animated wave visualization
- Boundary reflections

**Prerequisites:** Wave mechanics

---

### 06: Chemical Reactions 🧪

**Objective:** Model reaction networks with Catalyst.jl.

Explore chemical kinetics through classic models like the Brusselator and Lotka-Volterra systems.

**Highlights:**
- Mass action kinetics
- Oscillating reactions (Brusselator)
- Predator-prey dynamics
- Stochastic vs. deterministic simulations

**Prerequisites:** Chemical kinetics basics

---

### 07: Mechanical Systems ⚙️

**Objective:** Simulate multi-body dynamics systems.

Explore complex mechanical systems like the double pendulum using Lagrangian and Hamiltonian mechanics.

**Highlights:**
- Double pendulum chaos
- Energy conservation
- Lagrangian formulation
- Animated mechanical systems

**Prerequisites:** Classical mechanics

---

### 08: Optimization Problems 📈

**Objective:** Perform parameter estimation and optimal control.

Learn to fit models to data, estimate parameters, and solve optimization problems with ModelingToolkit.

**Highlights:**
- Parameter estimation from noisy data
- Loss landscape visualization
- Model calibration
- Convergence analysis

**Prerequisites:** Optimization basics

---

### 09: GPU Parallelization 🚀

**Objective:** Accelerate simulations with GPU computing.

Harness the power of GPU parallelization for ensemble simulations and learn when GPU acceleration provides benefits.

**Highlights:**
- CPU vs. GPU performance comparison
- Ensemble simulations
- CUDA integration
- Graceful fallback for systems without GPUs

**Prerequisites:** Basic parallel computing concepts

---

### 10: Code Generation 💻

**Objective:** Generate optimized code for maximum performance.

Learn how ModelingToolkit generates fast C/LLVM code and compare performance with interpreted execution.

**Highlights:**
- Symbolic to compiled code
- Performance benchmarking
- JIT compilation benefits
- Standalone function export

**Prerequisites:** Basic understanding of compilation

---

### Interactive Notebook 🎮

**Objective:** Explore systems interactively with Pluto.jl.

Use interactive sliders and controls to explore parameter spaces in real-time with beautiful reactive visualizations.

**Highlights:**
- Real-time parameter adjustment
- Interactive phase plots
- Multiple system comparisons
- Educational narrative

**Prerequisites:** None (most accessible demo)

## 🎯 Running the Demos

### Running Individual Demos

Each demo is self-contained and can be run independently:

```julia
# From the exploratory_showcase directory
include("02_lorenz_attractor.jl")
```

### Running All Demos

To run all demos sequentially (useful for generating all visualizations):

```julia
demos = [
    "01_symbolic_algebra.jl",
    "02_lorenz_attractor.jl",
    "03_nonlinear_pendulum.jl",
    "04_heat_equation_pde.jl",
    "05_wave_equation_pde.jl",
    "06_chemical_reactions.jl",
    "07_mechanical_systems.jl",
    "08_optimization_problems.jl",
    "09_gpu_parallelization.jl",
    "10_code_generation.jl"
]

for demo in demos
    println("\n" * "="^60)
    println("Running: $demo")
    println("="^60 * "\n")
    include(demo)
end
```

### Output Files

Demos save visualizations to the `output/` directory (created automatically):
- **Static plots:** PNG format (high DPI)
- **Animations:** GIF and MP4 format
- **Data files:** CSV format for further analysis

## 🎮 Interactive Notebook

The Pluto.jl interactive notebook provides a hands-on exploration experience:

```julia
using Pluto
Pluto.run()
# Then open: interactive_notebook.jl
```

**Features:**
- 🎚️ Real-time parameter sliders
- 📊 Live updating visualizations
- 🔄 Compare multiple configurations
- 📖 Inline educational content

## 📖 Educational Resources

### Official Documentation
- [ModelingToolkit.jl Docs](https://docs.sciml.ai/ModelingToolkit/stable/)
- [DifferentialEquations.jl Docs](https://docs.sciml.ai/DiffEqDocs/stable/)
- [Catalyst.jl Docs](https://docs.sciml.ai/Catalyst/stable/)
- [Makie.jl Docs](https://docs.makie.org/stable/)

### Recommended Textbooks
- *Ordinary Differential Equations* by Tenenbaum & Pollard
- *Numerical Methods for PDEs* by LeVeque
- *Nonlinear Dynamics and Chaos* by Strogatz
- *Classical Mechanics* by Goldstein

### Online Courses
- [Julia Academy - Scientific Computing](https://juliaacademy.com/)
- [Computational Thinking (MIT)](https://computationalthinking.mit.edu/)

### Research Papers
- [ModelingToolkit Paper](https://arxiv.org/abs/2103.05244)
- [SciML Ecosystem](https://sciml.ai/papers/)

## 🤝 Contributing

We welcome contributions to improve these demos!

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-demo`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing demo'`)
6. Push to the branch (`git push origin feature/amazing-demo`)
7. Open a Pull Request

### Contribution Ideas
- 🎨 Improve visualizations
- 📚 Add more examples
- 🐛 Fix bugs or typos
- 📝 Enhance documentation
- 🌐 Add translations
- 🎓 Create tutorial videos

## ⚠️ Troubleshooting

### Common Issues

**CUDA not available:**
```julia
# GPU demos will automatically fall back to CPU if CUDA is unavailable
# No action needed - the demos handle this gracefully
```

**Out of memory:**
```julia
# Reduce problem size or time span in demos
# Example: Change tspan = (0.0, 100.0) to tspan = (0.0, 10.0)
```

**Slow plotting:**
```julia
# First plot may be slow due to compilation
# Subsequent plots will be much faster
```

## 📜 License

This showcase is part of ModelingToolkit.jl and follows the same license (MIT).

## 🙏 Acknowledgments

- **SciML Community** for the amazing ecosystem
- **Julia Language** for making scientific computing beautiful
- **Contributors** who help improve these demos
- **Educators** who use these materials in teaching

## 📧 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/SciML/ModelingToolkit.jl/issues)
- **Discussions:** [GitHub Discussions](https://github.com/SciML/ModelingToolkit.jl/discussions)
- **Chat:** [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
- **Forum:** [Julia Discourse](https://discourse.julialang.org/)

---

<div align="center">

**Made with ❤️ by the SciML Community**

*Happy Computing! 🎉*

</div>
