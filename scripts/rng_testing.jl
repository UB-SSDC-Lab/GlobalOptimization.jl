using PyPlot
using Distributions
using GlobalOptimization

# Monte Carlo Trials
n = 1000000

# Create exponentia random vectors
λ = 2.0
expd = Exponential(1.0 / λ)
expd_vec = rand(Exponential(1.0 / λ), n)
expd_vec_go = [GlobalOptimization.exprate(λ) for _ in 1:n]

# Create Laplace random vectors for general distribution
μ = 10.2
θ = 2.5
lapg = rand(Laplace(μ, θ), n)
lapg_go = [GlobalOptimization.laplace(μ, θ) for _ in 1:n]

# Create Laplace random vectors for symmetric distribution
θ = 2.5
laps = rand(Laplace(0.0, θ), n)
laps_go = [GlobalOptimization.laplace(θ) for _ in 1:n]

# Create Laplace random vectors for unit distribution
lapu = rand(Laplace(0.0, 1.0), n)
lapu_go = [GlobalOptimization.laplace() for _ in 1:n]

PyPlot.figure()
PyPlot.hist(
    expd_vec; 
    bins = 100, alpha = 0.5, color = "green", label = "Distributions.jl",
)
PyPlot.hist(
    expd_vec_go;
    bins = 100, alpha = 0.5, color = "blue", label = "GlobalOptimization.jl",
)
PyPlot.title("Exponential Distribution")
PyPlot.legend()

PyPlot.figure()
PyPlot.hist(
    lapg; 
    bins = 100, alpha = 0.5, color = "green", label = "Distributions.jl",
)
PyPlot.hist(
    lapg_go;
    bins = 100, alpha = 0.5, color = "blue", label = "GlobalOptimization.jl",
)
PyPlot.title("Laplace General Distribution")
PyPlot.legend()

PyPlot.figure()
PyPlot.hist(
    laps; 
    bins = 100, alpha = 0.5, color = "green", label = "Distributions.jl",
)
PyPlot.hist(
    laps_go;
    bins = 100, alpha = 0.5, color = "blue", label = "GlobalOptimization.jl",
)
PyPlot.title("Laplace Symmetric Distribution")
PyPlot.legend()

PyPlot.figure()
PyPlot.hist(
    lapu; 
    bins = 100, alpha = 0.5, color = "green", label = "Distributions.jl",
)
PyPlot.hist(
    lapu_go;
    bins = 100, alpha = 0.5, color = "blue", label = "GlobalOptimization.jl",
)
PyPlot.title("Laplace Unit Distribution")
PyPlot.legend()