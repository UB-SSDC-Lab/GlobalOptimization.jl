module GlobalOptimization

using Format
using StaticArrays
using StructArrays
using ThreadsX
using Random: shuffle!

# Utilities
include("rng.jl")

# Base
include("Problem.jl")
include("Options.jl")
include("Optimizers.jl")
include("Results.jl")

# PSO
include("PSO/Particle.jl")
include("PSO/Swarm.jl")
include("PSO/PSO.jl")

export Problem
export Options
export PSO, StaticPSO
export optimize!

end
