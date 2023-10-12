module GlobalOptimization

using Format
using StaticArrays
using StructArrays
using ThreadsX
using Random: shuffle!

# Utilities
include("rng.jl")

# Base
include("SearchSpace.jl")
include("Problem.jl")
include("Population.jl")
include("Evaluator.jl")
include("Optimizers.jl")
include("Results.jl")

# PSO
include("PSO/Swarm.jl")
#include("PSO/PSO.jl")

# MBH
#include("MBH/Hopper.jl")
#include("MBH/MBH.jl")

#export Problem
#export Options
#export PSO, StaticPSO
#export MBH
#export optimize!

end
