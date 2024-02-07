module GlobalOptimization

using Format
using StaticArrays
using Random: shuffle!
using UnPack

# Utilities
include("rng.jl")

# Base
include("Options.jl")
include("SearchSpace.jl")
include("Problem.jl")
include("Candidate.jl")
include("Population.jl")
include("Evaluator.jl")
include("Optimizers.jl")
include("Results.jl")

# PSO
include("PSO/Swarm.jl")
include("PSO/PSO.jl")

# MBH
include("MBH/Distributions.jl")
include("MBH/Hopper.jl")
include("MBH/MBH.jl")

export ContinuousRectangularSearchSpace
export OptimizationProblem
export SerialPSO, ThreadedPSO
#export MBH
export optimize!

end
