module GlobalOptimization

using LinearAlgebra
using Format
using Polyester: @batch
using FunctionWrappersWrappers
using StaticArrays
using Random: shuffle!
using UnPack

import Base
import Optim
import LineSearches

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
include("MBH/LocalSearch.jl")
include("MBH/MBH.jl")

export ContinuousRectangularSearchSpace
export OptimizationProblem
export SerialPSO, ThreadedPSO, PolyesterPSO
#export MBH
export optimize!

end
