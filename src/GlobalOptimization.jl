module GlobalOptimization

using ADTypes
using ChunkSplitters
using Distributions
using LatinHypercubeSampling: LHCoptim, scaleLHC
using LinearAlgebra
using Format
using Polyester: @batch
using StaticArrays
using Statistics: cov
using Random: shuffle!
using UnPack

import Random: rand, rand!, shuffle!, AbstractRNG, GLOBAL_RNG
using Base: Base
using Optim: Optim
using LineSearches: LineSearches

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

# DE
include("DE/util.jl")
include("DE/Population.jl")
include("DE/mutation.jl")
include("DE/crossover.jl")
include("DE/DE.jl")

# MBH
include("MBH/Distributions.jl")
include("MBH/Hopper.jl")
include("MBH/LocalSearch.jl")
include("MBH/MBH.jl")

export ContinuousRectangularSearchSpace
export LatinHypercubeInitialization
export OptimizationProblem
export optimize!

export SerialPSO, ThreadedPSO, PolyesterPSO

export SerialDE, ThreadedDE, PolyesterDE
export SimpleSelector, RadiusLimitedSelector, RandomSubsetSelector
export Rand1, Rand2, Best1, Best2, CurrentToBest1, CurrentToBest2
export CurrentToRand1, CurrentToRand2, RandToBest1, RandToBest2, Unified
export MutationParameters, SelfMutationParameters
export SelfBinomialCrossoverParameters, BinomialCrossoverParameters
export CovarianceTransformation

export MBH, SerialCMBH, ThreadedCMBH, PolyesterCMBH
export MBHStaticDistribution, MBHAdaptiveDistribution
export LocalStochasticSearch, LBFGSLocalSearch

end
