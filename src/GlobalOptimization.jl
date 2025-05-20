module GlobalOptimization

using ADTypes: AbstractADType
using ChunkSplitters: chunks, ChunkSplitters, RoundRobin
using Distributions: Cauchy, Laplace, MixtureModel
using Format: printfmtln, FormatExpr
using LatinHypercubeSampling: scaleLHC, LHCoptim
using LinearAlgebra: dot, eigen!, mul!
using LineSearches: HagerZhang, InitialStatic
using Polyester: @batch
using StaticArrays: SA, SVector
using Statistics: cov
using Random: rand, rand!, shuffle!, AbstractRNG, GLOBAL_RNG
using UnPack: @unpack

import Base
import NonlinearSolve
import Optim

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
export SerialFunctionEvaluation, ThreadedFunctionEvaluation, PolyesterFunctionEvaluation
export LatinHypercubeInitialization
export OptimizationProblem, NonlinearProblem, NonlinearLeastSquaresProblem
export optimize!

export PSO, DE, MBH

export SimpleSelector, RadiusLimitedSelector, RandomSubsetSelector
export Rand1, Rand2, Best1, Best2, CurrentToBest1, CurrentToBest2
export CurrentToRand1, CurrentToRand2, RandToBest1, RandToBest2, Unified
export MutationParameters, SelfMutationParameters
export SelfBinomialCrossoverParameters, BinomialCrossoverParameters
export CovarianceTransformation

export SingleHopper, MCH
export MBHStaticDistribution, MBHAdaptiveDistribution
export LocalStochasticSearch, LBFGSLocalSearch, NonlinearSolveLocalSearch

end
