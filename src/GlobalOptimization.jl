module GlobalOptimization

using ADTypes: AbstractADType
using ChunkSplitters: chunks, ChunkSplitters, RoundRobin
using Distributions: Cauchy, Laplace, MixtureModel
using LatinHypercubeSampling: scaleLHC, LHCoptim
using LinearAlgebra: dot, eigen!, mul!, tril!
using LineSearches: HagerZhang, InitialStatic
using Polyester: @batch
using Printf: format, Format
using StaticArrays: SA, SVector
using Statistics: cov, mean, median!, std, cor
using Random: rand, rand!, shuffle!, AbstractRNG, GLOBAL_RNG
using UnPack: @unpack

using Base: Base
using NonlinearSolve: NonlinearSolve
using Optim: Optim

# Base
include("utils.jl")
include("enums.jl")
include("tracing.jl")
include("Options.jl")
include("Results.jl")
include("Optimizers.jl")
include("SearchSpace.jl")
include("Problem.jl")
include("Candidate.jl")
include("Population.jl")
include("Evaluator.jl")

# PSO
include("PSO/Swarm.jl")
include("PSO/velocity_update.jl")
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

export TraceMinimal, TraceDetailed, TraceAll

export ContinuousRectangularSearchSpace
export SerialFunctionEvaluation, ThreadedFunctionEvaluation, PolyesterFunctionEvaluation
export LatinHypercubeInitialization
export OptimizationProblem, NonlinearProblem, NonlinearLeastSquaresProblem
export optimize!

export PSO, DE, MBH

export MATLABVelocityUpdate, CSRNVelocityUpdate

export SimpleSelector, RadiusLimitedSelector, RandomSubsetSelector
export Rand1, Rand2, Best1, Best2, CurrentToBest1, CurrentToBest2
export CurrentToRand1, CurrentToRand2, RandToBest1, RandToBest2, Unified
export MutationParameters, SelfMutationParameters
export SelfBinomialCrossoverParameters, BinomialCrossoverParameters
export CovarianceTransformation, UncorrelatedCovarianceTransformation

export SingleHopper, MCH
export MBHStaticDistribution, MBHAdaptiveDistribution
export LocalStochasticSearch, LBFGSLocalSearch, NonlinearSolveLocalSearch

end
