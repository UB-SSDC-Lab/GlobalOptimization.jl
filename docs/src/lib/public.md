# Public API Documentation

Documentation for GlobalOptimization's public interface.

## Contents
```@contents
Pages = ["public.md"]
Depth = 2:3
```

## Index
```@index
Pages = ["public.md"]
```

## Optimization Problem
```@autodocs
Modules = [GlobalOptimization]
Pages = ["Problem.jl"]
Private = false
```

## Search Space
```@autodocs
Modules = [GlobalOptimization]
Pages = ["SearchSpace.jl"]
Private = false
```

## Optimization
```@autodocs
Modules = [GlobalOptimization]
Pages = ["Optimizers.jl"]
Private = false
```

## Population Initialization
```@docs
    LatinHypercubeInitialization
    LatinHypercubeInitialization(::Int)
```

## Function Evaluation Methods
```@docs
SerialFunctionEvaluation
SerialFunctionEvaluation()
ThreadedFunctionEvaluation
ThreadedFunctionEvaluation()
PolyesterFunctionEvaluation
PolyesterFunctionEvaluation()
```

## Algorithms
### Particle Swarm Optimization
```@autodocs
    Modules = [GlobalOptimization]
    Pages = ["PSO.jl"]
    Private = false
```

### Differential Evolution
```@autodocs
    Modules = [GlobalOptimization]
    Pages = ["DE.jl"]
    Private = false
```

#### Mutation Parameters
```@docs 
MutationParameters 
MutationParameters(::Any, ::Any, ::Any, ::Any; sel=::Any)
MutationParameters(::GlobalOptimization.AbstractMutationStrategy; dist=::Any, sel=::Any)
SelfMutationParameters 
SelfMutationParameters(::GlobalOptimization.AbstractMutationStrategy; dist=::Any, sel=:Any)
Rand1
Rand2
Best1
Best2
CurrentToBest1
CurrentToBest2
CurrentToRand1
CurrentToRand2
RandToBest1
RandToBest2
Unified
SimpleSelector
RadiusLimitedSelector
RandomSubsetSelector
```

#### DE Crossover Strategies
```@docs
BinomialCrossoverParameters
BinomialCrossoverParameters(::Float64; transform=::Any)
BinomialCrossoverParameters(; dist=::Any, transform=::Any)
SelfBinomialCrossoverParameters
SelfBinomialCrossoverParameters(; dist=::Any, transform=::Any)
CovarianceTransformation
CovarianceTransformation(::Any,::Any,::Any)
```

### Monotonic Basin Hopping
```@autodocs
    Modules = [GlobalOptimization]
    Pages = ["MBH.jl"]
    Private = false
```

#### Hopper Types
```@autodocs
    Modules = [GlobalOptimization]
    Pages = ["Hopper.jl"]
    Private = false
```

#### Hop Distributions
```@autodocs
    Modules = [GlobalOptimization]
    Pages = ["Distributions.jl"]
    Private = false
```

#### Local Search Methods
```@autodocs
    Modules = [GlobalOptimization]
    Pages = ["LocalSearch.jl"]
    Private = false
```