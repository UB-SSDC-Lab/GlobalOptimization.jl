# Public API Documentation

Documentation for GlobalOptimization's public interface.

## Contents
```@contents
Pages = ["public.md"]
Depth = 2:3
```

## Problems
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

#### Velocity Update Schemes
The PSO velocity update is the primary mechanism that drives the stochastic 
optimization process. The currently implemented velocity update schemes can 
be described by the following:

Consider a *swarm* of ``n`` particles ``\mathcal{S} = \{\mathbf{p}_i\}_{i=1,2,\dots,n}``.
Each particle ``\mathbf{p}_i`` has the following attributes associated with it:
 - position: ``\mathbf{x}_i``
 - velocity: ``\mathbf{v}_i``,
 - best position: ``\mathbf{x}_{i,b}``
 - best fitness: ``f_{i,b} = f(\mathbf{x}_{i,b})``

At each iteration of the PSO algorithm, the velocity of each particle is updated prior to 
updating the position of each particle with ``\mathbf{x}_i = \mathbf{x}_i + \mathbf{v}_i``.
This velocity update is described (for the ``i``-th particle) by the following expression:

``
\mathbf{v}_i = w \mathbf{v}_i +
    y_1 \mathbf{r}_1 (\mathbf{x}_{i,b} - \mathbf{x}_i) +
    y_2 \mathbf{r}_2 (\mathbf{x}_b - \mathbf{x}_i)
``

where ``w`` is the inertia,  ``r_1`` and ``r_2`` are realizations of a random vector
described by the multivariate uniform distribution
``\mathcal{U}(\mathbf{0}, \mathbf{1})``, ``y_1`` is the self-adjustment weight, ``y_2``
is the social adjustment weight, and ``\mathbf{x}_{b}`` is the best position in the
neighborhood of the ``i``-th particle ``\mathcal{N}_i``. That is,
``\mathbf{x}_b = \underset{x\in\mathcal{X}_b}{\mathrm{argmin}}(f(x))`` where
`` \mathcal{X}_{b,i} = \{ \mathbf{x}_{i,b} \}_{\mathbf{p}_i \in \mathcal{N}_i}`` and
``\mathcal{N}_i`` is a set containing a randomly selected subset of the particles in
``\mathcal{S}`` (not including ``\mathbf{p}_i``). Both the size of ``\mathcal{N}_i`` and 
the inertia ``w`` are handle differently depending on the velocity update scheme used.

```@docs
MATLABVelocityUpdate
MATLABVelocityUpdate(; kwargs...)
CSRNVelocityUpdate
CSRNVelocityUpdate(; kwargs...)
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

## Trace Options
Each algorithm provides the ability trace solve information to the terminal 
or a specified file through setting the keyword arguments `show_trace = Val(true)`
and `save_trace = Val(true)`, respectively. Additionally, the amount of information
provided in the trace can be controlled by setting the `trace_level` keyword argument
with one of the following `TraceLevel` constructors:
```@docs
TraceMinimal(;print_frequency=::Any, save_frequency=::Any)
TraceDetailed(;print_frequency=::Any, save_frequency=::Any)
TraceAll(;print_frequency=::Any, save_frequency=::Any)
```

## Index
```@index
Pages = ["public.md"]
```