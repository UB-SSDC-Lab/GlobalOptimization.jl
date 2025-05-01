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

## Algorithms
### Particle Swarm Optimization
```@docs
SerialPSO
ThreadedPSO
PolyesterPSO
```

### Differential Evolution
```@docs
SerialDE
ThreadedDE
PolyesterDE
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
```


#### DE Crossover Strategies
```@autodocs
Modules = [GlobalOptimization]
Pages = ["crossover.jl"]
Private = false
```