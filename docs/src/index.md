```@meta
CurrentModule = GlobalOptimization
```

# GlobalOptimization

Currently, GlobalOptimization provides Particle Swarm Optimization (PSO) and several variants of Differential Evolution (DE) as the only global optimization algorithms supported. Monotonic Basin Hopping (MBH) is in the works.

## Simple PSO Example
Let's use PSO to find the minimum to the non-convex Ackley function given by

``J(\mathbf{x}) = -a \exp\left(-b\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}\right) - \exp\left(\frac{1}{d}\sum_{i=1}^d \cos (cx_i)\right) + a + \exp(1)``

where ``a = 20``, ``b = 0.2``, ``c = 2\pi``, and `d` is the length of the decision vector ``\mathbf{x}``, subject to the constraint that ``-32.768 \leq x_i \leq 32.768 \hspace{1mm} \forall \hspace{1mm} x_i \hspace{1mm} \in \hspace{1mm} \mathbf{x}``.

To begin, we'll first define an Ackley function in Julia as follows:
```@example simple_ackley
function ackley(x)
    a = 20
    b = 0.2
    c = 2*π
    d = length(x)

    sum1 = 0.0
    sum2 = 0.0
    for val in x
        sum1 += val^2
        sum2 += cos(c*val)
    end
    return -a*exp(-b*sqrt(sum1/d)) - exp(sum2/d) + a + exp(1)
end
nothing # hide
```

Next, we'll define the `OptimizationProblem` by providing its constructor our new `ackley` function and bounds that define the search space. Then, we'll instantiate a `StaticPSO` (an implementation of the PSO algorithm that does not use parallel computing to evaluate the cost function) to perform the optimization!

```@example simple_ackley
using GlobalOptimization

N   = 10 # The number of decision variables
LB  = [-32.768 for _ in 1:10] # The lower bounds
UB  = [ 32.768 for _ in 1:10] # The upper bounds

# Construct the optimization problem
op  = OptimizationProblem(ackley, LB, UB)

# Instantiate SerialPSO instance
pso = SerialPSO(op)

# Perform optimization with pso
res = optimize!(pso)
nothing # hide
```

Finally, we can get the final optimal decision vector with
```@example simple_ackley
best_candidate = res.xbest
```

and the fitness of the final optimal decision vector with
```@example simple_ackley
best_candidate_fitness = res.fbest
```

## Index
```@index
```
