```@meta
CurrentModule = GlobalOptimization
```

# GlobalOptimization

Currently, GlobalOptimization.jl provides Particle Swarm Optimization (PSO) as the only global optimization algorithm supported. Monotonic Basin Hopping (MBH) will be added in the following weeks.

## A Simple PSO Example
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

To get an idea of just how non-convex the Ackley function it, let's create a contour plot for the 2-dimensional case as follows:
```@example simple_ackley
using CairoMakie

# Define a two-argument version of our ackley function to simplify broadcasting
ackley(x1,x2) = ackley((x1,x2))

# Create x, y, and z values
x = range(-5,5,length=1000)
y = copy(x)
z = @. ackley(x',y)

# Create plot
f = Figure()
Axis(f[1,1])
co = contourf!(x,y,z; levels = 10)
Colorbar(f[1,2], co)
f # hide
```
Clearly, the Ackley function is highly multi-modal! Gradient based optimization algorithms would, in general, fail to find the global minimum of this function unless an initial guess was provided that was very near the origin. Thankfully, many global optimization algorithms, like Particle Swarm Optimization for example, do not encounter these same difficulties and can often successfully find a solution that is at least near the globally optimal solution as we'll soon see... 

Next, we'll define the `OptimizationProblem` by providing its constructor our original `ackley` function and an bounds that define the search space. Then, we'll instantiate a `StaticPSO` (an implementation of the PSO algorithm that does not use paralle computing to evaluate the cost function) to perform the optimization!

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
