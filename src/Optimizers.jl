"""
    AbstractOptimizer

Abstract type of all optimization algorithms.

All subtypes must define the following methods:
- `initialize!`: Initialize the optimizer.
- `iterate!`: Perform iterations to optimize the problem and return the results of the
    optimization.
- `get_iteration`: Get the current iteration number of the optimizer.

All subtypes must have the following fields:
- `options`: The options for the optimizer.
"""
abstract type AbstractOptimizer end

# ===== Interface
"""
    optimize!(opt::AbstractOptimizer)

Perform optimization using the optimizer `opt`. Returns the results of the optimization.

# Arguments
- `opt::AbstractOptimizer`: The optimizer to use.

# Returns
- `Results`: The results of the optimization. See the [Results](@ref) docstring for details
    on its contents.

# Example
```julia-repl
julia> using GlobalOptimization
julia> f(x) = sum(x.^2) # Simple sphere function
julia> prob = OptimizationProblem(f, [-1.0, 0.0], [1.0, 2.0])
julia> pso = SerialPSO(prob)
julia> results = optimize!(pso)
Results:
 - Best function value: 6.696180996034206e-20
 - Best candidate: [-2.587698010980842e-10, 0.0]
 - Iterations: 26
 - Time: 0.004351139068603516 seconds
 - Exit flag: 3
```
"""
function optimize!(opt::AbstractOptimizer)
    # Initialize the optimizer
    initialize!(opt)

    # Perform iterations
    return iterate!(opt)
end

"""
    initialize!(opt::AbstractOptimizer)

Initialize the optimizer `opt`. All memory allocations that are not possible to do in the
constructor should be done here when possible.
"""
function initialize!(opt::AbstractOptimizer)
    # Initialize the optimizer
    throw(ArgumentError("initialize! not implemented for $(typeof(opt))."))
end

"""
    iterate!(opt::AbstractOptimizer)

Perform iterations to optimize the problem and returns the results of the optimization.
"""
function iterate!(opt::AbstractOptimizer)
    # Perform iterations
    throw(ArgumentError("iterate! not implemented for $(typeof(opt))."))
end

"""
    get_iteration(opt::AbstractOptimizer)

Get the current iteration number of the optimizer `opt`.
"""
function get_iteration(opt::AbstractOptimizer)
    throw(ArgumentError("get_iteration not implemented for $(typeof(opt))."))
end
