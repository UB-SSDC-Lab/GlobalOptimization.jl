"""
    AbstractOptimizer

Abstract type of all optimization algorithms.
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
    throw(ArgumentError("optimize! not implemented for $(typeof(opt))."))
end
