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
- `Results`: The results of the optimization.
"""
optimize!(opt::AbstractOptimizer) = throw(NotImplementedError("optimize! not implemented for $(typeof(opt))."))