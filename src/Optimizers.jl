"""
    AbstractOptimizer

Abstract type of all optimization algorithms.
"""
abstract type AbstractOptimizer end

# ===== Interface
optimize!(opt::AbstractOptimizer) = throw(NotImplementedError("optimize! not implemented for $(typeof(opt))."))