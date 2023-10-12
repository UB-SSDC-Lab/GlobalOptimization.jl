"""
    AbstractOptimizationProblem

Abstract type for optimization problems.
"""
abstract type AbstractOptimizationProblem end

"""
    OptimizationProblem

An optimization problem. Contains the objective function and search space.
"""
struct OptimizationProblem{F <: Function, SS <: SearchSpace} <: AbstractOptimizationProblem
    f::F    # Objective function
    ss::SS  # Search space

    function OptimizationProblem(f::F, ss::SS) where {F <: Function, SS <: SearchSpace}
        return new{F,SS}(f, ss)
    end
end

"""
    OptimizationProblem(f, LB, UB)

Constructs an optimization problem with objective function `f` and a 
`ContinuousRectangularSearchSpace` defined by `LB` and `UB`.
"""
function OptimizationProblem(
    f::F, LB::AbstractVector{<:Real}, UB::AbstractVector{<:Real},
) where F <: Function
    return OptimizationProblem(
        f, ContinuousRectangularSearchSpace(LB, UB),
    )
end

"""
    evaluate(prob::OptimizationProblem, x::AbstractArray)

Evaluates the objective function `f` of the optimization problem `prob` at `x`.
"""
function evaluate(prob::OptimizationProblem, x::AbstractArray)
    return prob.f(x)
end