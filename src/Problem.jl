"""
    AbstractOptimizationProblem

Abstract type for optimization problems.
"""
abstract type AbstractOptimizationProblem{SS} end

"""
    OptimizationProblem

An optimization problem. Contains the objective function and search space.
"""
struct OptimizationProblem{SS <: SearchSpace, F <: Function} <: AbstractOptimizationProblem{SS}
    f::F    # Objective function
    ss::SS  # Search space

    function OptimizationProblem(f::F, ss::SS) where {F <: Function, SS <: SearchSpace}
        return new{SS,F}(f, ss)
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
    search_space(prob::OptimizationProblem)

Returns the search space of the optimization problem `prob`.
"""
search_space(prob::OptimizationProblem) = prob.ss

"""
    numdims(prob::OptimizationProblem)

Returns the number of dimensions of the decision vector of the optimization problem `prob`.
"""
numdims(prob::OptimizationProblem) = numdims(search_space(prob))

"""
    evaluate(prob::OptimizationProblem, x::AbstractArray)

Evaluates the objective function `f` of the optimization problem `prob` at `x`.
"""
function evaluate(prob::OptimizationProblem, x::AbstractArray)
    return prob.f(x)
end