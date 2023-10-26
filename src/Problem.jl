"""
    AbstractOptimizationProblem

Abstract type for optimization problems.
"""
abstract type AbstractOptimizationProblem{SS} end

"""
    OptimizationProblem{SS, F}

An optimization problem. Contains the objective function and search space.

# Fields
- `f::F`: The objective function.
- `ss::SS`: The search space.
"""
struct OptimizationProblem{SS <: SearchSpace, F <: Function} <: AbstractOptimizationProblem{SS}
    f::F    # Objective function
    ss::SS  # Search space

    @doc """
        OptimizationProblem(f::F, ss::SS)

    Constructs an optimization problem with objective function `f` and search space `ss`.

    # Arguments
    - `f::F`: The objective function.
    - `ss::SS`: The search space.

    # Returns
    - `OptimizationProblem{SS, F}`

    # Examples
    ```julia-repl
    julia> using GlobalOptimization;
    julia> f(x) = sum(x.^2); # Simple sphere function
    julia> LB = [-1.0, 0.0];
    julia> UB = [ 1.0, 2.0];
    julia> ss = ContinuousRectangularSearchSpace(LB, UB);
    julia> prob = OptimizationProblem(f, ss)
    OptimizationProblem{ContinuousRectangularSearchSpace{Float64}, typeof(f)}(f, ContinuousRectangularSearchSpace{Float64}([-1.0, 0.0], [1.0, 2.0], [2.0, 2.0]))
    ```
    """
    function OptimizationProblem(f::F, ss::SS) where {F <: Function, SS <: SearchSpace}
        return new{SS,F}(f, ss)
    end
end

"""
    OptimizationProblem(f, LB, UB)

Constructs an optimization problem with objective function `f` and a 
`ContinuousRectangularSearchSpace` defined by `LB` and `UB`.

# Arguments
- `f::F`: The objective function.
- `LB::AbstractVector{<:Real}`: The lower bounds of the search space.
- `UB::AbstractVector{<:Real}`: The upper bounds of the search space.

# Returns
- `OptimizationProblem{ContinuousRectangularSearchSpace, F}`

# Examples
```julia-repl
julia> using GlobalOptimization;
julia> f(x) = sum(x.^2); # Simple sphere function
julia> LB = [-1.0, 0.0];
julia> UB = [ 1.0, 2.0];
julia> prob = OptimizationProblem(f, LB, UB)
OptimizationProblem{ContinuousRectangularSearchSpace{Float64}, typeof(f)}(f, ContinuousRectangularSearchSpace{Float64}([-1.0, 0.0], [1.0, 2.0], [2.0, 2.0]))
```
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