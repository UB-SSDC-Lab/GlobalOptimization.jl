"""
    AbstractProblem

Abstract type for all solvable problems.
"""
abstract type AbstractProblem{SS} end

"""
    AbstractOptimizationProblem

Abstract type for optimization problems.
"""
abstract type AbstractOptimizationProblem{SS} <: AbstractProblem{SS} end

"""
    OptimizationProblem{SS, F}

An optimization problem. Contains the objective function and search space.

# Fields
- `f::F`: The objective function.
- `ss::SS`: The search space.
"""
struct OptimizationProblem{has_penalty, SS <: SearchSpace, F <: Function, G <: Union{Nothing,Function}} <: AbstractOptimizationProblem{SS}
    f::F    # Objective function
    g!::G    # Gradient of the objective function
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
    function OptimizationProblem{has_penalty}(f::F, ss::SS) where {has_penalty, F <: Function, SS <: SearchSpace}
        return new{has_penalty,SS,F,Nothing}(f, nothing, ss)
    end
    function OptimizationProblem{has_penalty}(f::F, g::G, ss::SS) where {has_penalty, F <: Function, G <: Function, SS <: SearchSpace}
        return new{has_penalty,SS,F,G}(f, g, ss)
    end
end

"""
    OptimizationProblem(f, [g], ss)

Constructs an optimization problem with objective function `f`, optimal gradient `g`, and a search space.

# Arguments
- `f::F`: The objective function.
- `g::G`: The gradient of the objective function.
- `ss::SS`: The search space.

# Returns
- `OptimizationProblem{ContinuousRectangularSearchSpace, F}`

# Examples
```julia-repl
julia> using GlobalOptimization;
julia> f(x) = sum(x.^2); # Simple sphere function
julia> LB = [-1.0, 0.0];
julia> UB = [ 1.0, 2.0];
julia> prob = OptimizationProblem(f, ContinuousRectangularSearchSpace(LB, UB))
OptimizationProblem{ContinuousRectangularSearchSpace{Float64}, typeof(f)}(f, ContinuousRectangularSearchSpace{Float64}([-1.0, 0.0], [1.0, 2.0], [2.0, 2.0]))
```
"""
function OptimizationProblem(f::F, ss::SS) where {F <: Function, SS <: SearchSpace}
    has_penalty = Base.return_types(f)[1] <: Tuple ? Val(true) : Val(false)
    return OptimizationProblem{has_penalty}(f, ss)
end
function OptimizationProblem(f::F, g::G, ss::SS) where {F <: Function, G <: Function, SS <: SearchSpace}
    has_penalty = Base.return_types(f)[1] <: Tuple ? Val(true) : Val(false)
    return OptimizationProblem{has_penalty}(f, g, ss)
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
    scalar_function(prob::OptimizationProblem, x::AbstractArray)

Evaluates the objective function `f` of the optimization problem `prob` at `x`
and returns the cost function plus half the infeasibility squared.
"""
@inline function scalar_function(
    prob::OptimizationProblem{has_penalty,SS,F,G}, 
    x::AbstractArray,
) where {has_penalty, SS, F, G}
    return scalar_function(prob, x, has_penalty)
end
@inline function scalar_function(prob::OptimizationProblem, x::AbstractArray, ::Val{true})
    f, g = prob.f(x)
    return f + 0.5*g*g
end
@inline function scalar_function(prob::OptimizationProblem, x::AbstractArray, ::Val{false})
    return prob.f(x)
end

"""
    scalar_function_with_penalty(prob::OptimizationProblem, x::AbstractArray)

Evaluates the objective function `f` of the optimization problem `prob` at `x`
and returns the cost function and the infeasibility penalty term as tuple.
i.e., for an OptimizationProblem, this is simply the original function.
"""
@inline function scalar_function_with_penalty(
    prob::OptimizationProblem{has_penalty,SS,F,G},
    x::AbstractArray,
) where {has_penalty, SS, F, G}
    return scalar_function_with_penalty(prob, x, has_penalty)
end
@inline function scalar_function_with_penalty(prob::OptimizationProblem, x::AbstractArray, ::Val{true}) 
    return prob.f(x)
end
@inline function scalar_function_with_penalty(prob::OptimizationProblem, x::AbstractArray, ::Val{false})
    val = prob.f(x)
    penalty = zero(eltype(val)) 
    return val, penalty
end


"""
    get_scalar_function(prob::OptimizationProblem)

    Returns cost function plus the infeasibility penalty squared as a scalar value.
    This is used for PSO (GA, Differential Evolution, etc. if we ever get around to adding those)
"""
@inline function get_scalar_function(prob::OptimizationProblem)
    return x -> scalar_function(prob, x)
end

"""
    get_scalar_function_with_penalty(prob::OptimizationProblem)
"""
@inline function get_scalar_function_with_penalty(prob::OptimizationProblem)
    return x -> scalar_function_with_penalty(prob, x)
end
