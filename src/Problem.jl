"""
    AbstractProblem

Abstract type for all solvable problems.
"""
abstract type AbstractProblem{has_penalty,SS} end

"""
    AbstractOptimizationProblem

Abstract type for optimization problems.
"""
abstract type AbstractOptimizationProblem{has_penalty,SS} <: AbstractProblem{has_penalty,SS} end

"""
    AbstractNonlinearEquationProblem

Abstract type for problems involving a set of nonlinear equations to solve.
"""
abstract type AbstractNonlinearEquationProblem{has_penalty,SS} <: AbstractProblem{has_penalty,SS} end

"""
    OptimizationProblem{has_penalty, SS, F, G}

An optimization problem. Contains the objective function and search space.

# Fields
- `f::F`: The objective function.
- `g!::G`: The gradient of the objective function.
- `ss::SS`: The search space.
"""
struct OptimizationProblem{has_penalty, SS <: SearchSpace, F <: Function, G <: Union{Nothing,Function}} <: AbstractOptimizationProblem{has_penalty,SS}
    f::F    # Objective function
    g!::G    # Gradient of the objective function
    ss::SS  # Search space

    @doc """
        OptimizationProblem{has_penalty}(f::F, [g::G], ss::SS)

    Constructs an optimization problem with objective function `f`, optional gradient `g`, and search space `ss`.
    If has_penalty is specified as true, then the objective function must return a Tuple{T,T} for a given x of
    type AbstractArray{T}.

    # Arguments
    - `f::F`: The objective function.
    - `g::G`: The gradient of the objective function.
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
    function OptimizationProblem{has_penalty}(
        f::F, LB::AbstractArray{<:Real}, UB::AbstractArray{<:Real},
    ) where {has_penalty, F <: Function}
        return new{has_penalty,SS,F,Nothing}(f, nothing, ContinuousRectangularSearchSpace(LB, UB))
    end
    function OptimizationProblem{has_penalty}(
        f::F, g::G, LB::AbstractArray{<:Real}, UB::AbstractArray{<:Real},
    ) where {has_penalty, F <: Function, G <: Function}
        return new{has_penalty,SS,F,G}(f, g, ContinuousRectangularSearchSpace(LB, UB))
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
    has_penalty = Base.return_types(f)[1] <: Tuple ? Val(true) : Val(false)
    return OptimizationProblem{has_penalty}(f, LB, UB)
end
function OptimizationProblem(
    f::F, g::G, LB::AbstractVector{<:Real}, UB::AbstractVector{<:Real},
) where {F <: Function, G <: Function}
    has_penalty = Base.return_types(f)[1] <: Tuple ? Val(true) : Val(false)
    return OptimizationProblem{has_penalty}(f, g, LB, UB)
end

struct NonlinearProblem{has_penalty, SS <: SearchSpace, F <: Function, G <: Union{Nothing,Function}} <: AbstractNonlinearEquationProblem{has_penalty,SS}
    f::F   # The nonlinear equations
    g!::G  # The jacobian of the nonlinear equations
    ss::SS # The search space

    function NonlinearProblem{has_penalty}(f::F, ss::SS) where {has_penalty, F <: Function, SS <: SearchSpace}
        return new{has_penalty,SS,F,Nothing}(f, nothing, ss)
    end
    function NonlinearProblem{has_penalty}(f::F, g::G, ss::SS) where {has_penalty, F <: Function, G <: Function, SS <: SearchSpace}
        return new{has_penalty,SS,F,G}(f, g, ss)
    end
    function NonlinearProblem{has_penalty}(
        f::F, LB::AbstractArray{<:Real}, UB::AbstractArray{<:Real},
    ) where {has_penalty, F <: Function}
        return new{has_penalty,SS,F,Nothing}(f, nothing, ContinuousRectangularSearchSpace(LB, UB))
    end
    function NonlinearProblem{has_penalty}(
        f::F, g::G, LB::AbstractArray{<:Real}, UB::AbstractArray{<:Real},
    ) where {has_penalty, F <: Function, G <: Function}
        return new{has_penalty,SS,F,G}(f, g, ContinuousRectangularSearchSpace(LB, UB))
    end
end

function NonlinearProblem(f::F, ss::SS) where {F <: Function, SS <: SearchSpace}
    has_penalty = Base.return_types(f)[1] <: Tuple ? Val(true) : Val(false)
    return NonlinearProblem{has_penalty}(f, ss)
end
function NonlinearProblem(f::F, g::G, ss::SS) where {F <: Function, G <: Function, SS <: SearchSpace}
    has_penalty = Base.return_types(f)[1] <: Tuple ? Val(true) : Val(false)
    return NonlinearProblem{has_penalty}(f, g, ss)
end
function NonlinearProblem(
    f::F, LB::AbstractVector{<:Real}, UB::AbstractVector{<:Real},
) where F <: Function
    has_penalty = Base.return_types(f)[1] <: Tuple ? Val(true) : Val(false)
    return NonlinearProblem{has_penalty}(f, LB, UB)
end
function NonlinearProblem(
    f::F, g::G, LB::AbstractVector{<:Real}, UB::AbstractVector{<:Real},
) where {F <: Function, G <: Function}
    has_penalty = Base.return_types(f)[1] <: Tuple ? Val(true) : Val(false)
    return NonlinearProblem{has_penalty}(f, g, LB, UB)
end

struct NonlinearLeastSquaresProblem{has_penalty, SS <: SearchSpace, F <: Function, G <: Union{Nothing,Function}} <: AbstractNonlinearEquationProblem{has_penalty,SS}
    f::F   # The nonlinear equations
    g!::G  # The jacobian of the nonlinear equations
    ss::SS # The search space

    # For nonlinear least squres, we also need to know the number of residuals
    n::Int # The number of residuals

    function NonlinearLeastSquaresProblem{has_penalty}(f::F, ss::SS, num_resid::Int) where {has_penalty, F <: Function, SS <: SearchSpace}
        return new{has_penalty,SS,F,Nothing}(f, nothing, ss, num_resid)
    end
    function NonlinearLeastSquaresProblem{has_penalty}(f::F, g::G, ss::SS, num_resid::Int) where {has_penalty, F <: Function, G <: Function, SS <: SearchSpace}
        return new{has_penalty,SS,F,G}(f, g, ss, num_resid)
    end
    function NonlinearLeastSquaresProblem{has_penalty}(
        f::F, LB::AbstractArray{<:Real}, UB::AbstractArray{<:Real}, num_resid::Int,
    ) where {has_penalty, F <: Function}
        return new{has_penalty,SS,F,Nothing}(f, nothing, ContinuousRectangularSearchSpace(LB, UB), num_resid)
    end
    function NonlinearLeastSquaresProblem{has_penalty}(
        f::F, g::G, LB::AbstractArray{<:Real}, UB::AbstractArray{<:Real}, num_resid::Int,
    ) where {has_penalty, F <: Function, G <: Function}
        return new{has_penalty,SS,F,G}(f, g, ContinuousRectangularSearchSpace(LB, UB), num_resid)
    end
end

function NonlinearLeastSquaresProblem(f::F, ss::SS, num_resid::Int) where {F <: Function, SS <: SearchSpace}
    has_penalty = Base.return_types(f)[1] <: Tuple ? Val(true) : Val(false)
    return NonlinearLeastSquaresProblem{has_penalty}(f, ss, num_resid)
end
function NonlinearLeastSquaresProblem(f::F, g::G, ss::SS, num_resid::Int) where {F <: Function, G <: Function, SS <: SearchSpace}
    has_penalty = Base.return_types(f)[1] <: Tuple ? Val(true) : Val(false)
    return NonlinearLeastSquaresProblem{has_penalty}(f, g, ss, num_resid)
end
function NonlinearLeastSquaresProblem(
    f::F, LB::AbstractVector{<:Real}, UB::AbstractVector{<:Real}, num_resid::Int,
) where F <: Function
    has_penalty = Base.return_types(f)[1] <: Tuple ? Val(true) : Val(false)
    return NonlinearLeastSquaresProblem{has_penalty}(f, LB, UB, num_resid)
end
function NonlinearLeastSquaresProblem(
    f::F, g::G, LB::AbstractVector{<:Real}, UB::AbstractVector{<:Real}, num_resid::Int,
) where {F <: Function, G <: Function}
    has_penalty = Base.return_types(f)[1] <: Tuple ? Val(true) : Val(false)
    return NonlinearLeastSquaresProblem{has_penalty}(f, g, LB, UB, num_resid)
end

"""
    search_space(prob::AbstractProblem)

Returns the search space of the optimization problem `prob`.
"""
search_space(prob::AbstractProblem) = prob.ss

"""
    numdims(prob::AbstractProblem)

Returns the number of dimensions of the decision vector of the optimization problem `prob`.
"""
numdims(prob::AbstractProblem) = numdims(search_space(prob))

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
    scalar_function(prob::AbstractNonlinearEquationProblem, x::AbstractArray)

Evaluates the set of nonlinear equations `f` and returns the nonlinear least squares cost 
plus half the infeasibility squared.
"""
@inline function scalar_function(
    prob::AbstractNonlinearEquationProblem{has_penalty},
    x::AbstractArray,
) where has_penalty
    return scalar_function(prob, x, has_penalty)
end
@inline function scalar_function(prob::AbstractNonlinearEquationProblem, x::AbstractArray, ::Val{true})
    f, g = prob.f(x)
    return 0.5*(f'*f + g*g)
end
@inline function scalar_function(prob::AbstractNonlinearEquationProblem, x::AbstractArray, ::Val{false})
    f = prob.f(x)
    return 0.5*f'*f
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
    scalar_function_with_penalty(prob::AbstractNonlinearEquationProblem, x::AbstractArray)

Evaluates the set of nonlinear equations `f` and returns the nonlinear least squares cost 
and the infeasibility penalty term as a tuple.
"""
@inline function scalar_function_with_penalty(
    prob::AbstractNonlinearEquationProblem{has_penalty},
    x::AbstractArray,
) where has_penalty
    return scalar_function_with_penalty(prob, x, has_penalty)
end
@inline function scalar_function_with_penalty(prob::AbstractNonlinearEquationProblem, x::AbstractArray, ::Val{true})
    f, g = prob.f(x)
    cost = 0.5*f'*f
    return cost, g
end
@inline function scalar_function_with_penalty(prob::AbstractNonlinearEquationProblem, x::AbstractArray, ::Val{false})
    f = prob.f(x)
    cost = 0.5*f'*f
    penalty = zero(typeof(cost))
    return cost, penalty
end

"""
    get_scalar_function(prob::AbstractProblem)

    Returns cost function plus the infeasibility penalty squared as a scalar value.
    This is used for PSO (GA, Differential Evolution, etc. if we ever get around to adding those)
"""
@inline function get_scalar_function(prob::AbstractProblem)
    return x -> scalar_function(prob, x)
end

"""
    get_scalar_function_with_penalty(prob::AbstractProblem)
"""
@inline function get_scalar_function_with_penalty(prob::AbstractProblem)
    return x -> scalar_function_with_penalty(prob, x)
end
