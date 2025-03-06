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
struct OptimizationProblem{has_penalty, SS <: SearchSpace, F, G} <: AbstractOptimizationProblem{has_penalty,SS}
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
    function OptimizationProblem{has_penalty}(f::F, ss::SearchSpace{T}) where {T, has_penalty, F <: Function}
        fargtypes = (Tuple{Vector{T}},)
        frettypes = has_penalty isa Val{false} ? (T,) : (Tuple{T,T},)
        fwrap = FunctionWrappersWrapper(f, fargtypes, frettypes)
        return new{has_penalty,typeof(ss),typeof(fwrap),Nothing}(fwrap, nothing, ss)
    end
    function OptimizationProblem{has_penalty}(
        f::F, g::G, ss::SearchSpace{T},
    ) where {T, has_penalty, F <: Function, G <: Function}
        fargtypes = (Tuple{Vector{T}},)
        frettypes = has_penalty isa Val{false} ? (T,) : (Tuple{T,T},)
        fwrap = FunctionWrappersWrapper(f, fargtypes, frettypes)

        gargtypes = (Tuple{Vector{T},Vector{T}},)
        grettypes = (Nothing)
        gwrap = FunctionWrappersWrapper(g, gargtypes, grettypes)
        return new{has_penalty,typeof(ss),typeof(fwrap),typeof(gwrap)}(
            fwrap, gwrap, ss,
        )
    end
    function OptimizationProblem{has_penalty}(
        f::F, LB::AbstractArray{T}, UB::AbstractArray{T},
    ) where {has_penalty, F <: Function, T <: Real}
        ss = ContinuousRectangularSearchSpace(LB, UB)

        fargtypes = (Tuple{Vector{T}},)
        frettypes = has_penalty isa Val{false} ? (T,) : (Tuple{T,T},)
        fwrap = FunctionWrappersWrapper(f, fargtypes, frettypes)

        return new{has_penalty,typeof(ss),typeof(fwrap),Nothing}(fwrap, nothing, ss)
    end
    function OptimizationProblem{has_penalty}(
        f::F, g::G, LB::AbstractArray{T}, UB::AbstractArray{T},
    ) where {has_penalty, F <: Function, G <: Function, T <: Real}
        ss = ContinuousRectangularSearchSpace(LB, UB)

        fargtypes = (Tuple{Vector{T}},)
        frettypes = has_penalty isa Val{false} ? (T,) : (Tuple{T,T},)
        fwrap = FunctionWrappersWrapper(f, fargtypes, frettypes)

        gargtypes = (Tuple{Vector{T},Vector{T}},)
        grettypes = (Nothing)
        gwrap = FunctionWrappersWrapper(g, gargtypes, grettypes)

        return new{has_penalty,typeof(ss),typeof(fwrap),typeof(gwrap)}(
            fwrap, gwrap, ContinuousRectangularSearchSpace(LB, UB),
        )
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
    OptimizationProblem(f, [g], LB, UB)

Constructs an optimization problem with objective function `f`,
optional gradient `g`, and a `ContinuousRectangularSearchSpace`
defined by `LB` and `UB`.

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

"""
    NonlinearProblem{has_penalty, SS, F, G}

A nonlinear problem. Contains the nonlinear equations and search space.

# Fields
- `f::F`: The nonlinear equations.
- `g!::G`: The jacobian of the nonlinear equations.
- `ss::SS`: The search space.
"""
struct NonlinearProblem{has_penalty, SS <: SearchSpace, F, G} <: AbstractNonlinearEquationProblem{has_penalty,SS}
    f::F   # The nonlinear equations
    g!::G  # The jacobian of the nonlinear equations
    ss::SS # The search space

    @doc """
        NonlinearProblem{has_penalty}(f::F, [g::G], ss::SS)

    Constructs a nonlinear problem with nonlinear functions `f`, optional jacobian `g`, and search space `ss`.
    If has_penalty is specified as true, then the nonlinear function must return a Tuple{AbstractArray{T},T}
    for a given x of type AbstractArray{T}.

    # Arguments
    - `f::F`: The nonlinear function.
    - `g::G`: The Jacobian of the nonlinear function.
    - `ss::SS`: The search space.

    # Returns
    - `NonlinearProblem{has_penalty, SS, F, G}`

    # Examples
    ```julia-repl
    julia> using GlobalOptimization;
    julia> f(x) = [x[1] - 2.0, x[2] - 2.0]
    julia> LB = [-5.0, -5.0];
    julia> UB = [ 5.0, 5.0];
    julia> ss = ContinuousRectangularSearchSpace(LB, UB);
    julia> prob = NonlinearProblem(f, ss)
    NonlinearProblem{Val{false}(), ContinuousRectangularSearchSpace{Float64}, typeof(f), Nothing}(f, nothing, ContinuousRectangularSearchSpace{Float64}([-5.0, -5.0], [5.0, 5.0], [10.0, 10.0]))
    ```
    """
    function NonlinearProblem{has_penalty}(
        f::F, ss::SearchSpace{T},
    ) where {T, has_penalty, F <: Function}
        fargtypes = (Tuple{Vector{T}},)
        frettypes = has_penalty isa Val{false} ? (Tuple{Vector{T}},) : (Tuple{Vector{T},T},)
        fwrap = FunctionWrappersWrapper(f, fargtypes, frettypes)

        return new{has_penalty,typeof(ss),typeof(fwrap),Nothing}(fwrap, nothing, ss)
    end
    function NonlinearProblem{has_penalty}(
        f::F, g::G, ss::SearchSpace{T},
    ) where {T, has_penalty, F <: Function, G <: Function}
        fargtypes = (Tuple{Vector{T}},)
        frettypes = has_penalty isa Val{false} ? (Tuple{Vector{T}},) : (Tuple{Vector{T},T},)
        fwrap = FunctionWrappersWrapper(f, fargtypes, frettypes)

        gargtypes = (Tuple{Matrix{T}, Vector{T}},)
        grettypes = (Nothing,)
        gwrap = FunctionWrappersWrapper(g, gargtypes, grettypes)

        return new{has_penalty,typeof(ss),typeof(fwrap),typeof(gwrap)}(
            fwrap, gwrap, ss,
        )
    end
    function NonlinearProblem{has_penalty}(
        f::F, LB::AbstractArray{T}, UB::AbstractArray{T},
    ) where {has_penalty, F <: Function, T <: Real}
        ss = ContinuousRectangularSearchSpace(LB, UB)

        fargtypes = (Tuple{Vector{T}},)
        frettypes = has_penalty isa Val{false} ? (Tuple{Vector{T}},) : (Tuple{Vector{T},T},)
        fwrap = FunctionWrappersWrapper(f, fargtypes, frettypes)

        return new{has_penalty,typeof(ss),typeof(fwrap),Nothing}(
            fwrap, nothing, ss,
        )
    end
    function NonlinearProblem{has_penalty}(
        f::F, g::G, LB::AbstractArray{T}, UB::AbstractArray{T},
    ) where {has_penalty, F <: Function, G <: Function, T <: Real}
        ss = ContinuousRectangularSearchSpace(LB, UB)

        fargtypes = (Tuple{Vector{T}},)
        frettypes = has_penalty isa Val{false} ? (Tuple{Vector{T}},) : (Tuple{Vector{T},T},)
        fwrap = FunctionWrappersWrapper(f, fargtypes, frettypes)

        gargtypes = (Tuple{Matrix{T}, Vector{T}},)
        grettypes = (Nothing,)
        gwrap = FunctionWrappersWrapper(g, gargtypes, grettypes)

        return new{has_penalty,typeof(ss),typeof(fwrap),typeof(gwrap)}(
            fwrap, gwrap, ss,
        )
    end
end

"""
    NonlinearProblem(f, [g], ss)

Constructs a nonlinear problem with nonlinear function `f`, optional Jacobian `g`, and a search space.

# Arguments
- `f::F`: The nonlinear function.
- `g::G`: The Jacobian of the nonlinear function.
- `ss::SS`: The search space.

# Returns
- `NonlinearProblem{has_penalty, ContinuousRectangularSearchSpace, F, G}`

# Examples
```julia-repl
julia> using GlobalOptimization;
julia> f(x) = [x[1] - 2.0, x[2] - 2.0]
julia> LB = [-5.0, -5.0];
julia> UB = [ 5.0, 5.0];
julia> ss = ContinuousRectangularSearchSpace(LB, UB);
julia> prob = NonlinearProblem(f, ss)
NonlinearProblem{Val{false}(), ContinuousRectangularSearchSpace{Float64}, typeof(f), Nothing}(f, nothing, ContinuousRectangularSearchSpace{Float64}([-5.0, -5.0], [5.0, 5.0], [10.0, 10.0]))
```
"""
function NonlinearProblem(f::F, ss::SS) where {F <: Function, SS <: SearchSpace}
    has_penalty = Base.return_types(f)[1] <: Tuple ? Val(true) : Val(false)
    return NonlinearProblem{has_penalty}(f, ss)
end
function NonlinearProblem(f::F, g::G, ss::SS) where {F <: Function, G <: Function, SS <: SearchSpace}
    has_penalty = Base.return_types(f)[1] <: Tuple ? Val(true) : Val(false)
    return NonlinearProblem{has_penalty}(f, g, ss)
end

"""
    NonlinearProblem(f, [g], LB, UB)

Constructs a nonlinear problem with nonlinear function `f`, optional Jacobian `g`, and a
continuous rectangular search space defined by the bounds LB and UB.

# Arguments
- `f::F`: The nonlinear function.
- `g::G`: The Jacobian of the nonlinear function.
- `LB::AbstractVector{<:Real}`: The lower bounds of the search space.
- `UB::AbstractVector{<:Real}`: The upper bounds of the search space.

# Returns
- `NonlinearProblem{has_penalty, ContinuousRectangularSearchSpace, F, G}`

# Examples
```julia-repl
julia> using GlobalOptimization;
julia> f(x) = [x[1] - 2.0, x[2] - 2.0]
julia> LB = [-5.0, -5.0];
julia> UB = [ 5.0, 5.0];
julia> prob = NonlinearProblem(f, LB, UB)
NonlinearProblem{Val{false}(), ContinuousRectangularSearchSpace{Float64}, typeof(f), Nothing}(f, nothing, ContinuousRectangularSearchSpace{Float64}([-5.0, -5.0], [5.0, 5.0], [10.0, 10.0]))
```
"""
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

"""
    NonlinearLeastSquaresProblem{has_penalty, SS, F, G}

A nonlinear least squares problem. Contains the nonlinear equations and search space.

# Fields
- `f::F`: The nonlinear equations.
- `g!::G`: The jacobian of the nonlinear equations.
- `ss::SS`: The search space.
- `n::Int`: The number of residuals.
"""
struct NonlinearLeastSquaresProblem{has_penalty, SS <: SearchSpace, F <: Function, G <: Union{Nothing,Function}} <: AbstractNonlinearEquationProblem{has_penalty,SS}
    f::F   # The nonlinear equations
    g!::G  # The jacobian of the nonlinear equations
    ss::SS # The search space

    # For nonlinear least squres, we also need to know the number of residuals
    n::Int # The number of residuals

    @doc """
        NonlinearLeastSquaresProblem{has_penalty}(f::F, [g::G], ss::SS, num_resid::Int)

    Constructs a nonlinear least squares problem with nonlinear functions `f`, optional jacobian `g`,
    and search space `ss`. If has_penalty is specified as true, then the nonlinear function must return
    a Tuple{AbstractArray{T},T} for a given x of type AbstractArray{T}.

    # Arguments
    - `f::F`: The nonlinear function.
    - `g::G`: The Jacobian of the nonlinear function.
    - `ss::SS`: The search space.
    - `num_resid::Int`: The number of residuals.

    # Returns
    - `NonlinearLeastSquaresProblem{has_penalty, SS, F, G}`

    # Examples
    ```julia-repl
    julia> using GlobalOptimization;
    julia> f(x) = [x[1] - x[3], x[2] - x[3]]
    julia> LB = [-5.0, -5.0, -5.0];
    julia> UB = [ 5.0, 5.0, 5.0];
    julia> ss = ContinuousRectangularSearchSpace(LB, UB);
    julia> prob = NonlinearLeastSquaresProblem(f, ss, 2)
    NonlinearLeastSquaresProblem{Val{false}(), ContinuousRectangularSearchSpace{Float64}, typeof(f), Nothing}(f, nothing, ContinuousRectangularSearchSpace{Float64}([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], [10.0, 10.0, 10.0]), 2)
    ```
    """
    function NonlinearLeastSquaresProblem{has_penalty}(
        f::F, ss::SearchSpace{T}, num_resid::Int,
    ) where {T, has_penalty, F <: Function}
        fargtypes = (Tuple{Vector{T}},)
        frettypes = has_penalty isa Val{false} ? (Tuple{Vector{T}},) : (Tuple{Vector{T},T},)
        fwrap = FunctionWrappersWrapper(f, fargtypes, frettypes)

        return new{has_penalty,typeof(ss),typeof(fwrap),Nothing}(fwrap, nothing, ss, num_resid)
    end
    function NonlinearLeastSquaresProblem{has_penalty}(
        f::F, g::G, ss::SearchSpace{T}, num_resid::Int,
    ) where {T, has_penalty, F <: Function, G <: Function}
        fargtypes = (Tuple{Vector{T}},)
        frettypes = has_penalty isa Val{false} ? (Tuple{Vector{T}},) : (Tuple{Vector{T},T},)
        fwrap = FunctionWrappersWrapper(f, fargtypes, frettypes)

        gargtypes = (Tuple{Matrix{T}, Vector{T}},)
        grettypes = (Nothing,)
        gwrap = FunctionWrappersWrapper(g, gargtypes, grettypes)

        return new{has_penalty,typeof(ss),typeof(fwrap),typeof(gwrap)}(
            fwrap, gwrap, ss, num_resid,
        )
    end
    function NonlinearLeastSquaresProblem{has_penalty}(
        f::F, LB::AbstractArray{T}, UB::AbstractArray{T}, num_resid::Int,
    ) where {T, has_penalty, F <: Function}
        ss = ContinuousRectangularSearchSpace(LB, UB)

        fargtypes = (Tuple{Vector{T}},)
        frettypes = has_penalty isa Val{false} ? (Tuple{Vector{T}},) : (Tuple{Vector{T},T},)
        fwrap = FunctionWrappersWrapper(f, fargtypes, frettypes)

        return new{has_penalty,typeof(ss),typeof(fwrap),Nothing}(
            fwrap, nothing, ss, num_resid,
        )
    end
    function NonlinearLeastSquaresProblem{has_penalty}(
        f::F, g::G, LB::AbstractArray{T}, UB::AbstractArray{T}, num_resid::Int,
    ) where {T, has_penalty, F <: Function, G <: Function}
        ss = ContinuousRectangularSearchSpace(LB, UB)

        fargtypes = (Tuple{Vector{T}},)
        frettypes = has_penalty isa Val{false} ? (Tuple{Vector{T}},) : (Tuple{Vector{T},T},)
        fwrap = FunctionWrappersWrapper(f, fargtypes, frettypes)

        gargtypes = (Tuple{Matrix{T}, Vector{T}},)
        grettypes = (Nothing,)
        gwrap = FunctionWrappersWrapper(g, gargtypes, grettypes)

        return new{has_penalty,typeof(ss),typeof(fwrap),typeof(gwrap)}(
            f, g, ss, num_resid,
        )
    end
end

"""
    NonlinearLeastSquaresProblem(f, [g], ss)

Constructs a nonlinear least squares problem with nonlinear function `f`, optional Jacobian `g`, and a search space.

# Arguments
- `f::F`: The nonlinear function.
- `g::G`: The Jacobian of the nonlinear function.
- `ss::SS`: The search space.

# Returns
- `NonlinearLeastSquaresProblem{has_penalty, ContinuousRectangularSearchSpace, F, G}`

# Examples
```julia-repl
julia> using GlobalOptimization;
julia> f(x) = [x[1] - x[3], x[2] - x[3]]
julia> LB = [-5.0, -5.0, -5.0];
julia> UB = [ 5.0, 5.0, 5.0];
julia> ss = ContinuousRectangularSearchSpace(LB, UB);
julia> prob = NonlinearLeastSquaresProblem(f, ss, 2)
NonlinearLeastSquaresProblem{Val{false}(), ContinuousRectangularSearchSpace{Float64}, typeof(f), Nothing}(f, nothing, ContinuousRectangularSearchSpace{Float64}([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], [10.0, 10.0, 10.0]), 2)
```
"""
function NonlinearLeastSquaresProblem(f::F, ss::SS, num_resid::Int) where {F <: Function, SS <: SearchSpace}
    has_penalty = Base.return_types(f)[1] <: Tuple ? Val(true) : Val(false)
    return NonlinearLeastSquaresProblem{has_penalty}(f, ss, num_resid)
end
function NonlinearLeastSquaresProblem(f::F, g::G, ss::SS, num_resid::Int) where {F <: Function, G <: Function, SS <: SearchSpace}
    has_penalty = Base.return_types(f)[1] <: Tuple ? Val(true) : Val(false)
    return NonlinearLeastSquaresProblem{has_penalty}(f, g, ss, num_resid)
end

"""
    NonlinearLeastSquaresProblem(f, [g], LB, UB)

Constructs a nonlinear least squares problem with nonlinear function `f`, optional Jacobian `g`, and a search space.

# Arguments
- `f::F`: The nonlinear function.
- `g::G`: The Jacobian of the nonlinear function.
- `LB::AbstractVector{<:Real}`: The lower bounds of the search space.
- `UB::AbstractVector{<:Real}`: The upper bounds of the search space.

# Returns
- `NonlinearLeastSquaresProblem{has_penalty, ContinuousRectangularSearchSpace, F, G}`

# Examples
```julia-repl
julia> using GlobalOptimization;
julia> f(x) = [x[1] - x[3], x[2] - x[3]]
julia> LB = [-5.0, -5.0, -5.0];
julia> UB = [ 5.0, 5.0, 5.0];
julia> prob = NonlinearLeastSquaresProblem(f, ss, LB, UB, 2)
NonlinearLeastSquaresProblem{Val{false}(), ContinuousRectangularSearchSpace{Float64}, typeof(f), Nothing}(f, nothing, ContinuousRectangularSearchSpace{Float64}([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0], [10.0, 10.0, 10.0]), 2)
```
"""
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
    return 0.5*(dot(f,f) + g*g)
end
@inline function scalar_function(prob::AbstractNonlinearEquationProblem, x::AbstractArray, ::Val{false})
    f = prob.f(x)
    return 0.5*dot(f,f)
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
    cost = 0.5*dot(f,f)
    return cost, g
end
@inline function scalar_function_with_penalty(prob::AbstractNonlinearEquationProblem, x::AbstractArray, ::Val{false})
    f = prob.f(x)
    cost = 0.5*dot(f,f)
    penalty = zero(typeof(cost))
    return cost, penalty
end

"""
    get_scalar_function(prob::AbstractProblem)

    Returns cost function plus the infeasibility penalty squared as a scalar value.
    This is used for PSO (GA, Differential Evolution, etc. if we ever get around to adding those)
"""
@inline function get_scalar_function(prob::AbstractProblem)
    #return x -> scalar_function(prob, x)
    fun = let prob=prob
        x -> scalar_function(prob, x)
    end
    return fun
end

"""
    get_scalar_function_with_penalty(prob::AbstractProblem)
"""
@inline function get_scalar_function_with_penalty(prob::AbstractProblem)
    return x -> scalar_function_with_penalty(prob, x)
end
