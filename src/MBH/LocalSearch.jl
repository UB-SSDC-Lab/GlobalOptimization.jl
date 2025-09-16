abstract type AbstractLocalSearch{T} end
abstract type DerivativeBasedLocalSearch{T} <: AbstractLocalSearch{T} end

# Timeout function
struct TimeOutInterruptException <: Exception end
function timeout(f, arg, seconds, fail)
    tsk = @async f(arg)
    Timer(seconds) do timer
        istaskdone(tsk) || Base.throwto(tsk, TimeOutInterruptException())
    end
    try
        return fetch(tsk)
    catch e
        if isa(e, TaskFailedException)
            if isa(e.task.exception, TimeOutInterruptException)
                return fail
            else
                rethrow(e.task.exception)
            end
        else
            rethrow(e)
        end
    end
end

"""
    LocalStochasticSearch{T}

A local search algorithm that uses a stochastic approach to locally improve the candidate
solution.

Note that this local search algorithm is able to guarantee satisfaction of both the box
constraints and the nonlinear inequality constraint (if any).

# Fields
- `b::T`: The local step standard deviation.
- `iters::Int`: The number of iterations to perform.
- `step::Vector{T}`: The candidate step and candidate storage.
"""
struct LocalStochasticSearch{T} <: AbstractLocalSearch{T}
    # The local step standard deviation
    b::T

    # Number of iterations
    iters::Int

    # Candidate step and candidate storage
    step::Vector{T}

    @doc """
        LocalStochasticSearch{T}(b::Real, iters::Int) where {T<:AbstractFloat}

    Create a new `LocalStochasticSearch` object with the given step size and number of iterations.

    # Arguments
    - `b::Real`: The local step standard deviation.
    - `iters::Int`: The number of iterations to perform.
    """
    function LocalStochasticSearch{T}(b::Real, iters::Int) where {T<:AbstractFloat}
        return new{T}(T(b), iters, Vector{T}(undef, 0))
    end
end

"""
    UserLocalSearch{T,F<:Function}

    A user provided local search algorithm to locally improve the candidate solution.

    # Fields
    - `user_search_fun!::F`: The user provided function. This must accept a Hopper{T} as the
        single argument, mutating the Hopper{T} after performing the local search.
"""
struct UserLocalSearch{T,F<:Function} <: AbstractLocalSearch{T}
    # User provided local search function
    # Must be take a single argument that is a Hopper{T} and should mutate the Hopper{T}
    # after performing the local search.
    user_search_fun!::F

    @doc """
        UserLocalSearch{T}(user_search_fun!::F) where {T<:AbstractFloat,F<:Function}

    Create a new `UserLocalSearch` object with the provided `user_search_fun!`.

    # Arguments
    - `user_search_fun::F`: The user provided function. This must accept a Hopper{T} as the
        single argument, mutating the Hopper{T} after performing the local search.
    """
    function UserLocalSearch{T}(user_search_fun!::F) where {T<:AbstractFloat,F<:Function}
        return new{T,F}(user_search_fun!)
    end
end

# A simple cache for storing the solution from optimization with external extension solvers
mutable struct LocalSearchSolutionCache{T}
    x::Vector{T}
    cost::T
    function LocalSearchSolutionCache{T}() where {T}
        return new{T}(Vector{T}(undef, 0), zero(T))
    end
end
function initialize!(cache::LocalSearchSolutionCache, num_dims)
    resize!(cache.x, num_dims)
    return nothing
end

# Local search initialization
function initialize!(ls::LocalStochasticSearch, num_dims)
    resize!(ls.step, num_dims)
    return nothing
end
initialize!(ls::UserLocalSearch, num_dims) = nothing
function initialize!(ls_vec::Vector{<:AbstractLocalSearch}, num_dims)
    for ls in ls_vec
        initialize!(ls, num_dims)
    end
    return nothing
end

function draw_step!(
    step::AbstractVector{T}, ls::LocalStochasticSearch{T}
) where {T<:AbstractFloat}
    @inbounds for i in eachindex(step)
        step[i] = ls.b * randn(T)
    end
    return nothing
end

function local_search!(hopper, evaluator, ls::LocalStochasticSearch)
    @unpack b, iters, step = ls
    for _ in 1:iters
        # Draw step
        draw_step!(step, ls)
        step .+= hopper.candidate

        # Evaluate step
        if feasible(step, evaluator.prob.ss)
            fitness, penalty = evaluate_with_penalty(evaluator, step)
            if abs(penalty) - eps() <= 0.0 && fitness < hopper.candidate_fitness
                hopper.candidate_step .+= step .- hopper.candidate
                hopper.candidate .= step
                hopper.candidate_fitness = fitness
            end
        end
    end
    return nothing
end
function local_search!(hopper, evaluator, ls::UserLocalSearch)
    ls.user_search_fun!(hopper)
    return nothing
end

# Interface for external extension solvers
function get_solve_fun(eval, ::Any) end

function call(
    solve!::F, candidate, use_timeout::Val{true}, max_solve_time,
) where F <: Function
    # Call the solve function with a timeout
    return timeout(solve!, candidate, max_solve_time, false)
end
function call(
    solve!::F, candidate, use_timeout::Val{false}, max_solve_time,
) where F <: Function
    solve!(candidate)
    return true
end

function local_search!(hopper, evaluator, ls::DerivativeBasedLocalSearch)
    @unpack candidate, candidate_fitness = hopper
    @unpack percent_decrease_tolerance, max_solve_time, use_timeout, cache = ls

    # Create solve call
    solve! = get_solve_fun(evaluator, ls)

    # Perform local search
    current_fitness = candidate_fitness
    done = false
    while !done
        # Perform optimization with optim and terminate if we don't finish in max_solve_time seconds
        #solve_finished = timeout(solve!, candidate, max_solve_time, false)
        solve_finished = call(solve!, candidate, use_timeout, max_solve_time)

        if solve_finished && feasible(cache.x, evaluator, ls)
            # Solve finished in time, so check fitness
            new_fitness = cache.cost
            if new_fitness < current_fitness
                # Update hopper candidate since we've improved some
                hopper.candidate_step .+= cache.x .- hopper.candidate
                hopper.candidate .= cache.x
                hopper.candidate_fitness = new_fitness

                # Check if we should continue local search
                perc_decrease =
                    100.0 * (current_fitness - new_fitness) / abs(current_fitness)
                if perc_decrease < percent_decrease_tolerance
                    done = true
                else
                    current_fitness = new_fitness
                end
            else
                done = true
            end
        else
            # Solve did not finish in time, so don't do anything
            done = true
        end
    end
    return nothing
end

function feasible(x, eval, ls::DerivativeBasedLocalSearch{T}) where {T}
    if !feasible(x, eval.prob.ss)
        return false
    else
        _, penalty = evaluate_with_penalty(eval, x)
        if abs(penalty) - eps(T) <= zero(T)
            return true
        else
            return false
        end
    end
end
