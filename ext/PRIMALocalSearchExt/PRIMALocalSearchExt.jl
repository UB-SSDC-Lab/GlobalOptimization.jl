"""
    module PRIMALocalSearchExt

Provides a local search extension using the BOBYQA algorithm via PRIMA.jl to perform derivative-based local optimization with constraints.

"""
module PRIMALocalSearchExt

using GlobalOptimization, UnPack
using PRIMA: PRIMA

"""
    BOBYQALocalSearch{T,UTO} <: DerivativeBasedLocalSearch{T}

A local search algorithm that uses the BOBYQA algorithm from PRIMA.jl to locally improve the candidate solution.

# Fields
- `percent_decrease_tolerance::T`: The tolerance on the percent decrease of the objective function for performing another local search. If the decrease after a local search is less than this percentage, the search terminates.
- `max_fevals::Int`: Maximum number of function evaluations during the local search.
- `use_timeout::UTO`: Flag (Val{true} or Val{false}) indicating whether a time limit on solving should be enforced.
- `max_solve_time::Float64`: The maximum allowed time (in seconds) for each solve call.
- `cache::LocalSearchSolutionCache{T}`: Cache storing the solution found during the local search.

# Keyword Arguments
- `percent_decrease_tol::Number=50.0`: Default tolerance on percent decrease to continue searching.
- `max_fevals::Int=1000`: Maximum number of solve function evaluations.
- `use_timeout::Val{true}=Val{true}()`: Whether to use a timeout on the solver.
- `max_solve_time::Float64=0.1`: Max solve time in seconds.

"""
struct BOBYQALocalSearch{T,UTO<:Union{Val{true},Val{false}}} <: GlobalOptimization.BOBYQALocalSearch{T}
    # Tollerance on percent decrease of objective function for performing another local search
    percent_decrease_tolerance::T

    # Max solve function evaluations
    max_fevals::Int

    # Max time per solve
    use_timeout::UTO
    max_solve_time::Float64

    # Solution cache
    cache::GlobalOptimization.LocalSearchSolutionCache{T}

    function BOBYQALocalSearch{T}(;
        percent_decrease_tol::Number=50.0,
        max_fevals::Int=1000,
        use_timeout::VT= Val{true}(),
        max_solve_time::Float64=0.1,
    ) where {T<:AbstractFloat,VT<:Union{Val{true},Val{false}}}
        return new{T,VT}(
            T(percent_decrease_tol),
            max_fevals,
            use_timeout,
            max_solve_time,
            GlobalOptimization.LocalSearchSolutionCache{T}(),
        )
    end
end
function GlobalOptimization.BOBYQALocalSearch{T}(args...; kwargs...) where T
    return BOBYQALocalSearch{T}(args...; kwargs...)
end

function GlobalOptimization.initialize!(ls::BOBYQALocalSearch, num_dims)
    GlobalOptimization.initialize!(ls.cache, num_dims)
    return nothing
end

function bobyqa_solve!(
    cache::GlobalOptimization.LocalSearchSolutionCache, prob, x0, max_fevals
)
    x, info = PRIMA.bobyqa(
        GlobalOptimization.get_scalar_function(prob), x0;
        maxfun = max_fevals,
        xl = prob.ss.dim_min,
        xu = prob.ss.dim_max,
        #rhobeg = minimum(prob.ss.dim_delta) / 4.0,
    )
    cache.x .= x
    cache.cost = info.fx
    return true
end

function GlobalOptimization.get_solve_fun(evaluator, ls::BOBYQALocalSearch{T}) where {T}
    @unpack prob = evaluator
    @unpack max_fevals, cache = ls
    solve! = let cache = cache, prob = prob, max_fevals = max_fevals
        x -> bobyqa_solve!(cache, prob, x, max_fevals)
    end
    return solve!
end

end
