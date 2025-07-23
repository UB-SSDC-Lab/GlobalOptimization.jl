
"""
    module NonlinearSolveLocalSearchExt

Provides an MBH local search extension using NonlinearSolve.jl.
"""
module NonlinearSolveLocalSearchExt

using GlobalOptimization, UnPack
using NonlinearSolve: NonlinearSolve

"""
    NonlinearSolveLocalSearch{T,A} <: DerivativeBasedLocalSearch{T}

A local search algorithm that uses the NonlinearSolve.jl package to locally improve the
candidate solution. Note that this method only works for `NonlinearProblem` and
`NonlinearLeastSquaresProblem` types.

Additionally, this method is not able to guarantee satisfaction of the box constraints
or the penalty nonlinear inequality constraint (if any). However, if a new solution violates
either of these constraints, the new solution is discarded and the local search is
terminated.

# Fields
- `percent_decrease_tolerance::T`: The tolerance on the percent decrease of the objective
    function for performing another local search. I.e., if after a local search involving
    `iters_per_solve` iterations, the objective function value is reduced by more than
    `percent_decrease_tolerance` percent, then another local search is performed.
- `alg::A`: The NonlinearSolve.jl algorithm to use.
- `abs_tol::Float64`: The absolute tolerance for the solver. Default is `1e-8`.
- `max_solve_iters::Int`: The maximum number of iterations to perform in each local search.
    Default is `5`.
- `max_solve_time::Float64`: The maximum time per solve in seconds. If a solve does not
    finish in this time, the solve process is terminated. Default is `0.1`.
- `cache::LocalSearchSolutionCache{T}`: The solution cache for storing the solution from
    solving with NonlinearSolve.jl.
"""
struct NonlinearSolveLocalSearch{T,A,UTO<:Union{Val{true},Val{false}}} <: GlobalOptimization.NonlinearSolveLocalSearch{T}
    # Tollerance on percent decrease of objective function for performing another local search
    percent_decrease_tolerance::T

    alg::A

    # Absolute tolerance for the solver
    abs_tol::Float64

    # Max solve iters
    max_solve_iters::Int

    # Max time per solve
    use_timeout::UTO
    max_solve_time::Float64

    # Solution cache
    cache::GlobalOptimization.LocalSearchSolutionCache{T}

    @doc """
        NonlinearSolveLocalSearch{T,A}(
            alg::A;
            iters_per_solve::Int=5,
            time_per_solve::Float64=0.1,
            percent_decrease_tol::Number=50.0,
            abs_tol::Float64=1e-8,
        )

    Create a new `NonlinearSolveLocalSearch` object with the given parameters.

    # Arguments
    - `alg::A`: The NonlinearSolve.jl algorithm to use. For example,
        `NonlinearSolve.NewtonRaphson()` of `NonlinearSolve.TrustRegion()`.
    - `iters_per_solve::Int`: The number of iterations to perform in each local search.
    - `time_per_solve::Float64`: The maximum time per solve in seconds. If a solve does not
        finish in this time, the solve process is terminated.
    - `percent_decrease_tol::Number`: The tolerance on the percent decrease of the objective
        function for performing another local search. I.e., if after a local search involving
        `iters_per_solve` iterations, the objective function value is reduced by more than
        `percent_decrease_tol` percent, then another local search is performed.
    - `abs_tol::Float64`: The absolute tolerance for the solver. Default is `1e-8`.
    """
    function NonlinearSolveLocalSearch{T}(
        alg::A;
        iters_per_solve::Int=5,
        use_timeout::VT=Val{true}(),
        time_per_solve::Float64=0.1,
        percent_decrease_tol::Number=50.0,
        abs_tol::Float64=1e-8,
    ) where {T,A,VT<:Union{Val{true},Val{false}}}
        return new{T,A,VT}(
            T(percent_decrease_tol),
            alg,
            abs_tol,
            iters_per_solve,
            use_timeout,
            time_per_solve,
            GlobalOptimization.LocalSearchSolutionCache{T}(),
        )
    end
end
function GlobalOptimization.NonlinearSolveLocalSearch{T}(args...; kwargs...) where {T}
    return NonlinearSolveLocalSearch{T}(args...; kwargs...)
end

function GlobalOptimization.initialize!(ls::NonlinearSolveLocalSearch, num_dims)
    GlobalOptimization.initialize!(ls.cache, num_dims)
    return nothing
end

function nonlinear_solve!(
    cache::GlobalOptimization.LocalSearchSolutionCache, prob, x0, alg, abs_tol, max_iters
)
    nl_prob = NonlinearSolve.NonlinearProblem{false}((x, p) -> prob.f(x), x0)
    sol = NonlinearSolve.solve(nl_prob, alg; abstol=abs_tol, maxiters=max_iters)
    cache.x .= sol.u
    cache.cost = GlobalOptimization.scalar_function(prob, sol.u)
    return true
end

function GlobalOptimization.get_solve_fun(evaluator, ls::NonlinearSolveLocalSearch{T,A}) where {T,A}
    @unpack prob = evaluator
    @unpack alg, abs_tol, max_solve_iters, cache = ls
    solve! =
        let cache = cache,
            prob = prob,
            alg = alg,
            abs_tol = abs_tol,
            max_solve_iters = max_solve_iters

            x -> nonlinear_solve!(cache, prob, x, alg, abs_tol, max_solve_iters)
        end
    return solve!
end

end
