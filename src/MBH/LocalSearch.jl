abstract type AbstractLocalSearch{T} end
abstract type DerivativeBasedLocalSearch{T} <: AbstractLocalSearch{T} end
abstract type OptimLocalSearch{T,AD} <: DerivativeBasedLocalSearch{T} end

# Timeout function
struct TimeOutInterruptException <: Exception end
function timeout(f, arg, seconds, fail)
<<<<<<< Updated upstream
    tsk = @task f(arg)
    schedule(tsk)
=======
    tsk = @async f(arg)
    #schedule(tsk)
>>>>>>> Stashed changes
    Timer(seconds) do timer
        istaskdone(tsk) || Base.throwto(tsk, TimeOutInterruptException())
    end
    try
        return fetch(tsk)
    catch e
<<<<<<< Updated upstream
=======
        println("timeout")
>>>>>>> Stashed changes
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

# A simple cache for storing the solution from optimization with Optim.jl
mutable struct LocalSearchSolutionCache{T}
    x::Vector{T}
    cost::T
    function LocalSearchSolutionCache{T}() where {T}
        return new{T}(Vector{T}(undef, 0), zero(T))
    end
end

# Initalize optim cache
function initialize!(cache::LocalSearchSolutionCache, num_dims)
    resize!(cache.x, num_dims)
    return nothing
end

"""
    LBFGSLocalSearch{T,AT,OT,AD<:Union{ADTypes.AbstractADType, Nothing}}

A local search algorithm that uses the LBFGS algorithm with box constraints to locally
improve the candidate solution.

Note that this method employs the `LBFGS` algorithm with the `Fminbox` wrapper from
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

# Fields
- `percent_decrease_tolerance::T`: The tolerance on the percent decrease of the objective
    function for performing another local search. I.e., if after a local search involving
    `iters_per_solve` iterations, the objective function value is reduced by more than
    `percent_decrease_tolerance` percent, then another local search is performed.
- `alg::AT`: The `LBFGS` algorithm with the `Fminbox` wrapper.
- `options::OT`: The Optim.jl options. Only used to enforce the number of iterations
    performed in each local search.
- `max_solve_time::Float64`: The maximum time per solve in seconds. If a solve does not
    finish in this time, the solve process is terminated.
- `cache::LocalSearchSolutionCache{T}`: The solution cache for storing the solution from
    optimization with Optim.jl.
- `ad::AD`: The autodiff method to use. If `nothing`, then the default of ForwardDiff.jl is
    used. Can be any of the autodiff methods from
    [ADTypes.jl](https://github.com/SciML/ADTypes.jl).
"""
struct LBFGSLocalSearch{T,AT,OT,AD<:Union{AbstractADType,Nothing}} <: OptimLocalSearch{T,AD}

    # Tollerance on percent decrease of objective function for performing another local search
    percent_decrease_tolerance::T

    # The LBFGS algorithm
    alg::AT

    # The Optim.jl options
    options::OT

    # Max time per solve
    max_solve_time::Float64

    # Solution cache
    cache::LocalSearchSolutionCache{T}

    # Autodiff method
    ad::AD

    @doc """
        LBFGSLocalSearch{T}(;
            iters_per_solve::Int=5,
            percent_decrease_tol::Number=50.0,
            m::Int=10,
            alphaguess=LineSearches.InitialStatic(),
            linesearch=LineSearches.HagerZhang(),
            manifold=Optim.Flat(),
            max_solve_time::Float64=0.1,
            ad=nothing,
        )

    Create a new `LBFGSLocalSearch` object with the given parameters.

    # Keyword Arguments
    - `iters_per_solve::Int`: The number of iterations to perform in each local search.
    - `percent_decrease_tol::Number`: The tolerance on the percent decrease of the objective
        function for performing another local search. I.e., if after a local search involving
        `iters_per_solve` iterations, the objective function value is reduced by more than
        `percent_decrease_tol` percent, then another local search is performed.
    - `m::Int`: The number of recent steps to employ in approximating the Hessian.
    - `alphaguess`: The initial guess for the step length. Default is
        `LineSearches.InitialStatic()`.
    - `linesearch`: The line search method to use. Default is `LineSearches.HagerZhang()`.
    - `manifold`: The manifold to use. Default is `Optim.Flat()`.
    - `max_solve_time::Float64`: The maximum time per solve in seconds. If a solve does not
        finish in this time, the solve process is terminated.
    - `ad`: The autodiff method to use. If `nothing`, then the default of ForwardDiff.jl is
        used. Can be any of the autodiff methods from
        [ADTypes.jl](https://github.com/SciML/ADTypes.jl).
    """
    function LBFGSLocalSearch{T}(;
        iters_per_solve::Int=5,
        percent_decrease_tol::Number=50.0,
        m::Int=10,
        alphaguess=InitialStatic(),
        linesearch=HagerZhang(),
        manifold=Optim.Flat(),
        max_solve_time=0.1,
        ad=nothing,
    ) where {T<:AbstractFloat}
        alg = Optim.Fminbox(
            Optim.LBFGS(;
                m=m, alphaguess=alphaguess, linesearch=linesearch, manifold=manifold
            ),
        )
        opts = Optim.Options(; iterations=iters_per_solve)
        return new{T,typeof(alg),typeof(opts),typeof(ad)}(
            T(percent_decrease_tol),
            alg,
            opts,
            max_solve_time,
            LocalSearchSolutionCache{T}(),
            ad,
        )
    end
end

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
struct NonlinearSolveLocalSearch{T,A} <: DerivativeBasedLocalSearch{T}
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
    cache::LocalSearchSolutionCache{T}

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
        time_per_solve::Float64=0.1,
        percent_decrease_tol::Number=50.0,
        abs_tol::Float64=1e-8,
    ) where {T,A}
        return new{T,A}(
            T(percent_decrease_tol),
            alg,
            abs_tol,
            iters_per_solve,
            time_per_solve,
            LocalSearchSolutionCache{T}(),
        )
    end
end

struct BOBYQALocalSearch{T,UTO<:Union{Val{true},Val{false}}} <: DerivativeBasedLocalSearch{T}
    # Tollerance on percent decrease of objective function for performing another local search
    percent_decrease_tolerance::T

    # Max solve function evaluations
    max_fevals::Int

    # Max time per solve
    use_timeout::UTO
    max_solve_time::Float64

    # Solution cache
    cache::LocalSearchSolutionCache{T}

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
            LocalSearchSolutionCache{T}(),
        )
    end
end

function initialize!(ls::LocalStochasticSearch, num_dims)
    resize!(ls.step, num_dims)
    return nothing
end
function initialize!(ls::LBFGSLocalSearch, num_dims)
    initialize!(ls.cache, num_dims)
    return nothing
end
function initialize!(ls::NonlinearSolveLocalSearch, num_dims)
    initialize!(ls.cache, num_dims)
    return nothing
end
function initialize!(ls::BOBYQALocalSearch, num_dims)
    initialize!(ls.cache, num_dims)
    return nothing
end
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

function optim_solve!(cache::LocalSearchSolutionCache, prob, x0, alg, options)
    res = Optim.optimize(
        get_scalar_function(prob), prob.ss.dim_min, prob.ss.dim_max, x0, alg, options;
    )
    cache.x .= Optim.minimizer(res)
    cache.cost = Optim.minimum(res)
    return true
end
function optim_solve!(cache::LocalSearchSolutionCache, prob, x0, alg, ad, options)
    res = Optim.optimize(
        get_scalar_function(prob),
        prob.ss.dim_min,
        prob.ss.dim_max,
        x0,
        alg,
        options;
        autodiff=ad,
    )
    cache.x .= Optim.minimizer(res)
    cache.cost = Optim.minimum(res)
    return true
end

function nonlinear_solve!(
    cache::LocalSearchSolutionCache, prob, x0, alg, abs_tol, max_iters
)
    nl_prob = NonlinearSolve.NonlinearProblem{false}((x, p) -> prob.f(x), x0)
    sol = NonlinearSolve.solve(nl_prob, alg; abstol=abs_tol, maxiters=max_iters)
    cache.x .= sol.u
    cache.cost = scalar_function(prob, sol.u)
    return true
end

function bobyqa_solve!(
    cache::LocalSearchSolutionCache, prob, x0, max_fevals
)
    x, info = PRIMA.bobyqa(
        get_scalar_function(prob), x0;
        maxfun = max_fevals,
        xl = prob.ss.dim_min,
        xu = prob.ss.dim_max,
    )
    cache.x .= x
    cache.cost = info.fx
    return true
end

function get_solve_fun(evaluator, ls::OptimLocalSearch{T,Nothing}) where {T}
    @unpack prob = evaluator
    @unpack alg, options, cache = ls
    solve! = let cache = cache, prob = prob, alg = alg, options = options
        x -> optim_solve!(cache, prob, x, alg, options)
    end
    return solve!
end
function get_solve_fun(evaluator, ls::OptimLocalSearch{T,AD}) where {T,AD<:AbstractADType}
    @unpack prob = evaluator
    @unpack alg, options, cache, ad = ls
    solve! = let cache = cache, prob = prob, alg = alg, options = options, ad = ad
        x -> optim_solve!(cache, prob, x, alg, ad, options)
    end
    return solve!
end

function get_solve_fun(evaluator, ls::NonlinearSolveLocalSearch{T,A}) where {T,A}
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

<<<<<<< Updated upstream
=======
function get_solve_fun(evaluator, ls::BOBYQALocalSearch{T}) where {T}
    @unpack prob = evaluator
    @unpack max_fevals, cache = ls
    solve! = let cache = cache, prob = prob, max_fevals = max_fevals
        x -> bobyqa_solve!(cache, prob, x, max_fevals)
    end
    return solve!
end

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

>>>>>>> Stashed changes
function local_search!(hopper, evaluator, ls::DerivativeBasedLocalSearch)
    @unpack candidate, candidate_fitness = hopper
    @unpack percent_decrease_tolerance, max_solve_time, cache = ls

    # Create solve call
    solve! = get_solve_fun(evaluator, ls)

    # Perform local search
    current_fitness = candidate_fitness
    done = false
    while !done
        # Perform optimization with optim and terminate if we don't finish in max_solve_time seconds
        solve_finished = timeout(solve!, candidate, max_solve_time, false)

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

function feasible(x, eval, ls::Union{OptimLocalSearch{T}, BOBYQALocalSearch{T}}) where {T}
    _, penalty = evaluate_with_penalty(eval, x)
    if abs(penalty) - eps(T) <= zero(T)
        return true
    else
        return false
    end
end
function feasible(x, eval, ls::NonlinearSolveLocalSearch{T}) where {T}
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
