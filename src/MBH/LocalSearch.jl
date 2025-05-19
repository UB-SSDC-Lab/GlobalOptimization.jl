abstract type AbstractLocalSearch{T} end
abstract type DerivativeBasedLocalSearch{T} <: AbstractLocalSearch{T} end
abstract type OptimLocalSearch{T,AD} <: DerivativeBasedLocalSearch{T} end

# Timeout function
struct TimeOutInterruptException <: Exception end
function timeout(f, arg, seconds, fail)
    tsk = @task f(arg)
    schedule(tsk)
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

struct LocalStochasticSearch{T} <: AbstractLocalSearch{T}
    # The local step standard deviation
    b::T

    # Number of iterations
    iters::Int

    # Candidate step and candidate storage
    step::Vector{T}

    function LocalStochasticSearch{T}(
        b::Real, iters::Int
    ) where {T<:AbstractFloat}
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

struct LBFGSLocalSearch{
    T,AT,OT,AD<:Union{ADTypes.AbstractADType, Nothing},
} <: OptimLocalSearch{T,AD}

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

    function LBFGSLocalSearch{T}(;
        iters_per_solve::Int=5,
        percent_decrease_tol::Number=50.0,
        m::Int=10,
        alphaguess=LineSearches.InitialStatic(),
        linesearch=LineSearches.HagerZhang(),
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
            T(percent_decrease_tol), alg, opts, max_solve_time, LocalSearchSolutionCache{T}(), ad,
        )
    end
end

struct NonlinearSolveLocalSearch{T, A} <: DerivativeBasedLocalSearch{T}
    # Tollerance on percent decrease of objective function for performing another local search
    percent_decrease_tolerance::T

    alg::A

    # Absolute tolerance for the solver
    abs_tol::Float64

    # Max solve iters
    max_solve_iters::Int

    # Max time per solve
    max_solve_time::Float64

    # Solution cache
    cache::LocalSearchSolutionCache{T}

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
function initialize!(
    ls_vec::Vector{<:AbstractLocalSearch}, num_dims
)
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
        get_scalar_function(prob),
        prob.ss.dim_min,
        prob.ss.dim_max,
        x0,
        alg,
        options;
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
        autodiff = ad,
    )
    cache.x .= Optim.minimizer(res)
    cache.cost = Optim.minimum(res)
    return true
end

function nonlinear_solve!(cache::LocalSearchSolutionCache, prob, x0, alg, abs_tol, max_iters)
    nl_prob = NonlinearSolve.NonlinearProblem{false}((x,p) -> prob.f(x), x0)
    sol = NonlinearSolve.solve(nl_prob, alg; abstol=abs_tol, maxiters=max_iters)
    cache.x .= sol.u
    cache.cost = scalar_function(prob, sol.u)
    return true
end

function get_solve_fun(
    evaluator,
    ls::OptimLocalSearch{T,Nothing},
) where T
    @unpack prob = evaluator
    @unpack alg, options, cache = ls
    solve! = let cache = cache, prob = prob, alg = alg, options = options
        x -> optim_solve!(cache, prob, x, alg, options)
    end
    return solve!
end
function get_solve_fun(
    evaluator,
    ls::OptimLocalSearch{T,AD},
) where {T,AD<:ADTypes.AbstractADType}
    @unpack prob = evaluator
    @unpack alg, options, cache, ad = ls
    solve! = let cache = cache, prob = prob, alg = alg, options = options, ad = ad
        x -> optim_solve!(cache, prob, x, alg, ad, options)
    end
    return solve!
end

function get_solve_fun(
    evaluator,
    ls::NonlinearSolveLocalSearch{T,A},
) where {T,A}
    @unpack prob = evaluator
    @unpack alg, abs_tol, max_solve_iters, cache = ls
    solve! = let cache = cache, prob = prob, alg = alg, abs_tol = abs_tol, max_solve_iters = max_solve_iters
        x -> nonlinear_solve!(cache, prob, x, alg, abs_tol, max_solve_iters)
    end
    return solve!
end

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

function feasible(
    x, eval, ls::OptimLocalSearch{T}
) where T
    _, penalty = evaluate_with_penalty(eval, x)
    if abs(penalty) - eps(T) <= zero(T)
        return true
    else
        return false
    end
end
function feasible(
    x, eval, ls::NonlinearSolveLocalSearch{T}
) where T
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
