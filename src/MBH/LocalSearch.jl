abstract type AbstractLocalSearch end
abstract type GradientBasedLocalSearch <: AbstractLocalSearch end
abstract type OptimLocalSearch <: GradientBasedLocalSearch end

# Timeout function
function timeout(f, arg, seconds, fail)
    tsk = @task f(arg)
    schedule(tsk)
    Timer(seconds) do timer
        istaskdone(tsk) || Base.throwto(tsk, InterruptException())
    end
    try
        return fetch(tsk)
    catch _
        return fail
    end
end

struct LocalStochasticSearch{T} <: AbstractLocalSearch
    # The scale laplace distribution scale parameter
    b::T

    # Number of iterations
    iters::Int

    # Candidate step and candidate storage
    step::Vector{T}

    function LocalStochasticSearch{T}(
        ndim::Int, b::Real, iters::Int
    ) where {T<:AbstractFloat}
        return new{T}(T(b), iters, Vector{T}(undef, ndim))
    end
end

# A simple cache for storing the solution from optimization with Optim.jl
mutable struct OptimSolutionCache{T}
    initialized::Bool
    x::Vector{T}
    cost::T
    function OptimSolutionCache{T}() where {T}
        return new{T}(false, Vector{T}(undef, 0), zero(T))
    end
end

# Initalize optim cache
is_initialized(cache::OptimSolutionCache) = cache.initialized
function initialize!(cache::OptimSolutionCache{T}, ndim) where {T}
    cache.x = Vector{T}(undef, ndim)
    cache.initialized = true
    return nothing
end

struct LBFGSLocalSearch{T,AT,OT} <: OptimLocalSearch
    # Tollerance on percent decrease of objective function for performing another local search
    percent_decrease_tolerance::T

    # The LBFGS algorithm
    alg::AT

    # The Optim.jl options
    options::OT

    # Max time per solve
    max_solve_time::Float64

    # Solution cache
    cache::OptimSolutionCache{T}

    function LBFGSLocalSearch{T}(;
        iters_per_solve::Int=5,
        percent_decrease_tol::Number=50.0,
        m::Int=10,
        alphaguess=LineSearches.InitialStatic(),
        linesearch=LineSearches.HagerZhang(),
        manifold=Optim.Flat(),
        max_solve_time=0.1,
    ) where {T<:AbstractFloat}
        alg = Optim.Fminbox(
            Optim.LBFGS(;
                m=m, alphaguess=alphaguess, linesearch=linesearch, manifold=manifold
            ),
        )
        opts = Optim.Options(; iterations=iters_per_solve)
        return new{T,typeof(alg),typeof(opts)}(
            T(percent_decrease_tol), alg, opts, max_solve_time, OptimSolutionCache{T}()
        )
    end
end

function draw_step!(
    step::AbstractVector{T}, ls::LocalStochasticSearch{T}
) where {T<:AbstractFloat}
    @inbounds for i in eachindex(step)
        #step[i] = laplace(ls.b)
        step[i] = ls.b * randn(T)
    end
    return nothing
end

function local_search!(hopper, evaluator, ls::LocalStochasticSearch)
    @unpack b, iters, step = ls
    better_candidate_found = false
    for _ in 1:iters
        # Draw step
        draw_step!(step, ls)
        step .+= hopper.candidate

        # Evaluate step
        if feasible(step, evaluator.prob.ss)
            fitness, penalty = evaluate_with_penalty(evaluator, step)
            if abs(penalty) - eps() <= 0.0 && fitness < hopper.candidate_fitness
                better_candidate_found = true
                hopper.candidate .= step
                hopper.candidate_fitness = fitness
            end
        end
    end
    # Update candidate step if necessary
    if better_candidate_found
        hopper.candidate_step .= hopper.candidate .- hopper.best_candidate
    end
    return nothing
end

function optim_solve!(cache::OptimSolutionCache, prob, x0, alg, options)
    res = Optim.optimize(
        get_scalar_function(prob), prob.ss.dim_min, prob.ss.dim_max, x0, alg, options;
    )
    cache.x .= Optim.minimizer(res)
    cache.cost = Optim.minimum(res)
    return true
end

function local_search!(hopper, evaluator, ls::OptimLocalSearch)
    @unpack candidate, candidate_fitness = hopper
    @unpack prob = evaluator
    @unpack percent_decrease_tolerance, alg, options, max_solve_time, cache = ls

    # Initialize cache if necessary
    is_initialized(cache) || initialize!(cache, num_dims(prob))

    # Create solve call
    solve! = let cache = cache, prob = prob, alg = alg, options = options
        x -> optim_solve!(cache, prob, x, alg, options)
    end

    # Perform local search
    current_fitness = candidate_fitness
    done = false
    while !done
        # Perform optimization with optim and terminate if we don't finish in max_solve_time seconds
        solve_finished = timeout(solve!, candidate, max_solve_time, false)

        if solve_finished
            # Solve finished in time, so check fitness
            new_fitness = cache.cost
            if new_fitness < current_fitness
                # Update hopper candidate since we've improved some
                hopper.candidate .= cache.x
                hopper.candidate_fitness = new_fitness
                hopper.candidate_step .= hopper.candidate .- hopper.best_candidate

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
