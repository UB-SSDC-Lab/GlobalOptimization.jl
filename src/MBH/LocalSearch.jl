abstract type AbstractLocalSearch{T} end
abstract type GradientBasedLocalSearch{T} <: AbstractLocalSearch{T} end
abstract type OptimLocalSearch{T} <: GradientBasedLocalSearch{T} end

struct LocalStochasticSearch{T} <: AbstractLocalSearch{T}
    # The scale laplace distribution scale parameter
    b::T

    # Number of iterations
    iters::Int

    # Candidate step and candidate storage
    step::Vector{T}

    function LocalStochasticSearch{T}(ndim::Int, b::T, iters::Int) where {T <: AbstractFloat}
        return new{T}(b, iters, Vector{T}(undef, ndim))
    end
end

struct LBFGSLocalSearch{T, AT, OT} <: OptimLocalSearch{T}
    # Tollerance on percent decrease of objective function for performing another local search
    percent_decrease_tolerance::T

    # The LBFGS algorithm
    alg::AT

    # The Optim.jl options
    options::OT

    function LBFGSLocalSearch{T}(;
        iters_per_solve::Int = 5,     
        percent_decrease_tol::Number = 50.0,
        m::Int = 10,
        alphaguess = LineSearches.InitialStatic(),
        linesearch = LineSearches.HagerZhang(),
        manifold = Optim.Flat(),
    ) where {T <: AbstractFloat}
        alg = Optim.LBFGS(;
            m = m,
            alphaguess = alphaguess,
            linesearch = linesearch,
            manifold = manifold,
        )
        opts = Optim.Options(
            iterations = iters_per_solve,
        )
        return new{T, typeof(alg), typeof(opts)}(
            T(percent_decrease_tol), 
            alg,
            opts,
        )
    end
end

function draw_step!(step::AbstractVector{T}, ls::LocalStochasticSearch{T}) where {T <: AbstractFloat}
    @inbounds for i in eachindex(step)
        #step[i] = laplace(ls.b)
        step[i] = ls.b*randn(T)
    end
    return nothing
end

function local_search!(hopper, evaluator, ls::LocalStochasticSearch)
    @unpack prob = evaluator
    @unpack b, iters, step = ls
    better_candidate_found = false
    for _ in 1:iters
        # Draw step
        draw_step!(step, ls)
        step .+= hopper.candidate 

        # Evaluate step
        if feasible(step, prob.ss)
            fitness = evaluate(prob, step)
            if fitness < hopper.candidate_fitness
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

function local_search!(hopper, evaluator, ls::OptimLocalSearch)
    @unpack candidate, candidate_fitness = hopper
    @unpack prob = evaluator
    @unpack percent_decrease_tolerance, alg, options = ls

    # Perform local search
    current_fitness = candidate_fitness
    done = false
    while !done
        # Perform optimization
        res = Optim.optimize(
            get_local_search_function(prob), 
            candidate, 
            alg,
            options;
        )

        new_fitness = Optim.minimum(res)
        if new_fitness < current_fitness
            # Update hopper candidate since we've improved some
            hopper.candidate        .= Optim.minimizer(res)
            hopper.candidate_fitness = new_fitness
            hopper.candidate_step   .= hopper.candidate .- hopper.best_candidate

            # Check if we should continue local search
            perc_decrease = 100.0*(current_fitness - new_fitness)/abs(current_fitness)
            if perc_decrease < percent_decrease_tolerance
                done = true
            else
                current_fitness = new_fitness
            end
        else
            done = true
        end
    end
    return nothing
end