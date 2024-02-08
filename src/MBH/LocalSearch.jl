

abstract type LocalSearch{T} end

struct LocalStochasticSearch{T} <: LocalSearch{T}
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

struct LocalGradientSearch{T} <: LocalSearch{T}
    iters::Int

    function LocalGradientSearch{T}(iters::Int) where {T <: AbstractFloat}
        return new{T}(iters)
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

function local_search!(hopper, evaluator, ls::LocalGradientSearch)
    @unpack prob = evaluator
    @unpack iters = ls

    res = Optim.optimize(
        prob.f, 
        hopper.candidate, 
        Optim.LBFGS(),
        Optim.Options(
            iterations = iters,
        );
        autodiff = :forward,
    )

    # Update candidate step if necessary
    if Optim.minimum(res) < hopper.candidate_fitness
        hopper.candidate .= Optim.minimizer(res)
        hopper.candidate_fitness = Optim.minimum(res)
        hopper.candidate_step .= hopper.candidate .- hopper.best_candidate
    end
    return nothing
end