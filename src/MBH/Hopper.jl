abstract type AbstractHopper{T} <: AbstractCandidate end

mutable struct BasicHopper{T} <: AbstractHopper{T}
    # The Hopper's current solution
    candidate::Vector{T}
    candidate_step::Vector{T}
    candidate_fitness::T

    # The Hopper's best solution
    best_candidate::Vector{T}
    best_candidate_fitness::T

    function BasicHopper{T}(nDims::Integer) where {T}
        candidate = zeros(T, nDims)
        candidate_fitness = T(Inf)
        return new{T}(
            candidate, 
            copy(candidate),
            candidate_fitness, 
            copy(candidate), 
            candidate_fitness,
        )
    end
    function BasicHopper{T}(::UndefInitializer) where {T}
        candidate = Vector{T}(undef, 0)
        candidate_fitness = T(Inf)
        return new{T}(
            candidate, 
            copy(candidate),
            candidate_fitness, 
            copy(candidate), 
            candidate_fitness,
        )
    end
end

# Interface
Base.length(h::BasicHopper) = length(h.x)

# Methods
"""
    initialize!(hopper::BasicHopper{T}, search_space::ContinuousRectangularSearchSpace{T})

Initializes the hopper `hopper` with a uniform distribution in the search space.
"""
function initialize!(
    hopper::BasicHopper{T},
    search_space::ContinuousRectangularSearchSpace{T},
) where {T}
    # Unpack hopper
    @unpack candidate, candidate_step, best_candidate = hopper

    # Initialize the hopper position
    @inbounds for i in eachindex(candidate)
        dmin = dimmin(search_space, i)
        dΔ = dimdelta(search_space, i)

        # Set position
        candidate[i] = dmin + dΔ*rand(T)
        candidate_step[i] = zero(T)
        best_candidate[i] = candidate[i]
    end
    return nothing
end

"""
    initialize_fitness!(hopper::BasicHopper{T}, evaluator::BasicEvaluator{T})

Initializes the hopper's fitness using the given `evaluator`.
"""
function initialize_fitness!(
    hopper::BasicHopper{T},
    evaluator::BasicEvaluator{T},
) where {T}
    # Evaluate the cost function for the candidate
    evaluate!(hopper, evaluator)

    # Initialize the best candidate fitness
    hopper.best_candidate_fitness = hopper.candidate_fitness

    return nothing
end

"""
    evaluate_fitness!(hopper::BasicHopper{T}, distribution::AbstractMBHDistribution{T}, evaluator::BasicEvaluator{T})

Evaluates the fitness of the hopper `hopper` using the given `evaluator`. Also updates the distrubution if step is an improvement.
"""
function evaluate_fitness!(
    hopper::BasicHopper{T}, distribution::MBHStaticDistribution{T}, evaluator::BasicEvaluator{T},
) where {T}
    # Evaluate the cost function for the hopper
    evaluate!(hopper, evaluator)

    if hopper.candidate_fitness < hopper.best_candidate_fitness
        # Update hopper
        hopper.best_candidate .= hopper.candidate
        hopper.best_candidate_fitness = hopper.candidate_fitness
    end

    return nothing
end
function evaluate_fitness!(
    hopper::BasicHopper{T}, distribution::MBHAdaptiveDistribution{T}, evaluator::BasicEvaluator{T},
) where {T}
    # Evaluate the cost function for the hopper
    evaluate!(hopper, evaluator)

    if hopper.candidate_fitness < hopper.best_candidate_fitness
        # Update distribution
        push_accepted_step!(
            distribution,
            hopper.candidate_step,
            hopper.best_candidate_fitness,
            hopper.candidate_fitness,
        )

        # Update hopper
        hopper.best_candidate .= hopper.candidate
        hopper.best_candidate_fitness = hopper.candidate_fitness
    end

    return nothing
end

"""
    draw_update!(hopper::BasicHopper{T}, distribution::AbstractMBHDistribution{T})

Draws a perterbation from distribution and updates candidate for the hopper `hopper`.
"""
function draw_update!(hopper::BasicHopper{T}, distribution::AbstractMBHDistribution{T}) where {T}
    # Unpack hopper
    @unpack candidate, candidate_step, best_candidate = hopper

    # Draw step (dispatches based on Type(distribution))
    draw_step!(candidate_step, distribution)

    # Update candidate
    candidate .= best_candidate .+ candidate_step

    # Update hopper
    return nothing
end