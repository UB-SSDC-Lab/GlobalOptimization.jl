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
    evaluator::FeasibilityHandlingEvaluator{T},
) where {T}
    # Unpack hopper
    @unpack candidate, candidate_step, best_candidate = hopper

    # Itteratively initialize the hopper position
    # until we find one that is feasible
    # That is, we need the penalty term to be zero
    feasible = false
    while !feasible
        # Initialize the hopper position in search space
        @inbounds for i in eachindex(candidate)
            dmin = dimmin(search_space, i)
            dΔ = dimdelta(search_space, i)
            candidate[i] = dmin + dΔ * rand(T)
        end

        fitness, penalty = evaluate_with_penalty(evaluator, candidate)
        if abs(penalty) - eps() <= 0.0
            feasible = true

            # Set fitness
            hopper.candidate_fitness = fitness
            hopper.best_candidate_fitness = hopper.candidate_fitness

            # Set best candidate and initialize step to zero
            @inbounds for i in eachindex(candidate_step)
                candidate_step[i] = zero(T)
                best_candidate[i] = candidate[i]
            end
        end
    end

    return nothing
end

"""
    update_fitness!(hopper::BasicHopper{T}, distribution::AbstractMBHDistribution{T})

Updates the hopper fitness information after previously evaluating the fitness of the hopper.
"""
function update_fitness!(
    hopper::BasicHopper{T}, distribution::MBHStaticDistribution{T}
) where {T}
    if hopper.candidate_fitness < hopper.best_candidate_fitness
        # Update hopper
        hopper.best_candidate .= hopper.candidate
        hopper.best_candidate_fitness = hopper.candidate_fitness
    end
    return nothing
end
function update_fitness!(
    hopper::BasicHopper{T}, distribution::MBHAdaptiveDistribution{T}
) where {T}
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
function draw_update!(
    hopper::BasicHopper{T}, distribution::AbstractMBHDistribution{T}
) where {T}
    # Unpack hopper
    @unpack candidate, candidate_step, best_candidate = hopper

    # Draw step (dispatches based on Type(distribution))
    draw_step!(candidate_step, distribution)

    # Update candidate
    candidate .= best_candidate .+ candidate_step

    # Update hopper
    return nothing
end
