
mutable struct Hopper{T} <: AbstractCandidate{T}
    # The Hopper's current solution
    candidate::Vector{T}
    candidate_step::Vector{T}
    candidate_fitness::T

    function Hopper{T}(nDims::Integer) where {T}
        candidate = zeros(T, nDims)
        candidate_fitness = T(Inf)
        return new{T}(
            candidate,
            copy(candidate),
            candidate_fitness,
        )
    end
    function Hopper{T}(::UndefInitializer) where {T}
        candidate = Vector{T}(undef, 0)
        candidate_fitness = T(Inf)
        return new{T}(
            candidate,
            copy(candidate),
            candidate_fitness,
        )
    end
end

abstract type AbstractHopperSet{T} end
mutable struct SingleHopper{T} <: AbstractHopperSet{T}
    # The single hopper
    hopper::Hopper{T}

    # The hopper's best solution
    best_candidate::Vector{T}
    best_candidate_fitness::T

    function SingleHopper{T}(nDims::Integer) where T
        new{T}(
            Hopper{T}(nDims),
            zeros(T, nDims),
            T(Inf),
        )
    end
    function SingleHopper{T}(::UndefInitializer) where T
        new{T}(
            Hopper{T}(undef),
            Vector{T}(undef, 0),
            T(Inf),
        )
    end
end

mutable struct MultipleCommunicatingHoppers{T} <: AbstractHopperSet{T}
    # The communicating hoppers
    hoppers::Vector{Hopper{T}}

    # The communicating hoppers' best solution
    best_candidate::Vector{T}
    best_candidate_fitness::T

    function MultipleCommunicatingHoppers{T}(nDims::Integer, nHoppers::Integer) where T
        hoppers = [Hopper{T}(nDims) for _ in 1:nHoppers]
        best_candidate = Vector{T}(undef, nDims)
        best_candidate_fitness = T(Inf)
        return new{T}(hoppers, best_candidate, best_candidate_fitness)
    end
end

# Interface
Base.length(h::Hopper) = length(h.candidate)
num_dims(h::Hopper) = length(h)

Base.length(h::SingleHopper) = length(h.hopper)
num_dims(h::SingleHopper) = length(h.hopper)

Base.length(cm::MultipleCommunicatingHoppers) = length(cm.hoppers)
num_dims(cm::MultipleCommunicatingHoppers) = num_dims(cm.hoppers[1])

# Methods
"""
    initialize!(
        hopper::Hopper{T},
        search_space::ContinuousRectangularSearchSpace{T},
        evaluator::FeasibilityHandlingEvaluator{T},
    )

Initializes the hopper position in the search space.
"""
function initialize!(
    hopper::Hopper{T},
    search_space::ContinuousRectangularSearchSpace{T},
    evaluator::FeasibilityHandlingEvaluator,
) where {T}
    # Unpack hopper
    @unpack candidate, candidate_step = hopper

    # Itteratively initialize the hopper position
    # until we find one that is feasible
    # That is, we need the penalty term to be zero
    feasible = false
    while !feasible
        # Initialize the hopper position in search space
        @inbounds for i in eachindex(candidate)
            dmin = dim_min(search_space, i)
            dΔ = dim_delta(search_space, i)
            candidate[i] = dmin + dΔ * rand(T)
        end

        fitness, penalty = evaluate_with_penalty(evaluator, candidate)
        if abs(penalty) - eps() <= 0.0
            feasible = true

            # Set fitness
            hopper.candidate_fitness = fitness

            # Set best candidate and initialize step to zero
            @inbounds for i in eachindex(candidate_step)
                candidate_step[i] = zero(T)
            end
        end
    end

    return nothing
end

function initialize!(
    shopper::SingleHopper{T},
    search_space::ContinuousRectangularSearchSpace{T},
    evaluator::AbstractEvaluator,
    bhe::Nothing
) where {T}
    @unpack hopper, best_candidate = shopper

    initialize!(
        hopper,
        search_space,
        evaluator,
    )

    best_candidate .= hopper.candidate
    shopper.best_candidate_fitness = hopper.candidate_fitness

    return nothing
end

function initialize!(
    mch::MultipleCommunicatingHoppers{T},
    search_space::ContinuousRectangularSearchSpace{T},
    evaluator::AbstractEvaluator,
    bhe::BatchJobEvaluator
) where {T}
    @unpack hoppers, best_candidate = mch

    # Initialize individual hoppers
    job! = let ss = search_space, e = evaluator
        hpr -> initialize!(hpr, ss, e)
    end
    evaluate!(job!, hoppers, bhe)

    # Initialize best candidate
    best_hopper = argmin(hpr -> hpr.candidate_fitness, hoppers)
    best_candidate .= best_hopper.candidate
    mch.best_candidate_fitness = best_hopper.candidate_fitness

    # Set all communicating hoppers to the best candidate
    for hpr in hoppers
        hpr.candidate .= best_candidate
        hpr.candidate_fitness = mch.best_candidate_fitness
    end

    return nothing
end

"""
    initialize!(
        mch::MultipleCommunicatingHoppers{T},
        search_space::ContinuousRectangularSearchSpace{T},
        evaluator::FeasibilityHandlingEvaluator{T},
    )

Initializes the multiple communicating hoppers position in the search space.
"""
function initialize!(
    mch::MultipleCommunicatingHoppers{T},
    search_space::ContinuousRectangularSearchSpace{T},
    evaluator::FeasibilityHandlingEvaluator{T},
) where {T}
    # Unpack hopper
    @unpack hopper, best_candidate = hopper
    @unpack candidate, candidate_step, best_candidate = hopper

    # Itteratively initialize the hopper position
    # until we find one that is feasible
    # That is, we need the penalty term to be zero
    feasible = false
    while !feasible
        # Initialize the hopper position in search space
        @inbounds for i in eachindex(candidate)
            dmin = dim_min(search_space, i)
            dΔ = dim_delta(search_space, i)
            candidate[i] = dmin + dΔ * rand(T)
        end

        fitness, penalty = evaluate_with_penalty(evaluator, candidate)
        if abs(penalty) - eps() <= 0.0
            feasible = true

            # Set fitness
            hopper.candidate_fitness = fitness
            single_hopper.best_candidate_fitness = hopper.candidate_fitness

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
    check_fitness!(c::AbstractHopperSet, options::Union{GeneralOptions,Val{true},Val{false}})

Checks the fitness of the candidate `c` to ensure that it is valid
iff options <: Union{GeneralOptions{D,Val{true}}, Val{true}}, otherwise, does nothing.
"""
@inline function check_fitness!(
    hs::AbstractHopperSet, options::GeneralOptions{D,FVC},
) where {D,FVC}
    check_fitness!(hs, FVC)
end
@inline check_fitness!(hs::AbstractHopperSet, ::Type{Val{false}}) = nothing
function check_fitness!(hs::AbstractHopperSet, ::Type{Val{true}})
    isfinite(hs.best_candidate_fitness) || error(
        "Hopper set has an invalid fitness ($(hs.best_candidate_fitness))."
    )
    return nothing
end

"""
    update_fitness!(hopper::AbstractHopperSet{T}, distribution::AbstractMBHDistribution{T})

Updates the hopper fitness information after previously evaluating the fitness of the hopper.
"""
function update_fitness!(
    hopper_set::SingleHopper{T}, distribution::MBHStaticDistribution{T}
) where {T}
    if hopper_set.hopper.candidate_fitness < hopper_set.best_candidate_fitness
        # Update hopper set
        hopper_set.best_candidate .= hopper_set.hopper.candidate
        hopper_set.best_candidate_fitness = hopper_set.hopper.candidate_fitness
    else
        # Reset hopper
        hopper_set.hopper.candidate .= hopper_set.best_candidate
        hopper_set.hopper.candidate_fitness = hopper_set.best_candidate_fitness
    end
    return nothing
end
function update_fitness!(
    hopper_set::MultipleCommunicatingHoppers{T}, distribution::MBHStaticDistribution{T}
) where {T}
    # Get best candidate index
    best_hopper = argmin(hpr -> hpr.candidate_fitness, hoppers)

    if best_hopper.candidate_fitness < hopper_set.best_candidate_fitness
        # Update hopper
        hopper_set.best_candidate .= best_hopper.candidate
        hopper_set.best_candidate_fitness = best_hopper.candidate_fitness
    end

    # Reset hoppers
    for hpr in hopper_set.hoppers
        hpr.candidate .= hopper_set.best_candidate
        hpr.candidate_fitness = hopper_set.best_candidate_fitness
    end

    return nothing
end


function update_fitness!(
    hopper_set::SingleHopper{T}, distribution::MBHAdaptiveDistribution{T}
) where {T}
    if hopper_set.hopper.candidate_fitness < hopper_set.best_candidate_fitness
        # Update distribution
        push_accepted_step!(
            distribution,
            hopper_set.hopper.candidate_step,
            hopper_set.best_candidate_fitness,
            hopper_set.hopper.candidate_fitness,
        )

        # Update hopper
        hopper_set.best_candidate .= hopper_set.hopper.candidate
        hopper_set.best_candidate_fitness = hopper_set.hopper.candidate_fitness
    else
        # Reset hopper
        hopper_set.hopper.candidate .= hopper_set.best_candidate
        hopper_set.hopper.candidate_fitness = hopper_set.best_candidate_fitness
    end
    return nothing
end
function update_fitness!(
    hopper_set::MultipleCommunicatingHoppers{T}, distribution::MBHAdaptiveDistribution{T}
) where {T}
    # Get best candidate index
    best_hopper = argmin(hpr -> hpr.candidate_fitness, hopper_set.hoppers)

    if best_hopper.candidate_fitness < hopper_set.best_candidate_fitness
        # Update distribution
        push_accepted_step!(
            distribution,
            best_hopper.candidate_step,
            hopper_set.best_candidate_fitness,
            best_hopper.candidate_fitness,
        )

        # Update hopper
        hopper_set.best_candidate .= best_hopper.candidate
        hopper_set.best_candidate_fitness = best_hopper.candidate_fitness
    end

    # Reset hoppers
    for hpr in hopper_set.hoppers
        hpr.candidate .= hopper_set.best_candidate
        hpr.candidate_fitness = hopper_set.best_candidate_fitness
    end

    return nothing
end

"""
    draw_update!(hopper::Hopper{T}, distribution::AbstractMBHDistribution{T})

Draws a perterbation from distribution and updates candidate for the hopper `hopper`.
"""
function draw_update!(
    hopper::Hopper{T}, distribution::AbstractMBHDistribution{T}
) where {T}
    # Unpack hopper
    @unpack candidate, candidate_step = hopper

    # Draw step (dispatches based on Type(distribution))
    draw_step!(candidate_step, distribution)

    # Update candidate
    #candidate .= best_candidate .+ candidate_step
    candidate .+= candidate_step

    # Update hopper
    return nothing
end
