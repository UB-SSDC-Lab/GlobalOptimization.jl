
mutable struct Hopper{T} <: AbstractCandidate{T}
    # The Hopper's current solution
    candidate::Vector{T}
    candidate_step::Vector{T}
    candidate_fitness::T

    function Hopper{T}(nDims::Integer) where {T}
        candidate = zeros(T, nDims)
        candidate_fitness = T(Inf)
        return new{T}(candidate, copy(candidate), candidate_fitness)
    end
end

abstract type AbstractHopperType end

"""
    SingleHopper <: GlobalOptimization.AbstractHopperType

A single hopper that is used to explore the search space. Note that no parallelism is
employed.
"""
struct SingleHopper <: AbstractHopperType end

"""
    MCH{EM<:AbstractFunctionEvaluationMethod} <: GlobalOptimization.AbstractHopperType

Employs the method of *Multiple Communicating Hoppers* (MCH) to explore the search space as
described in Englander, Arnold C., "Speeding-Up a Random Search for the Global Minimum
of a Non-Convex, Non-Smooth Objective Function" (2021). *Doctoral Dissertations*. 2569.
[https://scholars.unh.edu/dissertation/2569](https://scholars.unh.edu/dissertation/2569/).

The struct fields are the same as the constructor arguments.
"""
struct MCH{EM<:AbstractFunctionEvaluationMethod} <: AbstractHopperType
    num_hoppers::Int
    eval_method::EM

    @doc """
        MCH(;
            num_hoppers::Integer=4,
            eval_method<:AbstractFunctionEvaluationMethod=SerialFunctionEvaluation(),
        )

    Constructs a new `MCH` object with the specified number of hoppers and evaluation method.

    # Keyword Arguments
    - `num_hoppers::Integer`: The number of hoppers to use. Default is 4.
    - `eval_method<:AbstractFunctionEvaluationMethod`: The evaluation method to use. Default
        is `SerialFunctionEvaluation()`.
    """
    function MCH(;
        num_hoppers::Integer=4, eval_method::EM=SerialFunctionEvaluation()
    ) where {EM<:AbstractFunctionEvaluationMethod}
        return new{EM}(num_hoppers, eval_method)
    end
end

abstract type AbstractHopperSet{T} end
mutable struct SingleHopperSet{T} <: AbstractHopperSet{T}
    # The single hopper
    hopper::Hopper{T}

    # The hopper's best solution
    best_candidate::Vector{T}
    best_candidate_fitness::T

    # The number of hops performed before accepting
    num_hops::Int

    function SingleHopperSet{T}(nDims::Integer) where {T}
        new{T}(Hopper{T}(nDims), zeros(T, nDims), T(Inf), 0)
    end
end

mutable struct MCHSet{T} <: AbstractHopperSet{T}
    # The communicating hoppers
    hoppers::Vector{Hopper{T}}

    # The communicating hoppers' best solution
    best_candidate::Vector{T}
    best_candidate_fitness::T

    # The number of hops performed by each hopper before accepting
    num_hops::Vector{Int}

    function MCHSet{T}(nDims::Integer, nHoppers::Integer) where {T}
        hoppers = [Hopper{T}(nDims) for _ in 1:nHoppers]
        best_candidate = Vector{T}(undef, nDims)
        best_candidate_fitness = T(Inf)
        num_hops = Vector{Int}(undef, nHoppers)
        return new{T}(hoppers, best_candidate, best_candidate_fitness, num_hops)
    end
end

# Interface
Base.length(h::Hopper) = length(h.candidate)
num_dims(h::Hopper) = length(h)

Base.length(h::SingleHopperSet) = length(h.hopper)
num_dims(h::SingleHopperSet) = length(h.hopper)

Base.length(cm::MCHSet) = length(cm.hoppers)
Base.eachindex(cm::MCHSet) = eachindex(cm.hoppers)
num_dims(cm::MCHSet) = num_dims(cm.hoppers[1])

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
            set_fitness!(hopper, fitness)

            # Set best candidate and initialize step to zero
            @inbounds for i in eachindex(candidate_step)
                candidate_step[i] = zero(T)
            end
        end
    end

    return nothing
end

function initialize!(
    shopper::SingleHopperSet{T},
    search_space::ContinuousRectangularSearchSpace{T},
    evaluator::FeasibilityHandlingEvaluator,
    bhe::Nothing,
) where {T}
    @unpack hopper, best_candidate = shopper

    initialize!(hopper, search_space, evaluator)

    best_candidate .= candidate(hopper)
    shopper.best_candidate_fitness = fitness(hopper)

    return nothing
end

function initialize!(
    mch::MCHSet{T},
    search_space::ContinuousRectangularSearchSpace{T},
    evaluator::FeasibilityHandlingEvaluator,
    bhe::BatchJobEvaluator,
) where {T}
    @unpack hoppers, best_candidate = mch

    # Initialize individual hoppers
    job! = let ss = search_space, e = evaluator
        hpr -> initialize!(hpr, ss, e)
    end
    evaluate!(job!, hoppers, bhe)

    # Initialize best candidate
    best_hopper = argmin(fitness, hoppers)
    best_candidate .= candidate(best_hopper)
    mch.best_candidate_fitness = fitness(best_hopper)

    # Set all communicating hoppers to the best candidate
    for hpr in hoppers
        set_candidate!(hpr, best_candidate)
        set_fitness!(hpr, mch.best_candidate_fitness)
    end

    return nothing
end

"""
    check_fitness!(c::AbstractHopperSet, options::Union{GeneralOptions,Val{true},Val{false}})

Checks the fitness of the candidate `c` to ensure that it is valid
iff the option has been enabled.
"""
check_fitness!(hs::AbstractHopperSet, ::Val{false}) = nothing
function check_fitness!(hs::AbstractHopperSet, ::Val{true})
    isfinite(hs.best_candidate_fitness) ||
        error("Hopper set has an invalid fitness ($(hs.best_candidate_fitness)).")
    return nothing
end

"""
    update_fitness!(hopper::AbstractHopperSet{T}, distribution::AbstractMBHDistribution{T})

Updates the hopper fitness information after previously evaluating the fitness of the hopper.
"""
function update_fitness!(
    hopper_set::SingleHopperSet{T}, distribution::MBHStaticDistribution{T}
) where {T}
    if fitness(hopper_set.hopper) < hopper_set.best_candidate_fitness
        # Update hopper set
        hopper_set.best_candidate .= candidate(hopper_set.hopper)
        hopper_set.best_candidate_fitness = fitness(hopper_set.hopper)
    else
        # Reset hopper
        set_candidate!(hopper_set.hopper, hopper_set.best_candidate)
        set_fitness!(hopper_set.hopper, hopper_set.best_candidate_fitness)
    end
    return nothing
end
function update_fitness!(
    hopper_set::MCHSet{T}, distribution::MBHStaticDistribution{T}
) where {T}
    # Get best candidate index
    best_hopper = argmin(fitness, hopper_set.hoppers)

    if fitness(best_hopper) < hopper_set.best_candidate_fitness
        # Update hopper
        hopper_set.best_candidate .= candidate(best_hopper)
        hopper_set.best_candidate_fitness = fitness(best_hopper)
    end

    # Reset hoppers
    for hpr in hopper_set.hoppers
        set_candidate!(hpr, hopper_set.best_candidate)
        set_fitness!(hpr, hopper_set.best_candidate_fitness)
    end

    return nothing
end

function update_fitness!(
    hopper_set::SingleHopperSet{T}, distribution::MBHAdaptiveDistribution{T}
) where {T}
    if fitness(hopper_set.hopper) < hopper_set.best_candidate_fitness
        # Update distribution
        push_accepted_step!(
            distribution,
            hopper_set.hopper.candidate_step,
            hopper_set.best_candidate_fitness,
            fitness(hopper_set.hopper),
        )

        # Update hopper
        hopper_set.best_candidate .= candidate(hopper_set.hopper)
        hopper_set.best_candidate_fitness = fitness(hopper_set.hopper)
    else
        # Reset hopper
        set_candidate!(hopper_set.hopper, hopper_set.best_candidate)
        set_fitness!(hopper_set.hopper, hopper_set.best_candidate_fitness)
    end
    return nothing
end
function update_fitness!(
    hopper_set::MCHSet{T}, distribution::MBHAdaptiveDistribution{T}
) where {T}
    # Get best candidate index
    best_hopper = argmin(fitness, hopper_set.hoppers)

    if fitness(best_hopper) < hopper_set.best_candidate_fitness
        # Update distribution
        push_accepted_step!(
            distribution,
            best_hopper.candidate_step,
            hopper_set.best_candidate_fitness,
            fitness(best_hopper),
        )

        # Update hopper
        hopper_set.best_candidate .= candidate(best_hopper)
        hopper_set.best_candidate_fitness = fitness(best_hopper)
    end

    # Reset hoppers
    for hpr in hopper_set.hoppers
        set_candidate!(hpr, hopper_set.best_candidate)
        set_fitness!(hpr, hopper_set.best_candidate_fitness)
    end

    return nothing
end

"""
    draw_update!(hopper::Hopper{T}, distribution::AbstractMBHDistribution{T})

Draws a perterbation from distribution and updates candidate for the hopper `hopper`.
"""
function draw_update!(hopper::Hopper{T}, distribution::AbstractMBHDistribution{T}) where {T}
    # Unpack hopper
    @unpack candidate, candidate_step = hopper

    # Draw step (dispatches based on Type(distribution))
    draw_step!(candidate_step, distribution)

    # Update candidate
    candidate .+= candidate_step

    # Update hopper
    return nothing
end

"""
    reset!(hopper::Hopper)

Resets the candidate to state prior to the last `draw_update!` call.

Note: This just subtracts the last step from the candidate.
"""
function reset!(hopper::Hopper)
    hopper.candidate .-= hopper.candidate_step
    return nothing
end

"""
    hop!(hopper::Union{Hopper, AbstractHopperSet}, ss, eval, dist, ls)

Performs a single hop for the hopper `hopper` in the search space `ss` using the
evaluator `eval`, the distribution `dist`, and the local search `ls`.
"""
function hop!(hopper::Hopper, ss, eval, dist, ls)
    step_accepted = false
    draw_count = 0
    while !step_accepted
        # Draw update
        # This perturbs the candidate by a realization from dist
        draw_update!(hopper, dist)

        # Update counter
        draw_count += 1

        # Check if we're in feasible search space
        if feasible(candidate(hopper), ss)
            # We're in the search space, so we're about to accept the step,
            # but we need to check if we're also
            # in the feasible region defined by the penalty parameter
            fitness, penalty = evaluate_with_penalty(eval, candidate(hopper))

            if abs(penalty) - eps() <= 0.0
                # We're in the feasible region, so we can accept the step
                step_accepted = true

                # Set fitness of candidate
                set_fitness!(hopper, fitness)

                # Break from the loop
                break
            end
        end

        # If we get here, we need to reject the step by calling reset!
        reset!(hopper)
    end

    # Perform local search
    local_search!(hopper, eval, ls)

    return draw_count
end

function hop!(hopper_set::SingleHopperSet, ss, eval, bhe, dist, ls)
    hopper_set.num_hops = hop!(hopper_set.hopper, ss, eval, dist, ls)
    return nothing
end

function hop!(hopper_set::MCHSet, ss, eval, bhe, dist, ls)
    # Unpack the hoppers
    @unpack hoppers = hopper_set

    job! = let hs=hopper_set, ss=ss, eval=eval, dist=dist, ls=ls
        i -> begin
            hs.num_hops[i] = hop!(hs.hoppers[i], ss, eval, dist, ls[i])
            nothing
        end
    end
    evaluate!(job!, eachindex(hopper_set), bhe)

    return nothing
end

function get_show_trace_elements(
    hopper_set::SingleHopperSet,
    trace_mode::Union{Val{:detailed}, Val{:all}}
)
    return (
        TraceElement("Hops", 'd', 8, 0, hopper_set.num_hops),
    )
end
function get_show_trace_elements(
    hopper_set::MCHSet,
    trace_mode::Union{Val{:detailed}, Val{:all}}
)
    min_hops = minimum(hopper_set.num_hops)
    max_hops = maximum(hopper_set.num_hops)
    return (
        TraceElement("Min Hops", 'd', 12, 0, min_hops),
        TraceElement("Max Hops", 'd', 12, 0, max_hops),
    )
end

function get_save_trace_elements(
    hopper_set::AbstractHopperSet,
    trace_mode::Union{Val{:detailed}, Val{:all}}
)
    return get_show_trace_elements(hopper_set, trace_mode)
end
