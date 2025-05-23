
"""
    MBHOptions <: AbstractAlgorithmSpecificOptions

Options for the Monotonic Basin Hopping (MBH) algorithm.
"""
struct MBHOptions{
    D<:AbstractMBHDistribution,
    LS<:Union{AbstractLocalSearch, Vector{<:AbstractLocalSearch}},
    ISS<:Union{Nothing,ContinuousRectangularSearchSpace},
    GO<:GeneralOptions
} <: AbstractAlgorithmSpecificOptions
    # The general options
    general::GO

    # The MBH distribution
    distribution::D

    # The local search algorithm
    local_search::LS

    # MBH specific options
    initial_space::ISS

    function MBHOptions(
        general::GO,
        distribution::D,
        local_search::LS,
        initial_space::ISS,
    ) where {D,LS,ISS,GO}
        return new{D,LS,ISS,GO}(
            general,
            distribution,
            local_search,
            initial_space,
        )
    end
end

"""
    MBH

Monotonic Basin Hopping (MBH) algorithm.

This implementation employs a single candidate rather than a population.
"""
struct MBH{
    D<:AbstractMBHDistribution,
    LS<:Union{AbstractLocalSearch, Vector{<:AbstractLocalSearch}},
    T<:AbstractFloat,
    E<:FeasibilityHandlingEvaluator,
    BHE<:Union{Nothing, BatchJobEvaluator},
    IBSS,
    H<:AbstractHopperSet{T},
    GO<:GeneralOptions,
} <: AbstractOptimizer
    # Monotonic Basin Hopping Options
    options::MBHOptions{D,LS,IBSS,GO}

    # The base evaluator
    evaluator::E

    # The batch hop evaluator (nothing if single hopper)
    bhe::BHE

    # The hopper set
    hopper_set::H

    # The MBH cache
    cache::MinimalOptimizerCache{T}
end

# Functions to help construct the MBH algorithm

"""
    handle_local_search(local_search::AbstractLocalSearch, hopper_type::AbstractHopperType)

Returns the local search algorithm for a given hopper type.
"""
function handle_local_search(local_search, hopper_type::SingleHopper)
    return local_search
end
function handle_local_search(local_search, hopper_type::MCH)
    return [deepcopy(local_search) for _ in 1:hopper_type.num_hoppers]
end

"""
    get_batch_evaluator(hopper_type::AbstractHopperType)

Returns the batch job evaluator for a given hopper type.
"""
get_batch_evaluator(hopper_type::SingleHopper) = nothing
get_batch_evaluator(hopper_type::MCH) = construct_batch_job_evaluator(hopper_type.eval_method)

"""
    get_hopper_set(prob::AbstractProblem, hopper_type::AbstractHopperType)

Returns the hopper set for a given problem and hopper type.
"""
function get_hopper_set(
    prob::AbstractProblem{has_penalty,ContinuousRectangularSearchSpace{T}},
    hopper_type::SingleHopper,
) where {has_penalty,T}
    return SingleHopperSet{T}(num_dims(prob))
end
function get_hopper_set(
    prob::AbstractProblem{has_penalty,ContinuousRectangularSearchSpace{T}},
    hopper_type::MCH,
) where {has_penalty,T}
    return MCHSet{T}(num_dims(prob), hopper_type.num_hoppers)
end

"""
    MBH(prob::AbstractOptimizationProblem{SS}; kwargs...)

Construct the standard Monotonic Basin Hopping (MBJ) algorithm with the specified options.

# Keyword Arguments
- `hopper_type::AbstractHopperType`: The type of hopper to use. Default is
    `SingleHopper()`.
- `hop_distribution::AbstractMBHDistribution{T}`: The distribution from which hops are
    drawn. Default is `MBHAdaptiveDistribution{T}(100, 5)`.
- `local_search::AbstractLocalSearch{T}`: The local search algorithm to use. Default is
    `LBFGSLocalSearch{T}()`.
- `initial_space::Union{Nothing,ContinuousRectangularSearchSpace}=nothing`: The initial bounds for the search space.
- `max_iterations::Integer=1000`: The maximum number of iterations.
- `function_tolerance::Real=1e-6`: The function tolerance (stall-based stopping criteria).
- `max_stall_time::Real=60.0`: The maximum stall time (in seconds).
- `max_stall_iterations::Integer=100`: The maximum number of stall iterations.
- `max_time::Real=60.0`: The maximum time (in seconds) to allow for optimization.
- `min_cost::Real=(-Inf)`: The minimum cost to allow for optimization.
- `function_value_check::Union{Val{false},Val{true}}=Val(true)`: Whether to check the function value
    for bad values (i.e., Inf or NaN).
- `show_trace::Union{Val{false},Val{true}}=Val(false)`: Whether to show the trace.
- `save_trace::Union{Val{false},Val{true}}=Val(false)`: Whether to save the trace.
- `save_file::String="no_file.txt"`: The file to save the trace to.
- `trace_level::TraceLevel=TraceMinimal(1)`: The trace level to use.
"""
function MBH(
    prob::AbstractProblem{has_penalty,ContinuousRectangularSearchSpace{T}};
    hopper_type::AbstractHopperType=SingleHopper(),
    hop_distribution::AbstractMBHDistribution{T}=MBHAdaptiveDistribution{T}(100, 5),
    local_search::AbstractLocalSearch{T}=LBFGSLocalSearch{T}(),
    initial_space::Union{Nothing,ContinuousRectangularSearchSpace}=nothing,
    max_iterations::Integer=1000,
    function_tolerance::Real=1e-6,
    max_stall_time::Real=60.0,
    max_stall_iterations::Integer=100,
    max_time::Real=60.0,
    min_cost::Real=(-Inf),
    function_value_check::Union{Val{false},Val{true}}=Val(true),
    show_trace::Union{Val{false},Val{true}}=Val(false),
    save_trace::Union{Val{false},Val{true}}=Val(false),
    save_file::String="no_file.txt",
    trace_level::TraceLevel=TraceMinimal(1),
) where {T<:Number,has_penalty}
    # Check arguments
    if isa(prob, OptimizationProblem) && isa(local_search, NonlinearSolveLocalSearch)
        throw(
            ArgumentError(
                "NonlinearSolveLocalSearch is not supported for OptimizationProblem! " *
                "Please consider using LBFGSLocalSearch or LocalStochasticSearch instead.",
            ),
        )
    end
    if max_time < 0.0
        throw(ArgumentError("max_time must be greater than 0.0!"))
    end
    if min_cost < -Inf
        throw(ArgumentError("min_cost must be greater than -Inf!"))
    end

    # Construct the options
    options = MBHOptions(
        GeneralOptions(
            GlobalOptimizationTrace(
                show_trace,
                save_trace,
                save_file,
                trace_level,
            ),
            function_value_check,
            min_cost,
            max_time,
            max_iterations,
            function_tolerance,
            max_stall_time,
            max_stall_iterations,
        ),
        hop_distribution,
        handle_local_search(local_search, hopper_type),
        intersection(search_space(prob), initial_space),
    )

    # Construct MBH
    return MBH(
        options,
        FeasibilityHandlingEvaluator(prob),
        get_batch_evaluator(hopper_type),
        get_hopper_set(prob, hopper_type),
        MinimalOptimizerCache{T}(),
    )
end

# ===== AbstractOptimizer interface
get_best_fitness(mbh::MBH) = mbh.hopper_set.best_candidate_fitness
get_best_candidate(mbh::MBH) = mbh.hopper_set.best_candidate

function initialize!(opt::MBH)
    # Unpack MBH
    @unpack options, evaluator, bhe, hopper_set, cache = opt
    @unpack distribution, local_search = options

    # Initialize the distribution
    initialize!(distribution, evaluator.prob.ss)

    # Initialize the local search
    initialize!(local_search, num_dims(hopper_set))

    # Initialize the hopper
    initialize!(hopper_set, options.initial_space, evaluator, bhe)
    check_fitness!(hopper_set, get_function_value_check(options))

    # Initialize the cache
    initialize!(cache, hopper_set.best_candidate_fitness)

    return nothing
end

function step!(opt::MBH)
    # Unpack MBH
    @unpack options, evaluator, bhe, hopper_set, cache = opt
    @unpack distribution, local_search = options
    search_space = evaluator.prob.ss

    # Take a hop
    draw_count = hop!(
        hopper_set, search_space, evaluator, bhe, distribution, local_search
    )
    check_fitness!(hopper_set, get_function_value_check(options))

    # Update fitness
    update_fitness!(hopper_set, distribution)

    return nothing
end

function show_trace(mbh::MBH, ::Union{Val{:minimal}, Val{:detailed}, Val{:all}})

end

function get_save_trace(mbh::MBH, ::Union{Val{:minimal}, Val{:detailed}, Val{:all}})

end

# ===== Implementation Specific Methods

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
    return hop!(hopper_set.hopper, ss, eval, dist, ls)
end
function hop!(hopper_set::MCHSet, ss, eval, bhe, dist, ls)
    # Unpack the hoppers
    @unpack hoppers = hopper_set

    job! = let hs=hopper_set, ss=ss, eval=eval, dist=dist, ls=ls
        i -> hop!(hs.hoppers[i], ss, eval, dist, ls[i])
    end
    evaluate!(job!, eachindex(hopper_set), bhe)

    return 0
end
