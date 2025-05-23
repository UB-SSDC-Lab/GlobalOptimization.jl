
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

    # Algorithm max iterations
    max_iterations::Int

    # Stall parameters
    function_tolerance::Float64
    max_stall_time::Float64
    max_stall_iterations::Int

    function MBHOptions(
        general::GO,
        distribution::D,
        local_search::LS,
        initial_space::ISS,
        max_iterations::Int,
        function_tolerance::Float64,
        max_stall_time::Float64,
        max_stall_iterations::Int,
    ) where {D,LS,ISS,GO}
        return new{D,LS,ISS,GO}(
            general,
            distribution,
            local_search,
            initial_space,
            max_iterations,
            function_tolerance,
            max_stall_time,
            max_stall_iterations,
        )
    end
end

"""
    MBHCache

Cache for the Monotonic Basin Hopping (MBH) algorithm.
"""
mutable struct MBHCache{T}
    iteration::Int
    start_time::Float64
    stall_start_time::Float64
    stall_iteration::Int
    stall_value::T
    function MBHCache{T}() where {T}
        return new{T}(0, 0.0, 0.0, 0, T(Inf))
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
    cache::MBHCache{T}
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
- `initial_space::Union{Nothing,ContinuousRectangularSearchSpace}`: The initial search space
    to use. Default is `nothing`.
- `function_value_check::Bool`: Whether to check the function value. Default is `true`.
- `max_time::Real`: The maximum time to run the algorithm in seconds. Default is `60.0`.
- `min_cost::Real`: The minimum cost to reach. Default is `-Inf`.
- `display::Bool`: Whether to display the status of the algorithm. Default is `false`.
- `display_interval::Int`: The interval at which to display the status of the algorithm.
    Default is `1`.
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
            max_time,
            min_cost,
        ),
        hop_distribution,
        handle_local_search(local_search, hopper_type),
        intersection(search_space(prob), initial_space),
        max_iterations,
        function_tolerance,
        max_stall_time,
        max_stall_iterations,
    )

    # Construct MBH
    return MBH(
        options,
        FeasibilityHandlingEvaluator(prob),
        get_batch_evaluator(hopper_type),
        get_hopper_set(prob, hopper_type),
        MBHCache{T}(),
    )
end

# ===== AbstractOptimizer interface

get_iteration(opt::MBH) = opt.cache.iteration

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
    cache.iteration = 0
    cache.start_time = time()
    cache.stall_start_time = cache.start_time
    cache.stall_iteration = 0
    cache.stall_value = hopper_set.best_candidate_fitness

    return nothing
end

function iterate!(opt::MBH)
    # Unpack MBH
    @unpack options, evaluator, bhe, hopper_set, cache = opt
    @unpack distribution, local_search = options
    search_space = evaluator.prob.ss

    # Begin loop
    status = IN_PROGRESS
    draw_count = 0
    while status == IN_PROGRESS
        # Update iteration counter
        cache.iteration += 1

        # Take a hop
        draw_count = hop!(
            hopper_set, search_space, evaluator, bhe, distribution, local_search
        )
        check_fitness!(hopper_set, get_function_value_check(options))

        # Update fitness
        update_fitness!(hopper_set, distribution)

        # Stopping criteria
        status = check_stopping_criteria(opt)

        # Tracing
        trace(opt)
    end

    # Return results
    return construct_results(opt, status)
end

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

function handle_stall!(mbh::MBH)
    @unpack cache, options = mbh
    if cache.stall_value - get_best_fitness(mbh) < options.function_tolerance
        # Currently stalled...
        cache.stall_iteration += 1
    else
        # Not stalled!!
        cache.stall_value = get_best_fitness(mbh)
        cache.stall_iteration = 0
        cache.stall_start_time = time()
    end
end

get_best_fitness(mbh::MBH) = mbh.hopper_set.best_candidate_fitness

function check_stopping_criteria(mbh::MBH)
    @unpack cache, options = mbh
    current_time = time()
    if get_best_fitness(mbh) <= get_min_cost(options)
        return MINIMUM_COST_ACHIEVED
    elseif current_time - cache.start_time >= get_max_time(options)
        return MAXIMUM_TIME_EXCEEDED
    elseif cache.iteration >= options.max_iterations
        return MAXIMUM_ITERATIONS_EXCEEDED
    elseif cache.stall_iteration >= options.max_stall_iterations
        return MAXIMUM_STALL_ITERATIONS_EXCEEDED
    elseif current_time - cache.stall_start_time >= options.max_stall_time
        return MAXIMUM_STALL_TIME_EXCEEDED
    end
    return IN_PROGRESS
end

function construct_results(mbh::MBH, status::Status)
    @unpack cache, hopper_set = mbh
    return Results(
        hopper_set.best_candidate_fitness,
        hopper_set.best_candidate,
        cache.iteration,
        time() - cache.start_time,
        status,
    )
end

function show_trace(mbh::MBH, ::Any)

end

function get_save_trace(mbh::MBH, ::Any)

end
