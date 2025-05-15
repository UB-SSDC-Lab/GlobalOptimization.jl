
"""
    MBHOptions <: AbstractAlgorithmSpecificOptions

Options for the Monotonic Basin Hopping (MBH) algorithm.
"""
struct MBHOptions{
    ISS<:Union{Nothing,ContinuousRectangularSearchSpace},GO<:GeneralOptions
} <: AbstractAlgorithmSpecificOptions
    # The general options
    general::GO

    # MBH specific options
    initial_space::ISS

    function MBHOptions(general::GO, initial_space::ISS) where {ISS,GO}
        return new{ISS,GO}(general, initial_space)
    end
end

"""
    MBH

Monotonic Basin Hopping (MBH) algorithm.

This implementation employs a single candidate rather than a population.
"""
struct MBH{
    T<:Number,
    H<:AbstractHopperSet{T},
    E<:FeasibilityHandlingEvaluator,
    BHE<:Union{Nothing,BatchJobEvaluator},
    D<:AbstractMBHDistribution,
    LS<:Union{AbstractLocalSearch, Vector{<:AbstractLocalSearch}},
} <: AbstractOptimizer

    # Monotonic Basin Hopping Options
    options::MBHOptions

    # The base evaluator
    evaluator::E

    # The hopper set
    hopper_set::H

    # The batch hop evaluator (nothing if single hopper)
    bhe::BHE

    # The MBH distribution
    distribution::D

    # The local search algorithm
    local_search::LS
end

"""
    MBH(prob::AbstractOptimizationProblem{SS})
"""
function MBH(
    prob::AbstractProblem{has_penalty,ContinuousRectangularSearchSpace{T}};
    hop_distribution::AbstractMBHDistribution{T}=MBHAdaptiveDistribution{T}(100, 5),
    local_search::AbstractLocalSearch{T}=LBFGSLocalSearch{T}(),
    initial_space::Union{Nothing,ContinuousRectangularSearchSpace}=nothing,
    function_value_check::Bool=true,
    max_time::Real=60.0,
    min_cost::Real=(-Inf),
    display::Bool=false,
    display_interval::Int=1,
) where {T<:Number,has_penalty}
    # Construct the options
    options = MBHOptions(
        GeneralOptions(
            function_value_check ? Val(true) : Val(false),
            display ? Val(true) : Val(false),
            display_interval,
            max_time,
            min_cost,
        ),
        intersection(search_space(prob), initial_space),
    )

    # Construct MBH
    return MBH(
        options,
        FeasibilityHandlingEvaluator(prob),
        SingleHopper{T}(num_dims(prob)),
        nothing,
        hop_distribution,
        local_search,
    )
end

"""
    SerialCMCH(prob::AbstractOptimizationProblem{SS})
"""
function SerialCMBH(
    prob::AbstractProblem{has_penalty,ContinuousRectangularSearchSpace{T}};
    num_hoppers::Integer = Threads.nthreads(),
    hop_distribution::AbstractMBHDistribution{T}=MBHAdaptiveDistribution{T}(100, 5),
    local_search::AbstractLocalSearch{T}=LBFGSLocalSearch{T}(),
    initial_space::Union{Nothing,ContinuousRectangularSearchSpace}=nothing,
    function_value_check::Bool=true,
    max_time::Real=60.0,
    min_cost::Real=(-Inf),
    display::Bool=false,
    display_interval::Int=1,
) where {T<:Number,has_penalty}
    # Construct the options
    options = MBHOptions(
        GeneralOptions(
            function_value_check ? Val(true) : Val(false),
            display ? Val(true) : Val(false),
            display_interval,
            max_time,
            min_cost,
        ),
        intersection(search_space(prob), initial_space),
    )

    # Construct MBH
    return MBH(
        options,
        FeasibilityHandlingEvaluator(prob),
        MultipleCommunicatingHoppers{T}(num_dims(prob), num_hoppers),
        SerialBatchJobEvaluator(),
        hop_distribution,
        [deepcopy(local_search) for _ in 1:num_hoppers],
    )
end

"""
    ThreadedCMCH(prob::AbstractOptimizationProblem{SS})
"""
function ThreadedCMBH(
    prob::AbstractProblem{has_penalty,ContinuousRectangularSearchSpace{T}};
    num_hoppers::Integer = Threads.nthreads(),
    hop_distribution::AbstractMBHDistribution{T}=MBHAdaptiveDistribution{T}(100, 5),
    local_search::AbstractLocalSearch{T}=LBFGSLocalSearch{T}(),
    initial_space::Union{Nothing,ContinuousRectangularSearchSpace}=nothing,
    function_value_check::Bool=true,
    max_time::Real=60.0,
    min_cost::Real=(-Inf),
    batch_n::Int=Threads.nthreads(),
    batch_split::ChunkSplitters.Split=ChunkSplitters.RoundRobin(),
    display::Bool=false,
    display_interval::Int=1,
) where {T<:Number,has_penalty}
    # Construct the options
    options = MBHOptions(
        GeneralOptions(
            function_value_check ? Val(true) : Val(false),
            display ? Val(true) : Val(false),
            display_interval,
            max_time,
            min_cost,
        ),
        intersection(search_space(prob), initial_space),
    )

    # Construct MBH
    return MBH(
        options,
        FeasibilityHandlingEvaluator(prob),
        MultipleCommunicatingHoppers{T}(num_dims(prob), num_hoppers),
        ThreadedBatchJobEvaluator(batch_n, batch_split),
        hop_distribution,
        [deepcopy(local_search) for _ in 1:num_hoppers],
    )
end

"""
    PolyesterCMCH(prob::AbstractOptimizationProblem{SS})
"""
function PolyesterCMBH(
    prob::AbstractProblem{has_penalty,ContinuousRectangularSearchSpace{T}};
    num_hoppers::Integer = Threads.nthreads(),
    hop_distribution::AbstractMBHDistribution{T}=MBHAdaptiveDistribution{T}(100, 5),
    local_search::AbstractLocalSearch{T}=LBFGSLocalSearch{T}(),
    initial_space::Union{Nothing,ContinuousRectangularSearchSpace}=nothing,
    function_value_check::Bool=true,
    max_time::Real=60.0,
    min_cost::Real=(-Inf),
    display::Bool=false,
    display_interval::Int=1,
) where {T<:Number,has_penalty}
    # Construct the options
    options = MBHOptions(
        GeneralOptions(
            function_value_check ? Val(true) : Val(false),
            display ? Val(true) : Val(false),
            display_interval,
            max_time,
            min_cost,
        ),
        intersection(search_space(prob), initial_space),
    )

    # Construct MBH
    return MBH(
        options,
        FeasibilityHandlingEvaluator(prob),
        MultipleCommunicatingHoppers{T}(num_dims(prob), num_hoppers),
        PolyesterBatchJobEvaluator(),
        hop_distribution,
        [deepcopy(local_search) for _ in 1:num_hoppers],
    )
end

# Methods
function optimize!(opt::MBH)
    # Initialize the MBH algorithm
    initialize!(opt)

    # Perform iterations and return results
    return iterate!(opt)
end

function initialize!(opt::MBH)
    # Unpack MBH
    @unpack options, evaluator, bhe, hopper_set, distribution, local_search = opt

    # Initialize the hopper
    initialize!(hopper_set, options.initial_space, evaluator, bhe)

    # Handle fitness
    check_fitness!(hopper_set, get_general(options))

    # Initialize the distribution
    initialize!(distribution, num_dims(hopper_set))

    # Initialize the local search
    initialize!(local_search, num_dims(hopper_set))

    return nothing
end

function iterate!(opt::MBH)
    # Unpack MBH
    @unpack options, evaluator, bhe, hopper_set, distribution, local_search = opt
    search_space = evaluator.prob.ss

    # Initialize algorithm stopping criteria requrements
    iteration = 0
    start_time = time()
    current_time = start_time

    # Begin loop
    exit_flag = 0
    draw_count = 0
    while exit_flag == 0
        # Update iteration counter
        iteration += 1

        # Take a hop
        draw_count = hop!(
            hopper_set,
            search_space,
            evaluator,
            bhe,
            distribution,
            local_search,
        )

        # Update fitness
        update_fitness!(hopper_set, distribution)

        # Stopping criteria
        current_time = time()
        if current_time - start_time >= options.general.max_time
            exit_flag = 1
        elseif hopper_set.best_candidate_fitness <= get_min_cost(options)
            exit_flag = 2
        end

        # Output status
        display_status_mbh(
            current_time - start_time,
            iteration,
            draw_count,
            hopper_set.best_candidate_fitness,
            get_general(options),
        )
    end

    # Return results
    return Results(
        hopper_set.best_candidate_fitness,
        hopper_set.best_candidate,
        iteration,
        current_time - start_time,
        exit_flag,
    )
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
function hop!(hopper_set::SingleHopper, ss, eval, bhe, dist, ls)
    return hop!(hopper_set.hopper, ss, eval, dist, ls)
end
function hop!(hopper_set::MultipleCommunicatingHoppers, ss, eval, bhe, dist, ls)
    # Unpack the hoppers
    @unpack hoppers = hopper_set

    job! = let hs=hopper_set, ss=ss, eval=eval, dist=dist, ls=ls
        i -> hop!(hs.hoppers[i], ss, eval, dist, ls[i])
    end
    evaluate!(job!, eachindex(hopper_set), bhe)

    return 0
end

function display_status_mbh(
    time, iteration, draw_count, fitness, options::GeneralOptions{D,FVC}
) where {D,FVC}
    display_status_mbh(
        time, iteration, draw_count, fitness, get_display_interval(options), D
    )
    return nothing
end
@inline display_status_mbh(
    time, iteration, draw_count, fitness, display_interval, ::Type{Val{false}}
) = nothing
function display_status_mbh(
    time, iteration, draw_count, fitness, display_interval, ::Type{Val{true}}
)
    if iteration % display_interval == 0
        fspec1 = FormatExpr("Time Elapsed: {1:f} sec, Iteration Number: {2:d}")
        fspec2 = FormatExpr("Draw Count: {1:d}, Best Fitness: {2:e}")
        printfmtln(fspec1, time, iteration)
        printfmtln(fspec2, draw_count, fitness)
    end
    return nothing
end
