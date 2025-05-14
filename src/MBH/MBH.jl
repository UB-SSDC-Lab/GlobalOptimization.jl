
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
    H<:AbstractHopper{T},
    E<:SingleEvaluator{T},
    D<:AbstractMBHDistribution,
    LS<:AbstractLocalSearch,
} <: AbstractOptimizer

    # Monotonic Basin Hopping Options
    options::MBHOptions

    # The MBH evaluator
    evaluator::E

    # The hopper
    hopper::H

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
    function_value_check::Bool=true,
    display::Bool=false,
    display_interval::Int=1,
    max_time::Real=60.0,
    min_cost::Real=(-Inf),
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
        search_space(prob),
    )

    # Construct MBH
    return MBH(
        options,
        FeasibilityHandlingEvaluator(prob),
        BasicHopper{T}(num_dims(prob)),
        hop_distribution,
        local_search,
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
    @unpack options, evaluator, hopper, distribution, local_search = opt

    # Initialize the hopper
    initialize!(hopper, options.initial_space, evaluator)

    # Handle fitness
    check_fitness!(hopper, get_general(options))

    # Initialize the distribution
    initialize!(distribution, num_dims(hopper))

    # Initialize the local search
    initialize!(local_search, num_dims(hopper))

    return nothing
end

function iterate!(opt::MBH)
    # Unpack MBH
    @unpack options, evaluator, hopper, distribution, local_search = opt
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

        # Begin search for feasible step
        step_accepted = false
        draw_count = 0
        while !step_accepted
            # Draw update
            draw_update!(hopper, distribution)

            # Update counter
            draw_count += 1

            # Check if we're in feasable search space
            if feasible(hopper.candidate, search_space)
                # We're in the search space, so we're about to accept the step,
                # but we need to check if we're also
                # in the feasible reagion defined by the penalty parameter
                fitness, penalty = evaluate_with_penalty(evaluator, hopper.candidate)

                if abs(penalty) - eps() <= 0.0
                    # We're in the feasible region, so we can accept the step
                    step_accepted = true

                    # Set fitness of candidate
                    hopper.candidate_fitness = fitness
                end
            end
        end

        # Perform local search (performing local search after checkng feasibility but before updating hopper fitness)
        local_search!(hopper, evaluator, local_search)

        # Update fitness
        update_fitness!(hopper, distribution)

        # Stopping criteria
        current_time = time()
        if current_time - start_time >= options.general.max_time
            exit_flag = 1
        elseif hopper.best_candidate_fitness <= get_min_cost(options)
            exit_flag = 2
        end

        # Output status
        display_status_mbh(
            current_time - start_time,
            iteration,
            draw_count,
            hopper.best_candidate_fitness,
            get_general(options),
        )
    end

    # Return results
    return Results(
        hopper.best_candidate_fitness,
        hopper.best_candidate,
        iteration,
        current_time - start_time,
        exit_flag,
    )
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
