
"""
    MBHOptions <: AbstractAlgorithmSpecificOptions

Options for the Monotonic Basin Hopping (MBH) algorithm.
"""
struct MBHOptions{ISS <: Union{Nothing, ContinuousRectangularSearchSpace}, GO <: GeneralOptions} <: AbstractAlgorithmSpecificOptions
    # The general options
    general::GO

    # MBH specific options
    # Defines the search space that the initial particles are drawn from
    initial_space::ISS

    function MBHOptions(
        general::GO,
        initial_space::ISS,
    ) where {ISS,GO}
        return new{ISS,GO}(
            general,
            initial_space,
        )
    end
end

"""
    MBH

Monotonic Basin Hopping (MBH) algorithm.

This implementation employs a single candidate rather than a population.
"""
struct MBH{T <: Number, H <: AbstractHopper{T}, E <: SingleEvaluator{T}} <: AbstractOptimizer
    # Monotonic Basin Hopping Options
    options::MBHOptions

    # The MBH evaluator
    evaluator::E

    # The hopper
    hopper::H

    # The MBH distribution
    distribution::AbstractMBHDistribution{T}
end

"""
    BasicMBH(prob::AbstractOptimizationProblem{SS})

Constructs a basic MBH algorithm for the optimization problem `prob`.
"""
function BasicMBH(
    prob::AbstractOptimizationProblem{SS};
    function_value_check::Bool = true,
    display::Bool = false,
    display_interval::Int = 1,
    max_time::Real = 60.0,
    a = 0.93,
    b = 0.05,
    c = 1.0,
    λ = 0.1,
) where {T <: Number, SS <: ContinuousRectangularSearchSpace{T}}
    # Construct the options
    options = MBHOptions(
        GeneralOptions(
            function_value_check ? Val(true) : Val(false),
            display ? Val(true) : Val(false),
            display_interval,
            max_time,
        ),
        search_space(prob),
    )

    # Construct MBH
    return MBH(
        options,
        BasicEvaluator(prob),
        BasicHopper{T}(numdims(prob)),
        MBHStaticDistribution{T}(;
            a = a,
            b = b,
            c = c,
            λ = λ,
        ),
    )
end

"""
    AdaptiveMBH(prob::AbstractOptimizationProblem{SS})

Constructs an adaptive MBH algorithm for the optimization problem `prob`.
"""
function AdaptiveMBH(
    prob::AbstractOptimizationProblem{SS};
    function_value_check::Bool = true,
    display::Bool = false,
    display_interval::Int = 1,
    max_time::Real = 60.0,
    a  = 0.93,
    b  = 0.05,
    c  = 1.0,
    λ  = 0.1,
    memory_len = 10,
) where {T <: Number, SS <: ContinuousRectangularSearchSpace{T}}
    # Construct the options
    options = MBHOptions(
        GeneralOptions(
            function_value_check ? Val(true) : Val(false),
            display ? Val(true) : Val(false),
            display_interval,
            max_time,
        ),
        search_space(prob),
    )

    # Construct MBH
    return MBH(
        options,
        BasicEvaluator(prob),
        BasicHopper{T}(numdims(prob)),
        MBHAdaptiveDistribution{T}(
            numdims(prob),
            memory_len;
            a = a,
            b = b,
            c = c,
            λhat0 = λ,
        ),
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
    @unpack options, evaluator, hopper = opt

    # Initialize the hopper
    initialize!(hopper, options.initial_space)

    # Handle fitness
    initialize_fitness!(hopper, evaluator)
    check_fitness!(hopper, get_general(options))

    return nothing
end

function iterate!(opt::MBH)
    # Unpack MBH
    @unpack options, evaluator, hopper, distribution = opt
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
                step_accepted = true
            end
        end

        # Evaluate the candidate
        evaluate_fitness!(hopper, distribution, evaluator)

        # Stopping criteria
        current_time = time()
        if current_time - start_time >= options.general.max_time
            exit_flag = 1
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

function display_status_mbh(time, iteration, draw_count, fitness, options::GeneralOptions{D,FVC}) where {D,FVC}
    display_status_mbh(time, iteration, draw_count, fitness, get_display_interval(options), D)
    return nothing
end
@inline display_status_mbh(time, iteration, draw_count, fitness, display_interval, ::Val{false}) = nothing
function display_status_mbh(time, iteration, draw_count, fitness, display_interval, ::Val{true})
    if iteration % display_interval == 0
        fspec1 = FormatExpr("Time Elapsed: {1:f} sec, Iteration Number: {2:d}")
        fspec2 = FormatExpr("Draw Count: {1:d}, Best Fitness: {2:e}")
        printfmtln(fspec1, time, iteration)
        printfmtln(fspec2, draw_count, fitness)
    end
    return nothing
end


