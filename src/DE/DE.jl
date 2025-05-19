"""
DEOptions

Options for the Differential Evolution (DE) algorithms.

# Fields:
- `general<:GeneralOptions`: The general options.
- `pop_init_method<:AbstractPopulationInitialization`: The population initialization method.
- `mutation_params<:AbstractMutationParameters`: The mutation strategy parameters.
- `crossover_params<:AbstractCrossoverParameters`: The crossover strategy parameters.
- `initial_space<:Union{Nothing,ContinuousRectangularSearchSpace}`: The initial space to initialize the population.
- `max_iterations::Int`: The maximum number of iterations.
- `function_tolerance::Float64`: The function tolerance for the stall condition.
- `max_stall_time::Float64`: The maximum stall time for the stall condition.
- `max_stall_iterations::Int`: The maximum number of stall iterations for the stall condition.
"""
struct DEOptions{
    MP<:AbstractMutationParameters,
    CP<:AbstractCrossoverParameters,
    ISS<:Union{Nothing,ContinuousRectangularSearchSpace},
    PI<:AbstractPopulationInitialization,
    GO<:GeneralOptions,
} <: AbstractAlgorithmSpecificOptions

    # The general options
    general::GO

    # The Population initialization method
    pop_init_method::PI

    # ===== DE specific options
    # Mutation strategy parameters
    mutation_params::MP

    # Crossover probability
    crossover_params::CP

    # Initial space to initialize the population
    initial_space::ISS

    # Algorithm maximum iterations
    max_iterations::Int

    # Stall parameters
    function_tolerance::Float64
    max_stall_time::Float64
    max_stall_iterations::Int

    """
    DEOptions(args...)

    Construct the Differential Evolution (DE) algorithms options.

    # Arguments
    - `general<:GeneralOptions`: The general options.
    - `pim<:AbstractPopulationInitialization`: The population initialization method.
    - `mutation_params<:AbstractMutationParameters`: The mutation strategy parameters.
    - `crossover_params<:AbstractCrossoverParameters`: The crossover strategy parameters.
    - `initial_space<:Union{Nothing,ContinuousRectangularSearchSpace}`: The initial space to initialize the population.
    - `max_iterations::Int`: The maximum number of iterations.
    - `function_tolerance::Float64`: The function tolerance for the stall condition.
    - `max_stall_time::Float64`: The maximum stall time for the stall condition.
    - `max_stall_iterations::Int`: The maximum number of stall iterations for the stall condition.
    """
    function DEOptions(
        general::GO,
        pim::PI,
        mutation::MP,
        crossover::CP,
        initial_space::ISS,
        max_iterations::Int,
        function_tolerance::Float64,
        max_stall_time::Float64,
        max_stall_iterations::Int,
    ) where {MP<:AbstractMutationParameters,CP<:AbstractCrossoverParameters,GO,PI,ISS}
        return new{MP,CP,ISS,PI,GO}(
            general,
            pim,
            mutation,
            crossover,
            initial_space,
            max_iterations,
            function_tolerance,
            max_stall_time,
            max_stall_iterations,
        )
    end
end

"""
    DECache

Cache for the DE algorithm.
"""
mutable struct DECache{T}
    global_best_candidate::Vector{T}
    global_best_fitness::T
    function DECache{T}(num_dims::Integer) where {T}
        return new{T}(zeros(T, num_dims), T(Inf))
    end
end

"""
DE

Differential Evolution (DE) algorithm.
"""
struct DE{
    MP<:AbstractMutationParameters,
    CP<:AbstractCrossoverParameters,
    T<:AbstractFloat,
    E<:BatchEvaluator,
    IBSS,
    PI<:AbstractPopulationInitialization,
    GO<:GeneralOptions,
} <: AbstractOptimizer

    # The DE algorithm options
    options::DEOptions{MP,CP,IBSS,PI,GO}

    # The DE evaluator
    evaluator::E

    # The population
    population::DEPopulation{T}

    # The DE cache
    cache::DECache{T}
end

"""
    SerialDE(prob::AbstractProblem{has_penalty,SS}; kwargs...)

Construct a serial Differential Evolution (DE) algorithm with the given options.

# Arguments
- `prob::AbstractProblem{has_penalty,SS}`: The problem to solve.

# Keyword Arguments
- `eval_method::AbstractFunctionEvaluationMethod=SerialFunctionEvaluation()`: The method to use for evaluating the objective function.
- `num_candidates::Integer=100`: The number of candidates in the population.
- `population_init_method::AbstractPopulationInitialization=UniformInitialization()`: The population initialization method.
- `mutation_params::MP=SelfMutationParameters(Rand1())`: The mutation strategy parameters.
- `crossover_params::CP=BinomialCrossoverParameters(0.6)`: The crossover strategy parameters.
- `initial_bounds::Union{Nothing,ContinuousRectangularSearchSpace}=nothing`: The initial bounds for the search space.
- `max_iterations::Integer=1000`: The maximum number of iterations.
- `max_time::Real=60.0`: The maximum time to run the algorithm.
- `function_tolerance::Real=1e-6`: The function tolerance for the stall condition.
- `max_stall_time::Real=60.0`: The maximum stall time for the stall condition.
- `max_stall_iterations::Integer=100`: The maximum number of stall iterations for the stall condition.
- `min_cost::Real=-Inf`: The minimum cost for the algorithm to stop.
- `function_value_check::Bool=true`: Whether to check the function value.
- `display::Bool=true`: Whether to display the algorithm status.
- `display_interval::Integer=1`: The interval at which to display the algorithm status.
"""
function DE(
    prob::AbstractProblem{has_penalty,SS};
    eval_method::AbstractFunctionEvaluationMethod=SerialFunctionEvaluation(),
    num_candidates::Integer=100,
    population_init_method::AbstractPopulationInitialization=UniformInitialization(),
    mutation_params::MP=SelfMutationParameters(Rand1()),
    crossover_params::CP=BinomialCrossoverParameters(0.6),
    initial_bounds::Union{Nothing,ContinuousRectangularSearchSpace}=nothing,
    max_iterations::Integer=1000,
    max_time::Real=60.0,
    function_tolerance::Real=1e-6,
    max_stall_time::Real=60.0,
    max_stall_iterations::Integer=100,
    min_cost::Real=(-Inf),
    function_value_check::Bool=true,
    display::Bool=false,
    display_interval::Integer=1,
) where {
    MP<:AbstractMutationParameters,
    CP<:AbstractCrossoverParameters,
    T<:AbstractFloat,
    SS<:ContinuousRectangularSearchSpace{T},
    has_penalty,
}
    # Construct options
    options = DEOptions(
        GeneralOptions(
            function_value_check ? Val(true) : Val(false),
            display ? Val(true) : Val(false),
            display_interval,
            max_time,
            min_cost,
        ),
        population_init_method,
        mutation_params,
        crossover_params,
        intersection(search_space(prob), initial_bounds),
        max_iterations,
        function_tolerance,
        max_stall_time,
        max_stall_iterations,
    )

    # Construct evaluator
    return DE(
        options,
        construct_batch_evaluator(eval_method, prob),
        DEPopulation{T}(num_candidates, num_dims(prob)),
        DECache{T}(num_dims(prob)),
    )
end

function optimize!(opt::DE)
    initialize!(opt)
    return iterate!(opt)
end

function initialize!(opt::DE)
    # Unpack DE
    @unpack options, evaluator, population = opt
    @unpack pop_init_method = options

    # Initialize mutation and crossover parameters
    N = length(population)
    initialize!(options.mutation_params, N)
    initialize!(options.crossover_params, num_dims(evaluator.prob), N)

    # Initialize population
    initialize!(population, pop_init_method, options.initial_space)

    # Handle fitness
    initialize_fitness!(population, evaluator)
    check_fitness!(population.current_generation, get_general(options))
    update_global_best!(opt)
    return nothing
end

function iterate!(opt::DE)
    # Unpack DE
    @unpack options, evaluator, population, cache = opt
    search_space = evaluator.prob.ss

    # Initialize DE algorithm parameters
    @unpack mutation_params, crossover_params = options

    # Initialize algorithm stopping criteria requirements
    iteration = 0
    start_time = time()
    current_time = start_time
    stall_start_time = start_time
    stall_value = Inf
    stall_count = 0

    # Begin loop
    exit_flag = 0
    while exit_flag == 0
        # Update iteration counter
        iteration += 1

        # Perform mutation
        mutate!(population, mutation_params)

        # Perform crossover
        crossover!(population, crossover_params, search_space)

        # Perform selection
        evaluate_mutant_fitness!(population, evaluator)
        check_fitness!(population.mutants, get_general(options))
        selection!(population)

        # Update global best
        new_best = update_global_best!(opt)

        # Adapt mutation and crossover parameters if necessary
        adapt!(mutation_params, population.improved, new_best)
        adapt!(crossover_params, population.improved, new_best)

        # Handle stall
        if stall_value - cache.global_best_fitness < options.function_tolerance
            stall_count += 1
        else
            stall_count = 0
            stall_value = cache.global_best_fitness
            stall_start_time = time()
        end

        # Check stopping criteria
        current_time = time()
        if current_time - start_time >= options.general.max_time
            exit_flag = 1
        elseif iteration >= options.max_iterations
            exit_flag = 2
        elseif stall_count >= options.max_stall_iterations
            exit_flag = 3
        elseif current_time - stall_start_time >= options.max_stall_time
            exit_flag = 4
        end

        # Print information
        display_de_status(
            current_time - start_time,
            iteration,
            stall_count,
            cache.global_best_fitness,
            get_general(options),
        )
    end

    return Results(
        cache.global_best_fitness,
        cache.global_best_candidate,
        iteration,
        current_time - start_time,
        exit_flag,
    )
end

function update_global_best!(opt::DE)
    # Grab info
    @unpack population, cache = opt
    @unpack current_generation = population

    # Find index and value of global best fitness if better than previous best
    global_best_fitness = cache.global_best_fitness
    global_best_idx = 0
    @inbounds for (i, fitness) in enumerate(current_generation.candidates_fitness)
        if fitness < global_best_fitness
            global_best_idx = i
            global_best_fitness = fitness
        end
    end

    # Check if we've found a new global best
    updated = false
    if global_best_idx > 0
        updated = true
        cache.global_best_candidate .= current_generation.candidates[global_best_idx]
        cache.global_best_fitness = global_best_fitness
    end
    return updated
end

function display_de_status(
    time, iteration, stall_count, global_fitness, options::GeneralOptions{D,FVC}
) where {D,FVC}
    return display_de_status(
        time, iteration, stall_count, global_fitness, get_display_interval(options), D
    )
end
function display_de_status(
    time, iteration, stall_count, global_fitness, display_interval, ::Type{Val{false}},
)
    return nothing
end
function display_de_status(
    time, iteration, stall_count, global_fitness, display_interval, ::Type{Val{true}},
)
    if iteration % display_interval == 0
        fspec1 = FormatExpr("Time Elapsed: {1:f} sec, Iteration Number: {2:d}")
        fspec2 = FormatExpr("Stall Iterations: {1:d}, Global Best: {2:e}")
        printfmtln(fspec1, time, iteration)
        printfmtln(fspec2, stall_count, global_fitness)
    end
end
