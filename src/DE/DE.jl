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
    # Global best candidate info
    global_best_candidate::Vector{T}
    global_best_fitness::T

    # Generic optimization state variables
    iteration::Int
    start_time::Float64
    stall_start_time::Float64
    stall_iteration::Int
    stall_value::T
    function DECache{T}(num_dims::Integer) where {T}
        return new{T}(zeros(T, num_dims), T(Inf), 0, 0.0, 0.0, 0, T(Inf))
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
    DE(prob::AbstractProblem{has_penalty,SS}; kwargs...)

Construct a serial Differential Evolution (DE) algorithm with the given options.

# Arguments
- `prob::AbstractProblem{has_penalty,SS}`: The problem to solve.

# Keyword Arguments
- `eval_method::AbstractFunctionEvaluationMethod=SerialFunctionEvaluation()`: The method to use for evaluating the objective function.
- `num_candidates::Integer=100`: The number of candidates in the population.
- `population_initialization::AbstractPopulationInitialization=UniformInitialization()`: The population initialization method.
- `mutation_params::MP=SelfMutationParameters(Rand1())`: The mutation strategy parameters.
- `crossover_params::CP=BinomialCrossoverParameters(0.6)`: The crossover strategy parameters.
- `initial_space::Union{Nothing,ContinuousRectangularSearchSpace}=nothing`: The initial bounds for the search space.
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
    population_initialization::AbstractPopulationInitialization=UniformInitialization(),
    mutation_params::MP=SelfMutationParameters(Rand1()),
    crossover_params::CP=BinomialCrossoverParameters(0.6),
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
        population_initialization,
        mutation_params,
        crossover_params,
        intersection(search_space(prob), initial_space),
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

# ===== AbstractOptimizer interface

get_iteration(opt::DE) = opt.cache.iteration


function initialize!(opt::DE)
    # Unpack DE
    @unpack options, evaluator, population, cache = opt
    @unpack pop_init_method = options

    # Initialize mutation and crossover parameters
    N = length(population)
    initialize!(options.mutation_params, N)
    initialize!(options.crossover_params, num_dims(evaluator.prob), N)

    # Initialize population
    initialize!(population, pop_init_method, options.initial_space)

    # Handle fitness
    initialize_fitness!(population, evaluator)
    check_fitness!(population.current_generation, get_function_value_check(options))

    # Initialize cache
    update_global_best!(opt)
    cache.iteration = 0
    cache.start_time = time()
    cache.stall_start_time = cache.start_time
    cache.stall_iteration = 0
    cache.stall_value = cache.global_best_fitness

    return nothing
end

function iterate!(opt::DE)
    # Unpack DE
    @unpack options, evaluator, population, cache = opt
    @unpack mutation_params, crossover_params = options
    search_space = evaluator.prob.ss

    # Begin loop
    status = IN_PROGRESS
    while status == IN_PROGRESS
        # Update iteration counter
        cache.iteration += 1

        # Perform mutation
        mutate!(population, mutation_params)

        # Perform crossover
        crossover!(population, crossover_params, search_space)

        # Perform selection
        evaluate_mutant_fitness!(population, evaluator)
        check_fitness!(population.mutants, get_function_value_check(options))
        selection!(population)

        # Update global best
        improved = update_global_best!(opt)

        # Adapt mutation and crossover parameters if necessary
        adapt!(mutation_params, population.improved, improved)
        adapt!(crossover_params, population.improved, improved)

        # Handle stall
        handle_stall!(opt)

        # Check stopping criteria
        status = check_stopping_criteria(opt)

        # Print information
        trace(opt)
    end

    return construct_results(opt, status)
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

get_best_fitness(de::DE) = de.cache.global_best_fitness

function handle_stall!(de::DE)
    @unpack cache, options = de
    if cache.stall_value - get_best_fitness(de) < options.function_tolerance
        # Currently stalled...
        cache.stall_iteration += 1
    else
        # Not stalled!!
        cache.stall_value = get_best_fitness(de)
        cache.stall_iteration = 0
        cache.stall_start_time = time()
    end
end

function check_stopping_criteria(de::DE)
    @unpack cache, options = de
    current_time = time()
    if get_best_fitness(de) <= get_min_cost(options)
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

function construct_results(de::DE, status::Status)
    @unpack cache = de
    return Results(
        cache.global_best_fitness,
        cache.global_best_candidate,
        cache.iteration,
        time() - cache.start_time,
        status,
    )
end

function show_trace(de::DE, ::Any)

end

function get_save_trace(de::DE, ::Any)

end
