"""
DEOptions

Options for the Differential Evolution (DE) algorithms.

# Fields:
- `general<:GeneralOptions`: The general options.
- `pop_init_method<:AbstractPopulationInitialization`: The population initialization method.
- `mutation_params<:AbstractMutationParameters`: The mutation strategy parameters.
- `crossover_params<:AbstractCrossoverParameters`: The crossover strategy parameters.
- `initial_space<:Union{Nothing,ContinuousRectangularSearchSpace}`: The initial space to initialize the population.
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

    """
    DEOptions(args...)

    Construct the Differential Evolution (DE) algorithms options.

    # Arguments
    - `general<:GeneralOptions`: The general options.
    - `pim<:AbstractPopulationInitialization`: The population initialization method.
    - `mutation_params<:AbstractMutationParameters`: The mutation strategy parameters.
    - `crossover_params<:AbstractCrossoverParameters`: The crossover strategy parameters.
    - `initial_space<:Union{Nothing,ContinuousRectangularSearchSpace}`: The initial space to initialize the population.
    """
    function DEOptions(
        general::GO,
        pim::PI,
        mutation::MP,
        crossover::CP,
        initial_space::ISS,
    ) where {MP<:AbstractMutationParameters,CP<:AbstractCrossoverParameters,GO,PI,ISS}
        return new{MP,CP,ISS,PI,GO}(
            general,
            pim,
            mutation,
            crossover,
            initial_space,
        )
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
} <: AbstractPopulationBasedOptimizer
    # The DE algorithm options
    options::DEOptions{MP,CP,IBSS,PI,GO}

    # The DE evaluator
    evaluator::E

    # The population
    population::DEPopulation{T}

    # The DE cache
    cache::MinimalPopulationBasedOptimizerCache{T}
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
            min_cost,
            max_time,
            max_iterations,
            function_tolerance,
            max_stall_time,
            max_stall_iterations,
        ),
        population_initialization,
        mutation_params,
        crossover_params,
        intersection(search_space(prob), initial_space),
    )

    # Construct evaluator
    return DE(
        options,
        construct_batch_evaluator(eval_method, prob),
        DEPopulation{T}(num_candidates, num_dims(prob)),
        MinimalPopulationBasedOptimizerCache{T}(num_dims(prob)),
    )
end

# ===== AbstractPopulationBasedOptimizer interface
function initialize!(opt::DE)
    # Unpack DE
    @unpack options, evaluator, population, cache = opt
    @unpack pop_init_method, mutation_params, crossover_params = options

    # Initialize mutation and crossover parameters
    N = length(population)
    initialize!(mutation_params, N)
    initialize!(crossover_params, num_dims(evaluator.prob), N)

    # Initialize population
    initialize!(population, pop_init_method, options.initial_space)

    # Handle fitness
    initialize_fitness!(population, evaluator)
    check_fitness!(population.current_generation, get_function_value_check(options))

    # Initialize cache
    update_global_best!(opt)
    initialize!(cache)

    return nothing
end

function step!(opt::DE)
    # Unpack DE
    @unpack options, evaluator, population, cache = opt
    @unpack mutation_params, crossover_params = options
    search_space = evaluator.prob.ss

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

    return nothing
end

function show_trace(de::DE, ::Union{Val{:minimal}, Val{:detailed}, Val{:all}})

end

function get_save_trace(de::DE, ::Union{Val{:minimal}, Val{:detailed}, Val{:all}})

end

# ===== Implementation Specific Methods

"""
    update_global_best!(opt::DE)

Updates the global best candidate in the DE algorithm `opt` if a better candidate is found.
"""
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
