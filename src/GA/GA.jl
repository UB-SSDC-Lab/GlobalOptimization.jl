"""
GAOptions

Options for the Genetic Algorithm (GA) algorithms.

# Fields:
- `general<:GeneralOptions`: The general options.
- `pop_init_method<:AbstractPopulationInitialization`: The population initialization method.
- `selection_params<:AbstractGASelectionParameters`: The selection parameters
- `crossover_params<:AbstractGACrossoverParameters`: The crossover strategy parameters.
- `mutation_params<:AbstractGAMutationParameters`: The mutation strategy parameters.
- `initial_space<:Union{Nothing,ContinuousRectangularSearchSpace}`: The initial space to initialize the population.
"""
struct GAOptions{
    SP<:AbstractGASelectionParameters,
    CP<:AbstractGACrossoverParameters,
    MP<:AbstractGAMutationParameters,
    EP<:AbstractGAElitismParameters,
    ISS<:Union{Nothing,ContinuousRectangularSearchSpace},
    PI<:AbstractPopulationInitialization,
    GO<:GeneralOptions,
} <: AbstractAlgorithmSpecificOptions
    # The general options
    general::GO

    # The Population initialization method
    pop_init_method::PI

    # Selection params
    selection_params::SP

    # Crossover params
    crossover_params::CP

    # Mutation strategy parameters
    mutation_params::MP

    elitism_params::EP

    # Initial space to initialize the population
    initial_space::ISS

    """
    GAOptions(args...)

    Construct the Genetic Algorithm (GA) algorithms options.

    # Arguments
    - `general<:GeneralOptions`: The general options.
    - `pim<:AbstractPopulationInitialization`: The population initialization method.
    - `selection<:AbstractGASelectionParameters`: The selection parameters
    - `crossover<:AbstractGACrossoverParameters`: The crossover strategy parameters.
    - `mutation<:AbstractGAMutationParameters`: The mutation strategy parameters.
    - `elitism<:AbstractGAElitismParameters`: Elitism Parameters
    - `initial_space<:Union{Nothing,ContinuousRectangularSearchSpace}`: The initial space to initialize the population.
    """
    function GAOptions(
        general::GO, pim::PI, selection::SP, crossover::CP, mutation::MP, elitism::EP, initial_space::ISS
    ) where {SP<:AbstractGASelectionParameters, MP<:AbstractGAMutationParameters,CP<:AbstractGACrossoverParameters,EP<:AbstractGAElitismParameters,GO,PI,ISS}
        return new{SP,CP,MP,EP,ISS,PI,GO}(general, pim, selection, crossover, mutation, elitism, initial_space)
    end
end


"""
    GA

The genetic algorithm
"""
struct GA{
    SP<:AbstractGASelectionParameters,
    CP<:AbstractGACrossoverParameters,
    MP<:AbstractGAMutationParameters,
    EP<:AbstractGAElitismParameters,
    T<:AbstractFloat,
    E<:BatchEvaluator,
    IBSS,
    PI<:AbstractPopulationInitialization,
    GO<:GeneralOptions,
} <: AbstractPopulationBasedOptimizer
    # The GA algorithm options
    options::GAOptions{SP,CP,MP,EP,IBSS,PI,GO}

    # The GA evaluator
    evaluator::E

    # The population
    population::GAPopulation{T}

    # The GA cache
    cache::MinimalPopulationBasedOptimizerCache{T}
end



"""
    GA(prob::AbstractProblem{has_penalty,SS}; kwargs...)

Construct a serial Genetic Algorithm (GA) algorithm with the given options.

# Arguments
- `prob::AbstractProblem{has_penalty,SS}`: The problem to solve.

# Keyword Arguments
- `eval_method::AbstractFunctionEvaluationMethod=SerialFunctionEvaluation()`: The method to use for evaluating the objective function.
- `num_candidates::Integer=100`: The number of candidates in the population.
- `population_initialization::AbstractPopulationInitialization=UniformInitialization()`: The population initialization method.
- `selection_params::SP=StochasticUniversalSampling(100)`: The selection parameters.
- `crossover_params::CP=BLXAlphaCrossover(0.8, 0.5)`: The crossover strategy parameters.
- `mutation_params::MP=RandomElementwiseMutation(0.05, 0.01)`: The mutation strategy parameters.
- `elitism_params::EP=SimpleElitism(0.03)`: the elitism parameters
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
- `save_file::String="trace.txt"`: The file to save the trace to.
- `trace_level::TraceLevel=TraceMinimal(1)`: The trace level to use.
"""
function GA(
    prob::AbstractProblem{has_penalty,SS};
    eval_method::AbstractFunctionEvaluationMethod=SerialFunctionEvaluation(),
    num_candidates::Integer=100,
    population_initialization::AbstractPopulationInitialization=UniformInitialization(),
    selection_params::SP=StochasticUniversalSampling(100),
    crossover_params::CP=BLXAlphaCrossover(0.8, 0.5),
    mutation_params::MP=RandomElementwiseMutation(0.05, 0.01),
    elitism_params::EP=NoElitism(),
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
    save_file::String="trace.txt",
    trace_level::TraceLevel=TraceMinimal(1),
) where {
    SP<:AbstractGASelectionParameters,
    MP<:AbstractGAMutationParameters,
    CP<:AbstractGACrossoverParameters,
    EP<:AbstractGAElitismParameters,
    T<:AbstractFloat,
    SS<:ContinuousRectangularSearchSpace{T},
    has_penalty,
} 

    # Construct options
    options = GAOptions(
        GeneralOptions(
            GlobalOptimizationTrace(show_trace, save_trace, save_file, trace_level),
            function_value_check,
            min_cost,
            max_time,
            max_iterations,
            function_tolerance,
            max_stall_time,
            max_stall_iterations,
        ),
        population_initialization,
        selection_params,
        crossover_params,
        mutation_params,
        elitism_params,
        intersection(search_space(prob), initial_space),
    )

    # Construct evaluator
    return GA(
        options,
        construct_batch_evaluator(eval_method, prob),
        GAPopulation{T}(num_candidates, num_dims(prob)),
        MinimalPopulationBasedOptimizerCache{T}(num_dims(prob)),
    )
end

# ===== AbstractPopBasedOptimizer interface
get_population(opt::GA) = opt.population.current_generation

# Initialize parameters and pop and evaluate pop once
function initialize!(opt::GA) 
    # Unpack DE
    @unpack options, evaluator, population, cache = opt
    @unpack pop_init_method, mutation_params, crossover_params, selection_params = options

    # Initialize population
    initialize!(population, pop_init_method, options.initial_space)

    # Initialize selection, crossover, and mutation
    initialize!(mutation_params)
    initialize!(crossover_params)
    initialize!(selection_params, population)


    # Handle fitness
    initialize_fitness!(population, evaluator)
    check_fitness!(population.current_generation, get_function_value_check(options))

    # Initialize cache
    update_global_best!(opt)
    initialize!(cache)

    return nothing
end

function step!(opt::GA)
    # Unpack GA
    @unpack options, evaluator, population, cache = opt
    @unpack mutation_params, crossover_params, selection_params, elitism_params = options
    search_space = evaluator.prob.ss

    # Perform selection
    selection!(population, selection_params)

    # Perform crossover
    crossover!(population, crossover_params)

    # Perform mutation
    mutate!(population, mutation_params, search_space)

    # Evaluate fitness
    evaluate_children!(population, evaluator)
    check_fitness!(population.children, get_function_value_check(options))

    set_next_generation!(population, elitism_params)

    # Update global best
    improved = update_global_best!(opt)
    
    # Adapt mutation and crossover parameters if necessary
    adapt!(mutation_params, improved)
    adapt!(crossover_params, improved)

    return nothing
end