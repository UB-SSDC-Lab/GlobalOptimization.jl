
"""
    PSOOptions <: AbstractAlgorithmSpecificOptions

Options for the PSO algorithm.
"""
struct PSOOptions{
    VU<:AbstractVelocityUpdateScheme,
    ISS<:Union{Nothing,ContinuousRectangularSearchSpace},
    PI<:AbstractPopulationInitialization,
    GO<:GeneralOptions,
} <: AbstractAlgorithmSpecificOptions
    # The general options
    general::GO

    # Population initialization method
    pop_init_method::PI

    # Velocity update scheme
    velocity_update::VU

    # The initial space to initialize the particles in. This is calculated as the
    # intersection of the search space and the separate user provided initial space.
    initial_space::ISS

    # Max iterations
    max_iterations::Int

    # Stall parameters
    function_tolerance::Float64
    max_stall_time::Float64
    max_stall_iterations::Int

    function PSOOptions(
        general::GO,
        pim::PI,
        velocity_update::VU,
        initial_space::ISS,
        max_iterations::Int,
        function_tolerance,
        max_stall_time,
        max_stall_iterations,
    ) where {VU,ISS,PI,GO}
        return new{VU,ISS,PI,GO}(
            general,
            pim,
            velocity_update,
            initial_space,
            max_iterations,
            function_tolerance,
            max_stall_time,
            max_stall_iterations,
        )
    end
end

"""
    PSOCache

Cache for PSO algorithm.
"""
mutable struct PSOCache{T}
    # Global best candidate info
    global_best_candidate::Vector{T}
    global_best_fitness::T

    # Generic optimization state variables
    iteration::Int
    start_time::Float64
    stall_start_time::Float64
    stall_iteration::Int
    stall_value::T

    function PSOCache{T}(num_dims::Integer) where {T}
        return new{T}(zeros(T, num_dims), T(Inf), 0, 0.0, 0.0, 0, T(Inf))
    end
end

"""
    PSO

Particle Swarm Optimization (PSO) algorithm.
"""
struct PSO{
    VU<:AbstractVelocityUpdateScheme,
    T<:AbstractFloat,
    E<:BatchEvaluator,
    IBSS,
    PI<:AbstractPopulationInitialization,
    GO<:GeneralOptions,
} <: AbstractOptimizer
    # The PSO algorithm options
    options::PSOOptions{VU,IBSS,PI,GO}

    # The PSO evaluator
    evaluator::E

    # The PSO swarm
    swarm::Swarm{T}

    # The PSO cache
    cache::PSOCache{T}
end

"""
    PSO(prob::AbstractProblem{has_penalty,SS}; kwargs...)

Constructs a PSO algorithm with the given options.

# Arguments
- `prob::AbstractProblem{has_penalty,SS}`: The problem to solve.

# Keyword Arguments
- `eval_method::AbstractFunctionEvaluationMethod=SerialFunctionEvaluation()`: The method to use for evaluating the objective function.
- `num_particles::Int = 100`: The number of particles to use.
- `population_initialization::AbstractPopulationInitialization = UniformInitialization()`: The method to use for initializing the population.
- `velocity_update::AbstractVelocityUpdateScheme = MATLABVelocityUpdate()`: The method to use for updating the velocity of the particles.
- `initial_space::Union{Nothing,ContinuousRectangularSearchSpace} = nothing`: The initial bounds to use when initializing particle positions.
- `max_iterations::Int = 1000`: The maximum number of iterations to perform.
- `function_tolerence::AbstractFloat = 1e-6`: The function value tolerence to use for stopping criteria.
- `max_stall_time::Real = Inf`: The maximum amount of time to allow for stall time.
- `max_stall_iterations::Int = 25`: The maximum number of stall iterations to allow.
- `max_time::Real = 60.0`: The maximum amount of time to allow for optimization.
- `min_cost::Real = -Inf`: The minimum cost to allow for optimization.
- `function_value_check::Union{Val{false},Val{true}} = Val(true)`: Whether to check for bad function values.
- `show_trace::Union{Val{false},Val{true}} = Val(false)`: Whether to show trace information.
- `save_trace::Union{Val{false},Val{true}} = Val(false)`: Whether to save trace information.
- `save_file::String = "no_file.txt"`: The file to save trace information to.
- `trace_level::TraceLevel = TraceMinimal(1)`: The level of trace information to show.

# Returns
- `PSO`: The PSO algorithm.
"""
function PSO(
    prob::AbstractProblem{has_penalty,SS};
    eval_method::AbstractFunctionEvaluationMethod=SerialFunctionEvaluation(),
    num_particles::Int=100,
    population_initialization::AbstractPopulationInitialization=UniformInitialization(),
    velocity_update::AbstractVelocityUpdateScheme=MATLABVelocityUpdate(),
    initial_space::Union{Nothing,ContinuousRectangularSearchSpace}=nothing,
    max_iterations::Int=1000,
    function_tolerance::AbstractFloat=1e-6,
    max_stall_time::Real=Inf,
    max_stall_iterations::Int=25,
    max_time::Real=60.0,
    min_cost::Real=(-Inf),
    function_value_check::Union{Val{false},Val{true}}=Val(true),
    show_trace::Union{Val{false},Val{true}}=Val(false),
    save_trace::Union{Val{false},Val{true}}=Val(false),
    save_file::String="no_file.txt",
    trace_level::TraceLevel=TraceMinimal(1),
) where {T<:AbstractFloat,SS<:ContinuousRectangularSearchSpace{T},has_penalty}
    # Construct the options
    options = PSOOptions(
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
        velocity_update,
        intersection(search_space(prob), initial_space),
        max_iterations,
        function_tolerance,
        max_stall_time,
        max_stall_iterations,
    )

    # Construct PSO
    return PSO(
        options,
        construct_batch_evaluator(eval_method, prob),
        Swarm{T}(num_particles, num_dims(prob)),
        PSOCache{T}(num_dims(prob)),
    )
end
function PSO(prob::AbstractProblem{hp,SS}; kwargs...) where {hp,SS<:SearchSpace}
    throw(
        ArgumentError(
            "PSO only supports OptimizationProblem defined with a ContinuousRectangularSearchSpace.",
        ),
    )
end

# ===== AbstractOptimizer interface

get_iteration(opt::PSO) = opt.cache.iteration

function initialize!(opt::PSO)
    # Unpack PSO
    @unpack options, evaluator, swarm, cache = opt

    # Initialize the velocity update scheme
    initialize!(options.velocity_update, length(swarm))

    # Initialize swarm
    initialize!(swarm, options.pop_init_method, options.initial_space)

    # Handle swarm fitness
    initialize_fitness!(swarm, evaluator)
    check_fitness!(swarm, get_function_value_check(options))

    # Initialize PSO cache
    update_global_best!(opt)
    cache.iteration = 0
    cache.start_time = time()
    cache.stall_start_time = cache.start_time
    cache.stall_iteration = 0
    cache.stall_value = cache.global_best_fitness

    return nothing
end

function iterate!(opt::PSO)
    # Unpack PSO
    @unpack options, evaluator, swarm, cache = opt
    velocity_update = options.velocity_update
    search_space = evaluator.prob.ss

    # Begin loop
    status = IN_PROGRESS
    while status == IN_PROGRESS
        # Update iteration counter
        cache.iteration += 1

        # Update velocity
        update_velocity!(swarm, velocity_update)

        # Step swarm forward and enforce bounds
        step!(swarm)
        enforce_bounds!(swarm, search_space)

        # Evaluate the objective function and check for bad values in fitness
        evaluate_fitness!(swarm, evaluator)
        check_fitness!(swarm, get_function_value_check(options))

        # Update global best
        improved = update_global_best!(opt)

        # Adapt the velocity update scheme
        adapt!(velocity_update, improved, cache.stall_iteration)

        # Handle stall check
        handle_stall!(opt)

        # Check stopping criteria
        status = check_stopping_criteria(opt)

        # Tracing
        trace(opt)
    end

    # Return results
    return construct_results(opt, status)
end

# ===== Implementation

"""
    update_global_best!(pso::PSO)

Updates the global best candidate in the PSO algorithm `pso` if a better candidate is found.
"""
function update_global_best!(pso::PSO)
    # Grab information
    @unpack swarm, cache = pso
    @unpack best_candidates, best_candidates_fitness = swarm
    @unpack global_best_candidate, global_best_fitness = cache

    # Find index and value of global best fitness if better than previous best
    global_best_idx = 0
    @inbounds for (i, fitness) in enumerate(best_candidates_fitness)
        if fitness < global_best_fitness
            global_best_idx = i
            global_best_fitness = fitness
        end
    end

    # Check if we've found a better solution
    updated = false
    if global_best_idx > 0
        updated = true
        global_best_candidate .= best_candidates[global_best_idx]
        cache.global_best_fitness = global_best_fitness
    end
    return updated
end

get_best_fitness(pso::PSO) = pso.cache.global_best_fitness

function handle_stall!(pso::PSO)
    @unpack cache, options = pso
    if cache.stall_value - get_best_fitness(pso) < options.function_tolerance
        # Currently stalled...
        cache.stall_iteration += 1
    else
        # Not stalled!!
        cache.stall_value = get_best_fitness(pso)
        cache.stall_iteration = 0
        cache.stall_start_time = time()
    end
end

function check_stopping_criteria(pso::PSO)
    @unpack cache, options = pso
    current_time = time()
    if get_best_fitness(pso) <= get_min_cost(options)
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

function construct_results(pso::PSO, status::Status)
    @unpack cache = pso
    return Results(
        cache.global_best_fitness,
        cache.global_best_candidate,
        cache.iteration,
        time() - cache.start_time,
        status,
    )
end

function show_trace(pso::PSO, ::Any)

end

function get_save_trace(pso::PSO, ::Any)

end
