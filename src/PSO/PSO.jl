
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
- `population_init_method::AbstractPopulationInitialization = UniformInitialization()`: The method to use for initializing the population.
- `initial_space::Union{Nothing,ContinuousRectangularSearchSpace} = nothing`: The initial bounds to use when initializing particle positions.
- `max_iterations::Int = 1000`: The maximum number of iterations to perform.
- `function_tolerence::AbstractFloat = 1e-6`: The function value tolerence to use for stopping criteria.
- `max_stall_time::Real = Inf`: The maximum amount of time to allow for stall time.
- `max_stall_iterations::Int = 25`: The maximum number of stall iterations to allow.
- `inertia_range::Tuple{AbstractFloat,AbstractFloat} = (0.1, 1.0)`: The range of allowable inertia weights.
- `minimum_neighborhood_fraction::AbstractFloat = 0.25`: The minimum neighborhood fraction to use.
- `self_adjustment_weight::Real = 1.49`: The self adjustment weight to use.
- `social_adjustment_weight::Real = 1.49`: The social adjustment weight to use.
- `display::Bool = false`: Whether or not to display the status of the algorithm.
- `display_interval::Int = 1`: The display interval to use.
- `function_value_check::Bool = true`: Whether or not to check for bad function values (Inf or NaN).
- `max_time::Real = 60.0`: The maximum amount of time to allow for optimization.

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
    function_value_check::Bool=true,
    display::Bool=false,
    display_interval::Int=1,
) where {T<:AbstractFloat,SS<:ContinuousRectangularSearchSpace{T},has_penalty}
    # Construct the options
    options = PSOOptions(
        GeneralOptions(
            function_value_check ? Val(true) : Val(false),
            display ? Val(true) : Val(false),
            display_interval,
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

function optimize!(opt::PSO)
    # Initialize PSO algorithm
    initialize!(opt)

    # Perform iterations and return results
    return iterate!(opt)
end

function initialize!(opt::PSO)
    # Unpack PSO
    @unpack options, evaluator, swarm, cache = opt

    # Initialize swarm
    initialize!(swarm, options.pop_init_method, options.initial_space)

    # Handel swarm fitness
    initialize_fitness!(swarm, evaluator)
    check_fitness!(swarm, get_general(options))

    # Initialize the velocity update scheme
    initialize!(options.velocity_update, length(swarm))

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
    exit_flag = 0
    while exit_flag == 0
        # Update iteration counter
        cache.iteration += 1

        # Update velocity
        update_velocity!(swarm, velocity_update)

        # Step swarm forward and enforce bounds
        step!(swarm)
        enforce_bounds!(swarm, search_space)

        # Evaluate the objective function and check for bad values in fitness
        evaluate_fitness!(swarm, evaluator)
        check_fitness!(swarm, get_general(options))

        # Update global best
        improved = update_global_best!(opt)

        # Handle stall check
        handle_stall!(opt)

        # Adapt the velocity update scheme
        adapt!(velocity_update, improved, cache.stall_iteration)

        # Check stopping criteria
        exit_flag = check_stopping_criteria(opt)

        # Output Status
        display_status(opt)
    end

    # Return results
    return construct_results(opt, exit_flag)
end

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

function handle_stall!(pso::PSO)
    @unpack cache, options = pso
    if cache.stall_value - cache.global_best_fitness < options.function_tolerance
        # Currently stalled...
        cache.stall_iteration += 1
    else
        # Not stalled!!
        cache.stall_value = cache.global_best_fitness
        cache.stall_iteration = 0
        cache.stall_start_time = time()
    end
end

function check_stopping_criteria(pso::PSO)
    @unpack cache, options = pso
    current_time = time()
    if current_time - cache.start_time >= get_max_time(options)
        return 1
    elseif cache.iteration >= options.max_iterations
        return 2
    elseif cache.stall_iteration >= options.max_stall_iterations
        return 3
    elseif current_time - cache.stall_start_time >= options.max_stall_time
        return 4
    end
    return 0
end

function construct_results(pso::PSO, exit_flag::Int)
    @unpack cache = pso
    return Results(
        cache.global_best_fitness,
        cache.global_best_candidate,
        cache.iteration,
        time() - cache.start_time,
        exit_flag,
    )
end

"""
    display_status(pso::PSO)

Displays the status of the PSO algorithm.
"""
@inline function display_status(
    pso::PSO{VU,T,E,IBSS,PI,GeneralOptions{Val{false},FVC}}
) where {VU,T,E,IBSS,PI,FVC}
    return nothing
end
@inline function display_status(
    pso::PSO{VU,T,E,IBSS,PI,GeneralOptions{Val{true},FVC}}
) where {VU,T,E,IBSS,PI,FVC}
    @unpack cache, options = pso
    go = get_general(options)

    if cache.iteration % go.display_interval == 0
        fspec1 = FormatExpr("Time Elapsed: {1:f} sec, Iteration Number: {2:d}")
        fspec2 = FormatExpr("Stall Iterations: {1:d}, Global Best: {2:e}")
        printfmtln(fspec1, time() - cache.start_time, cache.iteration)
        printfmtln(fspec2, cache.stall_iteration, cache.global_best_fitness)
    end
    return nothing
end
