
"""
    PSOOptions <: AbstractAlgorithmSpecificOptions

Options for the PSO algorithm.
"""
struct PSOOptions{
    ISS<:Union{Nothing,ContinuousRectangularSearchSpace},
    PI<:AbstractPopulationInitialization,
    GO<:GeneralOptions,
} <: AbstractAlgorithmSpecificOptions
    # The general options
    general::GO

    # Population initialization method
    pop_init_method::PI

    # ===== PSO specific options
    # Defines the search space that the initial particles are drawn from
    initial_space::ISS

    # Max iterations
    max_iterations::Int

    # Stall parameters
    function_tolerence::Float64
    max_stall_time::Float64
    max_stall_iterations::Int

    # Velocity update
    inertia_range::Tuple{Float64,Float64}
    minimum_neighborhood_fraction::Float64
    minimum_neighborhood_size::Int
    self_adjustment_weight::Float64
    social_adjustment_weight::Float64

    function PSOOptions(
        general::GO,
        num_particles,
        pim::PI,
        initial_space::ISS,
        max_iterations,
        function_tolerence,
        max_stall_time,
        max_stall_iterations,
        inertia_range,
        minimum_neighborhood_fraction,
        self_adjustment_weight,
        social_adjustment_weight,
    ) where {GO,PI,ISS}
        minimum_neighborhood_size = max(
            2, floor(Int, num_particles * minimum_neighborhood_fraction)
        )
        return new{ISS,PI,GO}(
            general,
            pim,
            initial_space,
            max_iterations,
            function_tolerence,
            max_stall_time,
            max_stall_iterations,
            inertia_range,
            minimum_neighborhood_fraction,
            minimum_neighborhood_size,
            self_adjustment_weight,
            social_adjustment_weight,
        )
    end
end

"""
    PSOCache

Cache for PSO algorithm.
"""
mutable struct PSOCache{T}
    global_best_candidate::Vector{T}
    global_best_fitness::T
    index_vector::Vector{UInt16}
    function PSOCache{T}(num_particles::Integer, num_dims::Integer) where {T}
        return new{T}(zeros(T, num_dims), T(Inf), collect(0x1:UInt16(num_particles)))
    end
end

"""
    PSO

Particle Swarm Optimization (PSO) algorithm.
"""
struct PSO{T<:AbstractFloat,E<:BatchEvaluator{T},IBSS,PI,GO} <: AbstractOptimizer
    # The PSO algorithm options
    options::PSOOptions{IBSS,PI,GO}

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
- `initial_bounds::Union{Nothing,ContinuousRectangularSearchSpace} = nothing`: The initial bounds to use when initializing particle positions.
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
    population_init_method::AbstractPopulationInitialization=UniformInitialization(),
    initial_bounds::Union{Nothing,ContinuousRectangularSearchSpace}=nothing,
    max_iterations::Int=1000,
    function_tolerence::AbstractFloat=1e-6,
    max_stall_time::Real=Inf,
    max_stall_iterations::Int=25,
    inertia_range::Tuple{AbstractFloat,AbstractFloat}=(0.1, 1.0),
    minimum_neighborhood_fraction::AbstractFloat=0.25,
    self_adjustment_weight::Real=1.49,
    social_adjustment_weight::Real=1.49,
    display::Bool=false,
    display_interval::Int=1,
    function_value_check::Bool=true,
    max_time::Real=60.0,
    min_cost::Real=(-Inf),
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
        num_particles,
        population_init_method,
        intersection(search_space(prob), initial_bounds),
        max_iterations,
        function_tolerence,
        max_stall_time,
        max_stall_iterations,
        inertia_range,
        minimum_neighborhood_fraction,
        self_adjustment_weight,
        social_adjustment_weight,
    )

    # Construct PSO
    return PSO(
        options,
        construct_batch_evaluator(eval_method, prob),
        Swarm{T}(num_particles, num_dims(prob)),
        PSOCache{T}(num_particles, num_dims(prob)),
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
    @unpack options, evaluator, swarm = opt

    # Initialize swarm
    initialize!(swarm, options.pop_init_method, options.initial_space)

    # Handel swarm fitness
    initialize_fitness!(swarm, evaluator)
    check_fitness!(swarm, get_general(options))
    update_global_best!(opt)

    return nothing
end

function iterate!(opt::PSO)
    # Unpack PSO
    @unpack options, evaluator, swarm, cache = opt
    search_space = evaluator.prob.ss

    # Initialize PSO algorithm parameters
    ns = options.minimum_neighborhood_size  # Neighborhood size
    sc = 0                                  # Stall counter
    w = options.inertia_range[2]           # Inertia weight
    y1 = options.self_adjustment_weight     # Self adjustment weight
    y2 = options.social_adjustment_weight   # Social adjustment weight

    # Initialize algorithm stopping criteria requirements
    iteration = 0
    start_time = time()
    current_time = start_time
    stall_start_time = start_time
    stall_value = Inf

    # Begin loop
    exit_flag = 0
    while exit_flag == 0
        # Update iteration counter
        iteration += 1

        # Update swarm velocity and step (enforcing particles are feasible after step)
        update_velocity!(swarm, cache, ns, w, y1, y2)
        step!(swarm)
        enforce_bounds!(swarm, search_space)

        # Evaluate the objective function and check for bad values in fitness
        evaluate_fitness!(swarm, evaluator)
        check_fitness!(swarm, get_general(options))

        # Update global best (STOPPED HERE!)
        update_global_best!(opt)

        # Update inertia
        w = update_inertia(w, options.inertia_range, sc)

        # Handle stall
        if stalled(opt, stall_value)
            sc += 1
        else
            stall_value = cache.global_best_fitness
            sc = 0
            stall_start_time = time()
        end

        # Stopping criteria
        current_time = time()
        if current_time - start_time >= options.general.max_time # Hit maximum time
            exit_flag = 1
        elseif iteration >= options.max_iterations # Hit maximum iterations
            exit_flag = 2
        elseif sc >= options.max_stall_iterations # Hit maximum stall iterations
            exit_flag = 3
        elseif current_time - stall_start_time >= options.max_stall_time # Hit maximum stall time
            exit_flag = 4
        end

        # Output Status
        display_status(
            current_time - start_time,
            iteration,
            sc,
            cache.global_best_fitness,
            get_general(options),
        )
    end

    # Return results
    return Results(
        cache.global_best_fitness,
        cache.global_best_candidate,
        iteration,
        current_time - start_time,
        exit_flag,
    )
end

"""
    update_global_best!(pso::PSO)

Updates the global best candidate in the PSO algorithm `pso` if a better candidate is found.
"""
function update_global_best!(pso::PSO)
    # Grab information
    @unpack swarm, cache = pso
    @unpack best_candidates, best_candidates_fitness = swarm
    @unpack global_best_candidate = cache

    # Find index and value of global best fitness if better than previous best
    global_best_fitness = cache.global_best_fitness
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

"""
    stalled(pso::PSO, stall_value)

Returns true if the PSO algorithm `pso` is stalled, false otherwise.
"""
function stalled(pso::PSO, stall_value)
    @unpack cache, options = pso
    return stall_value - cache.global_best_fitness < options.function_tolerence
end

"""
    update_inertia(inertia, range, stall_count)

Returns new inertia weight based on the current 'inertia', the `range`, and `stall_count`.
"""
function update_inertia(inertia, range, stall_count)
    w = inertia
    if stall_count < 2
        w *= 2.0
    elseif stall_count > 5
        w /= 2.0
    end
    return clamp(w, range[1], range[2])
end

"""
    display_status(time, iteration, stall_count, options)

Displays the status of the PSO algorithm.
"""
@inline function display_status(
    time, iteration, stall_count, global_fitness, options::GeneralOptions{D,FVC}
) where {D,FVC}
    display_status(
        time, iteration, stall_count, global_fitness, get_display_interval(options), D
    )
    return nothing
end
@inline display_status(
    time, iteration, stall_count, global_fitness, display_interval, ::Type{Val{false}}
) = nothing
function display_status(
    time, iteration, stall_count, global_fitness, display_interval, ::Type{Val{true}}
)
    if iteration % display_interval == 0
        fspec1 = FormatExpr("Time Elapsed: {1:f} sec, Iteration Number: {2:d}")
        fspec2 = FormatExpr("Stall Iterations: {1:d}, Global Best: {2:e}")
        printfmtln(fspec1, time, iteration)
        printfmtln(fspec2, stall_count, global_fitness)
    end
    return nothing
end
