"""
    AbstractOptimizer

Abstract type of all optimization algorithms.

All subtypes must have the following fields:
- `options<:AbstractAlgorithmSpecificOptions`: The options for the optimizer. See the
    `AbstractAlgorithmSpecificOptions` interface documentation for details.
- `cache<:AbstractOptimizerCache`: The cache for the optimizer. See the
    `AbstractOptimizerCache` interface documentation for details.

All subtypes must define the following methods:
- `initialize!(opt<:AbstractOptimizer)`: Initialize the optimizer.
- `step!(opt<:AbstractOptimizer)`: Perform a single step/iteration with the optimizer.
- `get_best_fitness(opt<:AbstractOptimizer)`: Get the best fitness of the optimizer.
- `get_best_candidate(opt<:AbstractOptimizer)`: Get the best candidate of the optimizer.
- `show_trace(opt<:AbstractOptimizer, Union{Val{:minimal},Val{:detailed},Val{:all}})`: Print
    the trace of the optimizer to the terminal at the current iteration. Note that a single
    method can by implemented using the `Union` type, or individual methods can be
    implemented for each trace level.
- `get_save_trace(opt<:AbstractOptimizer, Union{Val{:minimal},Val{:detailed},Val{:all}})`:
    Get the trace of the optimizer as a string at the current iteration, which will be
    appended to the optimally specified file. Note that a single method can by implemented
    using the `Union` type, or individual methods can be implemented for each trace level.
"""
abstract type AbstractOptimizer end

"""
    AbstractPopulationBasedOptimizer

Abstract type of all population based optimization algorithms.

All subtypes must have the following fields:
- `options<:AbstractAlgorithmSpecificOptions`: The options for the optimizer. See the
    `AbstractAlgorithmSpecificOptions` interface documentation for details.
- `cache<:AbstractPopulationBasedOptimizerCache`: The cache for the optimizer. See the
    `AbstractPopulationBasedOptimizerCache` interface documentation for details.

All subtypes must define the following methods:
- `initialize!(<:AbstractPopulationBasedOptimizer)`: Initialize the optimizer.
- `step!(<:AbstractPopulationBasedOptimizer)`: Perform a single step/iteration with the
    optimizer.
- `show_trace(opt<:AbstractOptimizer, Union{Val{:minimal},Val{:detailed},Val{:all}})`: Print
    the trace of the optimizer to the terminal at the current iteration. Note that a single
    method can by implemented using the `Union` type, or individual methods can be
    implemented for each trace level.
- `get_save_trace(opt<:AbstractOptimizer, Union{Val{:minimal},Val{:detailed},Val{:all}})`:
    Get the trace of the optimizer as a string at the current iteration, which will be
    appended to the optimally specified file. Note that a single method can by implemented
    using the `Union` type, or individual methods can be implemented for each trace level.

Note that the `get_best_fitness` and `get_best_candidate` methods required by the
`AbstractOptimizer` interface are provided for subtypes of
`AbstractPopulationBasedOptimizer`.
"""
abstract type AbstractPopulationBasedOptimizer <: AbstractOptimizer end

"""
    AbstractOptimizerCache

Abstract type of all optimization algorithm caches.

All subtypes must have the following fields:
- `iteration`: The current iteration number of the optimizer.
- `start_time`: The start time of the optimization.
- `stall_start_time`: THe start time of the stall.
- `stall_iteration`: The iteration number of the stall.
- `stall_value`: The objective function value at start of stall.

These fields must be initialized in the `initialize!(opt)` method of the optimizer.
"""
abstract type AbstractOptimizerCache end

"""
    initialize!(cache::AbstractOptimizerCache, best_fitness)

A helper function to initialize the cache of an optimizer. This function should be
called in the `initialize!(opt)` method of the optimizer.
"""
function initialize!(cache::AbstractOptimizerCache, best_fitness)
    cache.iteration = 0
    cache.start_time = time()
    cache.stall_start_time = cache.start_time
    cache.stall_iteration = 0
    cache.stall_value = best_fitness
    return nothing
end

"""
    AbstractPopulationBasedOptimizerCache

Abstract type of all population based optimization algorithm caches.

All subtypes must have the following fields:
- All fields of `AbstractOptimizerCache`.
- `global_best_candidate`: The global best candidate of the population.
- `global_best_fitness`: The global best fitness of the population.

These fields must be initialized in the `initialize!(opt)` method of the optimizer.
Additionally, the `global_best_candidate` and `global_best_fitness` fields must be
updated in each iteration when appropriate (i.e., when the global best candidate improves)
in `iterate!(opt)`.
"""
abstract type AbstractPopulationBasedOptimizerCache <: AbstractOptimizerCache end

"""
    initialize!(cache::AbstractPopulationBasedOptimizerCache)

A helper function to initialize the cache of a population based optimizer. This function
should be called in the `initialize!(opt)` method of the optimizer, after initializing the
`global_best_candidate` and `global_best_fitness` fields of the cache.
"""
function initialize!(cache::AbstractPopulationBasedOptimizerCache)
    cache.iteration = 0
    cache.start_time = time()
    cache.stall_start_time = cache.start_time
    cache.stall_iteration = 0
    cache.stall_value = cache.global_best_fitness
    return nothing
end

"""
    MinimalOptimizerCache{T}

A minimal implementation of the `AbstractOptimizerCache` type.
"""
mutable struct MinimalOptimizerCache{T} <: AbstractOptimizerCache
    iteration::Int
    start_time::Float64
    stall_start_time::Float64
    stall_iteration::Int
    stall_value::T
    function MinimalOptimizerCache{T}() where {T}
        return new{T}(0, 0.0, 0.0, 0, T(Inf))
    end
end

"""
    MinimalPopulationBasedOptimizerCache{T}

A minimal implementation of the `AbstractPopulationBasedOptimizerCache` type.

Algorithms employing this cache must initialize all fields in `initialize!(opt)` and
must update the `iteration` field each iteration in `iterate!(opt)`. Additionally, the
`global_best_candidate` and `global_best_fitness` fields must be updated in each iteration
when appropriate (i.e., when the global best candidate improves) in `iterate!(opt)`.
"""
mutable struct MinimalPopulationBasedOptimizerCache{T} <: AbstractPopulationBasedOptimizerCache
    # Global best candidate info
    global_best_candidate::Vector{T}
    global_best_fitness::T

    # Generic optimization state variables
    iteration::Int
    start_time::Float64
    stall_start_time::Float64
    stall_iteration::Int
    stall_value::T

    function MinimalPopulationBasedOptimizerCache{T}(num_dims::Integer) where {T}
        return new{T}(zeros(T, num_dims), T(Inf), 0, 0.0, 0.0, 0, T(Inf))
    end
end

# ===== Interface
"""
    optimize!(opt::AbstractOptimizer)

Perform optimization using the optimizer `opt`. Returns the results of the optimization.

# Arguments
- `opt::AbstractOptimizer`: The optimizer to use.

# Returns
- `Results`: The results of the optimization. See the [Results](@ref) docstring for details
    on its contents.

# Example
```julia-repl
julia> using GlobalOptimization
julia> f(x) = sum(x.^2) # Simple sphere function
julia> prob = OptimizationProblem(f, [-1.0, 0.0], [1.0, 2.0])
julia> pso = SerialPSO(prob)
julia> results = optimize!(pso)
Results:
 - Best function value: 6.696180996034206e-20
 - Best candidate: [-2.587698010980842e-10, 0.0]
 - Iterations: 26
 - Time: 0.004351139068603516 seconds
 - Exit flag: 3
```
"""
function optimize!(opt::AbstractOptimizer)
    # Initialize the optimizer
    initialize!(opt)

    # Perform iterations
    status = IN_PROGRESS
    while status == IN_PROGRESS
        # Update iteration counter
        opt.cache.iteration += 1

        # Step the optimizer
        step!(opt)

        # Handle stall
        handle_stall!(opt)

        # Check stopping criteria
        status = check_stopping_criteria(opt)

        # Tracing
        trace(opt)
    end
    return construct_results(opt, status)
end

"""
    initialize!(opt::AbstractOptimizer)

Initialize the optimizer `opt`. All memory allocations that are not possible to do in the
constructor should be done here when possible.
"""
function initialize!(opt::AbstractOptimizer)
    # Initialize the optimizer
    throw(ArgumentError("initialize! not implemented for $(typeof(opt))."))
end

"""
    step!(opt::AbstractOptimizer)

Perform a single step/iteration with the optimizer `opt`. This function should be
non-allocating if possible.
"""
function step!(opt::AbstractOptimizer)
    # Perform iterations
    throw(ArgumentError("step! not implemented for $(typeof(opt))."))
end

"""
    get_iteration(opt::AbstractOptimizer)

Get the current iteration number of the optimizer `opt`.
"""
function get_iteration(opt::AbstractOptimizer)
    return opt.cache.iteration
end

"""
    get_elapsed_time(opt::AbstractOptimizer)

Get the elapsed time of the optimizer `opt` in seconds.
"""
get_elapsed_time(opt::AbstractOptimizer) = time() - opt.cache.start_time

"""
    get_best_fitness(opt::AbstractOptimizer)

Get the fitness of the best candidate found by `opt`.
"""
function get_best_fitness(opt::AbstractOptimizer)
    throw(ArgumentError("get_best_fitness not implemented for $(typeof(opt))."))
end
function get_best_fitness(opt::AbstractPopulationBasedOptimizer)
    return opt.cache.global_best_fitness
end

"""
    get_best_candidate(opt::AbstractOptimizer)

Get the best candidate found by `opt`.
"""
function get_best_candidate(opt::AbstractOptimizer)
    throw(ArgumentError("get_best_candidate not implemented for $(typeof(opt))."))
end
function get_best_candidate(opt::AbstractPopulationBasedOptimizer)
    return opt.cache.global_best_candidate
end

"""
    handle_stall!(opt::AbstractOptimizer)

Handles updating the stall related fields of the `AbstractOptimizerCache` for `opt`.
"""
function handle_stall!(opt::AbstractOptimizer)
    @unpack cache, options = opt
    if cache.stall_value - get_best_fitness(opt) < get_function_tolerance(options)
        cache.stall_iteration += 1
    else
        cache.stall_value = get_best_fitness(opt)
        cache.stall_iteration = 0
        cache.stall_start_time = time()
    end
end


"""
    check_stopping_criteria(opt::AbstractOptimizer)

Check if `opt` satisfies any stopping criteria. Returns the appropriate `Status` enum.
"""
function check_stopping_criteria(opt::AbstractOptimizer)
    @unpack cache, options = opt
    current_time = time()
    if get_best_fitness(opt) <= get_min_cost(options)
        return MINIMUM_COST_ACHIEVED
    elseif current_time - cache.start_time >= get_max_time(options)
        return MAXIMUM_TIME_EXCEEDED
    elseif cache.iteration >= get_max_iterations(options)
        return MAXIMUM_ITERATIONS_EXCEEDED
    elseif cache.stall_iteration >= get_max_stall_iterations(options)
        return MAXIMUM_STALL_ITERATIONS_EXCEEDED
    elseif current_time - cache.stall_start_time >= get_max_stall_time(options)
        return MAXIMUM_STALL_TIME_EXCEEDED
    end
    return IN_PROGRESS
end

"""
    construct_results(opt::AbstractOptimizer, status::Status)

Construct the results from the optimizer `opt` with the appropriate `Status` to indicate
the stopping criteria.
"""
function construct_results(opt::AbstractOptimizer, status::Status)
    @unpack cache = opt
    return Results(
        get_best_fitness(opt),
        get_best_candidate(opt),
        get_iteration(opt),
        get_elapsed_time(opt),
        status,
    )
end
