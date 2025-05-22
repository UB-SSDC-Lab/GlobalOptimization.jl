"""
    Swarm{T <: AbstractFloat} <: AbstractPopulation

A population of particles for the PSO algorithm.
"""
struct Swarm{T<:AbstractFloat} <: AbstractPopulation{T}
    # Population information (requried for all AbstractPopulations)
    candidates::Vector{Vector{T}}
    candidates_fitness::Vector{T}

    # PSO specific information
    candidates_velocity::Vector{Vector{T}}
    best_candidates::Vector{Vector{T}}
    best_candidates_fitness::Vector{T}

    function Swarm{T}(num_particles::Integer, num_dims::Integer) where {T}
        num_dims > 0 || throw(ArgumentError("num_dims must be greater than 0."))
        num_particles > 0 || throw(ArgumentError("num_particles must be greater than 0."))
        return new{T}(
            [zeros(T, num_dims) for _ in 1:num_particles],
            zeros(T, num_particles),
            [zeros(T, num_dims) for _ in 1:num_particles],
            [zeros(T, num_dims) for _ in 1:num_particles],
            zeros(T, num_particles),
        )
    end
end

"""
    Swarm(num_particles::Integer, num_dims::Integer)

Constructs a `Swarm` with `num_particles` particles in `num_dims` dimensions.
"""
Swarm(num_particles::Integer, num_dims::Integer) = Swarm{Float64}(num_particles, num_dims)

"""
    Swarm_F64(num_particles::Integer, num_dims::Integer)

Constructs a Float64 `Swarm` with `num_particles` particles in `num_dims` dimensions.
"""
function Swarm_F64(num_particles::Integer, num_dims::Integer)
    Swarm{Float64}(num_particles, num_dims)
end

"""
    Swarm_F32(num_particles::Integer, num_dims::Integer)

Constructs a Float32 `Swarm` with `num_particles` particles in `num_dims` dimensions.
"""
function Swarm_F32(num_particles::Integer, num_dims::Integer)
    Swarm{Float32}(num_particles, num_dims)
end

"""
    initialize!(
        swarm::Swarm{T},
        pop_init_method::AbstractPopulationInitialization,
        search_space::ContinuousRectangularSearchSpace{T}
    )

Initializes the `swarm` population with `pop_init_method` in the `search_space`.
"""
function initialize!(
    swarm::Swarm{T},
    pop_init_method::AbstractPopulationInitialization,
    search_space::ContinuousRectangularSearchSpace{T},
) where {T}
    # Unpack swarm
    @unpack candidates,
    candidates_fitness, candidates_velocity, best_candidates,
    best_candidates_fitness = swarm
    @unpack dim_min, dim_max, dim_delta = search_space

    # Initialize velocities
    vel_min = candidates[1]; # We're using the first candidate here to avoid an allocation
    vel_min .= -dim_delta
    vel_max = dim_delta
    initialize_population_vector!(candidates_velocity, vel_min, vel_max, pop_init_method)

    # Initialize the positions
    initialize_population_vector!(candidates, dim_min, dim_max, pop_init_method)

    return nothing
end

"""
    initialize_fitness!(swarm::Swarm{T}, evaluator::BatchEvaluator)

Initializes the fitness of each candidate in the swarm `swarm` using the `evaluator`.
"""
function initialize_fitness!(swarm::Swarm{T}, evaluator::BatchEvaluator) where {T}
    # Evaluate the cost function for each candidate
    evaluate!(swarm, evaluator)

    # Initialize best candidate information to current candidate information
    # We do this because we have not performed any iterations, therefore,
    # our current information is our best information
    @unpack candidates, candidates_fitness, best_candidates, best_candidates_fitness = swarm
    @inbounds for i in eachindex(candidates)
        best_candidates[i] .= candidates[i]
        best_candidates_fitness[i] = candidates_fitness[i]
    end
    return nothing
end

"""
    evaluate_fitness!(swarm::Swarm{T}, evaluator::BatchEvaluator)

Evaluates the fitness of each candidate in the swarm `swarm` using the `evaluator`.
Updates the swarms best candidates if any are found.
"""
function evaluate_fitness!(swarm::Swarm{T}, evaluator::BatchEvaluator) where {T}
    # Evaluate the cost function for each candidate
    evaluate!(swarm, evaluator)

    # Update best candidate information
    @unpack candidates, candidates_fitness, best_candidates, best_candidates_fitness = swarm
    @inbounds for (i, new_fitness) in enumerate(candidates_fitness)
        if new_fitness < best_candidates_fitness[i]
            best_candidates[i] .= candidates[i]
            best_candidates_fitness[i] = new_fitness
        end
    end
    return nothing
end

"""
    step!(swarm::Swarm)

Steps the swarm `swarm` forward one iteration.
"""
function step!(swarm::Swarm)
    @unpack candidates, candidates_velocity = swarm
    @inbounds for (i, candidate) in enumerate(candidates)
        candidate .+= candidates_velocity[i]
    end
    return nothing
end

"""
    enforce_bounds!(swarm::Swarm{T}, evaluator::BatchEvaluator)

Enforces the bounds of the search space on each candidate in the swarm `swarm`. If a candidate
"""
function enforce_bounds!(
    swarm::Swarm{T}, search_space::ContinuousRectangularSearchSpace{T}
) where {T}
    @unpack candidates, candidates_velocity = swarm
    @inbounds for (i, candidate) in enumerate(candidates)
        for j in eachindex(candidate)
            if candidate[j] < dim_min(search_space, j)
                candidate[j] = dim_min(search_space, j)
                candidates_velocity[i][j] = 0.0
            elseif candidate[j] > dim_max(search_space, j)
                candidate[j] = dim_max(search_space, j)
                candidates_velocity[i][j] = 0.0
            end
        end
    end
    return nothing
end
