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
Swarm_F64(num_particles::Integer, num_dims::Integer) =
    Swarm{Float64}(num_particles, num_dims)

"""
    Swarm_F32(num_particles::Integer, num_dims::Integer)

Constructs a Float32 `Swarm` with `num_particles` particles in `num_dims` dimensions.
"""
Swarm_F32(num_particles::Integer, num_dims::Integer) =
    Swarm{Float32}(num_particles, num_dims)

"""
    initialize_uniform!(swarm::Swarm{T}, search_space::ContinuousRectangularSearchSpace{T})

Initializes the swarm `swarm` with a uniform particle distribution in the search space.
"""
function initialize_uniform!(
    swarm::Swarm{T}, search_space::ContinuousRectangularSearchSpace{T}
) where {T}
    # Unpack swarm
    @unpack candidates,
    candidates_fitness, candidates_velocity, best_candidates,
    best_candidates_fitness = swarm

    # Initialize each candidate
    @inbounds for i in eachindex(candidates)
        # Get candidate position and velocity
        pos = candidates[i]
        vel = candidates_velocity[i]

        # Iterate over each dimension
        for j in eachindex(pos)
            dmin = dimmin(search_space, j)
            dΔ = dimdelta(search_space, j)

            # Set position and velocity
            pos[j] = dmin + dΔ * rand(T)
            vel[j] = -dΔ + 2.0 * dΔ * rand(T)
        end
    end
    return nothing
end

"""
    initialize_fitness!(swarm::Swarm{T}, evaluator::BatchEvaluator{T})

Initializes the fitness of each candidate in the swarm `swarm` using the `evaluator`.
"""
function initialize_fitness!(swarm::Swarm{T}, evaluator::BatchEvaluator{T}) where {T}
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
    evaluate_fitness!(swarm::Swarm{T}, evaluator::BatchEvaluator{T})

Evaluates the fitness of each candidate in the swarm `swarm` using the `evaluator`.
Updates the swarms best candidates if any are found.
"""
function evaluate_fitness!(swarm::Swarm{T}, evaluator::BatchEvaluator{T}) where {T}
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
    update_velocity!(swarm::Swarm{T}, cache::Cache, ns::Integer, w, y1, y2)

Updates the velocity of each candidate in the swarm `swarm`,
"""
function update_velocity!(swarm::Swarm{T}, cache, ns, w, y1, y2) where {T}
    # Unpack data
    @unpack candidates, candidates_velocity, best_candidates, best_candidates_fitness =
        swarm
    @unpack index_vector = cache

    # Update velocity for each candidate
    wT = T(w)
    y1T = T(y1)
    y2T = T(y2)
    for (i, vel) in enumerate(candidates_velocity)
        # Shuffle vector containing integers 1:num_particles
        shuffle!(index_vector)

        # Defermine fbest in neighborhood
        fbest = Inf
        bestidx = 0
        for j in 1:ns
            # Get index of particle in neighborhood
            # If k in neighborhood is i, replace with index_vector[ns + 1]
            k = index_vector[j] != i ? index_vector[j] : index_vector[ns + 1]
            if best_candidates_fitness[k] < fbest
                fbest = best_candidates_fitness[k]
                bestidx = k
            end
        end

        # Update velocity
        for j in eachindex(vel)
            vel[j] =
                wT * vel[j] +
                y1T * rand(T) * (best_candidates[i][j] - candidates[i][j]) +
                y2T * rand(T) * (best_candidates[bestidx][j] - candidates[i][j])
        end
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
            if candidate[j] < dimmin(search_space, j)
                candidate[j] = dimmin(search_space, j)
                candidates_velocity[i][j] = 0.0
            elseif candidate[j] > dimmax(search_space, j)
                candidate[j] = dimmax(search_space, j)
                candidates_velocity[i][j] = 0.0
            end
        end
    end
    return nothing
end
