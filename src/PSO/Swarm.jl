"""
    Swarm{T <: AbstractFloat} <: AbstractPopulation

A population of particles for the PSO algorithm.
"""
struct Swarm{T <: AbstractFloat} <: AbstractPopulation
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
Swarm_F64(num_particles::Integer, num_dims::Integer) = Swarm{Float64}(num_particles, num_dims)

"""
    Swarm_F32(num_particles::Integer, num_dims::Integer)

Constructs a Float32 `Swarm` with `num_particles` particles in `num_dims` dimensions.
"""
Swarm_F32(num_particles::Integer, num_dims::Integer) = Swarm{Float32}(num_particles, num_dims)

"""
    initialize_uniform!(swarm::Swarm{T}, search_space::ContinuousRectangularSearchSpace{T})

Initializes the swarm `swarm` with a uniform particle distribution in the search space. 
"""
function initialize_uniform!(
    swarm::Swarm{T}, 
    search_space::ContinuousRectangularSearchSpace{T},
) where T
    # Unpack swarm
    @unpack candidates, candidates_fitness, candidates_velocity, 
        best_candidates, best_candidates_fitness = swarm

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
            pos[j] = dmin + dΔ*rand(T)
            vel[j] = -dΔ + 2.0*dΔ*rand(T)
        end
    end
    return nothing
end

"""
    initialize_uniform!(
        swarm::Swarm{T}, 
        search_space::ContinuousRectangularSearchSpace{T}, 
        initial_bounds::ContinuousRectangularSearchSpace)

    Initializes the swarm `swarm` with a uniform particle distribution in the intersection of the
    the search space and initial bounds.
"""
function initialize_uniform!(
    swarm, search_space, initial_bounds::ContinuousRectangularSearchSpace,
)
    initialize_uniform!(swarm, intersection(search_space, initial_bounds))
    return nothing
end
function initialize_uniform!(
    swarm, search_space, initial_bounds::Nothing,
)
    initialize_uniform!(swarm, search_space)
    return nothing
end

function initialize_fitness!(
    swarm::Swarm{T}, evaluator::BatchEvaluator{T},
) where {T}
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
