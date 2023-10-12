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

