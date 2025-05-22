
"""
    AbstractVelocityUpdateScheme

Abstract type for velocity update schemes in Particle Swarm Optimization (PSO).
"""
abstract type AbstractVelocityUpdateScheme end

"""
    AbstractRandomNeighborhoodVelocityUpdateScheme

Abstract type for a velocity update scheme that randomly selects the particles in the
neighborhood of a given particle.
"""
abstract type AbstractRandomNeighborhoodVelocityUpdateScheme <:
              AbstractVelocityUpdateScheme end

"""
    MATLABVelocityUpdate <: AbstractRandomNeighborhoodVelocityUpdateScheme

A velocity update scheme employed by the [MATLAB PSO algorithm](https://www.mathworks.com/help/gads/particle-swarm-optimization-algorithm.html).
"""
mutable struct MATLABVelocityUpdate <: AbstractRandomNeighborhoodVelocityUpdateScheme
    # Constant parameters
    swarm_size::Int
    inertia_range::Tuple{Float64,Float64}
    minimum_neighborhood_fraction::Float64
    minimum_neighborhood_size::Int
    self_adjustment_weight::Float64
    social_adjustment_weight::Float64

    # State variables
    w::Float64  # The inertia weight
    c::Int      # The stall iteration counter
    N::Int      # The neighborhood size

    # Cache
    index_vector::Vector{UInt16}

    # Constructor
    function MATLABVelocityUpdate(;
        inertia_range::Tuple{AbstractFloat,AbstractFloat}=(0.1, 1.0),
        minimum_neighborhood_fraction::AbstractFloat=0.25,
        self_adjustment_weight::AbstractFloat=1.49,
        social_adjustment_weight::AbstractFloat=1.49,
    )
        # Validate input parameters
        @assert minimum_neighborhood_fraction > 0.0 && minimum_neighborhood_fraction <= 1.0 "Invalid neighborhood fraction"
        ir_check =
            inertia_range[1] >= 0.0 &&
            inertia_range[2] <= 2.0 &&
            inertia_range[1] < inertia_range[2]
        @assert ir_check "Invalid inertia range"

        return new(
            0,
            (Float64(inertia_range[1]), Float64(inertia_range[2])),
            Float64(minimum_neighborhood_fraction),
            0,
            Float64(self_adjustment_weight),
            Float64(social_adjustment_weight),
            Float64(inertia_range[2]),
            0,
            0,
            Vector{UInt16}(undef, 0),
        )
    end
end

function initialize!(mvu::MATLABVelocityUpdate, swarm_size)
    # Set swarm size
    mvu.swarm_size = swarm_size

    # Set initial neighborhood parameters
    mvu.minimum_neighborhood_size = max(
        2, floor(Int, swarm_size * mvu.minimum_neighborhood_fraction)
    )
    mvu.N = mvu.minimum_neighborhood_size

    # Allocate index vector
    resize!(mvu.index_vector, swarm_size)
    mvu.index_vector .= 1:swarm_size

    return nothing
end

function update_velocity!(
    swarm::Swarm{T}, velocity_update::AbstractRandomNeighborhoodVelocityUpdateScheme
) where {T}
    # Unpack data
    @unpack candidates, candidates_velocity, best_candidates, best_candidates_fitness =
        swarm
    @unpack index_vector, w, N, self_adjustment_weight, social_adjustment_weight =
        velocity_update

    # Update velocity for each candidate
    wT = T(w)
    y1T = T(self_adjustment_weight)
    y2T = T(social_adjustment_weight)
    for (i, vel) in enumerate(candidates_velocity)
        # Shuffle vector containing integers 1:num_particles
        shuffle!(index_vector)

        # Determine best candidate in neighborhood
        best_f = Inf
        best_idx = 0
        for j in 1:N
            # Get index of particle in neighborhood
            # If k in neighborhood is i, replace with index_vector[neighborhood_size + 1]
            k = ifelse(index_vector[j] != i, index_vector[j], index_vector[N + 1])
            if best_candidates_fitness[k] < best_f
                best_f = best_candidates_fitness[k]
                best_idx = k
            end
        end

        # Update velocity
        @inbounds for j in eachindex(vel)
            vel[j] =
                wT * vel[j] +
                y1T * rand(T) * (best_candidates[i][j] - candidates[i][j]) +
                y2T * rand(T) * (best_candidates[best_idx][j] - candidates[i][j])
        end
    end
    return nothing
end

function adapt!(mvu::MATLABVelocityUpdate, improved::Bool, stall_iteration::Int)
    if improved
        # Reduce counter
        mvu.c = max(0, mvu.c - 1)

        # Set neighborhood size to minimum
        mvu.N = mvu.minimum_neighborhood_size

        # Update inertia weight
        w = mvu.w
        w = ifelse(mvu.c < 2, 2.0*w, w)
        w = ifelse(mvu.c > 5, 0.5*w, w)
        mvu.w = clamp(w, mvu.inertia_range[1], mvu.inertia_range[2])
    else
        # Increase counter and neighborhood size
        mvu.c += 1
        mvu.N = min(mvu.N + mvu.minimum_neighborhood_size, mvu.swarm_size - 1)
    end
    return nothing
end
