
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
abstract type AbstractRandomNeighborhoodVelocityUpdateScheme <: AbstractVelocityUpdateScheme end

"""
    MATLABVelocityUpdate <: AbstractRandomNeighborhoodVelocityUpdateScheme

A velocity update scheme employed by the [MATLAB PSO algorithm](https://www.mathworks.com/help/gads/particle-swarm-optimization-algorithm.html).
This scheme is described as follows:

In this velocity update scheme, the size of the neighborhood, as well as the inertia weight,
are adaptively updated as follows:

*Prior to First Iteration:*
1. Set the inertial weight ``w``: `w = inertia_range[2]`
2. Set the minimum neighborhood size: `minimum_neighborhood_size = max(2, floor(Int, swarm_size * minimum_neighborhood_fraction))`

3. Set the neighborhood size: `N = minimum_neighborhood_size`
4. Set counter: `c = 0`

*After Evaluating Swarm Fitness Each Iteration:*
- If the best fitness of the swarm has improved:
    1. Decrease the counter: `c = max(0, c - 1)`
    2. Set the neighborhood size to the minimum: `N = minimum_neighborhood_size`
    3. Update the inertia weight:
        - If `c < 2; w = 2.0 * w`
        - If `c > 5; w = 0.5 * w`
        - Clamp `w` to lie in `[inertia_range[1], inertia_range[2]]`
- If the best fitness of the swarm has not improved:
    1. Increase the counter: `c += 1`
    2. Increase the neighborhood size:
        `N = min(N + minimum_neighborhood_size, swarm_size - 1)`

# Fields
- `swarm_size::Int`: The size of the swarm.
- `inertia_range::Tuple{Float64,Float64}`: The range of inertia weights.
- `minimum_neighborhood_fraction::Float64`: The minimum fraction of the swarm size to be used as the neighborhood size.
- `minimum_neighborhood_size::Int`: The minimum neighborhood size.
- `self_adjustment_weight::Float64`: The self-adjustment weight.
- `social_adjustment_weight::Float64`: The social adjustment weight.
- `w::Float64`: The inertia weight.
- `c::Int`: The stall iteration counter.
- `N::Int`: The neighborhood size.
- `index_vector::Vector{UInt16}`: A vector used to store the indices of the particles in the
    swarm. Used for random neighborhood selection without allocations.
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

    @doc """
        MATLABVelocityUpdate(;
            inertia_range::Tuple{AbstractFloat,AbstractFloat}=(0.1, 1.0),
            minimum_neighborhood_fraction::AbstractFloat=0.25,
            self_adjustment_weight::AbstractFloat=1.49,
            social_adjustment_weight::AbstractFloat=1.49,
        )

    Create a new instance of the `MATLABVelocityUpdate` velocity update scheme.

    # Keyword Arguments
    - `inertia_range::Tuple{AbstractFloat,AbstractFloat}`: The range of inertia weights.
    - `minimum_neighborhood_fraction::AbstractFloat`: The minimum fraction of the swarm size to be used as the neighborhood size.
    - `self_adjustment_weight::AbstractFloat`: The self-adjustment weight.
    - `social_adjustment_weight::AbstractFloat`: The social adjustment weight.
    """
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

"""
    CSRNVelocityUpdate <: AbstractRandomNeighborhoodVelocityUpdateScheme

A velocity update scheme employed a Constant Size Random Neighborhood (CSRN).

In this velocity update scheme, the size of the neighborhood is constant and set based
on the specified `neighborhood_fraction` (i.e., the fraction of the swarm size to be
considered to lie in a neighborhood). However, the inertia is adaptively updated as follows:

*Prior to First Iteration:*  Set the inertial weight ``w``: `w = inertia_range[2]`

*After Evaluating Swarm Fitness Each Iteration:*
- If `stall_iteration < 2; w = 2.0 * w`
- If `stall_iteration > 5; w = 0.5 * w`
- Clamp `w` to lie in `[inertia_range[1], inertia_range[2]]`

Note that `stall_iteration` is the number of iterations since the global best position
found so far was improved by a specified `function_tolerance` (see PSO keyword arguments).

# Fields
- `inertia_range::Tuple{Float64,Float64}`: The range of inertia weights.
- `neighborhood_fraction::Float64`: The fraction of the swarm size to be used as the neighborhood size.
- `N::Int`: The neighborhood size.
- `self_adjustment_weight::Float64`: The self-adjustment weight.
- `social_adjustment_weight::Float64`: The social adjustment weight.
- `w::Float64`: The inertia weight.
- `index_vector::Vector{UInt16}`: A vector used to store the indices of the particles in the
    swarm. Used for random neighborhood selection without allocations.
"""
mutable struct CSRNVelocityUpdate <: AbstractRandomNeighborhoodVelocityUpdateScheme
    # Constant parameters
    inertia_range::Tuple{Float64,Float64}
    neighborhood_fraction::Float64
    N::Int # The neighborhood size
    self_adjustment_weight::Float64
    social_adjustment_weight::Float64

    # State variables
    w::Float64 # The inertia weight

    # Cache
    index_vector::Vector{UInt16}

    @doc """
        CSRNVelocityUpdate(;
            inertia_range::Tuple{AbstractFloat,AbstractFloat}=(0.1, 1.0),
            neighborhood_fraction::AbstractFloat=0.25,
            self_adjustment_weight::AbstractFloat=1.49,
            social_adjustment_weight::AbstractFloat=1.49,
        )

    Create a new instance of the `CSRNVelocityUpdate` velocity update scheme.
    """
    function CSRNVelocityUpdate(;
        inertia_range::Tuple{AbstractFloat,AbstractFloat}=(0.1, 1.0),
        neighborhood_fraction::AbstractFloat=0.25,
        self_adjustment_weight::AbstractFloat=1.49,
        social_adjustment_weight::AbstractFloat=1.49,
    )
        # Validate input parameters
        @assert neighborhood_fraction > 0.0 && neighborhood_fraction <= 1.0 "Invalid neighborhood fraction"
        ir_check =
            inertia_range[1] >= 0.0 &&
            inertia_range[2] <= 2.0 &&
            inertia_range[1] < inertia_range[2]
        @assert ir_check "Invalid inertia range"

        return new(
            (Float64(inertia_range[1]), Float64(inertia_range[2])),
            Float64(neighborhood_fraction),
            0,
            Float64(self_adjustment_weight),
            Float64(social_adjustment_weight),
            Float64(inertia_range[2]),
            Vector{UInt16}(undef, 0),
        )
    end
end

"""
    initialize!(vu::AbstractVelocityUpdateScheme, swarm_size::Int)

Initialize the velocity update scheme for a given swarm size.
"""
function initialize!(vu::MATLABVelocityUpdate, swarm_size)
    # Initialize state variables
    vu.w = vu.inertia_range[2]
    vu.c = 0

    # Set swarm size
    vu.swarm_size = swarm_size

    # Set initial neighborhood parameters
    vu.minimum_neighborhood_size = max(
        2, min(floor(Int, swarm_size * vu.minimum_neighborhood_fraction), swarm_size - 1)
    )
    vu.N = vu.minimum_neighborhood_size

    # Allocate index vector
    resize!(vu.index_vector, swarm_size)
    vu.index_vector .= 1:swarm_size

    return nothing
end
function initialize!(vu::CSRNVelocityUpdate, swarm_size)
    # Initialize state variables
    vu.w = vu.inertia_range[2]

    # Set neighborhood size
    vu.N = max(2, min(floor(Int, swarm_size * vu.neighborhood_fraction), swarm_size - 1))

    # Allocate index vector
    resize!(vu.index_vector, swarm_size)
    vu.index_vector .= 1:swarm_size

    return nothing
end

"""
    update_velocity!(swarm::Swarm, vu::AbstractVelocityUpdateScheme)

Update the velocity of each candidate in the swarm using the specified velocity update scheme.
"""
function update_velocity!(
    swarm::Swarm{T}, vu::AbstractRandomNeighborhoodVelocityUpdateScheme
) where {T}
    # Unpack data
    @unpack candidates, candidates_velocity, best_candidates, best_candidates_fitness =
        swarm
    @unpack index_vector, w, N, self_adjustment_weight, social_adjustment_weight = vu

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

"""
    adapt!(vu::AbstractVelocityUpdateScheme, improved::Bool, stall_iteration::Int)

Adapt the velocity update scheme based on the improvement status of the swarm and the
stall iteration counter.
"""
function adapt!(vu::MATLABVelocityUpdate, improved::Bool, stall_iteration::Int)
    if improved
        # Reduce counter
        vu.c = max(0, vu.c - 1)

        # Set neighborhood size to minimum
        vu.N = vu.minimum_neighborhood_size

        # Update inertia weight
        w = vu.w
        w = ifelse(vu.c < 2, 2.0*w, w)
        w = ifelse(vu.c > 5, 0.5*w, w)
        vu.w = clamp(w, vu.inertia_range[1], vu.inertia_range[2])
    else
        # Increase counter and neighborhood size
        vu.c += 1
        vu.N = min(vu.N + vu.minimum_neighborhood_size, vu.swarm_size - 1)
    end
    return nothing
end
function adapt!(vu::CSRNVelocityUpdate, improved::Bool, stall_iteration::Int)
    w = vu.w
    w = ifelse(stall_iteration < 2, 2.0*w, w)
    w = ifelse(stall_iteration > 5, 0.5*w, w)
    vu.w = clamp(w, vu.inertia_range[1], vu.inertia_range[2])
    return nothing
end
