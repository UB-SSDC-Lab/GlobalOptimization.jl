
"""
    MBHStepMemory{T}

Memory about MBH accepted steps.
"""
mutable struct MBHStepMemory{T}
    # The data (Matrix with each column corresponding to a step. In column, first N - 2 elements are the step, final two are pre- and post-step fitness)
    data::Matrix{T}

    # The memory length
    memory_len::Int

    # Current steps in memory
    steps_in_memory::Int

    function MBHStepMemory{T}(num_dims::Integer, memory_len::Integer) where {T}
        return new{T}(Matrix{T}(undef, num_dims + 2, memory_len), memory_len, 0)
    end
    function MBHStepMemory{T}(num_dims::UndefInitializer, memory_len) where {T}
        return new{T}(Matrix{T}(undef, 0, 0), memory_len, 0)
    end
end

function initialize!(step_memory::MBHStepMemory{T}, num_dims::Integer) where {T}
    # Unpack step memory
    @unpack memory_len = step_memory
    step_memory.data = Matrix{T}(undef, num_dims + 2, memory_len)
    return nothing
end

"""
    step_std(step_memory::MBHStepMemory{T}, var_idx::Integer)

Returns the standard deviation of the step memory. If `var_idx` is specified, then the standard deviation
of the step memory for the variable at index `var_idx` is returned.
"""
function step_std(step_memory::MBHStepMemory{T}, var_idx::Integer) where {T}
    # Get view of steps for var_idx
    steps = view(step_memory.data, var_idx, 1:(step_memory.steps_in_memory))

    # Compute mean and std
    mean = sum(steps) / step_memory.steps_in_memory
    std = zero(T)
    @inbounds for x in steps
        std += (x - mean)^2
    end
    std = sqrt(std / step_memory.steps_in_memory)

    # Return standard deviation
    return std
end

"""
    push!(step_memory::MBHStepMemoory{T}, step::Vector{T}, pre_step_fitness::T, post_step_fitness::T)

Pushes a step into the step memory.
"""
function push!(
    step_memory::MBHStepMemory{T},
    step::AbstractVector{T},
    pre_step_fitness::T,
    post_step_fitness::T,
) where {T}
    # Unpack step memory
    @unpack data, steps_in_memory = step_memory

    # Get size information
    N, M = size(data)

    # Slide steps in memory
    last_remembered_step = steps_in_memory
    if steps_in_memory == M
        last_remembered_step -= 1
    else
        step_memory.steps_in_memory += 1
    end
    @inbounds for col in reverse(1:last_remembered_step)
        @views data[:, col + 1] .= data[:, col]
    end

    # Update first column with new step
    data[1:(N - 2), 1] .= step
    data[N - 1, 1] = pre_step_fitness
    data[N, 1] = post_step_fitness

    return nothing
end

"""
    AbstractMBHDistribution{T}

Abstract type for MBH distributions.
"""
abstract type AbstractMBHDistribution{T} end

"""
    MBHStaticDistribution{T}

Static distribution for MBH.
"""
struct MBHStaticDistribution{T} <: AbstractMBHDistribution{T}
    # Parameters (Set in options but also here for convenience)
    a::T
    b::T
    c::T

    # Scale parameter for p
    λ::T

    # Constructor
    function MBHStaticDistribution{T}(; a=0.93, b=0.05, c=1.0, λ=1.0) where {T}
        return new{T}(T(a), T(b), T(c), T(λ))
    end
end

"""
    MBHAdaptiveDistribution{T}

Adaptive distribution for MBH.
"""
mutable struct MBHAdaptiveDistribution{T} <: AbstractMBHDistribution{T}
    # Hopper accepted step memory
    step_memory::MBHStepMemory{T}

    # THe minimum number of steps in memory before updating the scale parameter
    min_memory_update::Int

    # Parameters (Set in options but also here for convenience)
    a::T
    b::T
    c::T

    # Estimated scale parameter of q-hat
    λhat::Vector{T}

    # THe initial value of the scale parameter
    λhat0::T

    # Constructor
    function MBHAdaptiveDistribution{T}(
        memory_len::Int,
        min_memory_update::Int;
        a=0.93,
        b=0.05,
        c=1.0,
        λhat0=1.0,
    ) where {T}
        return new{T}(
            MBHStepMemory{T}(undef, memory_len),
            min_memory_update,
            T(a),
            T(b),
            T(c),
            Vector{T}(undef, 0),
            T(λhat0),
        )
    end
end

"""
    initialize!(dist::AbstractMBHDistribution, num_dims)

Initializes the distribution `dist` with the number of dimensions `num_dims`.
"""
initialize!(dist::AbstractMBHDistribution, num_dims) = nothing
function initialize!(dist::MBHAdaptiveDistribution, num_dims)
    # Unpack distribution
    @unpack step_memory = dist

    # Initialize step memory
    initialize!(step_memory, num_dims)

    # Initialize scale parameter vector
    resize!(dist.λhat, num_dims)
    dist.λhat .= dist.λhat0

    return nothing
end

"""
    push_accepted_step!(
        dist::MBHAdaptiveDistribution{T},
        step::AbstractVector{T},
        pre_step_fitness::T,
        post_step_fitness::T,
    ) where {T}
"""
function push_accepted_step!(
    dist::MBHAdaptiveDistribution{T},
    step::AbstractVector{T},
    pre_step_fitness::T,
    post_step_fitness::T,
) where {T}
    # Unpack distribution
    @unpack step_memory, λhat, a, b, c = dist

    # Push step into memory
    push!(step_memory, step, pre_step_fitness, post_step_fitness)

    # Update scale parameter vector if enough steps are contained in memory
    if step_memory.steps_in_memory >= 20
        @inbounds for i in eachindex(λhat)
            # Compute standard deviation of steps for var index i
            σ = step_std(step_memory, i)

            # Update scale parameter
            λhat[i] = (1.0 - a) * σ + a * λhat[i]
        end
    end

    return nothing
end

"""
    draw_step!(step::AbstractVector{T}, dist::AbstractMBHDistribution{T})

Draws a step from the distribution `dist` and stores it in `step`.
"""
function draw_step!(step::AbstractVector{T}, dist::MBHStaticDistribution{T}) where {T}
    # Unpack distribution
    @unpack a, b, c, λ = dist

    # Draw step
    #k = length(step) / 2.0
    k = 1.0
    l1 = Laplace(0.0, c * λ)
    l2 = Laplace()
    @inbounds for i in eachindex(step)
        step[i] = k * ((1.0 - b) * rand(l1) + b * rand(l2))
    end

    return nothing
end
function draw_step!(step::AbstractVector{T}, dist::MBHAdaptiveDistribution{T}) where {T}
    # Unpack distribution
    @unpack a, b, c, λhat = dist

    # Draw step
    #k = length(step) / 2.0
    k = 1.0
    @inbounds for i in eachindex(step)
        step[i] = k * ((1.0 - b) * rand(Laplace(0.0, c * λhat[i])) + b * rand(Laplace()))
    end

    return nothing
end
