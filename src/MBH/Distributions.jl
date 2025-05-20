
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

    # Memory cache for MAD median
    mad_cache::Vector{T}

    function MBHStepMemory{T}(num_dims::Integer, memory_len::Integer) where {T}
        return new{T}(
            Matrix{T}(undef, num_dims + 2, memory_len),
            memory_len,
            0,
            Vector{T}(undef, memory_len),
        )
    end
    function MBHStepMemory{T}(num_dims::UndefInitializer, memory_len) where {T}
        return new{T}(
            Matrix{T}(undef, 0, 0),
            memory_len,
            0,
            Vector{T}(undef, memory_len)
        )
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

    # Return standard deviation
    return std(steps)
end

"""
    step_MAD_median(step_memory::MBHStepMemory{T}, var_idx::Integer)

Returns the mean absolute deviation (MAD) around the median of the step memory. If `var_idx`
is specified, then the MAD median of the step memory for the variable at index `var_idx` is
returned.
"""
function step_MAD_median(step_memory::MBHStepMemory{T}, var_idx::Integer) where {T}
    # Get view of steps for var_idx
    steps = view(step_memory.data, var_idx, 1:(step_memory.steps_in_memory))

    # Copy steps
    steps_copy = step_memory.mad_cache
    copyto!(steps_copy, steps)

    # Compute median
    median = median!(steps_copy)

    # Compute MAD median
    steps_copy .= abs.(steps_copy .- median)
    mad = mean(steps_copy)

    return mad
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

A static distribution for MBH. In this implementation, each element of a *hop* is drawn
from a mixture model comprised of two Laplace distributions given by:

``f_{mix}(x; b, \\lambda) = k\\left[(1 - b) f(x;\\mu = 0,\\theta = \\lambda) + b f(x;\\mu = 0, \\theta = 1)\\right]``

where ``\\mu`` denotes the location parameter and ``\\theta`` the scale parameter of a
Laplace distribution (i.e., with probability density ``f(x;\\mu,\\theta)``).

# Fields
- `b`: The mixing parameter for the two Laplace distributions
- `λ`: The scale parameter for the first Laplace distribution in the mixture model
- `dim_delta`: The length of the search space in each dimension
"""
struct MBHStaticDistribution{T} <: AbstractMBHDistribution{T}
    # The mixing parameter for the mixture model
    b::T

    # Scale parameter for p
    λ::T

    # The length of the search space in each dimension
    dim_delta::Vector{T}

    @doc """
        MBHStaticDistribution{T}(; b=0.05, λ=0.7) where {T}

    Creates a new `MBHStaticDistribution` with the given parameters.

    # Keyword Arguments
    - `b`: The mixing parameter for the two Laplace distributions
    - `λ`: The scale parameter for the first Laplace distribution in the mixture model
    """
    function MBHStaticDistribution{T}(; b=0.05, λ=0.7) where {T}
        return new{T}(T(b), T(λ), Vector{T}(undef, 0))
    end
end

"""
    MBHAdaptiveDistribution{T}

An adaptive distribution for MBH. In this implementation, each *hop* is drawn from an
adaptive mixture model comprised of two Laplace distributions as defined in
Englander, Arnold C., "Speeding-Up a Random Search for the Global Minimum of a Non-Convex,
Non-Smooth Objective Function" (2021). *Doctoral Dissertations*. 2569.
[https://scholars.unh.edu/dissertation/2569](https://scholars.unh.edu/dissertation/2569/).

The mixture model is given by:

``f_{mix}(x; b, c, \\hat{\\boldsymbol{\\lambda}}) = (1 - b) f(x;\\mu = 0,\\theta = c*\\hat{\\lambda}_i) + b f(x;\\mu = 0, \\theta = 1)``

where ``\\mu`` denotes the location parameter and ``\\theta`` the scale parameter of a
Laplace distribution (i.e., with probability density ``f(x;\\mu,\\theta)``). The ``i``-th
element of a new hop is draw with the scale parameter ``\\hat{\\lambda}_i`` for the first
Laplace distribution. Please see the aforementioned dissertation for details on how
``\\hat{\\lambda}_i`` is updated.

# Fields
- `step_memory`: The step memory for the distribution
- `min_memory_update`: The minimum number of steps in memory before updating the scale parameter
- `a`: A parameter that defines the influence of a new successful step in the adaptation of
    the distribution.
- `b`: The mixing parameter for the two Laplace distributions
- `c`: The scale parameter for the first Laplace distribution
- `λhat`: The estimated scale parameter of the first Laplace distribution
- `λhat0`: The initial value of the scale parameter
- `use_mad`: Flag to indicate if we will use the STD (proposed by Englander) or MAD median (MVE) to
    update the estimated scale parameter
- `dim_delta`: The length of the search space in each dimension
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

    # Flag to indicate if we will use the STD (proposed by Englander) or MAD median (MVE) to
    # update the estimated scale parameter
    use_mad::Bool

    # The length of the search space in each dimension
    dim_delta::Vector{T}

    @doc """
        MBHAdaptiveDistribution{T}(
            memory_len::Int, min_memory_update::Int;
            a=0.93,
            b=0.05,
            c=1.0,
            λhat0=1.0,
        ) where T

    Creates a new `MBHAdaptiveDistribution` with the given parameters.

    # Arguments
    - `memory_len::Int`: The length of the memory for the distribution adaptation.
    - `min_memory_update::Int`: The minimum number of steps in memory before updating the scale parameter.

    # Keyword Arguments
    - `a`: A parameter that defines the influence of a new successful step in the adaptation of
        the distribution.
    - `b`: The mixing parameter for the two Laplace distributions
    - `c`: The scale parameter for the first Laplace distribution
    - `λhat0`: The initial value of the scale parameter
    - `use_mad::Bool`: Flag to indicate which metric to use for estimating the scale parameter.
        If `true`, the MAD median is used, which is the maximum likelihood estimator for a
        Laplace distribution's shape parameter. If `false`, the standard deviation is used
        as proposed by Englander (2021).
    """
    function MBHAdaptiveDistribution{T}(
        memory_len::Int,
        min_memory_update::Int;
        a=0.93,
        b=0.05,
        c=1.0,
        λhat0=1.0,
        use_mad::Bool=false,
    ) where {T}
        return new{T}(
            MBHStepMemory{T}(undef, memory_len),
            min_memory_update,
            T(a),
            T(b),
            T(c),
            Vector{T}(undef, 0),
            T(λhat0),
            use_mad,
            Vector{T}(undef, 0),
        )
    end
end

"""
    initialize!(dist::AbstractMBHDistribution, num_dims)

Initializes the distribution `dist` with the number of dimensions `num_dims`.
"""
initialize!(dist::AbstractMBHDistribution, ::ContinuousRectangularSearchSpace) = nothing
function initialize!(dist::MBHStaticDistribution, search_space::ContinuousRectangularSearchSpace)
    # Get search space info
    ndims = num_dims(search_space)
    ddims = dim_delta(search_space)

    # Set the length of the search space in each dimension
    resize!(dist.dim_delta, ndims)
    dist.dim_delta .= ddims
    return nothing
end
function initialize!(dist::MBHAdaptiveDistribution, search_space::ContinuousRectangularSearchSpace)
    # Unpack distribution
    @unpack step_memory = dist

    # Get search space info
    ndims = num_dims(search_space)
    ddims = dim_delta(search_space)

    # Initialize step memory
    initialize!(step_memory, ndims)

    # Initialize scale parameter vector
    resize!(dist.λhat, ndims)
    dist.λhat .= dist.λhat0

    # Set the length of the search space in each dimension
    resize!(dist.dim_delta, ndims)
    dist.dim_delta .= ddims

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
    @unpack step_memory, λhat, a = dist

    # Push step into memory
    push!(step_memory, step, pre_step_fitness, post_step_fitness)

    # Update scale parameter vector if enough steps are contained in memory
    if step_memory.steps_in_memory >= dist.min_memory_update
        @inbounds for i in eachindex(λhat)
            # Note: We should probably use the mean absolute deviation from the median
            # instead of the standard deviation as this is the correct maximum likelihood
            # estimator for the Laplace distribution shape parameter.

            # Compute standard deviation of steps for var index i
            Ψ = if dist.use_mad
                step_MAD_median(step_memory, i)
            else
                step_std(step_memory, i)
            end

            # Update scale parameter
            λhat[i] = (1.0 - a) * Ψ + a * λhat[i]
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
    @unpack b, λ = dist

    # Draw step
    l1 = Laplace{T}(0.0, λ)
    l2 = Laplace{T}(0.0, 1.0)
    @inbounds for i in eachindex(step)
        k = 0.5*dist.dim_delta[i]

        # Draw step element from mixture model
        r = rand(T)
        if r < b
            step[i] = k * rand(l2)
        else
            step[i] = k * rand(l1)
        end
    end

    return nothing
end
function draw_step!(step::AbstractVector{T}, dist::MBHAdaptiveDistribution{T}) where {T}
    # Unpack distribution
    @unpack b, c, λhat = dist

    # Draw step
    l2 = Laplace{T}(0.0, 1.0)
    @inbounds for i in eachindex(step)
        k = 0.5*dist.dim_delta[i]

        # Draw step element from mixture model
        r = rand(T)
        if r < b
            step[i] = k * rand(l2)
        else
            step[i] = k * rand(Laplace{T}(0.0, c * λhat[i]))
        end
    end

    return nothing
end
