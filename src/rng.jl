"""
    exprate(λ::T)

Generate an exponential random variable with rate `λ`.

# Arguments
- `λ::AbstractFloat`: the rate parameter of the exponential distribution.

# Returns
- `val::T`: an exponential random variable with rate `λ`. 
"""
function exprate(λ::T) where {T<:AbstractFloat}
    return -log(1.0 - rand(T)) / λ
end

"""
    expscale(β::T)

Generate an exponential random variable with scale `β`.

# Arguments
- `β::AbstractFloat`: the scale parameter of the exponential distribution.

# Returns
- `val::T`: an exponential random variable with scale `β`.
"""
function expscale(β::T) where {T<:AbstractFloat}
    return -β * log(1.0 - rand(T))
end

"""
    laplace(μ::AbstractFloat, b::Real)

Generate a Laplace random variable with location `μ` and scale `b`.

# Arguments
- `μ::AbstractFloat`: the location parameter of the Laplace distribution.
- `b::Real`: the scale parameter of the Laplace distribution.

# Returns
- `val::T`: a Laplace random variable with location `μ` and scale `b`.
"""
function laplace(μ::T, b::T) where {T<:AbstractFloat}
    u = rand(T) - 0.5
    return μ - b * sign(u) * log(1.0 - 2.0 * abs(u))
end

"""
    laplace(b::T)

Generate a Laplace random variable with location `0.0` and scale `b`.

# Arguments
- `b::AbstractFloat`: the scale parameter of the Laplace distribution.

# Returns
- `val::T`: a Laplace random variable with location `0.0` and scale `b`.
"""
laplace(b::T) where {T} = laplace(zero(T), b)

"""
    laplace([::Type{T}]) where T <: AbstractFloat

Generate a Laplace random variable with location `0.0` and scale `1.0`. If `T` is specified,
the random variable will be of type `T`.

# Arguments
- `T::Type{T}`: the type of the random variable to generate. If not specified, `Float64` is used.

# Returns
- `val::T`: a Laplace random variable with location `0.0` and scale `1.0`
"""
laplace(::Type{T}) where {T} = laplace(zero(T), one(T))
laplace() = laplace(0.0, 1.0)
