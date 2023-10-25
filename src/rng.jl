"""
    exprate(λ::AbstractFloat)

Generate an exponential random variable with rate `λ`.
"""
function exprate(λ::T) where T <: AbstractFloat
    return -log(1.0 - rand(T)) / λ
end

"""
    expscale(β::AbstractFloat)

Generate an exponential random variable with scale `β`.
"""
function expscale(β::T) where T <: AbstractFloat
    return -β*log(1.0 - rand(T))
end

"""
    laplace(μ::AbstractFloat, b::Real)

Generate a Laplace random variable with location `μ` and scale `b`.
"""
function laplace(μ::T, b::T) where T <: AbstractFloat
    u = rand(T) - 0.5
    return μ - b*sign(u)*log(1.0 - 2.0*abs(u))
end

"""
    laplace(b)

Generate a Laplace random variable with location `0.0` and scale `b`.
"""
laplace(b::T) where {T} = laplace(zero(T), b)

"""
    laplace([::Type{T}]) where T <: AbstractFloat

Generate a Laplace random variable with location `0.0` and scale `1.0`. If `T` is specified,
the random variable will be of type `T`.
"""
laplace(::Type{T}) where {T} = laplace(zero(T), one(T))
laplace() = laplace(0.0, 1.0)