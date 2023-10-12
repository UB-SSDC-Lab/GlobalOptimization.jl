function exprate(λ::T) where T <: AbstractFloat
    return -log(1.0 - rand(T)) / λ
end
function expscale(β::T) where T <: AbstractFloat
    return -β*log(1.0 - rand(T))
end

function laplace(μ::T, b::T) where T <: AbstractFloat
    u = rand(T) - 0.5
    return μ - b*sign(u)*log(1.0 - 2.0*abs(u))
end
laplace(b::T) where {T} = laplace(zero(T), b)
laplace(::Type{T}) where {T} = laplace(zero(T), one(T))
laplace() = laplace(0.0, 1.0)