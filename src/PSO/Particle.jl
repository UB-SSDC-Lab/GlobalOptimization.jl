
mutable struct Particle{T<:AbstractFloat}
    # Position, velocity, and personal best
    x::Vector{T}
    v::Vector{T}
    p::Vector{T}

    # Objective function values
    fx::T # Current
    fp::T # Personal best

    function Particle{T}(nDims::Integer) where {T <:AbstractFloat}
        if nDims < 0
            throw(ArgumentError("nDims cannot be less than 0."))
        end
        return new{T}(Vector{T}(undef, nDims),
                      Vector{T}(undef, nDims),
                      Vector{T}(undef, nDims),
                      zero(T), zero(T))
    end
    function Particle{T}(::UndefInitializer) where {T <: AbstractFloat}
        return Particle{T}(
            Vector{T}(undef, 0),
            Vector{T}(undef, 0),
            Vector{T}(undef, 0),
            zero(T), zero(T))
    end
end

# ===== Interface
Base.length(p::Particle) = length(p.x)

position(p::Particle) = p.x
velocity(p::Particle) = p.v
personal_best(p::Particle) = p.p
current_objective(p::Particle) = p.fx
personal_best_objective(p::Particle) = p.fp

function step!(p::Particle)
    p.x .+= p.v
    return nothing
end

function enforce_bounds!(p::Particle{T}, LB, UB) where {T}
    @inbounds for i in eachindex(p.x)
        if p.x[i] < LB[i]
            p.x[i] = LB[i]
            p.v[i] = zero(T)
        elseif p.x[i] > UB[i]
            p.x[i] = UB[i]
            p.v[i] = zero(T)
        end
    end
    return nothing
end

function eval_objective!(p::Particle, f::F) where {F <: Function}
    p.fx = f(p.x)
    return nothing
end

function initialize_best!(p::Particle) 
    p.fp = p.fx
    return nothing
end

function update_best!(p::Particle)
    if p.fx < p.fp
        p.p .= p.x
        p.fp = p.fx
    end
    return nothing
end

function check_objective_value(p::Particle)
    isinf(p.fx) && throw(ArgumentError("Objective function value is Inf."))
    isnan(p.fx) && throw(ArgumentError("Objective function value is NaN."))
    return nothing
end
