abstract type AbstractHopper{T} end

mutable struct BasicHopper{T} <: AbstractHopper{T}
    # MBH time step (iteration counter)
    t::Int

    # The Hopper's current best solution
    x::Vector{Float64}
    f::Float64

    # The Hopper's candidate solution
    xc::Vector{Float64}
    fc::Float64

    function BasicHopper{T}(nDims::Integer) where {T}
        t = zero(Int)
        x = zeros(T, nDims)
        f = T(Inf)
        return new{T}(t, x, f, copy(x), f)
    end
end