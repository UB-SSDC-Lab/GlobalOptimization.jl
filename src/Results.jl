"""
    Results

A simple struct for returning results.
"""
struct Results{T <: AbstractFloat}
    fbest::T
    xbest::Vector{T}

    iters::Int
    time::Float64
    exitFlag::Int

    function Results(fbest::T, xbest::AbstractVector{T}, iters, time, exitFlag) where T
        return new{T}(fbest, copy(xbest), iters, time, exitFlag)
    end
end