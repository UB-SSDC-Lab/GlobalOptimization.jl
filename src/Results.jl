"""
    Results{T}

A simple struct for returning results.

# Fields
- `fbest::T`: The best function value found.
- `xbest::Vector{T}`: The best candidate found.
- `iters::Int`: The number of iterations performed.
- `time::Float64`: The time taken to perform the optimization in seconds.
- `exitFlag::Int`: The exit flag of the optimization.
"""
struct Results{T <: AbstractFloat}
    fbest::T
    xbest::Vector{T}

    iters::Int
    time::Float64
    exitFlag::Int

    @doc """
        Results(fbest::T, xbest::AbstractVector{T}, iters, time, exitFlag)

    Constructs a new `Results` struct.

    # Arguments
    - `fbest::T`: The best function value found.
    - `xbest::AbstractVector{T}`: The best candidate found.
    - `iters::Int`: The number of iterations performed.
    - `time::AbstractFloat`: The time taken to perform the optimization in seconds.
    - `exitFlag::Int`: The exit flag of the optimization.

    # Returns
    - `Results{T}`
    """
    function Results(fbest::T, xbest::AbstractVector{T}, iters, time, exitFlag) where T
        return new{T}(fbest, copy(xbest), iters, time, exitFlag)
    end
end