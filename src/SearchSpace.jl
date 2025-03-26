"""
    SearchSpace

The base abstract type for a `Problem` search space. 
"""
abstract type SearchSpace{T} end

"""
    FixedDimensionSearchSpace

The base abstract type for a search space with a fixed finite number of dimensions.
Applicable to the vast majority of optimization problems.
"""
abstract type FixedDimensionSearchSpace{T} <: SearchSpace{T} end

"""
    RectangularSearchSpace

A `FixedDimensionSearchSpace` with `N` dimensional rectangle as the set of feasible points.
"""
abstract type RectangularSearchSpace{T} <: FixedDimensionSearchSpace{T} end

"""
    ContinuousRectangularSearchSpace{T <: AbstractFloat}

A `RectangularSearchSpace` formed by a single continuous set.

# Fields
- `dimmin::Vector{T}`: A vector of minimum values for each dimension.
- `dimmax::Vector{T}`: A vector of maximum values for each dimension.
- `dimdelta::Vector{T}`: A vector of the difference between the maximum and minimum values for each dimension.
"""
struct ContinuousRectangularSearchSpace{T<:AbstractFloat} <: RectangularSearchSpace{T}
    dimmin::Vector{T}
    dimmax::Vector{T}
    dimdelta::Vector{T}

    @doc """
        ContinuousRectangularSearchSpace(dimmin::AbstractVector{T}, dimmax::AbstractVector{T})

    Constructs a new `ContinuousRectangularSearchSpace` with minimum values `dimmin` and maximum values `dimmax`.

    # Arguments
    - `dimmin::AbstractVector{T}`: A vector of minimum values for each dimension.
    - `dimmax::AbstractVector{T}`: A vector of maximum values for each dimension.

    # Returns
    - `ContinuousRectangularSearchSpace{T}`

    # Examples
    ```julia-repl
    julia> using GlobalOptimization;
    julia> LB = [-1.0, 0.0];
    julia> UB = [ 1.0, 2.0];
    julia> ss = ContinuousRectangularSearchSpace(LB, UB)
    ContinuousRectangularSearchSpace{Float64}([-1.0, 0.0], [1.0, 2.0], [2.0, 2.0])
    ```
    """
    function ContinuousRectangularSearchSpace(
        dimmin::AbstractVector{T1}, dimmax::AbstractVector{T2}
    ) where {T1<:Real,T2<:Real}
        # Check dimensions
        length(dimmin) == length(dimmax) ||
            throw(DimensionMismatch("dimmin and dimmax must be the same length."))

        # Check values
        for i in eachindex(dimmin)
            if dimmin[i] > dimmax[i]
                throw(
                    ArgumentError(
                        "dimmin[i] must be less than or equal to dimmax[i] for all i."
                    ),
                )
            end
        end

        # Handle types
        Tp = promote_type(T1, T2)
        T = Tp <: AbstractFloat ? Tp : Float64

        # Return new search space
        return new{T}(
            convert(Vector{T}, dimmin),
            convert(Vector{T}, dimmax),
            convert(Vector{T}, dimmax - dimmin),
        )
    end
end

"""
    numdims(ss::ContinuousRectangularSearchSpace)

Returns the number of dimensions in the search space `ss`.

# Arguments
- `ss::ContinuousRectangularSearchSpace`

# Returns
- `Integer`
"""
numdims(ss::ContinuousRectangularSearchSpace) = length(ss.dimmin)

"""
    dimmin(ss::ContinuousRectangularSearchSpace{T}, [i::Integer])

Returns the minimum value for the `i`-th dimension of `ss`. If `i` is not specified, 
returns a vector of all minimum values.

# Arguments
- `ss::RontinuousRectangularSearchSpace{T}`
- `i::Integer`: the dimension to return the minimum value for.

# Returns
- `T` or `Vector{T}`: the minimum value for the `i`-th dimension of `ss` or a vector of all minimum values if `i` not provided.
"""
dimmin(ss::ContinuousRectangularSearchSpace) = ss.dimmin
dimmin(ss::ContinuousRectangularSearchSpace, i::Integer) = ss.dimmin[i]

"""
    dimmax(ss::ContinuousRectangularSearchSpace{T}, [i::Integer])

Returns the maximum value for the `i`-th dimension of `ss`. If `i` is not specified,
returns a vector of all maximum values.

# Arguments
- `ss::ContinuousRectangularSearchSpace{T}`
- `i::Integer`: the dimension to return the maximum value for.

# Returns
- `T` or `Vector{T}`: the minimum value for the `i`-th dimension of `ss` or a vector of all minimum values if `i` not provided.
"""
dimmax(ss::ContinuousRectangularSearchSpace) = ss.dimmax
dimmax(ss::ContinuousRectangularSearchSpace, i::Integer) = ss.dimmax[i]

"""
    dimdelta(ss::ContinuousRectangularSearchSpace{T}, [i::Integer])

Returns the difference between the maximum and minimum values for the `i`-th dimension of `ss`.
If `i` is not specified, returns a vector of all differences.

# Arguments
- `ss::ContinuousRectangularSearchSpace{T}``
- `i::Integer`: the dimension to return the difference between the maximum and minimum values for.

# Returns
- `T` or `Vector{T}` the difference between the maximum and minimum values for the `i`-th dimension of `ss` or a vector of all differences if `i` not provided.
"""
dimdelta(ss::ContinuousRectangularSearchSpace) = ss.dimdelta
dimdelta(ss::ContinuousRectangularSearchSpace, i::Integer) = ss.dimdelta[i]

"""
    dimrange(ss::ContinuousRectangularSearchSpace{T}, [i::Integer])

Returns the range of values for the `i`-th dimension of `ss`. If `i` is not specified,
returns a vector of all ranges.

# Arguments
- `ss::ContinuousRectangularSearchSpace{T}`
- `i::Integer`: the dimension to return the range of values for.

# Returns
- `Tuple{T, T}` or `Vector{Tuple{T, T}}`: the range of values for the `i`-th dimension of `ss` or a vector of all ranges if `i` not provided.
"""
dimrange(ss::ContinuousRectangularSearchSpace) = tuple.(dimmin(ss), dimmax(ss))
dimrange(ss::ContinuousRectangularSearchSpace, i::Integer) = (dimmin(ss, i), dimmax(ss, i))

"""
    intersection(
        ss1::ContinuousRectangularSearchSpace{T1}, 
        ss2::ContinuousRectangularSearchSpace{T2}
    )

Returns the intersection of the two search spaces `ss1` and `ss2` as a new search space.

# Arguments
- `ss1::ContinuousRectangularSearchSpace{T1}`
- `ss2::ContinuousRectangularSearchSpace{T2}`

# Returns
- `ContinuousRectangularSearchSpace{promote_type(T1, T2)}

# Throws
- `DimensionMismatch`: if `ss1` and `ss2` do not have the same number of dimensions.
"""
function intersection(
    ss1::ContinuousRectangularSearchSpace{T1}, ss2::ContinuousRectangularSearchSpace{T2}
) where {T1,T2}
    # Check inputs
    numdims(ss1) == numdims(ss2) ||
        throw(DimensionMismatch("ss1 and ss2 must have the same number of dimensions."))

    # Handel types
    T = promote_type(T1, T2)

    # Instaitiate new min and max vectors
    dimmin = zeros(T, numdims(ss1))
    dimmax = zeros(T, numdims(ss1))

    # Compute the intersection
    @inbounds for i in eachindex(dimmin)
        dimmin[i] = max(dimmin(ss1, i), dimmin(ss2, i))
        dimmax[i] = min(dimmax(ss1, i), dimmax(ss2, i))
    end
    return ContinuousRectangularSearchSpace(dimmin, dimmax)
end
intersection(ss1::ContinuousRectangularSearchSpace{T1}, ss2::Nothing) where {T1} = ss1
intersection(ss1::Nothing, ss2::ContinuousRectangularSearchSpace{T2}) where {T2} = ss2

"""
    feasible(x, ss::ContinuousRectangularSearchSpace)

Returns `true` if the point `x` is feasible in the search space `ss`, otherwise returns `false`.

# Arguments
- `x::AbstractVector{T}`: the point to check for feasibility.
- `ss::ContinuousRectangularSearchSpace{T}`: the search space to check for feasibility in.

# Returns
- `Bool`: `true` if `x` is in `ss`, otherwise `false`.

# Throws
- `DimensionMismatch`: if `x` does not have the same number of dimensions as `ss`.
"""
function feasible(x::AbstractVector{T}, ss::ContinuousRectangularSearchSpace{T}) where {T}
    length(x) == numdims(ss) ||
        throw(DimensionMismatch("x must have the same number of dimensions as ss."))

    # Check values
    @inbounds for i in eachindex(x)
        if x[i] < dimmin(ss, i) || x[i] > dimmax(ss, i)
            return false
        end
    end
    return true
end
