"""
    SearchSpace

The base abstract type for a `Problem` search space. 
"""
abstract type SearchSpace end

"""
    FixedDimensionSearchSpace

The base abstract type for a search space with a fixed finite number of dimensions.
Applicable to the vast majority of optimization problems.
"""
abstract type FixedDimensionSearchSpace <: SearchSpace end

"""
    RectangularSearchSpace

A `FixedDimensionSearchSpace` with `N` dimensional rectangle as the set of feasible points.
"""
abstract type RectangularSearchSpace <: FixedDimensionSearchSpace end

"""
    ContinuousRectangularSearchSpace

A `RectangularSearchSpace` formed by a single continuous set.
"""
struct ContinuousRectangularSearchSpace{T <: AbstractFloat} <: RectangularSearchSpace 
    dimmin::Vector{T}
    dimmax::Vector{T}
    dimdelta::Vector{T}

    function ContinuousRectangularSearchSpace(
        dimmin::AbstractVector{T1}, dimmax::AbstractVector{T2}
    ) where {T1 <: Real, T2 <: Real}
        # Check dimensions
        length(dimmin) == length(dimmax) ||
            throw(DimensionMismatch("dimmin and dimmax must be the same length."))

        # Check values
        for i in eachindex(dimmin)
            if dimmin[i] > dimmax[i]
                throw(ArgumentError("dimmin[i] must be less than or equal to dimmax[i] for all i."))
            end
        end

        # Handle types
        Tp = promote_type(T1, T2)
        T  = Tp <: AbstractFloat ? Tp : Float64

        # Return new search space
        new{T}(convert(Vector{T}, dimmin), convert(Vector{T}, dimmax), convert(Vector{T}, dimmax - dimmin))
    end
end

"""
    numdims(ss::ContinuousRectangularSearchSpace)

Returns the number of dimensions in the search space `ss`.
"""
numdims(ss::ContinuousRectangularSearchSpace) = length(ss.dimmin)

"""
    dimmin(ss::ContinuousRectangularSearchSpace, [i::Integer])

Returns the minimum value for the `i`-th dimension of `ss`. If `i` is not specified, 
returns a vector of all minimum values.
"""
dimmin(ss::ContinuousRectangularSearchSpace) = ss.dimmin
dimmin(ss::ContinuousRectangularSearchSpace, i::Integer) = ss.dimmin[i]

"""
    dimmax(ss::ContinuousRectangularSearchSpace, [i::Integer])

Returns the maximum value for the `i`-th dimension of `ss`. If `i` is not specified,
returns a vector of all maximum values.
"""
dimmax(ss::ContinuousRectangularSearchSpace) = ss.dimmax
dimmax(ss::ContinuousRectangularSearchSpace, i::Integer) = ss.dimmax[i]

"""
    dimdelta(ss::ContinuousRectangularSearchSpace, [i::Integer])

Returns the difference between the maximum and minimum values for the `i`-th dimension of `ss`.
If `i` is not specified, returns a vector of all differences.
"""
dimdelta(ss::ContinuousRectangularSearchSpace) = ss.dimdelta
dimdelta(ss::ContinuousRectangularSearchSpace, i::Integer) = ss.dimdelta[i]

"""
    dimrange(ss::ContinuousRectangularSearchSpace, [i::Integer])

Returns the range of values for the `i`-th dimension of `ss`. If `i` is not specified,
returns a vector of all ranges.
"""
dimrange(ss::ContinuousRectangularSearchSpace) = tuple.(dimmin(ss), dimmax(ss))
dimrange(ss::ContinuousRectangularSearchSpace, i::Integer) = (dimmin(ss, i), dimmax(ss, i))
