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
- `dim_min::Vector{T}`: A vector of minimum values for each dimension.
- `dim_max::Vector{T}`: A vector of maximum values for each dimension.
- `dim_delta::Vector{T}`: A vector of the difference between the maximum and minimum values for each dimension.
"""
struct ContinuousRectangularSearchSpace{T<:AbstractFloat} <: RectangularSearchSpace{T}
    dim_min::Vector{T}
    dim_max::Vector{T}
    dim_delta::Vector{T}

    @doc """
        ContinuousRectangularSearchSpace(dim_min::AbstractVector{T}, dim_max::AbstractVector{T})

    Constructs a new `ContinuousRectangularSearchSpace` with minimum values `dim_min` and maximum values `dim_max`.

    # Arguments
    - `dim_min::AbstractVector{T}`: A vector of minimum values for each dimension.
    - `dim_max::AbstractVector{T}`: A vector of maximum values for each dimension.

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
        dim_min::AbstractVector{T1}, dim_max::AbstractVector{T2}
    ) where {T1<:Real,T2<:Real}
        # Check dimensions
        length(dim_min) == length(dim_max) ||
            throw(DimensionMismatch("dim_min and dim_max must be the same length."))

        # Check values
        for i in eachindex(dim_min)
            if dim_min[i] > dim_max[i]
                throw(
                    ArgumentError(
                        "dim_min[i] must be less than or equal to dim_max[i] for all i."
                    ),
                )
            end
        end

        # Handle types
        Tp = promote_type(T1, T2)
        T = Tp <: AbstractFloat ? Tp : Float64

        # Return new search space
        return new{T}(
            convert(Vector{T}, dim_min),
            convert(Vector{T}, dim_max),
            convert(Vector{T}, dim_max - dim_min),
        )
    end
end

"""
    num_dims(ss::ContinuousRectangularSearchSpace)

Returns the number of dimensions in the search space `ss`.

# Arguments
- `ss::ContinuousRectangularSearchSpace`

# Returns
- `Integer`
"""
num_dims(ss::ContinuousRectangularSearchSpace) = length(ss.dim_min)

"""
    dim_min(ss::ContinuousRectangularSearchSpace{T}, [i::Integer])

Returns the minimum value for the `i`-th dimension of `ss`. If `i` is not specified,
returns a vector of all minimum values.

# Arguments
- `ss::RontinuousRectangularSearchSpace{T}`
- `i::Integer`: the dimension to return the minimum value for.

# Returns
- `T` or `Vector{T}`: the minimum value for the `i`-th dimension of `ss` or a vector of all minimum values if `i` not provided.
"""
dim_min(ss::ContinuousRectangularSearchSpace) = ss.dim_min
dim_min(ss::ContinuousRectangularSearchSpace, i::Integer) = ss.dim_min[i]

"""
    dim_max(ss::ContinuousRectangularSearchSpace{T}, [i::Integer])

Returns the maximum value for the `i`-th dimension of `ss`. If `i` is not specified,
returns a vector of all maximum values.

# Arguments
- `ss::ContinuousRectangularSearchSpace{T}`
- `i::Integer`: the dimension to return the maximum value for.

# Returns
- `T` or `Vector{T}`: the minimum value for the `i`-th dimension of `ss` or a vector of all minimum values if `i` not provided.
"""
dim_max(ss::ContinuousRectangularSearchSpace) = ss.dim_max
dim_max(ss::ContinuousRectangularSearchSpace, i::Integer) = ss.dim_max[i]

"""
    dim_delta(ss::ContinuousRectangularSearchSpace{T}, [i::Integer])

Returns the difference between the maximum and minimum values for the `i`-th dimension of `ss`.
If `i` is not specified, returns a vector of all differences.

# Arguments
- `ss::ContinuousRectangularSearchSpace{T}``
- `i::Integer`: the dimension to return the difference between the maximum and minimum values for.

# Returns
- `T` or `Vector{T}` the difference between the maximum and minimum values for the `i`-th dimension of `ss` or a vector of all differences if `i` not provided.
"""
dim_delta(ss::ContinuousRectangularSearchSpace) = ss.dim_delta
dim_delta(ss::ContinuousRectangularSearchSpace, i::Integer) = ss.dim_delta[i]

"""
    dim_range(ss::ContinuousRectangularSearchSpace{T}, [i::Integer])

Returns the range of values for the `i`-th dimension of `ss`. If `i` is not specified,
returns a vector of all ranges.

# Arguments
- `ss::ContinuousRectangularSearchSpace{T}`
- `i::Integer`: the dimension to return the range of values for.

# Returns
- `Tuple{T, T}` or `Vector{Tuple{T, T}}`: the range of values for the `i`-th dimension of `ss` or a vector of all ranges if `i` not provided.
"""
dim_range(ss::ContinuousRectangularSearchSpace) = tuple.(dim_min(ss), dim_max(ss))
function dim_range(ss::ContinuousRectangularSearchSpace, i::Integer)
    (dim_min(ss, i), dim_max(ss, i))
end

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
    num_dims(ss1) == num_dims(ss2) ||
        throw(DimensionMismatch("ss1 and ss2 must have the same number of dimensions."))

    # Handel types
    T = promote_type(T1, T2)

    # Instaitiate new min and max vectors
    new_min = zeros(T, num_dims(ss1))
    new_max = zeros(T, num_dims(ss1))

    # Compute the intersection
    @inbounds for i in eachindex(new_min)
        new_min[i] = max(dim_min(ss1, i), dim_min(ss2, i))
        new_max[i] = min(dim_max(ss1, i), dim_max(ss2, i))
    end
    return ContinuousRectangularSearchSpace(new_min, new_max)
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
    length(x) == num_dims(ss) ||
        throw(DimensionMismatch("x must have the same number of dimensions as ss."))

    # Check values
    @inbounds for i in eachindex(x)
        if x[i] < dim_min(ss, i) || x[i] > dim_max(ss, i)
            return false
        end
    end
    return true
end
