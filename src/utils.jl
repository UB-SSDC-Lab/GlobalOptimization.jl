
# Utility functions for GlobalOptimization.jl

"""
    all_correlated(cor, tol)

Checks if the absolute value of the lower-triangular part of the correlation matrix `cor`
are above the correlation tolerance `tol` and returns `true` if so, otherwise, returns
`false`.

# Arguments:
- `cor::AbstractMatrix`: The correlation matrix
- `tol::AbstractFloat`: The correlation tolerance. Elements of `cor` with an absolute value
    greater than `tol` are assumed to be correlated.

# Returns:
- `Bool`: `true` if all elements are correlated and `false` otherwise
"""
function all_correlated(cor, tol)
    @inbounds for col in first(axes(cor, 2), size(cor, 2) - 1)
        for row in last(axes(cor, 1), size(cor, 1) - col)
            abs_x = abs(cor[row, col])
            if abs_x < tol
                return false
            end
        end
    end
    return true
end

"""
    fill_identity!(mat)

Fills the `mat` in-place with the identity matrix.
"""
function fill_identity!(mat)
    @inbounds for j in axes(mat, 2)
        for i in axes(mat, 1)
            mat[i, j] = ifelse(i == j, 1.0, 0.0)
        end
    end
    return nothing
end
