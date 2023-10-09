struct Problem{F <: Function, S}
    # The objective function
    f::F 
    
    # The upper and lower bounds
    LB::Vector{S}
    UB::Vector{S}
end

# If LB and UB are not specified, use default of +/- 1000
function Problem(objFunc::F, numVars::Integer) where {F <: Function}
    # Check that nDims > 0
    numVars > 0 || throw(ArgumentError("N must be greater than zero."))

    # Initialize lower and upper bounds
    LB = fill(-1000, numVars)
    UB = fill(1000,  numVars)

    return Problem{F,eltype(LB)}(objFunc, LB, UB)
end

function Problem(
    objFunc::F, LB::AbstractArray{SLB}, UB::AbstractArray{SUB},
) where {F <: Function, SUB <: Real, SLB <: Real}
    # Check that LB and UB are the same length
    if length(LB) == length(UB) 
        throw(ArgumentError("Lower bounds and upper bound vectors must be the same length."))
    end
    @inbounds for i in 1:N
        LB[i] > UB[i] && throw(ArgumentError("LB[i] must be greater than UB[i] for all i."))
    end

    S = promote_type(SLB, SUB)
    return Problem{F,S}(objFunc, convert(Vector{S}, LB), convert(Vector{S}, UB))
end