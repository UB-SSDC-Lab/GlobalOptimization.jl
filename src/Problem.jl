struct Problem{F <: Function, BT <: AbstractVector, N}
    # The objective function
    f::F 
    
    # The upper and lower bounds
    LB::BT
    UB::BT
end

# If LB and UB are not specified, use default of +/- 1000
function Problem(objFunc::F, numVars::Integer) where {F <: Function}
    # Check that nDims > 0
    numVars > 0 || throw(ArgumentError("N must be greater than zero."))

    # Initialize lower and upper bounds
    LB = fill(-1000, numVars)
    UB = fill(1000,  numVars)

    return Problem{F,Vector{Int},numVars}(objFunc, LB, UB)
end
function Problem{N}(objFunc::F) where {N, F <: Function}
    # Check that nDims > 0
    N > 0 || throw(ArgumentError("N must be greater than zero."))

    # Initialize lower and upper bounds
    LB = fill(-1000, numVars)
    UB = fill(1000,  numVars)

    return Problem{F,Vector{Int},N}(objFunc, LB, UB)
end
function Problem(
    objFunc::F, LB::AbstractArray{SLB}, UB::AbstractArray{SUB},
) where {F <: Function, SUB <: Real, SLB <: Real}
    # Check that LB and UB are the same length
    if length(LB) != length(UB) 
        throw(ArgumentError("Lower bounds and upper bound vectors must be the same length."))
    end
    @inbounds for i in eachindex(LB)
        LB[i] > UB[i] && throw(ArgumentError("LB[i] must be greater than UB[i] for all i."))
    end

    N = length(LB)
    S = promote_type(SLB, SUB)
    return Problem{F,Vector{S},N}(objFunc, convert(Vector{S}, LB), convert(Vector{S}, UB))
end
function Problem(
    objFunc::F, LB::SVector{N,SLB}, UB::SVector{N,SUB},
) where {F <: Function, SUB <: Real, SLB <: Real, N}
    @inbounds for i in eachindex(LB)
        LB[i] > UB[i] && throw(ArgumentError("LB[i] must be greater than UB[i] for all i."))
    end

    S = promote_type(SLB, SUB)
    return Problem{F,SVector{N,S},N}(objFunc, SVector{N,S}(LB), SVector{N,S}(UB))
end