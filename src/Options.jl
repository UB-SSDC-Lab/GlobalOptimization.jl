
struct Options{T<:AbstractFloat, U<:AbstractVector, CF<:Union{Function,Nothing}}
    # Display Options
    display::Bool 
    displayInterval::Int

    # Function tolerance
    funcTol::T

    # Check function val for NaN and Inf 
    funValCheck::Bool

    # Initial bounds
    iUB::U
    iLB::U

    # Max Iterations
    maxIters::Int

    # Max Stall Iterations 
    maxStallIters::Int

    # Max Stall Time 
    maxStallTime::T

    # Max Time
    maxTime::T

    # Objective Limit 
    objLimit::T

    # Use parallel
    useParallel::Bool 

    # Callback function
    callback::CF

    #Reset Distance
    resetDistance::Int

    #Maximum Reset Iterations
    maxResetIters::Int
end

function Options(;display=true, displayInterval=1, funcTol::T=1e-6,
    funValCheck=true, iUB::Uu=nothing, iLB::Ul=nothing, maxIters=1000, maxStallIters=25,
    maxStallTime=500, maxTime=1800, objLimit=-Inf, useParallel=false, callback::CF=nothing,
    resetDistance=2, maxResetIters=40) where 
    {T<:Number, Uu<:Union{Nothing, Vector}, Ul<:Union{Nothing, Vector}, CF<:Union{Nothing, Function}}

    if iUB === nothing
        iUB = Vector{T}([])
        U   = Vector{T}
    else
        U   = Uu;
    end
    if iLB === nothing
        iLB = Vector{T}([])
    end

    return Options{T,U,CF}(display, displayInterval, funcTol,
        funValCheck, iUB, iLB, maxIters, maxStallIters, maxStallTime, 
        maxTime, objLimit, useParallel, callback, resetDistance, maxResetIters)
end
