
mutable struct MBH{T <: AbstractFloat, HT <: AbstractHopper{T}, BT <: AbstractVector, F <: Function, N} <: Optimizer
    # Optimization problem
    prob::Problem{F,BT,N}

    # Hopper
    hopper::HT
end

function MBH{T}(prob::Problem{F,BT,N}) where {T <: AbstractFloat, F <: Function, BT, N}
    # Instantiate Hopper
    hopper = BasicHopper{T}(N)

    return MBH{T,BasicHopper{T},BT,F,N}(prob, hopper)
end

# Methods
function _optimize!(mbh::MBH, opts::Options)
    _initialize!(mbh, opts)
    res = _iterate!(mbh, opts)
    return res
end

function _initialize!(mbh::MBH{T,HT,BT,F,N}, opts::Options) where {T,HT,BT,F<:Function,N}
    # Get Boundary Constraints
    LB  = mbh.prob.LB
    UB  = mbh.prob.UB

    # Check if initial bounds on positions have been set
    useInitBnds = false
    if length(opts.iLB) == N && length(opts.iUB) == N
        useInitBnds = true
        iLB = opts.iLB            
        iUB = opts.iUB
    end

    # Initialize Hopper time steps
    mbh.hopper.t = zero(Int)

    # Initialize Hopper solution
    x = mbh.hopper.x
    for i in eachindex(x)
        lLB = useInitBnds ? (LB[i] < iLB[i] ? iLB[i] : LB[i]) : LB[i]
        lUB = useInitBnds ? (UB[i] > iUB[i] ? iUB[i] : UB[i]) : UB[i]
        x[i] = lLB + (lUB - lLB)*rand(T)
    end

    # Initialize Hopper objective function value
    mbh.hopper.f = mbh.prob.f(x)
    return nothing
end

function _iterate!(mbh::MBH{T,HT,BT,F,N}, opts::Options) where {T,HT,BT,F<:Function,N}
    # Some simple options
    max_time = 5.0
    t0 = time()

    # Begin loop
    exitFlag = 0
    hopper = mbh.hopper
    fun = mbh.prob.f
    while exitFlag == 0
        # Update MBH time step
        hopper.t += 1

        # Draw from distribution until step is accepted
        attempted = 0
        accepted = false
        pert = MVector{N,T}(undef)
        while !accepted
            attempted += 1
            for i in eachindex(pert)
                pert[i] = 0.99*laplace(0.0, 0.0001) + 0.01*laplace(0.0, 1.0)
            end
            hopper.xc .= hopper.x .+ pert 
            hopper.fc = fun(hopper.xc)
            if inbounds(hopper.xc, mbh.prob.LB, mbh.prob.UB) && hopper.fc < hopper.f
                hopper.x .= hopper.xc
                hopper.f = hopper.fc
                accepted = true
            elseif time() - t0 > max_time
                exitFlag = 1 
                break
            end
        end

        if time() - t0 > max_time
            exitFlag = 2
        end
    end
    return hopper.x, hopper.f, exitFlag
end

function inbounds(x, LB, UB)
    flag = true
    @inbounds for i in eachindex(x)
        if x[i] < LB[i] || x[i] > UB[i]
            flag = false
            break
        end
    end
    return flag
end
