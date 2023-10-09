mutable struct Swarm{T<:AbstractFloat}

    # Vector of particles
    particles::Vector{Particle{T}}

    # Preallocated vector of UInt16 for neighborhood selection
    nVec::Vector{Int}

    # Global best objective function value
    b::T

    # Location of global best objective function value 
    d::Vector{T}

    n::Int      # Neighborhood size 
    w::T        # Inertia
    c::Int      # Adaptive inertia counter

    y₁::T   # Self adjustment weight
    y₂::T   # Social adjustment weight

    function Swarm{T}(nDims::Integer, nParticles::Integer) where {T}
        nDims < 0 && throw(ArgumentError("nDims must be greater than 0.")) 
        nParticles < 0 && throw(ArgumentError("nParticles must be greater than 0."))
        return new{T}(
            [Particle{T}(nDims) for _ in 1:nParticles],
            Vector{Int}(1:nParticles), 
            zero(T), 
            Vector{T}(undef, nDims), 
            zero(Int), 
            zero(T), 
            zero(Int), 
            zero(T),
            zero(T),
    )
    end
    function Swarm{T}(::UndefInitializer) where {T}
            return new{T}(
                Vector{Particle{T}}(undef, 0),
                Vector{Int}(undef, 0), 
                zero(T), 
                Vector{T}(undef, 0), 
                zero(Int), 
                zero(T), 
                zero(Int), 
                zero(T), 
                zero(T),
        )
    end
end

# ===== Interface
Base.length(s::Swarm) = length(s.particles)
Base.getindex(s::Swarm, i::Int) = s.particles[i]
Base.eachindex(s::Swarm) = eachindex(s.particles)
function Base.setindex!(s::Swarm, v, i::Int)
    s.particles[i] = v
    return nothing
end

global_best(s::Swarm) = s.d
global_best_objective(s::Swarm) = s.b

function eval_objective!(s::Swarm, f::F, opts::Options; init = false) where {F <: Function}
    # Evaluate objective functions
    if opts.useParallel
        Threads.@threads for i in eachindex(s)
            @inbounds eval_objective!(s[i], f)
        end
    else
        for i in eachindex(s)
            @inbounds eval_objective!(s[i], f)
        end
    end

    # Check objective function values if desired 
    opts.funValCheck && check_objective_value(s)

    # Update each particles best objective function value and its location
    if !init
        @inbounds for i in 1:length(s)
            update_best!(s[i])
        end
    else
        @inbounds for i in 1:length(s)
            initialize_best!(s[i])
        end
    end
end

function update_global_best!(s::Swarm)
    updated = false
    @inbounds for i in 1:length(s)
        if personal_best_objective(s[i]) < global_best_objective(s)
            update_global_best!(
                s, 
                personal_best(s[i]), 
                personal_best_objective(s[i]),
            )
            updated = true
        end
    end
    return updated
end

function update_global_best!(s::Swarm, best_pos, best_val)
    s.b = best_val
    s.d .= best_pos
    return nothing
end

function check_objective_value(s::Swarm)
    @inbounds for i in eachindex(s)
        check_objective_value(s[i])
    end      
end

function update_velocity!(s::Swarm)
    n = length(s[1])
    @inbounds for i in eachindex(s)
        # Shuffle vector containing integers 1:n
        # first m != i will be neighborhood
        shuffle!(s.nVec)

        # Determine fbest(S)
        fbest = Inf
        best = 0
        for j in range(1; stop = s.n)
            s.nVec[i] == j && continue
            if personal_best_objective(s[s.nVec[j]]) < fbest
                fbest = personal_best_objective(s[s.nVec[j]])
                best = s.nVec[j]
            end
        end

        # Update i's velocity 
        px = position(s[i])
        pv = velocity(s[i])
        pp = personal_best(s[i])
        np = personal_best(s[best])
        for j in eachindex(pv)
            pv[j] = s.w*pv[j] + 
                s.y₁*rand()*(pp[j] - px[j]) + 
                s.y₂*rand()*(np[j] - px[j])
        end 
    end
end

function step!(s::Swarm)
    @inbounds for i in 1:length(s)
        step!(s[i])
    end
    return nothing
end

function enforce_bounds!(s::Swarm, LB, UB)
    @inbounds for i in eachindex(s)
        enforce_bounds!(s[i], LB, UB)
    end
    return nothing
end

function print_status(s::Swarm, time::AbstractFloat, iter::Int, stallCount::Int)
    fspec1 = FormatExpr("Time Elapsed: {1:f} sec, Iteration Number: {2:d}, Function Evaluations: {3:d}")
    fspec2 = FormatExpr("Stall Iterations: {1:d}, Global Best: {2:e}")
    printfmtln(fspec1, time, iter, (iter + 1)*length(s))
    printfmtln(fspec2, stallCount, s.b)
    println(" ")
end

function print_status(s::Swarm, time::AbstractFloat, iter::Int, MFD::AbstractFloat, fAvg::AbstractFloat)
    fspec1 = FormatExpr("Time Elapsed: {1:f} sec, Iteration Number: {2:d}, Function Evaluations: {3:d}")
    fspec2 = FormatExpr("MFD: {1:e}, Avg. Fitness: {2:e}, Global Best: {3:e}")
    printfmtln(fspec1, time, iter, (iter + 1)*length(s))
    printfmtln(fspec2, MFD, fAvg, s.b)
    println(" ")
end

# Initializes position and velocities of particles by sampling from a 
# uniform distribution
function initialize_uniform!(swarm::Swarm, prob::Problem, opts::Options)

    # Get N: Number of diamensions and M: Swarm Size 
    M = length(swarm)
    N = length(swarm[1])

    # Get Boundary Constraints
    LB  = prob.LB
    UB  = prob.UB

    # Check if initial bounds on positions have been set
    useInitBnds = false
    if length(opts.iLB) == N && length(opts.iUB) == N
        useInitBnds = true
        iLB = opts.iLB            
        iUB = opts.iUB
    end

    # Initialize particle positions and velocities
    @inbounds begin
        for d in 1:N
            # Get local bounds for d-axis
            lLB = useInitBnds ? (LB[d] < iLB[d] ? iLB[d] : LB[d]) : LB[d]
            lUB = useInitBnds ? (UB[d] > iUB[d] ? iUB[d] : UB[d]) : UB[d]
            for p in 1:M
                # Position information
                swarm[p].x[d] = lLB + (lUB - lLB)*rand()
                swarm[p].p[d] = swarm[p].x[d]

                # Velocity 
                r = useInitBnds ? min(lUB-lLB,UB[d]-LB[d]) : lUB - lLB 
                swarm[p].v[d] = -r + 2*r*rand()
            end
        end
    end
    
    return nothing
end

# Initializes position and velocities of particles using logistic map
function initialize_logistics_map!(swarm::Swarm, prob::Problem, opts::Options)

    # Get N: Number of diamensions and M: Swarm Size 
    M = length(swarm)
    N = length(swarm[1])

    # Get Boundary Constraints
    LB  = prob.LB
    UB  = prob.UB

    # Check if initial bounds on positions have been set
    useInitBnds = false
    if length(opts.iLB) == N && length(opts.iUB) == N
        useInitBnds = true
        iLB = opts.iLB            
        iUB = opts.iUB
    end

    # Logistics map initialization
    fixedPointTol = 1e-14
    maxPert = 1e-12
    lMapIters = 3000
    @inbounds begin
        for j in 1:M
            for k in 1:N
                swarm[j].x[k] = 0.4567 + 2*(rand() - 0.5)*maxPert
            end
        end
        for i in 1:lMapIters
            for j in 1:M 
                for k in 1:N
                    val = swarm[j].x[k]
                    if val < fixedPointTol || 
                       abs(val - 0.25) < fixedPointTol || 
                       abs(val - 0.50) < fixedPointTol ||
                       abs(val - 0.75) < fixedPointTol ||
                       abs(val - 1.00) < fixedPointTol

                       swarm[j].x[k] += maxPert*rand()
                    end
                    swarm[j].x[k] = lMap(swarm[j].x[k])
                    if isinf(swarm[j].x[k]) 
                        throw(ErrorException("Inf or NaN"))
                    end
                end
            end
        end
    end

    # Scale particle positions and initialize velocities
    @inbounds begin
        for d in 1:N
            # Get local bounds for d-axis
            lLB = useInitBnds ? (LB[d] < iLB[d] ? iLB[d] : LB[d]) : LB[d]
            lUB = useInitBnds ? (UB[d] > iUB[d] ? iUB[d] : UB[d]) : UB[d]
            for p in 1:M
                # Position information
                swarm[p].x[d] = lLB + (lUB - lLB)*swarm[p].x[d]
                swarm[p].p[d] = swarm[p].x[d]

                # Velocity 
                r = useInitBnds ? min(lUB-lLB,UB[d]-LB[d]) : lUB - lLB 
                swarm[p].v[d] = -r + 2*r*rand()
            end
        end
    end
    
    return nothing
end