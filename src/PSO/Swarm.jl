abstract type AbstractSwarm{T <: AbstractFloat} end

mutable struct Swarm{T <: AbstractFloat} <: AbstractSwarm{T}

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
end

mutable struct StaticSwarm{N, T <: AbstractFloat} <: AbstractSwarm{T}
    # Particle information
    xs::Vector{SVector{N, T}}
    vs::Vector{SVector{N, T}}
    ps::Vector{SVector{N, T}}
    fxs::Vector{T}
    fps::Vector{T}

    # Preallocated vector of 1:N for neighborhood selection
    nVec::Vector{Int}

    # Global best objective function value
    b::T

    # Location of global best objective function value 
    d::SVector{N, T}

    n::Int      # Neighborhood size 
    w::T        # Inertia
    c::Int      # Adaptive inertia counter

    y₁::T   # Self adjustment weight
    y₂::T   # Social adjustment weight

    function StaticSwarm{N,T}(nParticles::Integer) where {N,T}
        N < 0 && throw(ArgumentError("N must be greater than 0.")) 
        nParticles < 0 && throw(ArgumentError("nParticles must be greater than 0."))
        return new{N,T}(
            [SVector{N,T}(zeros(T, N)) for _ in 1:nParticles],
            [SVector{N,T}(zeros(T, N)) for _ in 1:nParticles],
            [SVector{N,T}(zeros(T, N)) for _ in 1:nParticles],
            zeros(T, nParticles),
            zeros(T, nParticles),
            Vector{Int}(1:nParticles), 
            zero(T), 
            SVector{N,T}(zeros(T, N)), 
            zero(Int), 
            zero(T), 
            zero(Int), 
            zero(T),
            zero(T),
        )
    end
end

function Swarm{T}(nDims::Integer, nParticles::Integer) where {T}
    nDims < 0 && throw(ArgumentError("nDims must be greater than 0.")) 
    nParticles < 0 && throw(ArgumentError("nParticles must be greater than 0."))
    return Swarm(
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

# ===== Interface
Base.length(s::Swarm) = length(s.particles)
Base.length(s::StaticSwarm) = length(s.xs)

Base.eachindex(s::Swarm) = eachindex(s.particles)
Base.eachindex(s::StaticSwarm) = eachindex(s.xs)

Base.getindex(s::Swarm, i::Int) = s.particles[i]
Base.getindex(s::StaticSwarm, i::Int) = StaticParticle(s.xs[i], s.vs[i], s.ps[i], s.fxs[i], s.fps[i])

global_best(s::AbstractSwarm) = s.d
global_best_objective(s::AbstractSwarm) = s.b

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
function eval_objective!(s::StaticSwarm, f::F, opts::Options; init = false) where {F <: Function}
    # Evaluate objective functions
    if opts.useParallel
        ThreadsX.map!(f, s.fxs, s.xs)
    else
        @inbounds for i in eachindex(s)
            s.fxs[i] = f(s.xs[i])
        end
    end

    # Check objective function values if desired 
    opts.funValCheck && check_objective_value(s)

    # Update each particles best objective function value and its location
    init ? initialize_best!(s) : update_best!(s)
end

function update_best!(s::Swarm)
    @inbounds for i in 1:length(s)
        update_best!(s[i])
    end
    return nothing
end
function update_best!(s::StaticSwarm)
    @inbounds for i in eachindex(s)
        if s.fxs[i] < s.fps[i]
            s.ps[i] = s.xs[i]
            s.fps[i] = s.fxs[i]
        end
    end
    return nothing
end

function initialize_best!(s::Swarm)
    @inbounds for i in eachindex(s)
        initialize_best!(s[i])
    end
    return nothing
end
function initialize_best!(s::StaticSwarm)
    @inbounds for i in eachindex(s)
        s.ps[i] = s.xs[i]
        s.fps[i] = s.fxs[i]
    end
    return nothing
end

function update_global_best!(s::Swarm)
    updated = false
    @inbounds for i in eachindex(s)
        p = s[i]
        if personal_best_objective(p) < global_best_objective(s)
            update_global_best!(
                s, 
                personal_best(p), 
                personal_best_objective(p),
            )
            updated = true
        end
    end
    return updated
end
function update_global_best!(s::StaticSwarm)
     updated = false
    @inbounds for i in eachindex(s)
        if s.fps[i] < global_best_objective(s)
            update_global_best!(s, s.ps[i], s.fps[i])
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
function update_global_best!(
    s::StaticSwarm{N,T}, 
    best_pos::SVector{N,T}, 
    best_val::T,
) where {N,T}
    s.b = best_val
    s.d = best_pos
    return nothing
end

function check_objective_value(s::Swarm)
    @inbounds for i in eachindex(s)
        check_objective_value(s[i])
    end      
    return nothing
end
function check_objective_value(s::StaticSwarm)
    @inbounds for i in eachindex(s.fxs)
        if isinf(s.fxs[i]) || isnan(s.fxs[i])
            throw(ArgumentError("Objective function value is Inf or NaN."))
        end
    end
    return nothing
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
    return nothing
end
function update_velocity!(s::StaticSwarm{N,T}) where {N,T}
    @inbounds for i in eachindex(s)
        # Shuffle vector containing integers 1:n
        # first m != i will be neighborhood
        shuffle!(s.nVec)

        # Determine fbest(S)
        fbest = Inf
        best = 0
        for j in range(1; stop = s.n)
            s.nVec[i] == j && continue
            p = s[s.nVec[j]]
            if personal_best_objective(p) < fbest
                fbest = personal_best_objective(p)
                best = s.nVec[j]
            end
        end

        # Update i's velocity 
        s.vs[i] = s.w*s.vs[i] + 
            s.y₁*rand(SVector{N,T}).*(s.ps[i] - s.xs[i]) +
            s.y₂*rand(SVector{N,T}).*(s.ps[best] - s.xs[i])
    end
    return nothing
end

function step!(s::Swarm)
    @inbounds for i in eachindex(s)
        step!(s[i])
    end
    return nothing
end
function step!(s::StaticSwarm)
    @inbounds for i in eachindex(s)
        s.xs[i] = s.xs[i] + s.vs[i]
    end
    return nothing
end

function enforce_bounds!(s::Swarm, LB, UB)
    @inbounds for i in eachindex(s)
        enforce_bounds!(s[i], LB, UB)
    end
    return nothing
end
function enforce_bounds!(s::StaticSwarm{N,T}, LB, UB) where {N,T}
    @inbounds for i in eachindex(s)
        xm = MVector{N,T}(s.xs[i])
        vm = MVector{N,T}(s.vs[i])
        for j in eachindex(xm)
            if xm[j] < LB[j]
                xm[j] = LB[j]
                vm[j] = zero(T) 
            elseif xm[j] > UB[j]
                xm[j] = UB[j]
                vm[j] = zero(T)
            end
        end
        s.xs[i] = SVector{N,T}(xm)
        s.vs[i] = SVector{N,T}(vm)
    end
    return nothing
end

function print_status(s::AbstractSwarm, time::AbstractFloat, iter::Int, stallCount::Int)
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
function initialize_uniform!(swarm::StaticSwarm{N,T}, prob::Problem, opts::Options) where {N,T}
    M = length(swarm)

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
    @inbounds for p in 1:M
        # Create MVectors for storing information
        pos = MVector{N,T}(undef)
        vel = MVector{N,T}(undef)

        for d in eachindex(pos)
            # Get local bounds for d-axis
            lLB = useInitBnds ? (LB[d] < iLB[d] ? iLB[d] : LB[d]) : LB[d]
            lUB = useInitBnds ? (UB[d] > iUB[d] ? iUB[d] : UB[d]) : UB[d]

            # Position
            pos[d] = lLB + (lUB - lLB)*rand()
    
            # Velocity
            r = useInitBnds ? min(lUB-lLB,UB[d]-LB[d]) : lUB - lLB
            vel[d] = -r + 2.0*r*rand()
        end

        # Set values in swarm
        swarm.xs[p] = SVector{N,T}(pos)
        swarm.ps[p] = SVector{N,T}(pos)
        swarm.vs[p] = SVector{N,T}(vel)
    end
    return nothing
end

