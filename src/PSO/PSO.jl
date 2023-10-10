mutable struct PSO{T <: AbstractFloat, ST <: AbstractSwarm{T}, S, F <: Function} <: Optimizer
    # Optimization problem
    prob::Problem{F,S}

    # Swarm of particles
    swarm::ST

    # PSO specific parameters/options
    inertiaRange::Tuple{T,T}
    minNeighborFrac::T 
    minNeighborSize::Int
    selfAdjustWeight::T
    socialAdjustWeight::T 

    # Optimizer state flag
    # state = 0 : Not initialized
    # state = 1 : Initialized
    # state = 2 : Optimizing
    # state = 3 : Converged
    state::Int

    # Optimizer time, iteration, and stall parameters
    t0::Float64 
    stallT0::Float64
    iters::Int 
    stallIters::Int
    fStall::T

    function PSO{T}(
        prob::Problem{F,S}, 
        numParticles::Integer,
        inertiaRange::Tuple{T,T}, 
        minNeighborFrac::T, 
        selfAdjustWeight::T, 
        socialAdjustWeight::T,
    ) where {T,S,F <: Function}

        # Compute minimum neighborhood size
        minNeighborSize = max(2, floor(Int, numParticles * minNeighborFrac))

        # Instantiate Swarm 
        N = length(prob.LB)
        #swarm = Swarm{T}(N, numParticles)
        swarm = StaticSwarm{N,T}(numParticles)

        return new{T,StaticSwarm{N,T},S,F}(
            prob, 
            swarm, 
            inertiaRange, 
            minNeighborFrac, 
            minNeighborSize,
            selfAdjustWeight, 
			socialAdjustWeight, 
            zero(Int),
            zero(T), 
            zero(T), 
            zero(Int), 
            zero(Int),
            zero(T),
        )
    end
end

# ===== Interface

function PSO(
    prob::Problem{F,S}; 
    numParticles = 100, 
    inertiaRange = (0.1, 1.1), 
    minNeighborFrac = 0.25, 
    selfAdjustWeight = 1.49, 
    socialAdjustWeight = 1.49,
) where {S,F <: Function}

    # Error checking
    length(inertiaRange) == 2 || throw(ArgumentError("inertiaRange must be of length 2."))
    minNeighborFrac > 0       || throw(ArgumentError("minNeighborFrac must be > 0."))

    # Type info
    T = typeof(inertiaRange[1]) == typeof(inertiaRange[2]) ? 
        typeof(inertiaRange[1]) : 
        promote_type(inertiaRange[1], inertiaRange[2])
    nIRange = (T(inertiaRange[1]), T(inertiaRange[2]))

    # Call constructor
    return PSO{T}(
        prob, 
        numParticles, 
        nIRange, 
        T(minNeighborFrac),
        T(selfAdjustWeight), 
        T(socialAdjustWeight))
end

# ===== Methods

function _optimize!(pso::PSO, opts::Options)
    _initialize!(pso, opts)
    res = _iterate!(pso, opts)
    return res
end

function _initialize!(pso::PSO, opts::Options)
    # Set optimizer state 
    pso.state    = 1

    initialize_uniform!(pso.swarm, pso.prob, opts)
    eval_objective!(pso, opts; init = true)
    initialize_global_best!(pso) 
    initialize_neighborhood!(pso) 
    initialize_inertia!(pso) 
    initialize_update_weights!(pso) 

    # Call callback function
    eval_callback!(pso, opts) 

    # Print Status
    opts.display && print_status(pso.swarm, 0.0, 0, 0)

    return nothing
end

function _iterate!(pso::PSO, opts::Options)
    # Prepare PSO for iterations
    prepare_for_iteration!(pso)

    # Begin loop
    exitFlag = 0
    while exitFlag == 0
        # Update iteration counter 
        pso.iters += 1

        # Evolve particles
        update_velocity!(pso)
        step!(pso)
        enforce_bounds!(pso)
        eval_objective!(pso, opts)
        update_global_best!(pso)
        update_inertia!(pso) 

        # Handle stall iterations
        check_stall!(pso, opts) 

        # Stopping criteria
        exitFlag = check_stop_criteria!(pso, opts) 

        # Output Status
        if opts.display && pso.iters % opts.displayInterval == 0
            print_status(pso.swarm, time() - pso.t0, pso.iters, pso.stallIters)
        end

        # Call callback function
        eval_callback!(pso, opts)
    end

    # Return results
    return construct_results(pso, exitFlag)
end

function construct_results(pso::PSO{T,S,F}, exitFlag::Int) where {T,S,F}
    return Results(pso.swarm.b, pso.swarm.d, pso.iters, time() - pso.t0, exitFlag)
end

function eval_objective!(pso::PSO, opts; init = false)
    eval_objective!(pso.swarm, pso.prob.f, opts; init = init)
    return nothing
end

function eval_callback!(pso::PSO, opts::Options{T,U,CF}) where {T,U,CF <: Function}
    opts.callback(pso, opts)
    return nothing
end

function eval_callback!(pso::PSO, opts::Options{T,U,CF}) where {T,U,CF <: Nothing}
    # Do nothing if we don't have a callback function
    return nothing
end

function initialize_global_best!(pso::PSO)
    pso.swarm.b = Inf
    update_global_best!(pso.swarm)
    return nothing
end

function initialize_neighborhood!(pso::PSO)
    pso.swarm.n = max(
        2, floor(length(pso.swarm) * pso.minNeighborFrac),
    )
    return nothing
end

function initialize_inertia!(pso::PSO)
    if pso.inertiaRange[2] > 0
        pso.swarm.w = pso.inertiaRange[2] > pso.inertiaRange[1] ? 
            pso.inertiaRange[2] : pso.inertiaRange[1]
    else
        pso.swarm.w = pso.inertiaRange[2] < pso.inertiaRange[1] ?
            pso.inertiaRange[2] : pso.inertiaRange[1]
    end
    return nothing
end

function initialize_update_weights!(pso::PSO)
    pso.swarm.y₁ = pso.selfAdjustWeight
    pso.swarm.y₂ = pso.socialAdjustWeight
    return nothing
end

function prepare_for_iteration!(pso::PSO)
    pso.t0 = time()
    pso.stallT0 = pso.t0
    pso.fStall = Inf
    return nothing
end

function handle_update!(pso::PSO, update_found::Bool)
    if update_found 
        pso.swarm.c = max(0, pso.swarm.c - 1)
        pso.swarm.n = pso.minNeighborSize
    else
        pso.swarm.c += 1
        pso.swarm.n = min(
            pso.swarm.n + pso.minNeighborSize, 
            length(pso.swarm) - 1,
        )
    end
    return nothing
end

function update_inertia!(pso::PSO)
    if pso.swarm.c < 2
        pso.swarm.w *= 2.0
    elseif pso.swarm.c > 5
        pso.swarm.w /= 2.0
    end

    # Ensure new inertia is in bounds
    if pso.swarm.w < pso.inertiaRange[1]
        pso.swarm.w = pso.inertiaRange[1]
    elseif pso.swarm.w > pso.inertiaRange[2]
        pso.swarm.w = pso.inertiaRange[2]
    end
    return nothing
end

function update_global_best!(pso::PSO)
    update_found = update_global_best!(pso.swarm)
    handle_update!(pso, update_found)
    return nothing
end

update_velocity!(pso::PSO) = update_velocity!(pso.swarm)

step!(pso::PSO) = step!(pso.swarm)

function enforce_bounds!(pso::PSO)
    need_check = false
    @inbounds for i in eachindex(pso.prob.LB)
        if !isinf(pso.prob.LB[i]) || !isinf(pso.prob.UB[i])
            need_check = true
            break
        end
    end
    if need_check
        enforce_bounds!(pso.swarm, pso.prob.LB, pso.prob.UB)
    end
    return nothing
end

function check_stall!(pso::PSO, opts::Options)
    if pso.fStall - pso.swarm.b > opts.funcTol
        pso.fStall = pso.swarm.b
        pso.stallIters = 0
        pso.stallT0 = time()
    else
        pso.stallIters += 1
    end
    return nothing
end

function check_stop_criteria!(pso::PSO, opts::Options)
    if pso.stallIters >= opts.maxStallIters
        pso.state = 3
        return 1
    elseif pso.iters >= opts.maxIters
        pso.state = 3
        return 2
    elseif pso.swarm.b <= opts.objLimit
        pso.state = 3
        return 3
    elseif time() - pso.stallT0 >= opts.maxStallTime 
        pso.state = 3
        return 4
    elseif time() - pso.t0 >= opts.maxTime 
        pso.state = 3
        return 5
    else
        pso.state = 2
        return 0
    end
end
