
"""
    PSOOptions{T <: AbstractFloat, IBT <: InitialBounds} <: AbstractAlgorithmOptions

Options for the PSO algorithm.
"""
struct PSOOptions{T <: AbstractFloat, IBT <: Union{Tuple,Nothing}} <: AbstractAlgorithmOptions
    # The general options
    general::GeneralOptions{T}

    # ===== PSO specific options
    # Initial bounds search space
    initial_bounds::IBT

    # Max iterations
    max_iterations::Int

    # Stall parameters
    function_tolerence::T
    max_stall_time::T
    max_stall_iterations::Int

    # Velocity update
    inertia_range::Tuple{T,T}
    minimum_neighborhood_fraction::T
    minimum_neighborhood_size::Int
    self_adjustment_weight::T
    social_adjustment_weight::T
end

# NOTE: THis constructor needs some work but is functional 
function PSOOptions(
    general::GeneralOptions{T}, 
    initial_bounds, 
    max_iterations, 
    function_tolerence, 
    max_stall_time, 
    max_stall_iterations, 
    inertia_range, 
    minimum_neighborhood_fraction, 
    minimum_neighborhood_size, 
    self_adjustment_weight, 
    social_adjustment_weight,
) where {T <: AbstractFloat}
    # Check initial bounds
    if initial_bounds !== nothing
        if length(initial_bounds) != 2
            throw(ArgumentError("initial_bounds must be a tuple of length 2."))
        end
    end
    return PSOOptions(general, initial_bounds, max_iterations, function_tolerence, max_stall_time, max_stall_iterations, inertia_range, minimum_neighborhood_fraction, minimum_neighborhood_size, self_adjustment_weight, social_adjustment_weight)
end

"""
    PSO

Particle Swarm Optimization (PSO) algorithm.
"""
struct PSO{T <: AbstractFloat, E <: BatchEvaluator{T}} <: AbstractOptimizer
    # The PSO algorithm options
    opts::PSOOptions{T}

    # The PSO evaluator
    eval::E

    # The PSO swarm
    swarm::Swarm{T}

    function PSO(
        evaluator::BatchEvaluator{T},
        num_particles::Int,
        initial_bounds::Union{Tuple,Nothing},
        max_iterations::Int,
        function_tolerence::Real,
        max_stall_time::Real,
        max_stall_iterations::Int,
        inertia_range::Tuple{Real,Real},
        minimum_neighborhood_fraction::Real,
        self_adjustment_weight::Real,
        social_adjustment_weight::Real,
        general::GeneralOptions{T},
    ) where {T <: AbstractFloat}
        # Construct PSO options
        minimum_neighborhood_size = max(2, floor(Int, num_particles * minimum_neighborhood_fraction))
        options = PSOOptions(
            general, 
            initial_bounds, 
            max_iterations, 
            function_tolerence, 
            max_stall_time, 
            max_stall_iterations, 
            inertia_range, 
            minimum_neighborhood_fraction, 
            minimum_neighborhood_size, 
            self_adjustment_weight, 
            social_adjustment_weight,
        )

        # Construct swarm
        swarm = Swarm{T}(num_particles, numdims(evaluator.prob.ss))

        # Instantiate PSO
        return new{T, typeof(evaluator)}(options, evaluator, swarm)
    end
end

"""
    PSO

Constructs a serial PSO algorithm with the given options.
"""
function PSO(
    prob::AbstractOptimizationProblem{SS};
    num_particles = 100,
    initial_bounds = nothing,
    max_iterations = 1000,
    function_tolerence = 1e-6,
    max_stall_time = Inf,
    max_stall_iterations = 25,
    inertia_range = (0.1, 1.0),
    minimum_neighborhood_fraction = 0.25,
    self_adjustment_weight = 1.49,
    social_adjustment_weight = 1.49,
    display = false,
    display_interval = 1,
    function_value_check = true,
    max_time = 60.0,
) where {T <: AbstractFloat, SS <: ContinuousRectangularSearchSpace{T}}
    # Construct the general options
    general = GeneralOptions(display, display_interval, function_value_check, max_time)

    # Construct evaluator (default to serial evaluator)
    evaluator = SerialBatchEvaluator(prob)

    # Construct PSO
    return PSO(evaluator, num_particles, initial_bounds, max_iterations, function_tolerence,
        max_stall_time, max_stall_iterations, inertia_range, minimum_neighborhood_fraction,
        self_adjustment_weight, social_adjustment_weight, general)
end

"""
    ThreadedPSO

Constructs a threaded PSO algorithm with the given options.
"""
function ThreadedPSO(
    prob::AbstractOptimizationProblem{SS};
    num_particles = 100,
    initial_bounds = nothing,
    max_iterations = 1000,
    function_tolerence = 1e-6,
    max_stall_time = Inf,
    max_stall_iterations = 25,
    inertia_range = (0.1, 1.0),
    minimum_neighborhood_fraction = 0.25,
    self_adjustment_weight = 1.49,
    social_adjustment_weight = 1.49,
    display = false,
    display_interval = 1,
    function_value_check = true,
    max_time = 60.0,
) where {T <: AbstractFloat, SS <: ContinuousRectangularSearchSpace{T}}
    # Construct the general options
    general = GeneralOptions(display, display_interval, function_value_check, max_time)

    # Construct evaluator (default to serial evaluator)
    evaluator = ThreadedBatchEvaluator(prob)

    # Construct PSO
    return PSO(evaluator, num_particles, initial_bounds, max_iterations, function_tolerence,
        max_stall_time, max_stall_iterations, inertia_range, minimum_neighborhood_fraction,
        self_adjustment_weight, social_adjustment_weight, general)
end

function PSO(prob::AbstractOptimizationProblem{SS}; kwargs...) where {SS <: SearchSpace}
    throw(ArgumentError("PSO only supports OptimizationProblem defined with a ContinuousRectangularSearchSpace."))
end
function ThreadedPSO(prob::AbstractOptimizationProblem{SS}; kwargs...) where {SS <: SearchSpace}
    throw(ArgumentError("PSO only supports OptimizationProblem defined with a ContinuousRectangularSearchSpace."))
end

# ===== Methods

# function _optimize!(pso::PSO, opts::Options)
#     _initialize!(pso, opts)
#     res = _iterate!(pso, opts)
#     return res
# end

# function _initialize!(pso::PSO, opts::Options)
#     # Set optimizer state 
#     pso.state    = 1

#     initialize_uniform!(pso.swarm, pso.prob, opts)
#     eval_objective!(pso, opts; init = true)
#     initialize_global_best!(pso) 
#     initialize_neighborhood!(pso) 
#     initialize_inertia!(pso) 
#     initialize_update_weights!(pso) 

#     # Call callback function
#     eval_callback!(pso, opts) 

#     # Print Status
#     opts.display && print_status(pso.swarm, 0.0, 0, 0)

#     return nothing
# end

# function _iterate!(pso::PSO, opts::Options)
#     # Prepare PSO for iterations
#     prepare_for_iteration!(pso)

#     # Begin loop
#     exitFlag = 0
#     while exitFlag == 0
#         # Update iteration counter 
#         pso.iters += 1

#         # Evolve particles
#         update_velocity!(pso)
#         step!(pso)
#         enforce_bounds!(pso)
#         eval_objective!(pso, opts)
#         update_global_best!(pso)
#         update_inertia!(pso) 

#         # Handle stall iterations
#         check_stall!(pso, opts) 

#         # Stopping criteria
#         exitFlag = check_stop_criteria!(pso, opts) 

#         # Output Status
#         if opts.display && pso.iters % opts.displayInterval == 0
#             print_status(pso.swarm, time() - pso.t0, pso.iters, pso.stallIters)
#         end

#         # Call callback function
#         eval_callback!(pso, opts)
#     end

#     # Return results
#     return construct_results(pso, exitFlag)
# end

# function construct_results(pso::PSO, exitFlag::Int)
#     return Results(pso.swarm.b, copy(pso.swarm.d), pso.iters, time() - pso.t0, exitFlag)
# end

# function eval_objective!(pso::PSO, opts; init = false)
#     eval_objective!(pso.swarm, pso.prob.f, opts; init = init)
#     return nothing
# end

# function eval_callback!(pso::PSO, opts::Options{T,U,CF}) where {T,U,CF <: Function}
#     opts.callback(pso, opts)
#     return nothing
# end

# function eval_callback!(pso::PSO, opts::Options{T,U,CF}) where {T,U,CF <: Nothing}
#     # Do nothing if we don't have a callback function
#     return nothing
# end

# function initialize_global_best!(pso::PSO)
#     pso.swarm.b = Inf
#     update_global_best!(pso.swarm)
#     return nothing
# end

# function initialize_neighborhood!(pso::PSO)
#     pso.swarm.n = max(
#         2, floor(length(pso.swarm) * pso.minNeighborFrac),
#     )
#     return nothing
# end

# function initialize_inertia!(pso::PSO)
#     if pso.inertiaRange[2] > 0
#         pso.swarm.w = pso.inertiaRange[2] > pso.inertiaRange[1] ? 
#             pso.inertiaRange[2] : pso.inertiaRange[1]
#     else
#         pso.swarm.w = pso.inertiaRange[2] < pso.inertiaRange[1] ?
#             pso.inertiaRange[2] : pso.inertiaRange[1]
#     end
#     return nothing
# end

# function initialize_update_weights!(pso::PSO)
#     pso.swarm.y₁ = pso.selfAdjustWeight
#     pso.swarm.y₂ = pso.socialAdjustWeight
#     return nothing
# end

# function prepare_for_iteration!(pso::PSO)
#     pso.t0 = time()
#     pso.stallT0 = pso.t0
#     pso.fStall = Inf
#     return nothing
# end

# function handle_update!(pso::PSO, update_found::Bool)
#     if update_found 
#         pso.swarm.c = max(0, pso.swarm.c - 1)
#         pso.swarm.n = pso.minNeighborSize
#     else
#         pso.swarm.c += 1
#         pso.swarm.n = min(
#             pso.swarm.n + pso.minNeighborSize, 
#             length(pso.swarm) - 1,
#         )
#     end
#     return nothing
# end

# function update_inertia!(pso::PSO)
#     if pso.swarm.c < 2
#         pso.swarm.w *= 2.0
#     elseif pso.swarm.c > 5
#         pso.swarm.w /= 2.0
#     end

#     # Ensure new inertia is in bounds
#     if pso.swarm.w < pso.inertiaRange[1]
#         pso.swarm.w = pso.inertiaRange[1]
#     elseif pso.swarm.w > pso.inertiaRange[2]
#         pso.swarm.w = pso.inertiaRange[2]
#     end
#     return nothing
# end

# function update_global_best!(pso::PSO)
#     update_found = update_global_best!(pso.swarm)
#     handle_update!(pso, update_found)
#     return nothing
# end

# update_velocity!(pso::PSO) = update_velocity!(pso.swarm)

# step!(pso::PSO) = step!(pso.swarm)

# function enforce_bounds!(pso::PSO)
#     need_check = false
#     @inbounds for i in eachindex(pso.prob.LB)
#         if !isinf(pso.prob.LB[i]) || !isinf(pso.prob.UB[i])
#             need_check = true
#             break
#         end
#     end
#     if need_check
#         enforce_bounds!(pso.swarm, pso.prob.LB, pso.prob.UB)
#     end
#     return nothing
# end

# function check_stall!(pso::PSO, opts::Options)
#     if pso.fStall - pso.swarm.b > opts.funcTol
#         pso.fStall = pso.swarm.b
#         pso.stallIters = 0
#         pso.stallT0 = time()
#     else
#         pso.stallIters += 1
#     end
#     return nothing
# end

# function check_stop_criteria!(pso::PSO, opts::Options)
#     if pso.stallIters >= opts.maxStallIters
#         pso.state = 3
#         return 1
#     elseif pso.iters >= opts.maxIters
#         pso.state = 3
#         return 2
#     elseif pso.swarm.b <= opts.objLimit
#         pso.state = 3
#         return 3
#     elseif time() - pso.stallT0 >= opts.maxStallTime 
#         pso.state = 3
#         return 4
#     elseif time() - pso.t0 >= opts.maxTime 
#         pso.state = 3
#         return 5
#     else
#         pso.state = 2
#         return 0
#     end
# end
