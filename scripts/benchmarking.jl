
using GlobalOptimization
using BenchmarkTools
using Random
using Statistics
using Distributions
using Infiltrator
using DataFrames
using DataFramesMeta
using JLD2
using GLMakie

import BlackBoxOptim as BBO

function get_problem_sets()
    # Define problem sets
    ProblemSets = Dict{String,Any}(
        "easy" => [
            # ProblemName, NumDims, PopSize, MaxFevals
            ("Sphere",           5,      20,    5e3),
            ("Sphere",          10,      20,    1e4),
            ("Sphere",          30,      20,    3e4),

            ("Schwefel2.22",    5,       20,    5e3),
            ("Schwefel2.22",   10,       20,    1e4),
            ("Schwefel2.22",   30,       20,    3e4),

            ("Schwefel2.21",    5,       20,    5e3),
            ("Schwefel2.21",   10,       20,    1e4),
            ("Schwefel2.21",   30,       20,    3e4)
        ],

        "harder" => [
            # Harder problems
            ("Schwefel1.2",     5,       20,    5e3),
            ("Schwefel1.2",    10,       50,    5e4),
            ("Schwefel1.2",    30,       50,    2e5),
            ("Schwefel1.2",    50,       50,    3e5),

            ("Rosenbrock",      5,       20,    1e4),
            ("Rosenbrock",     10,       50,    5e4),
            ("Rosenbrock",     30,       50,    2e5),
            ("Rosenbrock",     50,       40,    3e5),

            ("Rastrigin",      50,       50,    5e5),
            ("Rastrigin",     100,       90,    8e5),

            ("Ackley",         50,       50,    5e5),
            ("Ackley",        100,       90,    8e5),

            ("Griewank",       50,       50,    5e5),
            ("Griewank",      100,       90,    8e5),
        ],

        "lowdim" => [
            ("Schwefel1.2",     2,       25,    1e4),
            ("Rosenbrock",      2,       25,    1e4),
            ("Rastrigin",       2,       25,    1e4),
            ("Ackley",          2,       25,    1e4),
            ("Griewank",        2,       25,    1e4),
        ],

        "test" => [
            ("Rosenbrock",     30,       50,    2e5),
        ]
    )
    ProblemSets["all"] = vcat(ProblemSets["easy"], ProblemSets["harder"])
    return ProblemSets
end

function construct_pso(prob, pop_size, max_iters)
    return SerialPSO(
        prob;
        num_particles = pop_size,
        max_iterations = max_iters,
        max_stall_iterations = max_iters,
        max_time = Inf,
    )
end

function construct_default_de(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display = false,
        num_candidates = pop_size,
        max_iterations = max_iters,
        max_stall_iterations = max_iters,
        mutation_params = SelfMutationParameters(
            mut_strat
        ),
        crossover_params = SelfBinomialCrossoverParameters()
    )
end

function construct_de(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display = false,
        num_candidates = pop_size,
        max_iterations = max_iters,
        max_stall_iterations = max_iters,
        mutation_params = SelfMutationParameters(
            mut_strat;
            dist = Uniform(0.0, 1.0),
        ),
        crossover_params = SelfBinomialCrossoverParameters(;
            dist = Uniform(0.0, 1.0),
        )
    )
end

function construct_radius_limited_de(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display = false,
        num_candidates = pop_size,
        max_iterations = max_iters,
        max_stall_iterations = max_iters,
        mutation_params = SelfMutationParameters(
            mut_strat;
            dist = Uniform(0.0, 1.0),
            sel = GlobalOptimization.RadiusLimitedSelector(8),
        ),
        crossover_params = SelfBinomialCrossoverParameters(;
            dist = Uniform(0.0, 1.0),
        )
    )
end

function construct_covbin_de(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display = false,
        num_candidates = pop_size,
        max_iterations = max_iters,
        max_stall_iterations = max_iters,
        mutation_params = SelfMutationParameters(
            mut_strat;
            dist = Uniform(0.0, 1.0),
        ),
        crossover_params = SelfBinomialCrossoverParameters(;
            dist = Uniform(0.0, 1.0),
            transform = GlobalOptimization.CovarianceTransformation(
                0.1, 0.5, GlobalOptimization.numdims(prob.ss)
            )
        )
    )
end

function construct_radius_limited_covbin_de(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display = false,
        num_candidates = pop_size,
        max_iterations = max_iters,
        max_stall_iterations = max_iters,
        mutation_params = SelfMutationParameters(
            mut_strat;
            dist = Uniform(0.0, 1.0),
            sel = GlobalOptimization.RadiusLimitedSelector(8),
        ),
        crossover_params = SelfBinomialCrossoverParameters(;
            dist = Uniform(0.0, 1.0),
            transform = GlobalOptimization.CovarianceTransformation(
                0.1, 0.5, GlobalOptimization.numdims(prob.ss)
            )
        )
    )
end

function algorithm_set()
    return [
        ("pso",  construct_pso),
        ("default_de_rand_1_bin", (p, s, i) -> construct_default_de(p,Rand1(), s, i)),
        ("default_de_best_1_bin", (p, s, i) -> construct_default_de(p,Best1(), s, i)),
        ("default_de_current_to_best_1_bin", (p, s, i) -> construct_default_de(p,CurrentToBest1(), s, i)),
        ("default_de_current_to_rand_1_bin", (p, s, i) -> construct_default_de(p,CurrentToRand1(), s, i)),
        ("default_de_randto_best_1_bin", (p, s, i) -> construct_default_de(p,RandToBest1(), s, i)),
        ("default_de_unified_bin", (p, s, i) -> construct_de(p,Unified(), s, i)),
        ("uniform_de_rand_1_bin", (p, s, i) -> construct_de(p,Rand1(), s, i)),
        ("uniform_de_best_1_bin", (p, s, i) -> construct_de(p,Best1(), s, i)),
        ("uniform_de_current_to_best_1_bin", (p, s, i) -> construct_de(p,CurrentToBest1(), s, i)),
        ("uniform_de_current_to_rand_1_bin", (p, s, i) -> construct_de(p,CurrentToRand1(), s, i)),
        ("uniform_de_randto_best_1_bin", (p, s, i) -> construct_de(p,RandToBest1(), s, i)),
        ("uniform_de_unified_bin", (p, s, i) -> construct_de(p,Unified(), s, i)),
        ("rl_de_rand_1_bin", (p, s, i) -> construct_radius_limited_de(p,Rand1(), s, i)),
        ("rl_de_best_1_bin", (p, s, i) -> construct_radius_limited_de(p,Best1(), s, i)),
        ("rl_de_current_to_best_1_bin", (p, s, i) -> construct_radius_limited_de(p,CurrentToBest1(), s, i)),
        ("rl_de_current_to_rand_1_bin", (p, s, i) -> construct_radius_limited_de(p,CurrentToRand1(), s, i)),
        ("rl_de_randto_best_1_bin", (p, s, i) -> construct_radius_limited_de(p,RandToBest1(), s, i)),
        ("rl_de_unified_bin", (p, s, i) -> construct_radius_limited_de(p,Unified(), s, i)),
        ("covbin_de_rand_1_bin", (p, s, i) -> construct_covbin_de(p,Rand1(), s, i)),
        ("covbin_de_best_1_bin", (p, s, i) -> construct_covbin_de(p,Best1(), s, i)),
        ("covbin_de_current_to_best_1_bin", (p, s, i) -> construct_covbin_de(p,CurrentToBest1(), s, i)),
        ("covbin_de_current_to_rand_1_bin", (p, s, i) -> construct_covbin_de(p,CurrentToRand1(), s, i)),
        ("covbin_de_randto_best_1_bin", (p, s, i) -> construct_covbin_de(p,RandToBest1(), s, i)),
        ("covbin_de_unified_bin", (p, s, i) -> construct_covbin_de(p,Unified(), s, i)),
        ("rl_covbin_de_rand_1_bin", (p, s, i) -> construct_radius_limited_covbin_de(p,Rand1(), s, i)),
        ("rl_covbin_de_best_1_bin", (p, s, i) -> construct_radius_limited_covbin_de(p,Best1(), s, i)),
        ("rl_covbin_de_current_to_best_1_bin", (p, s, i) -> construct_radius_limited_covbin_de(p,CurrentToBest1(), s, i)),
        ("rl_covbin_de_current_to_rand_1_bin", (p, s, i) -> construct_radius_limited_covbin_de(p,CurrentToRand1(), s, i)),
        ("rl_covbin_de_randto_best_1_bin", (p, s, i) -> construct_radius_limited_covbin_de(p,RandToBest1(), s, i)),
        ("rl_covbin_de_unified_bin", (p, s, i) -> construct_radius_limited_covbin_de(p,Unified(), s, i)),
    ]
end

function main()
    prob_set = get_problem_sets()["all"]
    algs = algorithm_set()

    N = 100
    data = DataFrame(
        ProblemName     = String[],
        NumDims         = Int[],
        PopSize         = Int[],
        MaxIters        = Int[],
        AlgorithmName   = String[],
        OptFitness      = Float64[],
        BestFitness     = Float64[],
        AvgFitness      = Float64[],
    )

    l = ReentrantLock()
    for prob in prob_set
        # Get info
        prob_name   = prob[1]
        num_dims    = prob[2]
        pop_size    = prob[3]
        max_iters   = round(Int, prob[4] / pop_size)

        println("Running $prob_name with $num_dims dimensions")

        # Get BBO problem
        bbo_prob = BBO.example_problems[prob[1]]
        opt_val = bbo_prob.opt_value

        # Construct problem
        opt_prob = OptimizationProblem(
            bbo_prob.objfunc,
            ContinuousRectangularSearchSpace(
                fill(bbo_prob.range_per_dim[1], num_dims),
                fill(bbo_prob.range_per_dim[2], num_dims),
            )
        )

        for alg in algs
            println("\t$(alg[1])")
            fitness = Vector{Float64}(undef, N)
            Threads.@threads for i in 1:N
                solver = alg[2](opt_prob, pop_size, max_iters)
                res = optimize!(solver)
                fitness[i] = res.fbest
            end
            avg_fitness = mean(fitness)
            best_fitness = minimum(fitness)
            push!(data, (prob_name, num_dims, pop_size, max_iters, alg[1], opt_val, best_fitness, avg_fitness))
        end
    end
    jldsave("benchmark_data.jld2"; df = data)
    return data
end

data = main()
