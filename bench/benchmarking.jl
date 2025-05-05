
using GlobalOptimization
using Dates
using Distributions
using DataFrames
using DataFramesMeta
using JLD2
using CairoMakie

import BlackBoxOptim as BBO

# Include benchmarking utilities
include(joinpath(@__DIR__, "utils.jl"))

function get_problem_sets()
    # Define problem sets
    ProblemSets = Dict{String,Any}(
        "easy" => [
            # ProblemName, NumDims, PopSize, MaxIters
            ("Sphere", 5, 20, 250),
            ("Sphere", 10, 20, 500),
            ("Sphere", 30, 20, 1500),
            ("Schwefel 2.22", 5, 20, 250),
            ("Schwefel 2.22", 10, 20, 500),
            ("Schwefel 2.22", 30, 20, 1500),
            ("Schwefel 2.21", 5, 20, 250),
            ("Schwefel 2.21", 10, 20, 500),
            ("Schwefel 2.21", 30, 20, 1500),
        ],
        "harder" => [
            # Harder problems
            ("Schwefel 1.2", 5, 20, 250),
            ("Schwefel 1.2", 10, 50, 1000),
            ("Schwefel 1.2", 30, 50, 4000),
            ("Schwefel 1.2", 50, 50, 6000),
            ("Rosenbrock", 5, 20, 500),
            ("Rosenbrock", 10, 50, 1000),
            ("Rosenbrock", 30, 50, 4000),
            ("Rosenbrock", 50, 40, 3e5),
            ("Rastrigin", 50, 50, 10000),
            ("Rastrigin", 100, 90, 8889),
            ("Ackley", 50, 50, 10000),
            ("Ackley", 100, 90, 8889),
            ("Griewank", 50, 50, 10000),
            ("Griewank", 100, 90, 8889),
        ],
        "lowdim" => [
            ("Schwefel 1.2", 2, 25, 400),
            ("Rosenbrock", 2, 25, 400),
            ("Rastrigin", 2, 25, 400),
            ("Ackley", 2, 25, 400),
            ("Griewank", 2, 25, 400),
        ],
        "test" => [("Rosenbrock", 30, 50, 4000)],
    )
    ProblemSets["all"] = vcat(ProblemSets["easy"], ProblemSets["harder"])
    return ProblemSets
end

function construct_pso(prob, pop_size, max_iters)
    return SerialPSO(
        prob;
        num_particles=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        max_time=Inf,
    )
end

function construct_default_de(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(mut_strat),
        crossover_params=SelfBinomialCrossoverParameters(),
    )
end

function construct_de(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(mut_strat; dist=Uniform(0.0, 1.0)),
        crossover_params=SelfBinomialCrossoverParameters(; dist=Uniform(0.0, 1.0)),
    )
end

function construct_radius_limited_de(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(
            mut_strat;
            dist=Uniform(0.0, 1.0),
            sel=GlobalOptimization.RadiusLimitedSelector(8),
        ),
        crossover_params=SelfBinomialCrossoverParameters(; dist=Uniform(0.0, 1.0)),
    )
end

function construct_random_subset_de(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(
            mut_strat;
            dist=Uniform(0.0, 1.0),
            sel=GlobalOptimization.RandomSubsetSelector(8),
        ),
        crossover_params=SelfBinomialCrossoverParameters(; dist=Uniform(0.0, 1.0)),
    )
end

function construct_covbin_de(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(mut_strat; dist=Uniform(0.0, 1.0)),
        crossover_params=SelfBinomialCrossoverParameters(;
            dist=Uniform(0.0, 1.0),
            transform=GlobalOptimization.CovarianceTransformation(
                0.1, 0.5, GlobalOptimization.numdims(prob.ss)
            ),
        ),
    )
end

function construct_radius_limited_covbin_de(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(
            mut_strat;
            dist=Uniform(0.0, 1.0),
            sel=GlobalOptimization.RadiusLimitedSelector(8),
        ),
        crossover_params=SelfBinomialCrossoverParameters(;
            dist=Uniform(0.0, 1.0),
            transform=GlobalOptimization.CovarianceTransformation(
                0.1, 0.5, GlobalOptimization.numdims(prob.ss)
            ),
        ),
    )
end

function algorithm_set()
    return [
        ("pso", construct_pso),
        ("default_de_rand_1_bin", (p, s, i) -> construct_default_de(p, Rand1(), s, i)),
        ("default_de_best_1_bin", (p, s, i) -> construct_default_de(p, Best1(), s, i)),
        (
            "default_de_current_to_best_1_bin",
            (p, s, i) -> construct_default_de(p, CurrentToBest1(), s, i),
        ),
        (
            "default_de_current_to_rand_1_bin",
            (p, s, i) -> construct_default_de(p, CurrentToRand1(), s, i),
        ),
        (
            "default_de_randto_best_1_bin",
            (p, s, i) -> construct_default_de(p, RandToBest1(), s, i),
        ),
        ("default_de_unified_bin", (p, s, i) -> construct_de(p, Unified(), s, i)),
        ("uniform_de_rand_1_bin", (p, s, i) -> construct_de(p, Rand1(), s, i)),
        ("uniform_de_best_1_bin", (p, s, i) -> construct_de(p, Best1(), s, i)),
        (
            "uniform_de_current_to_best_1_bin",
            (p, s, i) -> construct_de(p, CurrentToBest1(), s, i),
        ),
        (
            "uniform_de_current_to_rand_1_bin",
            (p, s, i) -> construct_de(p, CurrentToRand1(), s, i),
        ),
        ("uniform_de_randto_best_1_bin", (p, s, i) -> construct_de(p, RandToBest1(), s, i)),
        ("uniform_de_unified_bin", (p, s, i) -> construct_de(p, Unified(), s, i)),
        ("rl_de_rand_1_bin", (p, s, i) -> construct_radius_limited_de(p, Rand1(), s, i)),
        ("rl_de_best_1_bin", (p, s, i) -> construct_radius_limited_de(p, Best1(), s, i)),
        (
            "rl_de_current_to_best_1_bin",
            (p, s, i) -> construct_radius_limited_de(p, CurrentToBest1(), s, i),
        ),
        (
            "rl_de_current_to_rand_1_bin",
            (p, s, i) -> construct_radius_limited_de(p, CurrentToRand1(), s, i),
        ),
        (
            "rl_de_randto_best_1_bin",
            (p, s, i) -> construct_radius_limited_de(p, RandToBest1(), s, i),
        ),
        ("rl_de_unified_bin", (p, s, i) -> construct_radius_limited_de(p, Unified(), s, i)),
        ("rs_de_rand_1_bin", (p, s, i) -> construct_random_subset_de(p, Rand1(), s, i)),
        ("rs_de_best_1_bin", (p, s, i) -> construct_random_subset_de(p, Best1(), s, i)),
        (
            "rs_de_current_to_best_1_bin",
            (p, s, i) -> construct_random_subset_de(p, CurrentToBest1(), s, i),
        ),
        (
            "rs_de_current_to_rand_1_bin",
            (p, s, i) -> construct_random_subset_de(p, CurrentToRand1(), s, i),
        ),
        (
            "rs_de_randto_best_1_bin",
            (p, s, i) -> construct_random_subset_de(p, RandToBest1(), s, i),
        ),
        ("rs_de_unified_bin", (p, s, i) -> construct_random_subset_de(p, Unified(), s, i)),
        # ("covbin_de_rand_1_bin", (p, s, i) -> construct_covbin_de(p, Rand1(), s, i)),
        # ("covbin_de_best_1_bin", (p, s, i) -> construct_covbin_de(p, Best1(), s, i)),
        # (
        #     "covbin_de_current_to_best_1_bin",
        #     (p, s, i) -> construct_covbin_de(p, CurrentToBest1(), s, i),
        # ),
        # (
        #     "covbin_de_current_to_rand_1_bin",
        #     (p, s, i) -> construct_covbin_de(p, CurrentToRand1(), s, i),
        # ),
        # (
        #     "covbin_de_randto_best_1_bin",
        #     (p, s, i) -> construct_covbin_de(p, RandToBest1(), s, i),
        # ),
        # ("covbin_de_unified_bin", (p, s, i) -> construct_covbin_de(p, Unified(), s, i)),
        # (
        #     "rl_covbin_de_rand_1_bin",
        #     (p, s, i) -> construct_radius_limited_covbin_de(p, Rand1(), s, i),
        # ),
        # (
        #     "rl_covbin_de_best_1_bin",
        #     (p, s, i) -> construct_radius_limited_covbin_de(p, Best1(), s, i),
        # ),
        # (
        #     "rl_covbin_de_current_to_best_1_bin",
        #     (p, s, i) -> construct_radius_limited_covbin_de(p, CurrentToBest1(), s, i),
        # ),
        # (
        #     "rl_covbin_de_current_to_rand_1_bin",
        #     (p, s, i) -> construct_radius_limited_covbin_de(p, CurrentToRand1(), s, i),
        # ),
        # (
        #     "rl_covbin_de_randto_best_1_bin",
        #     (p, s, i) -> construct_radius_limited_covbin_de(p, RandToBest1(), s, i),
        # ),
        # (
        #     "rl_covbin_de_unified_bin",
        #     (p, s, i) -> construct_radius_limited_covbin_de(p, Unified(), s, i),
        # ),
    ]
end

function main()
    # Get problems and algorithms
    prob_set = get_problem_sets()["all"]
    algs = algorithm_set()

    # Number of trials per case
    N = 500

    # Initialize DataFrame to store results
    data = DataFrame(;
        ProblemName=String[],
        NumDims=Int[],
        PopSize=Int[],
        MaxIters=Int[],
        AlgorithmName=String[],
        OptFitness=Float64[],
        BestFitness=Float64[],
        AvgFitness=Float64[],
    )

    for prob in prob_set
        # Get info
        prob_name = prob[1]
        num_dims = prob[2]
        pop_size = prob[3]
        max_iters = prob[4]

        println("Running $prob_name with $num_dims dimensions")

        # Get problem
        test_prob = base_test_problems[prob_name]

        # Construct problem
        opt_prob = OptimizationProblem(
            test_prob.fun,
            ContinuousRectangularSearchSpace(
                fill(test_prob.lb_per_dim, num_dims),
                fill(test_prob.ub_per_dim, num_dims),
            ),
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
            push!(
                data,
                (
                    prob_name,
                    num_dims,
                    pop_size,
                    max_iters,
                    alg[1],
                    test_prob.min,
                    best_fitness,
                    avg_fitness,
                ),
            )
        end

        # Run with BlackBoxOptiml adaptive DE/rand/1/bin radius limited
        println("\tBBO adaptive DE/rand/1/bin radius limited")
        fitness = Vector{Float64}(undef, N)
        Threads.@threads for i in 1:N
            res = BBO.bboptimize(
                test_prob.fun;
                Method=:adaptive_de_rand_1_bin_radiuslimited,
                SearchRange=(test_prob.lb_per_dim, test_prob.ub_per_dim),
                NumDimensions=num_dims,
                PopulationSize=pop_size,
                MaxFuncEvals=max_iters * pop_size,
                TraceMode=:silent,
            )
            fitness[i] = BBO.best_fitness(res)
        end
        avg_fitness = mean(fitness)
        best_fitness = minimum(fitness)
        push!(
            data,
            (
                prob_name,
                num_dims,
                pop_size,
                max_iters,
                "BBO_adaptive_de_rand_1_bin_radius_limited",
                test_prob.min,
                best_fitness,
                avg_fitness,
            ),
        )
    end

    # Save data
    date_str = replace(Dates.today(), "-" => "_")
    jldsave(
        joinpath(@__DIR__, "data", "benchmark_data_$(date_str).jld2"); 
        df=data,
        commit=get_git_commit_hash(),
    )

    return data
end

data = main()
