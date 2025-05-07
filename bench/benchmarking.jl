
using GlobalOptimization
using Statistics
using Distributions
using DataFrames
using DataFramesMeta
using JLD2
using CairoMakie

import BlackBoxOptim as BBO

# Include benchmarking utilities
include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "alg_constructers.jl"))

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
            ("Rosenbrock", 50, 40, 7500),
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

function algorithm_set()
    return [
        ("pso", construct_pso),
        (
            "default_adaptive_de_rand_1_bin",
            (p, s, i) -> construct_default_adaptive_de_mutstrat_bin(p, Rand1(), s, i),
        ),
        (
            "default_adaptive_de_best_1_bin",
            (p, s, i) -> construct_default_adaptive_de_mutstrat_bin(p, Best1(), s, i),
        ),
        (
            "default_adaptive_de_current_to_best_1_bin",
            (p, s, i) -> construct_default_adaptive_de_mutstrat_bin(p, CurrentToBest1(), s, i),
        ),
        (
            "default_adaptive_de_current_to_rand_1_bin",
            (p, s, i) -> construct_default_adaptive_de_mutstrat_bin(p, CurrentToRand1(), s, i),
        ),
        (
            "default_adaptive_de_rand_to_best_1_bin",
            (p, s, i) -> construct_default_adaptive_de_mutstrat_bin(p, RandToBest1(), s, i),
        ),
        (
            "default_adaptive_de_unified_bin",
            (p, s, i) -> construct_default_adaptive_de_mutstrat_bin(p, Unified(), s, i),
        ),
        (
            "uniform_adaptive_de_rand_1_bin",
            (p, s, i) -> construct_uniform_adaptive_de_mutstrat_bin(p, Rand1(), s, i),
        ),
        (
            "uniform_adaptive_de_best_1_bin",
            (p, s, i) -> construct_uniform_adaptive_de_mutstrat_bin(p, Best1(), s, i),
        ),
        (
            "uniform_adaptive_de_current_to_best_1_bin",
            (p, s, i) -> construct_uniform_adaptive_de_mutstrat_bin(p, CurrentToBest1(), s, i),
        ),
        (
            "uniform_adaptive_de_current_to_rand_1_bin",
            (p, s, i) -> construct_uniform_adaptive_de_mutstrat_bin(p, CurrentToRand1(), s, i),
        ),
        (
            "uniform_adaptive_de_rand_to_best_1_bin",
            (p, s, i) -> construct_uniform_adaptive_de_mutstrat_bin(p, RandToBest1(), s, i),
        ),
        (
            "uniform_adaptive_de_unified_bin",
            (p, s, i) -> construct_uniform_adaptive_de_mutstrat_bin(p, Unified(), s, i),
        ),
        (
            "rl_default_adaptive_de_rand_1_bin",
            (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_bin(p, Rand1(), s, i),
        ),
        (
            "rl_default_adaptive_de_best_1_bin",
            (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_bin(p, Best1(), s, i),
        ),
        (
            "rl_default_adaptive_de_current_to_best_1_bin",
            (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_bin(p, CurrentToBest1(), s, i),
        ),
        (
            "rl_default_adaptive_de_current_to_rand_1_bin",
            (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_bin(p, CurrentToRand1(), s, i),
        ),
        (
            "rl_default_adaptive_de_rand_to_best_1_bin",
            (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_bin(p, RandToBest1(), s, i),
        ),
        (
            "rl_default_adaptive_de_unified_bin",
            (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_bin(p, Unified(), s, i),
        ),
        (
            "rl_uniform_adaptive_de_rand_1_bin",
            (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_bin(p, Rand1(), s, i),
        ),
        (
            "rl_uniform_adaptive_de_best_1_bin",
            (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_bin(p, Best1(), s, i),
        ),
        (
            "rl_uniform_adaptive_de_current_to_best_1_bin",
            (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_bin(p, CurrentToBest1(), s, i),
        ),
        (
            "rl_uniform_adaptive_de_current_to_rand_1_bin",
            (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_bin(p, CurrentToRand1(), s, i),
        ),
        (
            "rl_uniform_adaptive_de_rand_to_best_1_bin",
            (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_bin(p, RandToBest1(), s, i),
        ),
        (
            "rl_uniform_adaptive_de_unified_bin",
            (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_bin(p, Unified(), s, i),
        ),
        (
            "default_adaptive_de_rand_1_covbin",
            (p, s, i) -> construct_default_adaptive_de_mutstrat_covbin(p, Rand1(), s, i),
        ),
        (
            "default_adaptive_de_best_1_covbin",
            (p, s, i) -> construct_default_adaptive_de_mutstrat_covbin(p, Best1(), s, i),
        ),
        (
            "default_adaptive_de_current_to_best_1_covbin",
            (p, s, i) -> construct_default_adaptive_de_mutstrat_covbin(p, CurrentToBest1(), s, i),
        ),
        (
            "default_adaptive_de_current_to_rand_1_covbin",
            (p, s, i) -> construct_default_adaptive_de_mutstrat_covbin(p, CurrentToRand1(), s, i),
        ),
        (
            "default_adaptive_de_rand_to_best_1_covbin",
            (p, s, i) -> construct_default_adaptive_de_mutstrat_covbin(p, RandToBest1(), s, i),
        ),
        (
            "default_adaptive_de_unified_covbin",
            (p, s, i) -> construct_default_adaptive_de_mutstrat_covbin(p, Unified(), s, i),
        ),
        (
            "uniform_adaptive_de_rand_1_covbin",
            (p, s, i) -> construct_uniform_adaptive_de_mutstrat_covbin(p, Rand1(), s, i),
        ),
        (
            "uniform_adaptive_de_best_1_covbin",
            (p, s, i) -> construct_uniform_adaptive_de_mutstrat_covbin(p, Best1(), s, i),
        ),
        (
            "uniform_adaptive_de_current_to_best_1_covbin",
            (p, s, i) -> construct_uniform_adaptive_de_mutstrat_covbin(p, CurrentToBest1(), s, i),
        ),
        (
            "uniform_adaptive_de_current_to_rand_1_covbin",
            (p, s, i) -> construct_uniform_adaptive_de_mutstrat_covbin(p, CurrentToRand1(), s, i),
        ),
        (
            "uniform_adaptive_de_rand_to_best_1_covbin",
            (p, s, i) -> construct_uniform_adaptive_de_mutstrat_covbin(p, RandToBest1(), s, i),
        ),
        (
            "uniform_adaptive_de_unified_covbin",
            (p, s, i) -> construct_uniform_adaptive_de_mutstrat_covbin(p, Unified(), s, i),
        ),
        (
            "rl_default_adaptive_de_rand_1_covbin",
            (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_covbin(p, Rand1(), s, i),
        ),
        (
            "rl_default_adaptive_de_best_1_covbin",
            (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_covbin(p, Best1(), s, i),
        ),
        (
            "rl_default_adaptive_de_current_to_best_1_covbin",
            (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_covbin(p, CurrentToBest1(), s, i),
        ),
        (
            "rl_default_adaptive_de_current_to_rand_1_covbin",
            (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_covbin(p, CurrentToRand1(), s, i),
        ),
        (
            "rl_default_adaptive_de_rand_to_best_1_covbin",
            (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_covbin(p, RandToBest1(), s, i),
        ),
        (
            "rl_default_adaptive_de_unified_covbin",
            (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_covbin(p, Unified(), s, i),
        ),
        (
            "rl_uniform_adaptive_de_rand_1_covbin",
            (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_covbin(p, Rand1(), s, i),
        ),
        (
            "rl_uniform_adaptive_de_best_1_covbin",
            (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_covbin(p, Best1(), s, i),
        ),
        (
            "rl_uniform_adaptive_de_current_to_best_1_covbin",
            (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_covbin(p, CurrentToBest1(), s, i),
        ),
        (
            "rl_uniform_adaptive_de_current_to_rand_1_covbin",
            (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_covbin(p, CurrentToRand1(), s, i),
        ),
        (
            "rl_uniform_adaptive_de_rand_to_best_1_covbin",
            (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_covbin(p, RandToBest1(), s, i),
        ),
        (
            "rl_uniform_adaptive_de_unified_covbin",
            (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_covbin(p, Unified(), s, i),
        ),
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
        MeanFitness=Float64[],
        MedianFitness=Float64[],
        AllFitness=Vector{Vector{Float64}}(undef, 0)
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

            # Compute fitness over N trials
            fitness = Vector{Float64}(undef, N)
            bsolver = alg[2](opt_prob, pop_size, max_iters)
            Threads.@threads for i in 1:N
                solver = deepcopy(bsolver)
                res = optimize!(solver)
                fitness[i] = res.fbest
            end

            # Get fitness statistics
            mean_fitness = mean(fitness)
            median_fitness = median(fitness)
            best_fitness = minimum(fitness)

            # Append to data
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
                    mean_fitness,
                    median_fitness,
                    fitness,
                ),
            )
        end

        # Run with BlackBoxOptiml adaptive DE/rand/1/bin radius limited
        println("\tBBO adaptive DE/rand/1/bin radius limited")

        # Compute fitness over N trials
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

        # Get fitness statistics
        mean_fitness = mean(fitness)
        median_fitness = median(fitness)
        best_fitness = minimum(fitness)

        push!(
            data,
            (
                prob_name,
                num_dims,
                pop_size,
                max_iters,
                "BBO_adaptive_de_rand_1_bin_radiuslimited",
                test_prob.min,
                best_fitness,
                mean_fitness,
                median_fitness,
                fitness,
            ),
        )
    end

    # Save data
    short_hash = get_git_commit_hash(; abbrev=true)
    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)
    jldsave(
        joinpath(data_dir, "benchmark_data_$(short_hash).jld2");
        df=data,
        commit_hash=get_git_commit_hash(),
    )

    return data, short_hash
end

function plot(data, short_hash)
    # Create directory for plots
    plot_dir = joinpath(@__DIR__, "data", "plots_$(short_hash)")
    mkpath(plot_dir)

    # Get unique problem configurations
    unique_probs = unique(data[:,1:2])

    # Loop over unique problem configs
    for i in axes(unique_probs, 1)
        # Get subset
        pname = unique_probs[i, :ProblemName]
        ndims = unique_probs[i, :NumDims]
        data_subset = @subset(data,
            :ProblemName .== pname,
            :NumDims .== ndims,
        )

        # Create figure
        fig = Figure(;size=(1920,1080))
        ax = Axis(fig[1,1]; ylabel="Avg. Fitness", yscale=log10)
        ax.xticks = (axes(data_subset, 1), data_subset[!, :AlgorithmName])
        ax.xticklabelrotation = 70.0

        # Plot data
        barplot!(ax, axes(data_subset,1), data_subset[!, :MeanFitness])

        # Save
        save(joinpath(plot_dir, "$(pname)_$(ndims)dims.pdf"), fig)
    end
end

data, shash = main()

#plot(data, shash)
