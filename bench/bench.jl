using ArgParse
using GlobalOptimization
using Statistics
using Distributions
using DataFrames
using JLD2

import BlackBoxOptim as BBO

import LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)

# Include benchmarking utility functions
include(joinpath(@__DIR__, "src", "utils.jl"))
include(joinpath(@__DIR__, "src", "alg_constructors.jl"))

# Include benchmark problem set and algorithm set
include(joinpath(@__DIR__, "bench_config.jl"))

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--problem-set", "-p"
            help = "The problem set to run. Default is all problems."
            arg_type = String
            default = "all"
        "--num-trials", "-n"
            help = "The number of trials to run for each problem and algorithm. Default is 50."
            arg_type = Int
            default = 50
    end

    return parse_args(s)
end

function main()
    # Parse command line arguments
    parsed_args = parse_commandline()

    # Get problems and algorithms
    prob_set = get_problem_sets()[parsed_args["problem-set"]]
    algs = get_algorithm_sets()["all"]

    # Get commit hash
    short_hash = get_git_commit_hash(; abbrev=true)
    full_hash = get_git_commit_hash()

    # Number of trials per case
    N = parsed_args["num-trials"]

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
    data_dir = joinpath(@__DIR__, "data")
    file_name = if parsed_args["problem-set"] == "all"
        "benchmark_data_$(short_hash).jld2"
    else
        "benchmark_data_$(parsed_args["problem-set"])_$(short_hash).jld2"
    end
    mkpath(data_dir)
    jldsave(
        joinpath(data_dir, file_name);
        df=data,
        commit_hash=full_hash,
    )

    return data, short_hash
end

data, shash = main()
