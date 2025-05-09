using ArgParse
using DataFrames
using DataFramesMeta
using Statistics
using HypothesisTests
using JLD2
using CSV

using Infiltrator

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--hash", "-h"
        help = "The abbreviated commit hash of the benchmark data to load."
        required = true
        arg_type = String
        "--problem-set", "-p"
        help = "Specify if analyzing benchmark data for a specific problem set."
        arg_type = String
    end

    return parse_args(s)
end

function main()
    # Parse command line arguments
    parsed_args = parse_commandline()

    # Load the data
    data = JLD2.load(
        joinpath(@__DIR__, "data", "benchmark_data_$(parsed_args["hash"]).jld2"),
    )["df"]

    # Get unique algorithms in benchmark data
    algs = unique(data.AlgorithmName)

    # Get unique problems in benchmark data
    # Note: ProblemName + NumDims + PopSize defines a unique problem
    problems = unique(data[!, [:ProblemName, :NumDims, :PopSize]])

    # ==== Rank each algorithm from best to worst median, mean, and best fitness ====
    transform!(
        groupby(data, [:ProblemName, :NumDims, :PopSize]),
        :MeanFitness => sortperm => :MeanFitnessRank,
        :MedianFitness => sortperm => :MedianFitnessRank,
    )

    # ==== Rank each algorithm for each problem based on Mann Whitney U test ====
    grouped_data = groupby(data, [:ProblemName, :NumDims, :PopSize])
    for group in grouped_data
        # Adjust ranks based on Mann Whitney U test
        # Note: If two or more sequentially ranked algorithms are not statistically different,
        # they are assigned the same rank

        # ==== Mean fitness ranks
        mean_fitness_rank_idxs = sortperm(group.MeanFitnessRank)
        for i in 1:length(mean_fitness_rank_idxs) - 1
            # Perform Mann Whitney U test between the two groups
            pval = pvalue(
                MannWhitneyUTest(
                    group.AllFitness[mean_fitness_rank_idxs[i]],
                    group.AllFitness[mean_fitness_rank_idxs[i + 1]]
                )
            )

            # If distributions are statistically similar, assign the same rank and adjust
            # the remaining ranks
            if pval >= 0.05
                # Assign the same rank to both groups
                group.MeanFitnessRank[mean_fitness_rank_idxs[i + 1]] =
                    group.MeanFitnessRank[mean_fitness_rank_idxs[i]]

                # Adjust the remaining ranks
                for j in (i + 2):length(mean_fitness_rank_idxs)
                    group.MeanFitnessRank[mean_fitness_rank_idxs[j]] -= 1
                end
            end
        end

        # ==== Median fitness ranks
        median_fitness_rank_idxs = sortperm(group.MedianFitnessRank)
        for i in 1:length(median_fitness_rank_idxs) - 1
            # Perform Mann Whitney U test between the two groups
            pval = pvalue(
                MannWhitneyUTest(
                    group.AllFitness[median_fitness_rank_idxs[i]],
                    group.AllFitness[median_fitness_rank_idxs[i + 1]]
                )
            )

            # If distributions are statistically similar, assign the same rank and adjust
            # the remaining ranks
            if pval >= 0.05
                # Assign the same rank to both groups
                group.MedianFitnessRank[median_fitness_rank_idxs[i + 1]] =
                    group.MedianFitnessRank[median_fitness_rank_idxs[i]]

                # Adjust the remaining ranks
                for j in (i + 2):length(median_fitness_rank_idxs)
                    group.MedianFitnessRank[median_fitness_rank_idxs[j]] -= 1
                end
            end
        end
    end

    # ==== Summary statistics for each algorithm
    num_rank_1s(x) = sum(xx -> ifelse(xx==1, 1, 0), x)
    alg_summary_data =
        combine(
            groupby(data, :AlgorithmName),
            :MeanFitnessRank => mean => :MeanRankByMeanFitness,
            :MeanFitnessRank => num_rank_1s => :NumRank1ByMeanFitness,
            :MedianFitnessRank => mean => :MeanRankByMedianFitness,
            :MedianFitnessRank => num_rank_1s => :NumRank1ByMedianFitness,
        )


    CSV.write(
        joinpath(@__DIR__, "data", "benchmark_summary_$(hash).csv"),
        alg_summary_data,
    )
end

main()
