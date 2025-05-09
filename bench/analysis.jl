
using DataFrames
using DataFramesMeta
using Statistics
using JLD2
using CSV

using Infiltrator

function main(hash)
    # Load the data
    data = JLD2.load(
        joinpath(@__DIR__, "data", "benchmark_data_$(hash).jld2"),
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
        :BestFitness => sortperm => :BestFitnessRank,
    )

    # ==== Summary statistics for each algorithm
    num_rank_1s(x) = sum(xx -> ifelse(xx==1, 1, 0), x)
    alg_summary_data =
        combine(
            groupby(data, :AlgorithmName),
            :MeanFitnessRank => mean => :MeanRankByMeanFitness,
            :MeanFitnessRank => num_rank_1s => :NumRank1ByMeanFitness,
            :MedianFitnessRank => mean => :MeanRankByMedianFitness,
            :MedianFitnessRank => num_rank_1s => :NumRank1ByMedianFitness,
            :BestFitnessRank => mean => :MeanRankByBestFitness,
            :BestFitnessRank => num_rank_1s => :NumRank1ByBestFitness,
        )


    CSV.write(
        joinpath(@__DIR__, "data", "benchmark_summary_$(hash).csv"),
        alg_summary_data,
    )
end

hash = "9aff13e"
main(hash)
