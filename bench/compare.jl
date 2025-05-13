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
        "--hash-a"
        help = "The abbreviated commit hash of the first benchmark data set to load."
        required = true
        arg_type = String
        "--hash-b"
        help = "The abbreviated commit hash of the second benchmark data set to load."
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
    data_a = JLD2.load(
        joinpath(@__DIR__, "data", "benchmark_data_$(parsed_args["hash-a"]).jld2"),
    )["df"]
    data_b = JLD2.load(
        joinpath(@__DIR__, "data", "benchmark_data_$(parsed_args["hash-b"]).jld2"),
    )["df"]

    # Get intersection of algorithms in benchmarks
    algs = intersect(
        unique(data_a.AlgorithmName),
        unique(data_b.AlgorithmName),
    )

    # Get intersection of problems in benchmarks
    probs = innerjoin(
        unique(data_a[!, [:ProblemName, :NumDims]]),
        unique(data_b[!, [:ProblemName, :NumDims]]);
        on = [:ProblemName, :NumDims],
    )

    # For all algorithm/problem combos, check if the mean fitness is statistically different
    a_versus_b = DataFrame(
        ProblemName = String[],
        NumDims = Int[],
        PopSize = Int[],
        AlgorithmName = String[],
        PValue = Float64[],
        Change = Char[],
    )
    for alg in algs
        for prob in eachrow(probs)
            # Get the data for the algorithm and problem
            filter_fun(row) = begin
                row.AlgorithmName == alg &&
                row.ProblemName == prob.ProblemName &&
                row.NumDims == prob.NumDims &&
                row.PopSize == prob.PopSize
            end
            alg_prob_a = filter(filter_fun, data_a)
            alg_prob_b = filter(filter_fun, data_b)

            # Perform Mann Whitney U test
            pval = pvalue(
                MannWhitneyUTest(
                    alg_prob_a.AllFitness[1],
                    alg_prob_b.AllFitness[1]
                )
            )

            # Append to the results
            change = '0'
            if pval < 0.05
                if alg_prob_a.MedianFitness < alg_prob_b.MedianFitness
                    change = '-'
                else
                    change = '+'
                end
            end
            push!(
                a_versus_b,
                (
                    prob.ProblemName,
                    prob.NumDims,
                    prob.PopSize,
                    alg,
                    pval,
                    change,
                )
            )
        end
    end

    # Print comparison summary
    sig_diffs = filter(row -> row.Change != '0', a_versus_b)

    n_diffs = size(sig_diffs, 1)
    total = size(a_versus_b, 1)
    percent = round(Int, n_diffs / total * 100.0)

    n_improved = sum(c -> ifelse(c == '-', 1, 0), sig_diffs.Change)
    n_worse = sum(c -> ifelse(c == '+', 1, 0), sig_diffs.Change)

    println(
        "Found $(n_diffs) different experiments out of $(total) ($(percent)%)."
    )
    println(
        " - Improved:\t$(n_improved) ($(round(Int, n_improved / n_diffs * 100.0))% of diffs)",
    )
    println(
        " - Worse:\t$(n_worse) ($(round(Int, n_worse / n_diffs * 100.0))% of diffs)",
    )

    # Write differences to CSV
    file_name = "$(parsed_args["hash-a"])_vs_$(parsed_args["hash-b"]).csv"
    CSV.write(
        joinpath(@__DIR__, "data", "diffs", file_name),
        filter(
            row -> row.Change != '0',
            a_versus_b
        )
    )
end

if isempty(ARGS)
    push!(ARGS, "--hash-a", "4f969b2")
    push!(ARGS, "--hash-b", "9aff13e")
end

main()
