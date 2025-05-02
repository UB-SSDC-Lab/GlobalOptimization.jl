
using GlobalOptimization
using DataFrames
using DataFramesMeta
using JLD2
using CairoMakie
using Infiltrator

function main()
    # Load the benchmark data
    data = load("benchmark_data.jld2")["df"]

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
        barplot!(ax, axes(data_subset,1), data_subset[!, :AvgFitness])

        # Save
        save("figs/$(pname)_$(ndims)dims.pdf", fig)
    end
end

main()
