
abstract type AbstractCrossoverParameters{AS <: AbstractAdaptationStrategy} end
abstract type AbstractBinomialCrossoverParameters{AS} <: AbstractCrossoverParameters{AS} end

mutable struct BinomialCrossoverParameters{AS <: AbstractAdaptationStrategy, D} <: AbstractBinomialCrossoverParameters{AS}
    CR::Float64
    dist::D

    function BinomialCrossoverParameters(CR::Float64)
        new{NoAdaptation, Nothing}(CR, nothing)
    end

    function BinomialCrossoverParameters(;
        dist = default_binomial_crossover_dist
    )
        new{RandomAdaptation, typeof(dist)}(0.0, dist)
    end
end
struct SelfBinomialCrossoverParameters{AS <: AbstractAdaptationStrategy, D} <: AbstractBinomialCrossoverParameters{AS}
    CRs::Vector{Float64}
    dist::D

    function SelfBinomialCrossoverParameters(;
        dist = default_binomial_crossover_dist
    )
        CRs = Vector{Float64}(undef, 0)
        new{RandomAdaptation, typeof(dist)}(CRs, dist)
    end
end

get_parameter(params::BinomialCrossoverParameters, i) = params.CR
get_parameter(params::SelfBinomialCrossoverParameters, i) = params.CRs[i]

initialize!(params::AbstractCrossoverParameters{NoAdaptation}, population_size) = nothing
function initialize!(params::BinomialCrossoverParameters{RandomAdaptation}, population_size)
    params.CR = semiclamped_rand(params.dist)
    return nothing
end
function initialize!(params::SelfBinomialCrossoverParameters{RandomAdaptation}, population_size)
    resize!(params.CRs, population_size)
    @inbounds for i in eachindex(params.CRs)
        params.CRs[i] = semiclamped_rand(params.dist)
    end
    return nothing
end

adapt!(params::AbstractCrossoverParameters{NoAdaptation}, improved, global_best_improved) = nothing
function adapt!(params::BinomialCrossoverParameters{RandomAdaptation}, improved, global_best_improved)
    if !global_best_improved
        params.CR = semiclamped_rand(params.dist)
    end
    return nothing
end
function adapt!(params::SelfBinomialCrossoverParameters{RandomAdaptation}, improved, global_best_improved)
    @inbounds for i in eachindex(params.CRs)
        if !improved[i]
            params.CRs[i] = semiclamped_rand(params.dist)
        end
    end
    return nothing
end

"""
    crossover!(population::DEPopulation{T}, crossover_params, search_space)

Performs the crossover operation on the population `population` using the DE crossover strategy.

This function also ensures that, after crossover, the mutants are within the search space.
"""
function crossover!(
    population::DEPopulation,
    crossover_params::AbstractBinomialCrossoverParameters,
    search_space,
)
    @unpack current_generation, mutants = population

    @inbounds for i in eachindex(mutants)
        CR = get_parameter(crossover_params, i)

        candidate = current_generation.candidates[i]
        mutant = mutants.candidates[i]

        mbr_i = rand(1:length(mutant))
        for j in eachindex(mutant)
            # Perform crossover
            if rand() > CR && j != mbr_i
                mutant[j] = candidate[j]
            end

            # Ensure mutant is within search space
            if mutant[j] < dimmin(search_space, j)
                mutant[j] = dimmin(search_space, j)
            elseif mutant[j] > dimmax(search_space, j)
                mutant[j] = dimmax(search_space, j)
            end
        end
    end

    return nothing
end
