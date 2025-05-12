
abstract type AbstractCrossoverParameters{AS<:AbstractAdaptationStrategy} end
abstract type AbstractBinomialCrossoverParameters{AS} <: AbstractCrossoverParameters{AS} end

abstract type AbstractCrossoverTransformation end
struct NoTransformation <: AbstractCrossoverTransformation end
struct CovarianceTransformation <: AbstractCrossoverTransformation
    ps::Float64
    pb::Float64
    B::Matrix{Float64}

    # Preallocate storage for transformed candidate and mutant
    ct::Vector{Float64}
    mt::Vector{Float64}

    # Preallocate storage for calculating transformation
    idxs::Vector{UInt16}

    function CovarianceTransformation(ps, pb, num_dims)
        if ps <= 0.0 || ps > 1.0
            throw(ArgumentError("ps must be in the range (0, 1]."))
        end
        if pb <= 0.0 || pb > 1.0
            throw(ArgumentError("pb must be in the range (0, 1]."))
        end
        B = Matrix{Float64}(undef, num_dims, num_dims)
        return new(ps, pb, B, zeros(num_dims), zeros(num_dims), Vector{UInt16}(undef, 0))
    end
end

initialize!(transformation::NoTransformation, population_size) = nothing
function initialize!(transformation::CovarianceTransformation, population_size)
    resize!(transformation.idxs, population_size)
    transformation.idxs .= 1:population_size
    return nothing
end

update_transformation!(transformation::NoTransformation, population) = nothing
function update_transformation!(transformation::CovarianceTransformation, population)
    # Get number of candidates to consider in the covariance matrix
    n = clamp(ceil(Int, transformation.ps * length(population)), 2, length(population))

    # Get indices of n best candidates
    sortperm!(transformation.idxs, population.current_generation.candidates_fitness)
    idxs = view(transformation.idxs, 1:n)

    # Calculate the covariance matrix for n best candidates
    C = cov(view(population.current_generation.candidates, idxs))

    # Compute eigen decomposition
    E = eigen!(C)
    transformation.B .= real.(E.vectors)

    return nothing
end

to_transformed(transformation::NoTransformation, c, m) = c, m, false
function to_transformed(transformation::CovarianceTransformation, c, m)
    if rand() < transformation.pb
        mul!(transformation.ct, transpose(transformation.B), c)
        mul!(transformation.mt, transpose(transformation.B), m)
        return transformation.ct, transformation.mt, true
    else
        return c, m, false
    end
end

from_transformed!(transformation::NoTransformation, mt, m) = nothing
function from_transformed!(transformation::CovarianceTransformation, mt, m)
    mul!(m, transformation.B, mt)
    return nothing
end

mutable struct BinomialCrossoverParameters{
    AS<:AbstractAdaptationStrategy,T<:AbstractCrossoverTransformation,D
} <: AbstractBinomialCrossoverParameters{AS}
    CR::Float64
    transform::T
    dist::D

    function BinomialCrossoverParameters(CR::Float64; transform=NoTransformation())
        return new{NoAdaptation,typeof(transform),Nothing}(CR, transform, nothing)
    end

    function BinomialCrossoverParameters(;
        dist=default_binomial_crossover_dist, transform=NoTransformation()
    )
        return new{RandomAdaptation,typeof(transform),typeof(dist)}(0.0, transform, dist)
    end
end
struct SelfBinomialCrossoverParameters{
    AS<:AbstractAdaptationStrategy,T<:AbstractCrossoverTransformation,D
} <: AbstractBinomialCrossoverParameters{AS}
    CRs::Vector{Float64}
    transform::T
    dist::D

    function SelfBinomialCrossoverParameters(;
        dist=default_binomial_crossover_dist, transform=NoTransformation()
    )
        CRs = Vector{Float64}(undef, 0)
        return new{RandomAdaptation,typeof(transform),typeof(dist)}(CRs, transform, dist)
    end
end

get_parameter(params::BinomialCrossoverParameters, i) = params.CR
get_parameter(params::SelfBinomialCrossoverParameters, i) = params.CRs[i]

function initialize!(
    params::AbstractCrossoverParameters{NoAdaptation}, num_dims, population_size
)
    initialize!(params.transform, population_size)
    return nothing
end
function initialize!(
    params::BinomialCrossoverParameters{RandomAdaptation}, num_dims, population_size
)
    initialize!(params.transform, population_size)
    params.CR = one_clamped_rand(params.dist)
    return nothing
end
function initialize!(
    params::SelfBinomialCrossoverParameters{RandomAdaptation}, num_dims, population_size
)
    initialize!(params.transform, population_size)
    resize!(params.CRs, population_size)
    @inbounds for i in eachindex(params.CRs)
        params.CRs[i] = one_clamped_rand(params.dist)
    end
    return nothing
end

function adapt!(
    params::AbstractCrossoverParameters{NoAdaptation}, improved, global_best_improved
)
    return nothing
end
function adapt!(
    params::BinomialCrossoverParameters{RandomAdaptation}, improved, global_best_improved
)
    if !global_best_improved
        params.CR = one_clamped_rand(params.dist)
    end
    return nothing
end
function adapt!(
    params::SelfBinomialCrossoverParameters{RandomAdaptation},
    improved,
    global_best_improved,
)
    @inbounds for i in eachindex(params.CRs)
        if !improved[i]
            params.CRs[i] = one_clamped_rand(params.dist)
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
    @unpack transform = crossover_params

    # Update and get transformation
    update_transformation!(crossover_params.transform, population)

    @inbounds for i in eachindex(mutants)
        CR = get_parameter(crossover_params, i)

        # Get candidate and mutant
        candidate = current_generation.candidates[i]
        mutant = mutants.candidates[i]

        # Get transformed candidate and mutant
        candidate_t, mutant_t, transformed = to_transformed(transform, candidate, mutant)

        # Perform crossover
        mbr_i = rand(1:length(mutant_t))
        for j in eachindex(mutant_t)
            if rand() > CR && j != mbr_i
                mutant_t[j] = candidate_t[j]
            end
        end

        # Transform back to original space
        if transform isa CovarianceTransformation && transformed
            from_transformed!(crossover_params.transform, mutant_t, mutant)
        end

        for j in eachindex(mutant)
            # Ensure mutant is within search space
            if mutant[j] < dim_min(search_space, j)
                mutant[j] = dim_min(search_space, j)
            elseif mutant[j] > dim_max(search_space, j)
                mutant[j] = dim_max(search_space, j)
            end
        end
    end

    return nothing
end
