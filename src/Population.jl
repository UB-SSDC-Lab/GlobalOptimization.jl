"""
    AbstractPopulation

Abstract type for a population of candidates.
"""
abstract type AbstractPopulation{T} end

"""
    eachindex(pop::AbstractPopulation)

Returns an iterator for the indices of the population.
"""
Base.eachindex(pop::AbstractPopulation) = Base.eachindex(pop.candidates)

"""
    length(pop::AbstractPopulation)

Returns the number of candidates in the population.
"""
Base.length(pop::AbstractPopulation) = Base.length(pop.candidates)

"""
    size(pop::AbstractPopulation)

Returns the size of the population.
"""
Base.size(pop::AbstractPopulation) = size(pop.candidates)

"""
    candidates(pop::AbstractPopulation, [i::Integer])

Returns the candidates from a population. If `i` is specified, returns the `i`-th candidate.
"""
candidates(pop::AbstractPopulation) = pop.candidates
candidates(pop::AbstractPopulation, i::Integer) = pop.candidates[i]

"""
    fitness(pop::AbstractPopulation, [i::Integer])

Returns the fitness of the candidates from a population. If `i` is specified, returns the `i`-th fitness.
"""
fitness(pop::AbstractPopulation) = pop.candidates_fitness
fitness(pop::AbstractPopulation, i::Integer) = pop.candidates_fitness[i]

"""
    set_fitness(pop::AbstractPopulation, fitness, [i::Integer])

Sets the fitness of the candidates from a population. If `i` is specified, sets the `i`-th fitness.
"""
function set_fitness!(pop::AbstractPopulation, fitness::Vector)
    length(fitness) == length(candidates(pop)) ||
        throw(DimensionMismatch("fitness must be the same length as candidates."))
    GlobalOptimization.fitness(pop) .= fitness
    return nothing
end
@inline function set_fitness!(pop::AbstractPopulation, fitness::Real, i::Integer)
    pop.candidates_fitness[i] = fitness
    return nothing
end

"""
    check_fitness!(pop::AbstractPopulation, options::Union{GeneralOptions,Val{true},Val{false}})

Checks the fitness of each candidate in the population `pop` to ensure that it is valid
iff options <: Union{GeneralOptions{D,Val{true}}, Val{true}}, otherwise, does nothing.
"""
check_fitness!(pop::AbstractPopulation, ::Val{false}) = nothing
function check_fitness!(pop::AbstractPopulation, ::Val{true})
    @unpack candidates_fitness = pop
    @inbounds for (i, fitness) in enumerate(candidates_fitness)
        isfinite(fitness) || error("Candidate $i has an invalid fitness.")
    end
    return nothing
end

# ==== Some utilities for initialization of populations

abstract type AbstractPopulationInitialization end

"""
    UniformInitialization

Initializes a population from a uniform distribution in the search space.
"""
struct UniformInitialization <: AbstractPopulationInitialization end

"""
    LatinHypercubeInitialization

Initializes a population using optimal Latin hypercube sampling as implemented in
[LatinHypercubeSampling.jl](https://github.com/MrUrq/LatinHypercubeSampling.jl/tree/master).

# Fields:
- `gens::Int`: Number of GA generations to use to generate the Latin hypercube samples.
- `rng::U`: Random number generator to use for the Latin hypercube sampling.
- `pop_size::Int`: Size of the GA population used to generate the Latin hypercube samples.
- `n_tour::Int`: Number of tours to use in the GA.
- `p_tour::Float64`: Probability of tour to use in the GA.
- `inter_sample_weight::Float64`: Weight of the inter-sample distance in the GA.
- `periodic_ae::Bool`: Whether to use periodic adaptive evolution in the GA.
- `ae_power::Float64`: Power of the adaptive evolution in the GA.
"""
struct LatinHypercubeInitialization{U<:AbstractRNG} <: AbstractPopulationInitialization
    gens::Int
    rng::U
    pop_size::Int
    n_tour::Int
    p_tour::Float64
    inter_sample_weight::Float64
    periodic_ae::Bool
    ae_power::Float64

    @doc """
        LatinHypercubeInitialization(gens::Int = 10; kwargs...)

    Initializes a Latin hypercube sampling method with the given parameters.

    # Arguments
    - `gens::Int`: Number of GA generations to use to generate the Latin hypercube samples.
        Defaults to 10.

    # Keyword Arguments
    - `rng::U`: Random number generator to use for the Latin hypercube sampling.
        Defaults to `Random.GLOBAL_RNG`.
    - `pop_size::Int`: Size of the GA population used to generate the Latin hypercube samples.
        Defaults to 100.
    - `n_tour::Int`: Number of tours to use in the GA. Defaults to 2.
    - `p_tour::Float64`: Probability of tour to use in the GA. Defaults to 0.8.
    - `inter_sample_weight::Float64`: Weight of the inter-sample distance in the GA.
        Defaults to 1.0.
    - `periodic_ae::Bool`: Whether to use periodic adaptive evolution in the GA.
        Defaults to false.
    - `ae_power::Float64`: Power of the adaptive evolution in the GA. Defaults to 2.0.
    """
    function LatinHypercubeInitialization(
        gens::Int=10;
        rng::U=GLOBAL_RNG,
        pop_size::Int=100,
        n_tour::Int=2,
        p_tour::Float64=0.8,
        inter_sample_weight::Float64=1.0,
        periodic_ae::Bool=false,
        ae_power::Float64=2.0,
    ) where {U<:AbstractRNG}
        return new{U}(
            gens, rng, pop_size, n_tour, p_tour, inter_sample_weight, periodic_ae, ae_power
        )
    end
end

function initialize_population_vector!(
    pop_vec::Vector{<:AbstractVector{T}},
    min::AbstractVector,
    max::AbstractVector,
    method::UniformInitialization,
) where {T}
    @inbounds for i in eachindex(pop_vec)
        vec = pop_vec[i]
        for j in eachindex(vec)
            vec[j] = min[j] + (max[j] - min[j]) * rand(T)
        end
    end
    return nothing
end

function initialize_population_vector!(
    pop_vec::Vector{<:AbstractVector{T}},
    min::AbstractVector,
    max::AbstractVector,
    method::LatinHypercubeInitialization,
) where {T}
    # Generate the optimal Latin hypercube samples
    scaled_plan = begin
        plan, _ = LHCoptim(
            length(pop_vec),
            length(pop_vec[1]),
            method.gens;
            rng=method.rng,
            ntour=method.n_tour,
            ptour=method.p_tour,
            interSampleWeight=method.inter_sample_weight,
            periodic_ae=method.periodic_ae,
            ae_power=method.ae_power,
        )
        scaleLHC(plan, map(identity, zip(min, max)))
    end

    # Initialize the population vector with the samples
    for i in eachindex(pop_vec)
        vec = pop_vec[i]
        for j in eachindex(vec)
            vec[j] = scaled_plan[i, j]
        end
    end

    return nothing
end
