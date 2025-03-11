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
    fitness(pop) .= fitness
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
@inline check_fitness!(pop::AbstractPopulation, options::GeneralOptions{D,FVC}) where {D, FVC} = check_fitness!(pop, FVC)
@inline check_fitness!(pop::AbstractPopulation, ::Val{false}) = nothing
function check_fitness!(pop::AbstractPopulation, ::Val{true})
    @unpack candidates_fitness = pop
    @inbounds for (i, fitness) in enumerate(candidates_fitness)
        isfinite(fitness) || error("Candidate $i has an invalid fitness.")
    end
    return nothing
end
