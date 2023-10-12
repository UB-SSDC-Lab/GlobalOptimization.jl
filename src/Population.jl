"""
    AbstractPopulation

Abstract type for a population of candidates.
"""
abstract type AbstractPopulation end

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
function set_fitness(pop::AbstractPopulation, fitness::Vector)
    length(fitness) == length(candidates(pop)) ||
        throw(DimensionMismatch("fitness must be the same length as candidates."))
    fitness(pop) .= fitness
    return nothing
end
function set_fitness(pop::AbstractPopulation, fitness::Real, i::Integer)
    pop.candidates_fitness[i] = fitness
    return nothing
end

