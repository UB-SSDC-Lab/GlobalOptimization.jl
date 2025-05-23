
"""
    AbstractCandidate

Abstract type for a candidate
"""
abstract type AbstractCandidate{T} end

"""
    candidate(c::AbstractCandidate)

Returns the candidate `c`.
"""
candidate(c::AbstractCandidate) = c.candidate

"""
    fitness(c::AbstractCandidate)

Returns the fitness of the candidate `c`.
"""
fitness(c::AbstractCandidate) = c.candidate_fitness

"""
    set_fitness!(c::AbstractCandidate, fitness)
"""
@inline function set_fitness!(c::AbstractCandidate, fitness)
    c.candidate_fitness = fitness
    return nothing
end

@inline function set_candidate!(c::AbstractCandidate, candidate)
    c.candidate .= candidate
    return nothing
end

"""
    check_fitness!(c::AbstractCandidate, ::Val)

Checks the fitness if the option has been enabled.
"""
check_fitness!(c::AbstractCandidate, ::Val{false}) = nothing
function check_fitness!(c::AbstractCandidate, ::Val{true})
    isfinite(fitness(c)) || error("Candidate has an invalid fitness.")
    return nothing
end
