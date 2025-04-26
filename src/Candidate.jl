
"""
    AbstractCandidate

Abstract type for a candidate
"""
abstract type AbstractCandidate end

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

"""
    check_fitness!(c::AbstractCandidate, options::Union{GeneralOptions,Val{true},Val{false}})

Checks the fitness of the candidate `c` to ensure that it is valid
iff options <: Union{GeneralOptions{D,Val{true}}, Val{true}}, otherwise, does nothing.
"""
@inline check_fitness!(c::AbstractCandidate, options::GeneralOptions{D,FVC}) where {D,FVC} = check_fitness!(
    c, FVC
)
@inline check_fitness!(c::AbstractCandidate, ::Val{false}) = nothing
function check_fitness!(c::AbstractCandidate, ::Val{true})
    isfinite(fitness(c)) || error("Candidate has an invalid fitness.")
    return nothing
end
