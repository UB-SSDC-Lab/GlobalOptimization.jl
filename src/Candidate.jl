
"""
    AbstractCandidate

Abstract type for a candidate
"""
abstract type AbstractCandidate end


"""
    FitnessAwareCandidate

Abstract type for a candidate that knows (or at least has the ability to know)
its fitness.

Note: Name was selected to leave room for subtypes of AbstractCandidate that are not
aware of their fitness (i.e., just a vector representing the candidate)
"""
abstract type FitnessAwareCandidate <: AbstractCandidate end

"""
    set_fitness!(candidate,fitness)

Sets the fitness of a candidate.
"""
@inline function set_fitness!(candidate::FitnessAwareCandidate, fitness)
    candidate.fitness = fitness 
    return nothing
end

"""
    BasicCandidate

A basic candidate that knows who it is and its fitness
"""
mutable struct BasicCandidate{T} <: FitnessAwareCandidate
    value::Vector{T}
    fitness::T
    function BasicCandidate{T}(num_dims::Integer) where {T <: Number}
        num_dims > 0 || throw(ArgumentError("num_dims must be greater than 0."))
        return new{T}(Vector{T}(undef, num_dims), zero(T))
    end
end

