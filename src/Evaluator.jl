"""
    AbstractEvaluator

Abstract type for an evaluator. An evaluator is responsible for evaluating the fitness
of a population or candidate.
"""
abstract type AbstractEvaluator{T} end

"""
    SingleEvaluator

Abstract type for an evaluator that evaluates the fitness of a single candidate
"""
abstract type SingleEvaluator{T} <: AbstractEvaluator{T} end

"""
    BasicEvaluator

A basic evaluator that computes the fitness of a single candidate. 
"""
struct BasicEvaluator{T, has_penalty, SS <: SearchSpace{T}, F <: Function, G <: Union{Nothing, Function}} <: SingleEvaluator{T} 
    # The optimization problem
    prob::OptimizationProblem{has_penalty, SS,F,G}

    function BasicEvaluator(
        prob::OptimizationProblem{has_penalty,SS,F,G},
    ) where {T, has_penalty, SS <: SearchSpace{T}, F <: Function, G <: Union{Nothing, Function}}
        return new{T,has_penalty,SS,F,G}(prob)
    end
end

"""
    FeasibilityHandlingEvaluator

An evaluator that handled a functions returned infeasibility penalty
"""
struct FeasibilityHandlingEvaluator{T, has_penalty, SS <: SearchSpace{T}, F <: Function, G <: Union{Nothing, Function}} <: SingleEvaluator{T} 
    # The optimization problem
    prob::OptimizationProblem{has_penalty,SS,F,G}

    function FeasibilityHandlingEvaluator(
        prob::OptimizationProblem{has_penalty,SS,F,G},
    ) where {T, has_penalty, SS <: SearchSpace{T}, F <: Function, G <: Union{Nothing, Function}}
        return new{T,has_penalty,SS,F,G}(prob)
    end
end

"""
    BatchEvaluator

Abstract type for an evaluator that evaluates the fitness of an entire population.
"""
abstract type BatchEvaluator{T} <: AbstractEvaluator{T} end


"""
    AsyncEvaluator

Abstract type for an evaluator that evaluates the fitness of a single candidate asyncronously.
"""
abstract type AsyncEvaluator{T} <: SingleEvaluator{T} end


"""
    SerialBatchEvaluator

An evaluator that evaluates the fitness of a population in serial.
"""
struct SerialBatchEvaluator{T, has_penalty, SS <: SearchSpace{T}, F <: Function, G <: Union{Nothing, Function}} <: BatchEvaluator{T}
    # The optimization problem
    prob::OptimizationProblem{has_penalty,SS,F,G}

    function SerialBatchEvaluator(
        prob::OptimizationProblem{has_penalty,SS,F,G},
    ) where {T, has_penalty, SS <: SearchSpace{T}, F <: Function, G <: Union{Nothing, Function}}
        return new{T,has_penalty,SS,F,G}(prob)
    end
end

"""
    ThreadedBatchEvaluator

An evaluator that evaluates the fitness of a population in parallel using multi-threading.
"""
struct ThreadedBatchEvaluator{T, has_penalty, SS <: SearchSpace{T}, F <: Function, G <: Union{Nothing, Function}} <: BatchEvaluator{T}
    # The optimization problem
    prob::OptimizationProblem{has_penalty,SS,F,G}

    function ThreadedBatchEvaluator(
        prob::OptimizationProblem{has_penalty,SS,F,G},
    ) where {T, has_penalty, SS <: SearchSpace{T}, F <: Function, G <: Union{Nothing, Function}}
        return new{T,has_penalty,SS,F,G}(prob)
    end
end

"""
    has_gradient(evaluator::BasicEvaluator)

Returns `true` if the evaluator has a gradient, otherwise, `false`.
"""
function has_gradient(evaluator::BasicEvaluator{SS,F,G}) where {SS <: SearchSpace, F <: Function, G <: Function}
    return true
end
function has_gradient(evaluator::BasicEvaluator{SS,F,G}) where {SS <: SearchSpace, F <: Function, G <: Nothing}
    return false
end

"""
    has_gradient(evaluator::AbstractEvaluator)

Returns `true` if the evaluator has a gradient, otherwise, `false`.
"""
function has_gradient(evaluator::AbstractEvaluator)
    return false
end


"""
    evaluate!(can::AbstractCandidate, evaluator::BasicEvaluator)

Evaluates the fitness of a candidat using the given evaluator
"""
function evaluate!(c::AbstractCandidate, evaluator::BasicEvaluator{T,SS,F}) where {T,SS,F <: Function}
    @unpack candidate, candidate_fitness = c
    fun = get_scalar_function(evaluator.prob)
    set_fitness!(c, fun(candidate)) 
    return nothing
end

"""
    evaluate!(pop::AbstractPopulation, evaluator::BatchEvaluator)

Evaluates the fitness of a population using the given `evaluator`.
"""
function evaluate!(pop::AbstractPopulation, evaluator::SerialBatchEvaluator{T,SS,F}) where {T,SS,F <: Function}
    fun = get_scalar_function(evaluator.prob)
    @inbounds for (idx, candidate) in enumerate(candidates(pop))
        set_fitness!(pop, fun(candidate), idx)
    end
    return nothing
end
function evaluate!(pop::AbstractPopulation, evaluator::ThreadedBatchEvaluator)
    fun = get_scalar_function(evaluator.prob)
    Threads.@threads for idx in eachindex(candidates(pop))
        candidate = candidates(pop, idx)
        set_fitness!(pop, fun(candidate), idx)
    end
    return nothing
end


"""
    evaluate_with_penalty(evaluator::FeasibilityHandlingEvaluator, candidate::AbstractArray)
"""
function evaluate_with_penalty(evaluator::FeasibilityHandlingEvaluator, candidate::AbstractArray)
    fun = get_scalar_function_with_penalty(evaluator.prob)
    return fun(candidate)
end