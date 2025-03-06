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
struct BasicEvaluator{T, SS <: SearchSpace{T}, F <: Function} <: SingleEvaluator{T}
    # The optimization problem
    prob::OptimizationProblem{SS,F}

    function BasicEvaluator(prob::OptimizationProblem{SS,F}) where {T, SS <: SearchSpace{T}, F <: Function}
        return new{T,SS,F}(prob)
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
struct SerialBatchEvaluator{T, SS <: SearchSpace{T}, F <: Function} <: BatchEvaluator{T}
    # The optimization problem
    prob::OptimizationProblem{SS,F}

    function SerialBatchEvaluator(prob::OptimizationProblem{SS,F}) where {T, SS <: SearchSpace{T}, F <: Function}
        return new{T,SS,F}(prob)
    end
end


"""
    ThreadedBatchEvaluator

An evaluator that evaluates the fitness of a population in parallel using multi-threading.
"""
struct ThreadedBatchEvaluator{T, SS <: SearchSpace{T}, F <: Function} <: BatchEvaluator{T}
    # The optimization problem
    prob::OptimizationProblem{SS,F}

    function ThreadedBatchEvaluator(prob::OptimizationProblem{SS,F}) where {T, SS <: SearchSpace{T}, F <: Function}
        return new{T,SS,F}(prob)
    end
end

"""
    PolyesterBatchEvaluator

An evaluator that evaluates the fitness of a population in parallel using multi-threading using Polyester.jl.
"""
struct PolyesterBatchEvaluator{T, SS <: SearchSpace{T}, F <: Function} <: BatchEvaluator{T}
    # The optimization problem
    prob::OptimizationProblem{SS,F}

    function PolyesterBatchEvaluator(prob::OptimizationProblem{SS,F}) where {T, SS <: SearchSpace{T}, F <: Function}
        return new{T,SS,F}(prob)
    end
end

"""
    evaluate!(can::AbstractCandidate, evaluator::BasicEvaluator)

Evaluates the fitness of a candidat using the given evaluator
"""
function evaluate!(c::AbstractCandidate, evaluator::BasicEvaluator{T,SS,F}) where {T,SS,F <: Function}
    @unpack candidate, candidate_fitness = c
    set_fitness!(c, evaluate(evaluator.prob, candidate))
    return nothing
end

"""
    evaluate!(pop::AbstractPopulation, evaluator::BatchEvaluator)

Evaluates the fitness of a population using the given `evaluator`.
"""
function evaluate!(pop::AbstractPopulation, evaluator::SerialBatchEvaluator{T,SS,F}) where {T,SS,F <: Function}
    @inbounds for idx in eachindex(pop)
        candidate = candidates(pop, idx)
        set_fitness!(pop, evaluate(evaluator.prob, candidate), idx)
    end
    return nothing
end
function evaluate!(pop::AbstractPopulation, evaluator::ThreadedBatchEvaluator)
    Threads.@threads for idx in eachindex(pop)
        candidate = candidates(pop, idx)
        set_fitness!(pop, evaluate(evaluator.prob, candidate), idx)
    end
    return nothing
end
function evaluate!(pop::AbstractPopulation, evaluator::PolyesterBatchEvaluator)
    # Define fitness evaluation function for Polyester
    # NOTE: This is necessary for the @batch macro to work properly
    # on Arm.
    eval_fitness = let cs=candidates(pop), prob=evaluator.prob, pop=pop
        (idx) -> begin
            candidate = cs[idx]
            fitness   = evaluate(prob, candidate)
            set_fitness!(pop, fitness, idx)
        end
    end

    @batch for idx in eachindex(pop)
        eval_fitness(idx)
    end
    return nothing
end
