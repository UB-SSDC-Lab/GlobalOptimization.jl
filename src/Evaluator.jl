"""
    AbstractEvaluator

Abstract type for an evaluator. An evaluator is responsible for evaluating the fitness
of a population or candidate.
"""
abstract type AbstractEvaluator end

"""
    BatchEvaluator

Abstract type for an evaluator that evaluates the fitness of an entire population.
"""
abstract type BatchEvaluator end 

"""
    AsyncEvaluator

Abstract type for an evaluator that evaluates the fitness of a single candidate asyncronously.
"""
abstract type AsyncEvaluator end

"""
    SerialBatchEvaluator

An evaluator that evaluates the fitness of a population in serial.
"""
struct SerialBatchEvaluator{P <: AbstractOptimizationProblem} <: BatchEvaluator
    # The optimization problem
    prob::P

    function SerialBatchEvaluator(prob::P) where {P <: AbstractOptimizationProblem}
        return new{P}(prob)
    end
end

"""
    ThreadedBatchEvaluator

An evaluator that evaluates the fitness of a population in parallel using multi-threading.
"""
struct ThreadedBatchEvaluator{P <: AbstractOptimizationProblem} <: BatchEvaluator
    # The optimization problem
    prob::P

    function ThreadedBatchEvaluator(prob::P) where {P <: AbstractOptimizationProblem}
        return new{P}(prob)
    end
end

"""
    evaluate(pop::AbstractPopulation, evaluator::BatchEvaluator)

Evaluates the fitness of a population using the given `evaluator`.
"""
function evaluate!(pop::AbstractPopulation, evaluator::SerialBatchEvaluator)
    @inbounds for (idx, candidate) in enumerate(candidates(pop))
        set_fitness(pop, evaluate(evaluator.prob, candidate), idx)
    end
end
function evaluate!(pop::AbstractPopulation, evaluator::ThreadedBatchEvaluator)
    Threads.@threads for idx in eachindex(candidates(pop))
        candidate = candidates(pop, idx)
        set_fitness(pop, evaluate(evaluator.prob, candidate), idx)
    end
end
