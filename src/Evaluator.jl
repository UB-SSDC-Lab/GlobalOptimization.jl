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
    FeasibilityHandlingEvaluator

An evaluator that handled a functions returned infeasibility penalty
"""
struct FeasibilityHandlingEvaluator{T, PT <: AbstractProblem} <: SingleEvaluator{T}
    # The optimization problem
    prob::PT

    function FeasibilityHandlingEvaluator(
        prob::AbstractProblem{has_penalty, SS},
    ) where {T, has_penalty, SS <: SearchSpace{T}}
        return new{T,typeof(prob)}(prob)
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
struct SerialBatchEvaluator{T, has_penalty, SS <: SearchSpace{T}, F, G} <: BatchEvaluator{T}
    # The optimization problem
    prob::OptimizationProblem{has_penalty,SS,F,G}

    function SerialBatchEvaluator(
        prob::OptimizationProblem{has_penalty,SS,F,G},
    ) where {T, has_penalty, SS <: SearchSpace{T}, F, G}
        return new{T,has_penalty,SS,F,G}(prob)
    end
end

"""
    ThreadedBatchEvaluator

An evaluator that evaluates the fitness of a population in parallel using multi-threading.
"""
struct ThreadedBatchEvaluator{
    T, has_penalty, SS <: SearchSpace{T}, F, G,
    S <: ChunkSplitters.Split
} <: BatchEvaluator{T}
    # The optimization problem
    prob::OptimizationProblem{has_penalty,SS,F,G}

    # Chunk splitting args
    n::Int
    split::S

    function ThreadedBatchEvaluator(
        prob::OptimizationProblem{has_penalty,SS,F,G},
        n::Int = Threads.nthreads(),
        split::S = ChunkSplitters.RoundRobin(),
    ) where {T, has_penalty, SS <: SearchSpace{T}, F, G, S <: ChunkSplitters.Split}
        return new{T,has_penalty,SS,F,G,S}(prob, n, split)
    end
end

"""
    PolyesterBatchEvaluator

An evaluator that evaluates the fitness of a population in parallel using multi-threading using Polyester.jl.
"""
struct PolyesterBatchEvaluator{T, has_penalty, SS <: SearchSpace{T}, F, G} <: BatchEvaluator{T}
    # The optimization problem
    prob::OptimizationProblem{has_penalty,SS,F,G}

    function PolyesterBatchEvaluator(
        prob::OptimizationProblem{has_penalty,SS,F,G},
    ) where {T, has_penalty, SS <: SearchSpace{T}, F, G}
        return new{T,has_penalty,SS,F,G}(prob)
    end
end

"""
    has_gradient(evaluator::AbstractEvaluator)

Returns `true` if the evaluator has a gradient, otherwise, `false`.
"""
function has_gradient(evaluator::AbstractEvaluator)
    return false
end

"""
    evaluate!(pop::AbstractPopulation, evaluator::BatchEvaluator)

Evaluates the fitness of a population using the given `evaluator`.
"""
function evaluate!(pop::AbstractPopulation, evaluator::SerialBatchEvaluator)
    @inbounds for (idx, candidate) in enumerate(candidates(pop))
        fitness = scalar_function(evaluator.prob, candidate)
        set_fitness!(pop, fitness, idx)
    end
    return nothing
end
function evaluate!(pop::AbstractPopulation, evaluator::ThreadedBatchEvaluator)
    citer = ChunkSplitters.chunks(
        eachindex(pop);
        n = evaluator.n,
        split = evaluator.split,
    )

    cs = candidates(pop)
    @sync for idxs in citer
        Threads.@spawn begin
            for idx in idxs
                candidate = cs[idx]
                fitness = scalar_function(evaluator.prob, candidate)
                set_fitness!(pop, fitness, idx)
            end
        end
    end
    return nothing
end
function evaluate!(pop::AbstractPopulation, evaluator::PolyesterBatchEvaluator)
    # Define fitness evaluation function for Polyester
    # NOTE: This is necessary for the @batch macro to work properly
    # on Arm.
    eval_fitness = let cs=candidates(pop), pop=pop, prob=evaluator.prob
        (idx) -> begin
            candidate = cs[idx]
            fitness = scalar_function(prob, candidate)
            set_fitness!(pop, fitness, idx)
        end
    end

    @batch for idx in eachindex(pop)
        eval_fitness(idx)
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
