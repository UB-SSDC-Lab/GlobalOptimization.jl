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
        # Check if cfunction closures are supported
        cfunction_closure_unavailable = Sys.ARCH in (
            :aarch64, :aarch64_be, :aarch64_32,  # isAArch64
            :arm, :armeb,       # isARM
            :ppc64, :ppc64le    # isPPC64
        )
        if cfunction_closure_unavailable
            @warn(
                "C function closures are not available on $(Sys.ARCH). PolyesterBatchEvaluator will likely throw an error " *
                "the first time it is used due."
            )
        end

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
    @inbounds for (idx, candidate) in enumerate(candidates(pop))
        set_fitness!(pop, evaluate(evaluator.prob, candidate), idx)
    end
    return nothing
end
function evaluate!(pop::AbstractPopulation, evaluator::ThreadedBatchEvaluator)
    Threads.@threads for idx in eachindex(candidates(pop))
        candidate = candidates(pop, idx)
        set_fitness!(pop, evaluate(evaluator.prob, candidate), idx)
    end
    return nothing
end
function evaluate!(pop::AbstractPopulation, evaluator::PolyesterBatchEvaluator)
    # We need to get the candidates before we can use @batch
    # as the @batch macro fails for eachindex(candiates(pop))
    cs = candidates(pop); N = length(cs) 
    @batch for idx in 1:N
        #candidate = candidates(pop, idx)
        candidate = cs[idx]
        fitness   = evaluate(evaluator.prob, candidate)
        set_fitness!(pop, fitness, idx)
    end
    return nothing
end


