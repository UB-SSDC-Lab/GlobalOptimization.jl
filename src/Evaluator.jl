"""
    AbstractFunctionEvaluationMethod

A function evaluation method is a strategy for evaluating the fitness/objective, as well as
    possibly other algorithm specific things.
"""
abstract type AbstractFunctionEvaluationMethod end

"""
    SerialFunctionEvaluation

A function evaluation method that evaluates the fitness of a candidate in serial.
"""
struct SerialFunctionEvaluation <: AbstractFunctionEvaluationMethod
    @doc """
        SerialFunctionEvaluation()

    Construct a `SerialFunctionEvaluation` object.
    """
    function SerialFunctionEvaluation()
        return new()
    end
end

"""
    ThreadedFunctionEvaluation{S <: ChunkSplitters.Split}

A function evaluation method that evaluates the fitness of a candidate in parallel using
    multi-threading from Base.Threads.jl.

# Fields
- `n::Int`: The number of batch jobs to split the workload into using
    [ChunkSplitters.jl](https://github.com/JuliaFolds2/ChunkSplitters.jl).
- `split::S`: The chunk splitter to use. See [ChunkSplitters.jl](https://github.com/JuliaFolds2/ChunkSplitters.jl)
    for more information.
"""
struct ThreadedFunctionEvaluation{S<:ChunkSplitters.Split} <: AbstractFunctionEvaluationMethod
    n::Int
    split::S
    @doc """
        ThreadedFunctionEvaluation(
            n::Int=Threads.nthreads(),
            split::S=ChunkSplitters.RoundRobin(),
        )

    Construct a `ThreadedFunctionEvaluation` object.

    # Keyword Arguments
    - `n::Int`: The number of batch jobs to split the workload into using
        [ChunkSplitters.jl](https://github.com/JuliaFolds2/ChunkSplitters.jl).
    - `split::S`: The chunk splitter to use. See [ChunkSplitters.jl](https://github.com/JuliaFolds2/ChunkSplitters.jl)
        for more information.
    """
    function ThreadedFunctionEvaluation(;
        n::Int=Threads.nthreads(),
        split::S=ChunkSplitters.RoundRobin(),
    ) where {S<:ChunkSplitters.Split}
        return new{S}(n, split)
    end
end

"""
    PolyesterFunctionEvaluation

A function evaluation method that evaluates the fitness of a candidate in parallel using
    [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl).
"""
struct PolyesterFunctionEvaluation <: AbstractFunctionEvaluationMethod
    @doc """
        PolyesterFunctionEvaluation()

    Construct a `PolyesterFunctionEvaluation` object.
    """
    function PolyesterFunctionEvaluation()
        return new()
    end
end

"""
    AbstractEvaluator

Abstract type for an evaluator. An evaluator is responsible for evaluating the fitness
of a population or candidate.
"""
abstract type AbstractEvaluator end

"""
    SingleEvaluator

Abstract type for an evaluator that evaluates the fitness of a single candidate
"""
abstract type SingleEvaluator <: AbstractEvaluator end

"""
    FeasibilityHandlingEvaluator

An evaluator that handled a functions returned infeasibility penalty
"""
struct FeasibilityHandlingEvaluator{PT<:AbstractProblem} <: SingleEvaluator
    # The optimization problem
    prob::PT

    function FeasibilityHandlingEvaluator(
        prob::AbstractProblem{has_penalty,SS}
    ) where {has_penalty,SS<:SearchSpace}
        return new{typeof(prob)}(prob)
    end
end

"""
    BatchEvaluator

Abstract type for an evaluator that evaluates the fitness of an entire population.
"""
abstract type BatchEvaluator <: AbstractEvaluator end

"""
    AsyncEvaluator

Abstract type for an evaluator that evaluates the fitness of a single candidate asyncronously.
"""
abstract type AsyncEvaluator <: SingleEvaluator end

"""
    SerialBatchEvaluator

An evaluator that evaluates the fitness of a population in serial.
"""
struct SerialBatchEvaluator{has_penalty,SS<:SearchSpace,F,G} <: BatchEvaluator
    # The optimization problem
    prob::OptimizationProblem{has_penalty,SS,F,G}

    function SerialBatchEvaluator(
        prob::OptimizationProblem{has_penalty,SS,F,G}
    ) where {has_penalty,SS<:SearchSpace,F,G}
        return new{has_penalty,SS,F,G}(prob)
    end
end

"""
    BatchJobEvaluator

An evaluator that evaluates a batch of jobs
"""
abstract type BatchJobEvaluator <: AbstractEvaluator end

"""
    SerialBatchJobEvaluator

An evaluator that evaluates a batch of jobs in serial.
"""
struct SerialBatchJobEvaluator <: BatchJobEvaluator end

"""
    ThreadedBatchJobEvaluator

An evaluator that evaluates a batch of jobs in parallel using Threads.jl and
ChunkSplitters.jl.
"""
struct ThreadedBatchJobEvaluator{S<:ChunkSplitters.Split} <: BatchJobEvaluator
    # Chunk splitting args
    n::Int
    split::S
    function ThreadedBatchJobEvaluator(
        n=Threads.nthreads(),
        split::S=ChunkSplitters.RoundRobin(),
    ) where {S<:ChunkSplitters.Split}
        return new{S}(n, split)
    end
end

"""
    PolyesterBatchJobEvaluator

An evaluator that evaluates a batch of jobs in parallel using Polyester.jl.
"""
struct PolyesterBatchJobEvaluator <: BatchJobEvaluator end

"""
    ThreadedBatchEvaluator

An evaluator that evaluates the fitness of a population in parallel using multi-threading.
"""
struct ThreadedBatchEvaluator{
    has_penalty,SS<:SearchSpace,F,G,S<:ChunkSplitters.Split
} <: BatchEvaluator
    # The optimization problem
    prob::OptimizationProblem{has_penalty,SS,F,G}

    # Chunk splitting args
    n::Int
    split::S

    function ThreadedBatchEvaluator(
        prob::OptimizationProblem{has_penalty,SS,F,G}, n::Int, split::S,
    ) where {has_penalty,SS<:SearchSpace,F,G,S<:ChunkSplitters.Split}
        return new{has_penalty,SS,F,G,S}(prob, n, split)
    end
end

"""
    PolyesterBatchEvaluator

An evaluator that evaluates the fitness of a population in parallel using multi-threading using Polyester.jl.
"""
struct PolyesterBatchEvaluator{has_penalty,SS<:SearchSpace,F,G} <: BatchEvaluator
    # The optimization problem
    prob::OptimizationProblem{has_penalty,SS,F,G}

    function PolyesterBatchEvaluator(
        prob::OptimizationProblem{has_penalty,SS,F,G}
    ) where {has_penalty,SS<:SearchSpace,F,G}
        return new{has_penalty,SS,F,G}(prob)
    end
end

"""
    construct_batch_evaluator(
        method::AbstractFunctionEvaluationMethod,
        prob::OptimizationProblem,
    )
"""
function construct_batch_evaluator(method::SerialFunctionEvaluation, prob)
    return SerialBatchEvaluator(prob)
end
function construct_batch_evaluator(method::ThreadedFunctionEvaluation, prob)
    return ThreadedBatchEvaluator(prob, method.n, method.split)
end
function construct_batch_evaluator(method::PolyesterFunctionEvaluation, prob)
    return PolyesterBatchEvaluator(prob)
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
    citer = ChunkSplitters.chunks(eachindex(pop); n=evaluator.n, split=evaluator.split)

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
    eval_fitness = let pop = pop, prob = evaluator.prob
        (idx) -> begin
            cs = candidates(pop)
            @inbounds set_fitness!(pop, scalar_function(prob, cs[idx]), idx)
        end
    end

    @batch for idx in eachindex(pop)
        eval_fitness(idx)
    end
    return nothing
end

"""
    evaluate!(job::Function, job_ids::AbstractVector{Int}, evaluator::BatchJobEvaluator)

Evaluates `job` for each element of `job_args` using the given `evaluator`.
"""
function evaluate!(
    job::Function, job_args::AbstractVector, evaluator::SerialBatchJobEvaluator
)
    @inbounds for arg in job_args
        job(arg)
    end
    return nothing
end
function evaluate!(
    job::Function, job_args::AbstractVector, evaluator::ThreadedBatchJobEvaluator
)
    batched_args = ChunkSplitters.chunks(
        job_args; n=evaluator.n, split=evaluator.split
    )
    @sync for args in batched_args
        Threads.@spawn begin
            for arg in args
                job(arg)
            end
        end
    end
    return nothing
end
function evaluate!(
    job::Function, job_args::AbstractVector, evaluator::PolyesterBatchJobEvaluator
)
    @batch for arg in job_args
        job(arg)
    end
    return nothing
end

"""
    evaluate_with_penalty(evaluator::FeasibilityHandlingEvaluator, candidate::AbstractArray)
"""
function evaluate_with_penalty(
    evaluator::FeasibilityHandlingEvaluator, candidate::AbstractArray
)
    fun = get_scalar_function_with_penalty(evaluator.prob)
    return fun(candidate)
end
