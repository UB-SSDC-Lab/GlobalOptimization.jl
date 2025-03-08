abstract type AbstractMutationStrategy end
struct Rand1 <: AbstractMutationStrategy end
struct Rand2 <: AbstractMutationStrategy end
struct Best1 <: AbstractMutationStrategy end
struct Best2 <: AbstractMutationStrategy end
struct CurrentToBest1 <: AbstractMutationStrategy end
struct CurrentToBest2 <: AbstractMutationStrategy end
struct CurrentToRand1 <: AbstractMutationStrategy end
struct CurrentToRand2 <: AbstractMutationStrategy end
struct RandToBest1 <: AbstractMutationStrategy end
struct RandToBest2 <: AbstractMutationStrategy end
struct Unified <: AbstractMutationStrategy end

function get_parameters(strategy::Rand1, dist)
    F2 = 1.0
    F3 = semiclamped_rand(dist)
    return SA[0.0, F2, F3, 0.0]
end
function get_parameters(strategy::Rand2, dist)
    F2 = 1.0
    F3 = semiclamped_rand(dist)
    F4 = F3
    return SA[0.0, F2, F3, F4]
end
function get_parameters(strategy::Best1, dist)
    F1 = 1.0
    F3 = semiclamped_rand(dist)
    return SA[F1, 0.0, F3, 0.0]
end
function get_parameters(strategy::Best2, dist)
    F1 = 1.0
    F3 = semiclamped_rand(dist)
    F4 = F3
    return SA[F1, 0.0, F3, F4]
end
function get_parameters(strategy::CurrentToBest1, dist)
    F1 = semiclamped_rand(dist)
    F3 = semiclamped_rand(dist)
    return SA[F1, 0.0, F3, 0.0]
end
function get_parameters(strategy::CurrentToBest2, dist)
    F1 = semiclamped_rand(dist)
    F3 = semiclamped_rand(dist)
    F4 = F3
    return SA[F1, 0.0, F3, F4]
end
function get_parameters(strategy::CurrentToRand1, dist)
    F2 = semiclamped_rand(dist)
    F3 = semiclamped_rand(dist)
    return SA[0.0, F2, F3, 0.0]
end
function get_parameters(strategy::CurrentToRand2, dist)
    F2 = semiclamped_rand(dist)
    F3 = semiclamped_rand(dist)
    F4 = F3
    return SA[0.0, F2, F3, F4]
end
function get_parameters(strategy::RandToBest1, dist)
    F1 = semiclamped_rand(dist)
    F2 = 1.0
    F3 = semiclamped_rand(dist)
    return SA[F1, F2, F3, 0.0]
end
function get_parameters(strategy::RandToBest2, dist)
    F1 = semiclamped_rand(dist)
    F2 = 1.0
    F3 = semiclamped_rand(dist)
    F4 = F3
    return SA[F1, F2, F3, F4]
end
function get_parameters(strategy::Unified, dist)
    F1 = semiclamped_rand(dist)
    F2 = semiclamped_rand(dist)
    F3 = semiclamped_rand(dist)
    F4 = semiclamped_rand(dist)
    return SA[F1, F2, F3, F4]
end

abstract type AbstractMutationParameters{AS <: AbstractAdaptationStrategy} end
mutable struct MutationParameters{
    AS <: AbstractAdaptationStrategy,
    MS <: AbstractMutationStrategy,
    D,
} <: AbstractMutationParameters{AS}

    F1::Float64
    F2::Float64
    F3::Float64
    F4::Float64
    dist::D

    # For constant parameters
    function MutationParameters(F1, F2, F3, F4)
        new{NoAdaptation, Unified, Nothing}(F1, F2, F3, F4, nothing)
    end

    # For adaptive parameters with distribution `dist`
    function MutationParameters(
        strategy::MS; dist = default_mutation_dist
    ) where {MS <: AbstractMutationStrategy}
        Fs = get_parameters(strategy, dist)
        new{RandomAdaptation, MS, typeof(dist)}(Fs[1], Fs[2], Fs[3], Fs[4], dist)
    end
end

struct SelfMutationParameters{
    AS <: AbstractAdaptationStrategy,
    MS <: AbstractMutationStrategy,
    D,
} <: AbstractMutationParameters{AS}

    Fs::Vector{SVector{4, Float64}}
    dist::D

    function SelfMutationParameters(
        strategy::MS; dist = default_mutation_dist
    ) where {MS <: AbstractMutationStrategy}
        Fs = Vector{SVector{4, Float64}}(undef, 0)
        new{RandomAdaptation, MS, typeof(dist)}(Fs, dist)
    end
end

get_parameters(params::MutationParameters, i) = (params.F1, params.F2, params.F3, params.F4)
function get_parameters(params::SelfMutationParameters, i)
    Fs = params.Fs[i]
    return (Fs[1], Fs[2], Fs[3], Fs[4])
end

initialize!(params::AbstractMutationParameters{NoAdaptation}, population_size) = nothing
function initialize!(params::MutationParameters{RandomAdaptation, MS}, population_size) where MS
    Fs = get_parameters(MS(), params.dist)
    params.F1 = Fs[1]
    params.F2 = Fs[2]
    params.F3 = Fs[3]
    params.F4 = Fs[4]
    return nothing
end
function initialize!(
    params::SelfMutationParameters{RandomAdaptation, MS},
    population_size,
) where MS
    resize!(params.Fs, population_size)
    @inbounds for i in eachindex(params.Fs)
        params.Fs[i] = get_parameters(MS(), params.dist)
    end
end

adapt!(params::AbstractMutationParameters{NoAdaptation}, improved, global_best_improved) = nothing
function adapt!(params::MutationParameters{RandomAdaptation, MS}, improved, global_best_improved) where MS
    if !global_best_improved
        Fs = get_parameters(MS(), params.dist)
        params.F1 = Fs[1]
        params.F2 = Fs[2]
        params.F3 = Fs[3]
        params.F4 = Fs[4]
    end
    return nothing
end
function adapt!(params::SelfMutationParameters{RandomAdaptation, MS}, improved, global_best_improved) where MS
    @inbounds for i in eachindex(params.Fs)
        if !improved[i]
            params.Fs[i] = get_parameters(MS(), params.dist)
        end
    end
    return nothing
end

"""
    mutate!(population::DEPopulation{T}, F, best_candidate)

Mutates the population `population` using the DE mutation strategy.

This is an implementation of the unified mutation strategy proposed by Ji Qiang and
Chad Mitchell in "A Unified Differential Evolution Algorithm for Global Optimization".
"""
function mutate!(population::DEPopulation, F::MP, best_candidate) where {MP <: AbstractMutationParameters}
    @unpack current_generation, mutants = population
    N = length(current_generation)

    # Iterate over each candidate and mutate
    @inbounds for i in eachindex(current_generation)
        # Get parameters
        F1, F2, F3, F4 = get_parameters(F, i)

        # Initialize random integers
        r1=0; r2=0; r3=0; r4=0; r5=0

        # Initialize mutant with current candidate
        mutants.candidates[i] .= current_generation.candidates[i]

        # Update mutant based on values of F1, F2, F3, F4
        if F1 > 0.0
            @. mutants.candidates[i] += F1*(best_candidate - current_generation.candidates[i])
        end
        if F2 > 0.0
            # Generate unique r1 (just need to ensure r1 != i)
            r1 = rand(1:N)
            while r1 == i
                r1 = rand(1:N)
            end

            @. mutants.candidates[i] += F2*(current_generation.candidates[r1] - current_generation.candidates[i])
        end
        if F3 > 0.0
            # Generate unique r2 (need to ensure r2 != i and r2 != r1)
            r2 = rand(1:N)
            while r2 in (i, r1)
                r2 = rand(1:N)
            end

            # Generate unique r3 (need to ensure r3 != i, r1, and r2)
            r3 = rand(1:N)
            while r3 in (i, r1, r2)
                r3 = rand(1:N)
            end

            @. mutants.candidates[i] += F3*(current_generation.candidates[r2] - current_generation.candidates[r3])
        end
        if F4 > 0.0
            # Generate unique r4 (need to ensure r4 != i, r1, r2, and r3)
            r4 = rand(1:N)
            while r4 in (i, r1, r2, r3)
                r4 = rand(1:N)
            end

            # Generate unique r5 (need to ensure r5 != i, r1, r2, r3, and r4)
            r5 = rand(1:N)
            while r5 in (i, r1, r2, r3, r4)
                r5 = rand(1:N)
            end

            @. mutants.candidates[i] += F4*(current_generation.candidates[r4] - current_generation.candidates[r5])
        end
    end
    return nothing
end
