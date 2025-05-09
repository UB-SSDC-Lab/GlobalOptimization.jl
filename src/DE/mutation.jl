abstract type AbstractMutationStrategy end

"""
    Rand1

The DE/rand/1 mutation strategy given by:

``\\mathbf{v}_i = \\mathbf{x}_{r_1} + F\\left(\\mathbf{x}_{r_2} - \\mathbf{x}_{r_3}\\right)``

where ``\\mathbf{v}_i`` is the target (``i``-th) mutant, ``\\mathbf{x}_j`` denotes the
``j``-th candidate, ``F`` is a scaling factor, and ``r_1``, ``r_2``, and ``r_3`` are
randomly selected integers in the set returned by the selector.
"""
struct Rand1 <: AbstractMutationStrategy end

"""
    Rand2

The DE/rand/2 mutation strategy given by:

``
\\mathbf{v}_i = \\mathbf{x}_{r_1} + F\\left(\\mathbf{x}_{r_2} - \\mathbf{x}_{r_3}\\right) +
F\\left(\\mathbf{x}_{r_4} - \\mathbf{x}_{r_5}\\right)``

where ``\\mathbf{v}_i`` is the target (``i``-th) mutant, ``\\mathbf{x}_j`` denotes the
``j``-th candidate, ``F`` is a scaling factor, and ``r_1``, ``r_2``, ``r_3``, ``r_4``,
and ``r_5`` are randomly selected integers in the set returned by the selector.
"""
struct Rand2 <: AbstractMutationStrategy end

"""
    Best1

The DE/best/1 mutation strategy given by:

``\\mathbf{v}_i = \\mathbf{x}_{b} + F\\left(\\mathbf{x}_{r_1} - \\mathbf{x}_{r_2}\\right)``

where ``\\mathbf{v}_i`` is the target (``i``-th) mutant, ``\\mathbf{x}_j`` denotes the
``j``-th candidate, ``F`` is a scaling factor, subscript ``b`` denotes the best candidate
(in terms of the objective/fitness function), and ``r_1`` and ``r_2`` are randomly selected
integers in the set returned by the selector.
"""
struct Best1 <: AbstractMutationStrategy end

"""
    Best2

The DE/best/2 mutation strategy given by:

``\\mathbf{v}_i = \\mathbf{x}_{b} + F\\left(\\mathbf{x}_{r_1} - \\mathbf{x}_{r_2}\\right) +
F\\left(\\mathbf{x}_{r_3} - \\mathbf{x}_{r_4}\\right)``

where ``\\mathbf{v}_i`` is the target (``i``-th) mutant, ``\\mathbf{x}_j`` denotes the
``j``-th candidate, ``F`` is a scaling factor, subscript ``b`` denotes the best candidate,
and ``r_1``, ``r_2``, ``r_3``, and ``r_4`` are randomly selected integers in the set
returned by the selector.
"""
struct Best2 <: AbstractMutationStrategy end

"""
    CurrentToBest1

The DE/current-to-best/1 mutation strategy given by:

``\\mathbf{v}_i = \\mathbf{x}_{i} + F_{cr}\\left(\\mathbf{x}_{b} - \\mathbf{x}_{i}\\right) +
F\\left(\\mathbf{x}_{r_1} - \\mathbf{x}_{r_2}\\right)``

where ``\\mathbf{v}_i`` is the target (``i``-th) mutant, ``\\mathbf{x}_j`` denotes the
``j``-th candidate, ``F_{cs}`` and ``F`` are a scaling factors, subscript ``b`` denotes the
best candidate, and ``r_1`` and ``r_2`` are randomly selected integers in the set
returned by the selector.
"""
struct CurrentToBest1 <: AbstractMutationStrategy end

"""
    CurrentToBest2

The DE/current-to-best/2 mutation strategy given by:

``\\mathbf{v}_i = \\mathbf{x}_{i} + F_{cr}\\left(\\mathbf{x}_{b} - \\mathbf{x}_{i}\\right) +
F\\left(\\mathbf{x}_{r_1} - \\mathbf{x}_{r_2}\\right) +
F\\left(\\mathbf{x}_{r_3} - \\mathbf{x}_{r_4}\\right)``

where ``\\mathbf{v}_i`` is the target (``i``-th) mutant, ``\\mathbf{x}_j`` denotes the
``j``-th candidate, ``F_{cs}`` and ``F`` are a scaling factors, subscript ``b`` denotes the
best candidate, and ``r_1``, ``r_2``, ``r_3``, and ``r_4`` are randomly selected integers
in the set returned by the selector.
"""
struct CurrentToBest2 <: AbstractMutationStrategy end

"""
    CurrentToRand1

The DE/current-to-rand/1 mutation strategy given by:

``\\mathbf{v}_i = \\mathbf{x}_{i} + F_{cr}\\left(\\mathbf{x}_{r_1} - \\mathbf{x}_{i}\\right) +
F\\left(\\mathbf{x}_{r_2} - \\mathbf{x}_{r_3}\\right)``

where ``\\mathbf{v}_i`` is the target (``i``-th) mutant, ``\\mathbf{x}_j`` denotes the
``j``-th candidate, ``F_{cs}`` and ``F`` are a scaling factors, and ``r_1``, ``r_2``, and
``r_3`` are randomly selected integers in the set returned by the selector.
"""
struct CurrentToRand1 <: AbstractMutationStrategy end

"""
    CurrentToRand2

The DE/current-to-rand/2 mutation strategy given by:

``\\mathbf{v}_i = \\mathbf{x}_{i} + F_{cr}\\left(\\mathbf{x}_{r_1} - \\mathbf{x}_{i}\\right) +
F\\left(\\mathbf{x}_{r_2} - \\mathbf{x}_{r_3}\\right) +
F\\left(\\mathbf{x}_{r_4} - \\mathbf{x}_{r_5}\\right)``

where ``\\mathbf{v}_i`` is the target (``i``-th) mutant, ``\\mathbf{x}_j`` denotes the
``j``-th candidate, ``F_{cs}`` and ``F`` are a scaling factors, and ``r_1``, ``r_2``,
``r_3``, ``r_4``, and ``r_5`` are randomly selected integers in the set returned by the
selector.
"""
struct CurrentToRand2 <: AbstractMutationStrategy end

"""
    RandToBest1

The DE/rand-to-best/1 mutation strategy given by:

``\\mathbf{v}_i = \\mathbf{x}_{r_1} + F_{cr}\\left(\\mathbf{x}_{b} - \\mathbf{x}_i\\right) +
F\\left(\\mathbf{x}_{r_2} - \\mathbf{x}_{r_3}\\right)``

where ``\\mathbf{v}_i`` is the target (``i``-th) mutant, ``\\mathbf{x}_j`` denotes the
``j``-th candidate, ``F_{cs}`` and ``F`` are a scaling factors, subscript ``b`` denotes the
best candidate, and ``r_1``, ``r_2``, and ``r_3`` are randomly selected integers in the set
returned by the selector.
"""
struct RandToBest1 <: AbstractMutationStrategy end

"""
    RandToBest2

The DE/rand-to-best/2 mutation strategy given by:

``\\mathbf{v}_i = \\mathbf{x}_{r_1} + F_{cr}\\left(\\mathbf{x}_{b} - \\mathbf{x}_i\\right) +
F\\left(\\mathbf{x}_{r_2} - \\mathbf{x}_{r_3}\\right) +
F\\left(\\mathbf{x}_{r_4} - \\mathbf{x}_{r_5}\\right)``

where ``\\mathbf{v}_i`` is the target (``i``-th) mutant, ``\\mathbf{x}_j`` denotes the
``j``-th candidate, ``F_{cs}`` and ``F`` are a scaling factors, subscript ``b`` denotes the
best candidate, and ``r_1``, ``r_2``, ``r_3``, ``r_4``, and ``r_5`` are randomly selected
integers in the set returned by the selector.
"""
struct RandToBest2 <: AbstractMutationStrategy end

"""
    Unified

The unified DE mutation strategy proposed by Ji Qiang and Chad Mitchell in "A Unified
Differential Evolution Algorithm for Global Optimization," 2014,
[https://www.osti.gov/servlets/purl/1163659](https://www.osti.gov/servlets/purl/1163659).

This mutation strategy is given by:

``\\mathbf{v}_i = \\mathbf{x}_i + F_1\\left(\\mathbf{x}_b - \\mathbf{x}_i\\right) +
F_2\\left(\\mathbf{x}_{r_1} - \\mathbf{x}_i\\right) +
F_3\\left(\\mathbf{x}_{r_2} - \\mathbf{x}_{r_3}\\right) +
F_4\\left(\\mathbf{x}_{r_4} - \\mathbf{x}_{r_5}\\right)``

where ``\\mathbf{v}_i`` is the target (``i``-th) mutant, ``\\mathbf{x}_j`` denotes the
``j``-th candidate, ``F_1``, ``F_2``, ``F_3``, and ``F_4`` are scaling factors, subscript
``b`` denotes the best candidate, and ``r_1``, ``r_2``, ``r_3``, ``r_4``, and ``r_5`` are
randomly selected integers in the set returned by the selector.

Note that in the underlying implementation, all mutation strategies are implemented with
this formulation, where each unique strategy has a different set of
``\\{F_i : i \\in \\{1,2,3,4\\}\\}`` that are set to 0.0.
"""
struct Unified <: AbstractMutationStrategy end

function get_parameters(strategy::Rand1, dist)
    F2 = 1.0
    F3 = one_clamped_rand(dist)
    return SA[0.0, F2, F3, 0.0]
end
function get_parameters(strategy::Rand2, dist)
    F2 = 1.0
    F3 = one_clamped_rand(dist)
    F4 = F3
    return SA[0.0, F2, F3, F4]
end
function get_parameters(strategy::Best1, dist)
    F1 = 1.0
    F3 = one_clamped_rand(dist)
    return SA[F1, 0.0, F3, 0.0]
end
function get_parameters(strategy::Best2, dist)
    F1 = 1.0
    F3 = one_clamped_rand(dist)
    F4 = F3
    return SA[F1, 0.0, F3, F4]
end
function get_parameters(strategy::CurrentToBest1, dist)
    F1 = one_clamped_rand(dist)
    F3 = one_clamped_rand(dist)
    return SA[F1, 0.0, F3, 0.0]
end
function get_parameters(strategy::CurrentToBest2, dist)
    F1 = one_clamped_rand(dist)
    F3 = one_clamped_rand(dist)
    F4 = F3
    return SA[F1, 0.0, F3, F4]
end
function get_parameters(strategy::CurrentToRand1, dist)
    F2 = one_clamped_rand(dist)
    F3 = one_clamped_rand(dist)
    return SA[0.0, F2, F3, 0.0]
end
function get_parameters(strategy::CurrentToRand2, dist)
    F2 = one_clamped_rand(dist)
    F3 = one_clamped_rand(dist)
    F4 = F3
    return SA[0.0, F2, F3, F4]
end
function get_parameters(strategy::RandToBest1, dist)
    F1 = one_clamped_rand(dist)
    F2 = 1.0
    F3 = one_clamped_rand(dist)
    return SA[F1, F2, F3, 0.0]
end
function get_parameters(strategy::RandToBest2, dist)
    F1 = one_clamped_rand(dist)
    F2 = 1.0
    F3 = one_clamped_rand(dist)
    F4 = F3
    return SA[F1, F2, F3, F4]
end
function get_parameters(strategy::Unified, dist)
    F1 = one_clamped_rand(dist)
    F2 = one_clamped_rand(dist)
    F3 = one_clamped_rand(dist)
    F4 = one_clamped_rand(dist)
    return SA[F1, F2, F3, F4]
end

abstract type AbstractMutationParameters{AS<:AbstractAdaptationStrategy} end

"""
    MutationParameters{
        AS<:AbstractAdaptationStrategy,
        MS<:AbstractMutationStrategy,
        S<:AbstractSelector,
        D,
    }

The parameters for a DE mutation strategy that applies to all current and future candidates
in the population.

# Fields
- `F1::Float64`: The F₁ weight in the unified mutation strategy.
- `F2::Float64`: The F₂ weight in the unified mutation strategy.
- `F3::Float64`: The F₃ weight in the unified mutation strategy.
- `F4::Float64`: The F₄ weight in the unified mutation strategy.
- `sel<:AbstractSelector`: The selector used to select the candidates considered in
    mutation.
- `dist<:Distribution{Univariate,Continuous}`: The distribution used to adapt the mutation
    parameters. Note that this should generally be a distribution from
    [Distributions.jl](https://juliastats.org/Distributions.jl/latest/), but the only strict
    requirement is that rand(dist) returns a floating point value.
"""
mutable struct MutationParameters{
    AS<:AbstractAdaptationStrategy,MS<:AbstractMutationStrategy,S<:AbstractSelector,D
} <: AbstractMutationParameters{AS}
    F1::Float64
    F2::Float64
    F3::Float64
    F4::Float64
    sel::S
    dist::D

    @doc """
        MutationParameters(F1, F2, F3, F4; sel=SimpleSelector())

    Creates a `MutationParameters` object with the specified (constant) mutation parameters.
    These constant mutation parameters are used for all candidates in the population and
    define a unified mutation strategy as defined in Ji Qiang and Chad Mitchell "A Unified
    Differential Evolution Algorithm for Global Optimization," 2014,
    [https://www.osti.gov/servlets/purl/1163659](https://www.osti.gov/servlets/purl/1163659)

    # Arguments
    - `F1::Float64`: The F₁ weight in the unified mutation strategy.
    - `F2::Float64`: The F₂ weight in the unified mutation strategy.
    - `F3::Float64`: The F₃ weight in the unified mutation strategy.
    - `F4::Float64`: The F₄ weight in the unified mutation strategy.

    # Keyword Arguments
    - `sel::AbstractSelector`: The selector used to select the candidates considered in
        mutation. Defaults to `SimpleSelector()`.

    # Returns
    - `MutationParameters{NoAdaptation,Unified,typeof(sel),Nothing}`: A mutation parameters
        object with the specified mutation parameters and selector.

    # Examples
    ```julia-repl
    julia> using GlobalOptimization
    julia> params = MutationParameters(0.5, 0.5, 0.5, 0.5)
    MutationParameters{GlobalOptimization.NoAdaptation, Unified, SimpleSelector, Nothing}(0.5, 0.5, 0.5, 0.5, SimpleSelector(), nothing)
    ```
    ```julia-repl
    julia> using GlobalOptimization
    julia> params = MutationParameters(0.5, 0.5, 0.5, 0.5; sel=RadiusLimitedSelector(2))
    MutationParameters{GlobalOptimization.NoAdaptation, Unified, RadiusLimitedSelector, Nothing}(0.5, 0.5, 0.5, 0.5, RadiusLimitedSelector(2, UInt16[0x6cf0, 0x0c33, 0x0001, 0x0000, 0x0560]), nothing)
    ```
    """
    function MutationParameters(F1, F2, F3, F4; sel=SimpleSelector())
        S = typeof(sel)
        return new{NoAdaptation,Unified,S,Nothing}(F1, F2, F3, F4, sel, nothing)
    end

    @doc """
        MutationParameters(
            strategy::MS;
            dist=default_mutation_dist,
            sel=SimpleSelector(),
        )

    Creates a MutationParameters object with the specified mutation strategy with mutation
    parameter random adaptation. The mutation parameters are adaptively sampled from the
    provided `dist`, clamped to the range (0, 1].

    # Arguments
    - `strategy::MS`: The mutation strategy to use. This should be one of the mutation
        strategies defined in this module (e.g., `Rand1`, `Best2`, etc.).

    # Keyword Arguments
    - `dist::Distribution{Univariate,Continuous}`: The distribution used to adapt the
        mutation parameters each iteration. Note that this should generally be a
        distribution from
        [Distributions.jl](https://juliastats.org/Distributions.jl/latest/), but the only
        strict requirement is that rand(dist) returns a floating point value. Defaults to
        `GlobalOptimization.default_mutation_dist`, which is a mixture model comprised of
        two Cauchy distributions, with probability density given by:

        ``f_{mix}(x; \\mu, \\sigma) = 0.5 f(x;\\mu_1,\\sigma_1) + 0.5 f(x;\\mu_2,\\sigma_2)``.

        where ``\\mu = \\{0.65, 1.0\\}`` and ``\\sigma = \\{0.1, 0.1\\}``.

    - `sel::AbstractSelector`: The selector used to select the candidates considered in
        mutation. Defaults to `SimpleSelector()`.

    # Returns
    - `MutationParameters{RandomAdaptation,typeof(strategy),typeof(sel),typeof(dist)}`: A
        mutation parameters object with the specified mutation strategy and selector.

    # Examples
    ```julia-repl
    julia> using GlobalOptimization
    julia> params = MutationParameters(Rand1())
    MutationParameters{GlobalOptimization.RandomAdaptation, Rand1, SimpleSelector, Distributions.MixtureModel{Distributions.Univariate, Distributions.Continuous, Distributions.Cauchy, Distributions.Categorical{Float64, Vector{Float64}}}}(0.0, 1.0, 0.8450801042502032, 0.0, SimpleSelector(), MixtureModel{Distributions.Cauchy}(K = 2)
    components[1] (prior = 0.5000): Distributions.Cauchy{Float64}(μ=0.65, σ=0.1)
    components[2] (prior = 0.5000): Distributions.Cauchy{Float64}(μ=1.0, σ=0.1)
    )
    ```
    ```julia-repl
    julia> using GlobalOptimization
    julia> using Distributions
    julia> params = MutationParameters(Rand1(); dist=Normal(0.5, 0.1))
    MutationParameters{GlobalOptimization.RandomAdaptation, Rand1, SimpleSelector, Normal{Float64}}(0.0, 1.0, 0.5061103661726901, 0.0, SimpleSelector(), Normal{Float64}(μ=0.5, σ=0.1))
    ```
    """
    function MutationParameters(
        strategy::MS; dist=default_mutation_dist, sel=SimpleSelector()
    ) where {MS<:AbstractMutationStrategy}
        Fs = get_parameters(strategy, dist)
        return new{RandomAdaptation,MS,typeof(sel),typeof(dist)}(
            Fs[1], Fs[2], Fs[3], Fs[4], sel, dist
        )
    end
end

"""
    SelfMutationParameters{
        AS<:AbstractAdaptationStrategy,
        MS<:AbstractMutationStrategy,
        S<:AbstractSelector,
        D,
    }

The parameters for a DE mutation strategy that applies a mutation strategy with unique
parameters for each candidate in the population.

# Fields
- `Fs::Vector{SVector{4,Float64}}`: The mutation parameters for each candidate in the
    population. Each element of the vector is an SVector{4} containing the F₁, F₂, F₃, and
    F₄ weights for the unified mutation strategy.
- `sel<:AbstractSelector`: The selector used to select the candidates considered in mutation.
- `dist<:Distribution{Univariate,Continuous}`: The distribution used to adapt the mutation
    parameters. Note that this should generally be a distribution from
    [Distributions.jl](https://juliastats.org/Distributions.jl/latest/), but the only strict
    requirement is that rand(dist) returns a floating point value.
"""
struct SelfMutationParameters{
    AS<:AbstractAdaptationStrategy,MS<:AbstractMutationStrategy,S<:AbstractSelector,D
} <: AbstractMutationParameters{AS}
    Fs::Vector{SVector{4,Float64}}
    sel::S
    dist::D

    @doc """
        SelfMutationParameters(
            strategy::MS;
            dist=default_mutation_dist,
            sel=SimpleSelector(),
        )

    Creates a SelfMutationParameters object with the specified mutation strategy and
    mutation parameter random adaptation. The mutation parameters are adaptively sampled
    from the provided `dist`, clamped to the range (0, 1].

    # Arguments
    - `strategy::MS`: The mutation strategy to use. This should be one of the mutation
        strategies defined in this module (e.g., `Rand1`, `Best2`, etc.).

    # Keyword Arguments
    - `dist::Distribution{Univariate,Continuous}`: The distribution used to adapt the
        mutation parameters each iteration. Note that this should generally be a
        distribution from
        [Distributions.jl](https://juliastats.org/Distributions.jl/latest/), but the only
        strict requirement is that rand(dist) returns a floating point value. Defaults to
        `GlobalOptimization.default_mutation_dist`, which is a mixture model comprised of
        two Cauchy distributions, with probability density given by:

        ``f_{mix}(x; \\mu, \\sigma) = 0.5 f(x;\\mu_1,\\sigma_1) + 0.5 f(x;\\mu_2,\\sigma_2)``.

        where ``\\mu = \\{0.65, 1.0\\}`` and ``\\sigma = \\{0.1, 0.1\\}``.

    - `sel::AbstractSelector`: The selector used to select the candidates considered in
        mutation. Defaults to `SimpleSelector()`.

    # Returns
    - `SelfMutationParameters{RandomAdaptation,typeof(strategy),typeof(sel),typeof(dist)}`:
        A mutation parameters object with the specified mutation strategy and selector.

    # Examples
    ```julia-repl
    julia> using GlobalOptimization
    julia> params = SelfMutationParameters(Rand1())
    SelfMutationParameters{GlobalOptimization.RandomAdaptation, Rand1, SimpleSelector, MixtureModel{Univariate, Continuous, Cauchy, Categorical{Float64, Vector{Float64}}}}(StaticArraysCore.SVector{4, Float64}[], SimpleSelector(), MixtureModel{Cauchy}(K = 2)
    components[1] (prior = 0.5000): Cauchy{Float64}(μ=0.65, σ=0.1)
    components[2] (prior = 0.5000): Cauchy{Float64}(μ=1.0, σ=0.1)
    )
    ```
    """
    function SelfMutationParameters(
        strategy::MS; dist=default_mutation_dist, sel=SimpleSelector()
    ) where {MS<:AbstractMutationStrategy}
        Fs = Vector{SVector{4,Float64}}(undef, 0)
        return new{RandomAdaptation,MS,typeof(sel),typeof(dist)}(Fs, sel, dist)
    end
end

get_parameters(params::MutationParameters, i) = (params.F1, params.F2, params.F3, params.F4)
function get_parameters(params::SelfMutationParameters, i)
    Fs = params.Fs[i]
    return (Fs[1], Fs[2], Fs[3], Fs[4])
end

function initialize!(params::AbstractMutationParameters{NoAdaptation}, population_size)
    initialize!(params.sel, population_size)
    return nothing
end
function initialize!(
    params::MutationParameters{RandomAdaptation,MS}, population_size
) where {MS}
    initialize!(params.sel, population_size)
    Fs = get_parameters(MS(), params.dist)
    params.F1 = Fs[1]
    params.F2 = Fs[2]
    params.F3 = Fs[3]
    params.F4 = Fs[4]
    return nothing
end
function initialize!(
    params::SelfMutationParameters{RandomAdaptation,MS}, population_size
) where {MS}
    initialize!(params.sel, population_size)
    resize!(params.Fs, population_size)
    @inbounds for i in eachindex(params.Fs)
        params.Fs[i] = get_parameters(MS(), params.dist)
    end
end

function adapt!(
    params::AbstractMutationParameters{NoAdaptation}, improved, global_best_improved
)
    return nothing
end
function adapt!(
    params::MutationParameters{RandomAdaptation,MS}, improved, global_best_improved
) where {MS}
    if !global_best_improved
        Fs = get_parameters(MS(), params.dist)
        params.F1 = Fs[1]
        params.F2 = Fs[2]
        params.F3 = Fs[3]
        params.F4 = Fs[4]
    end
    return nothing
end
function adapt!(
    params::SelfMutationParameters{RandomAdaptation,MS}, improved, global_best_improved
) where {MS}
    @inbounds for i in eachindex(params.Fs)
        if !improved[i]
            params.Fs[i] = get_parameters(MS(), params.dist)
        end
    end
    return nothing
end

"""
    get_best_candidate_in_selection(population::DEPopulation, idxs)

Get the best candidate in the selected subset of population (as specified by the indices
in `idxs`).
"""
function get_best_candidate_in_selection(population::DEPopulation, idxs)
    # Get the index of the best candidate in the selected subset of population
    best_idx = argmin(
        let gen_fitness = population.current_generation.candidates_fitness
            i -> gen_fitness[i]
        end,
        idxs,
    )

    return population.current_generation.candidates[best_idx]
end

"""
    mutate!(population::DEPopulation{T}, F)

Mutates the population `population` using the DE mutation strategy.

This is an implementation of the unified mutation strategy proposed by Ji Qiang and
Chad Mitchell in "A Unified Differential Evolution Algorithm for Global Optimization".
"""
function mutate!(population::DEPopulation, F::MP) where {MP<:AbstractMutationParameters}
    @unpack current_generation, mutants = population
    N = length(current_generation)

    # Iterate over each candidate and mutate
    @inbounds for i in eachindex(current_generation)
        # Get parameters
        F1, F2, F3, F4 = get_parameters(F, i)

        # Select idxs for mutation
        idxs = select(F.sel, i, N)

        # Initialize random integers
        r1 = 0
        r2 = 0
        r3 = 0
        r4 = 0
        r5 = 0

        # Initialize mutant with current candidate
        mutants.candidates[i] .= current_generation.candidates[i]

        # Update mutant based on values of F1, F2, F3, F4
        if F1 > 0.0
            # Get the best candidate in the selected subset of population
            best_candidate = get_best_candidate_in_selection(population, idxs)

            @. mutants.candidates[i] +=
                F1 * (best_candidate - current_generation.candidates[i])
        end
        if F2 > 0.0
            # Generate unique r1 (just need to ensure r1 != i)
            r1 = rand(idxs)
            while r1 == i
                r1 = rand(idxs)
            end

            @. mutants.candidates[i] +=
                F2 * (current_generation.candidates[r1] - current_generation.candidates[i])
        end
        if F3 > 0.0
            # Generate unique r2 (need to ensure r2 != i and r2 != r1)
            r2 = rand(idxs)
            while r2 in (i, r1)
                r2 = rand(idxs)
            end

            # Generate unique r3 (need to ensure r3 != i, r1, and r2)
            r3 = rand(idxs)
            while r3 in (i, r1, r2)
                r3 = rand(idxs)
            end

            @. mutants.candidates[i] +=
                F3 * (current_generation.candidates[r2] - current_generation.candidates[r3])
        end
        if F4 > 0.0
            # Generate unique r4 (need to ensure r4 != i, r1, r2, and r3)
            r4 = rand(idxs)
            while r4 in (i, r1, r2, r3)
                r4 = rand(idxs)
            end

            # Generate unique r5 (need to ensure r5 != i, r1, r2, r3, and r4)
            r5 = rand(idxs)
            while r5 in (i, r1, r2, r3, r4)
                r5 = rand(idxs)
            end

            @. mutants.candidates[i] +=
                F4 * (current_generation.candidates[r4] - current_generation.candidates[r5])
        end
    end
    return nothing
end
