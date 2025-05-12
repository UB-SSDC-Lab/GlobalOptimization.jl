
"""
    DEBasePopulation{T <: AbstractFloat} <: AbstractPopulation{T}

The base representation of some population for the DE algorithms.
"""
struct DEBasePopulation{T<:AbstractFloat} <: AbstractPopulation{T}
    candidates::Vector{Vector{T}}
    candidates_fitness::Vector{T}

    function DEBasePopulation{T}(num_candidates::Integer, num_dims::Integer) where {T}
        return new{T}(
            [zeros(T, num_dims) for _ in 1:num_candidates], zeros(T, num_candidates)
        )
    end
end

"""
    DEPopulation{T <: AbstractFloat} <: AbstractPopulation{T}

The full population representation for the DE algorithms, including both the candidates
and the mutants.
"""
struct DEPopulation{T<:AbstractFloat}
    # The current population of candidates
    current_generation::DEBasePopulation{T}

    # The population of mutants
    mutants::DEBasePopulation{T}

    # Booleans indicating improvement
    improved::Vector{Bool}

    function DEPopulation{T}(num_candidates::Integer, num_dims::Integer) where {T}
        num_dims > 0 || throw(ArgumentError("num_dims must be greater than 0."))
        num_candidates > 0 || throw(ArgumentError("num_candidates must be greater than 0."))
        return new{T}(
            DEBasePopulation{T}(num_candidates, num_dims),
            DEBasePopulation{T}(num_candidates, num_dims),
            fill(false, num_candidates),
        )
    end
end

"""
    DEPopulation(num_candidates::Integer, num_dims::Integer)

Constructs a `DEPopulation` with `num_candidates` candidates in `num_dims` dimensions.
"""
function DEPopulation(num_candidates::Integer, num_dims::Integer)
    DEPopulation{Float64}(num_candidates, num_dims)
end

"""
    DEPopulation_F64(num_candidates::Integer, num_dims::Integer)

Constructs a Float64 `DEPopulation` with `num_candidate` candidates in `num_dims` dimensions.
"""
function DEPopulation_F64(num_candidates::Integer, num_dims::Integer)
    DEPopulation{Float64}(num_candidates, num_dims)
end

Base.length(population::DEPopulation) = length(population.current_generation)

function initialize_uniform!(
    population::DEPopulation{T}, search_space::ContinuousRectangularSearchSpace{T}
) where {T}
    # Unpack population
    candidates = population.current_generation.candidates

    # Initialize each candidate
    @inbounds for i in eachindex(candidates)
        candidate = candidates[i]

        # Iterate over dimensions
        for j in eachindex(candidate)
            dmin = dim_min(search_space, j)
            dΔ = dim_delta(search_space, j)

            # Set candidate
            candidate[j] = dmin + dΔ * rand(T)
        end
    end
    return nothing
end

function initialize_fitness!(
    population::DEPopulation{T}, evaluator::BatchEvaluator{T}
) where {T}
    # Evaluate the cost function for each candidate
    return evaluate!(population.current_generation, evaluator)
end

function evaluate_mutant_fitness!(
    population::DEPopulation{T}, evaluator::BatchEvaluator{T}
) where {T}
    # Evaluate the cost function for each mutant
    return evaluate!(population.mutants, evaluator)
end

"""
    selection!(population::DEPopulation{T}, evaluator::BatchEvaluator{T})

Replace candidates with mutants if they have a better fitness.
"""
function selection!(population::DEPopulation)
    # Update the current generation with the mutants
    @unpack current_generation, mutants = population
    @inbounds for i in eachindex(current_generation)
        if mutants.candidates_fitness[i] < current_generation.candidates_fitness[i]
            current_generation.candidates[i] .= mutants.candidates[i]
            current_generation.candidates_fitness[i] = mutants.candidates_fitness[i]
            population.improved[i] = true
        else
            population.improved[i] = false
        end
    end
    return nothing
end
