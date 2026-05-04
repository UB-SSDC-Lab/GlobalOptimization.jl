"""
    GABasePopulation{T<:AbstractFloat} <: AbstractPopulation

Base representation of population for Genetic Algorithms. This allcoates storage for a single set of candidates. 
Multiple of these are used within the GA to represent the population, mating pool, and children.
"""
struct GABasePopulation{T<:AbstractFloat} <: AbstractPopulation{T}

    candidates::Vector{Vector{T}}
    candidates_fitness::Vector{T}

    function GABasePopulation{T}(num_candidates::Integer, num_dims::Integer) where {T}
        return new{T}(
            [zeros(T, num_dims) for _ in 1:num_candidates], zeros(T, num_candidates)
        )
    end

end

"""
    GAPopulation{T<:AbstractFloat}
Full representation for GA population, with space for candidates and mating pool
"""
struct GAPopulation{T<:AbstractFloat}
    # The current population of candidates
    current_generation::GABasePopulation{T}

    # The mating pool, to be populated in the selection step
    mating_pool::GABasePopulation{T}

    # children: generated and operated on in the crossover step
    children::GABasePopulation{T}

    # Vectors of integers to store indexes for elitism/selection purposes
    parent_idx_array::Vector{Int} # parents
    child_idx_array::Vector{Int} # children
    
    function GAPopulation{T}(num_candidates::Integer, num_dims::Integer) where {T}
        num_dims > 0 || throw(ArgumentError("num_dims must be greater than 0."))
        num_candidates > 0 || throw(ArgumentError("num_candidates must be greater than 0."))
        return new{T}(
            GABasePopulation{T}(num_candidates, num_dims),
            GABasePopulation{T}(num_candidates, num_dims),
            GABasePopulation{T}(num_candidates, num_dims),
            Vector{Int}(zeros(num_candidates)),
            Vector{Int}(zeros(num_candidates))
        )
    end
end

"""
    function GAPopulation(num_candidates::Integer, num_dims::Integer)
    Creates a `GAPopulation` with `num_candidates`, each having `num_dims`.
"""
function GAPopulation(num_candidates::Integer, num_dims::Integer)
    return GAPopulation{Float64}(num_candidates, num_dims)
end

Base.length(population::GAPopulation) = length(population.current_generation)

function initialize!(
    population::GAPopulation{T},
    pop_init_method::AbstractPopulationInitialization,
    search_space::ContinuousRectangularSearchSpace{T},
) where {T}
    # Unpack population
    @unpack candidates = population.current_generation
    @unpack dim_min, dim_max = search_space

    initialize_population_vector!(candidates, dim_min, dim_max, pop_init_method)

    return nothing
end

function initialize_fitness!(
    population::GAPopulation{T}, evaluator::BatchEvaluator
) where {T}
    # Evaluate the cost function for each candidate
    evaluate!(population.current_generation, evaluator)
    return nothing
end

function evaluate_fitness!(population, evaluator)
    evaluate!(population.current_generation, evaluator)
    return nothing
end

function evaluate_children!(population, evaluator)
    evaluate!(population.children, evaluator)
    return nothing
end