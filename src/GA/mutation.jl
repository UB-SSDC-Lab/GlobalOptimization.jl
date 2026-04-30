"""
    AbstractGAMutationParameters

Abstract type for all GA Mutations
"""
abstract type AbstractGAMutationParameters end

mutable struct RandomElementwiseMutation{AS<:AbstractAdaptationStrategy, D} <: AbstractGAMutationParameters
    MP::Float64 # Mutation Probability
    delta::Float64 # mutation perturbation range, i.e., an element can change by Δi/2 in either direction
    dist::D # distribution, for adaptive mutation parameters
    function RandomElementwiseMutation(MP, delta)
        return new{NoAdaptation, Nothing}(MP, delta, nothing)
    end

    function RandomElementwiseMutation(delta; dist=default_ga_mutation_dist)
        return new{RandomAdaptation, typeof(dist)}(0.0, delta, dist)
    end

end

function initialize!(params::RandomElementwiseMutation{NoAdaptation})
    return nothing
end

function initialize!(params::RandomElementwiseMutation{RandomAdaptation})
    params.MP = rand(params.dist)
    return nothing
end

function adapt!(params::RandomElementwiseMutation{NoAdaptation}, global_best_improved)
    return nothing
end

function adapt!(params::RandomElementwiseMutation{RandomAdaptation}, global_best_improved)
    params.MP = rand(params.dist)
    return nothing
end

function mutate!(
    population::GAPopulation,
    params::RandomElementwiseMutation,
    search_space
)

    # Unpack subpopulations. We will be mutating the children population, which is generated in the crossover step. The children will then replace the parents.
    @unpack current_generation, mating_pool, children = population

    children = children.candidates

    for i in eachindex(children)
        for j in eachindex(children[i])
            if rand() < params.MP
                children[i][j] += (0.5-rand())*params.delta # add random perturbation
            end

            # clamp to bounds, regardless of if mutation occurred
            children[i][j] = clamp(children[i][j], dim_min(search_space, j), dim_max(search_space, j))

        end
    end

    return nothing

end