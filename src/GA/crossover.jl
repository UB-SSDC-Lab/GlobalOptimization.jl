
"""
    AbstractGACrossoverParameters{AS<:AbstractAdaptationStrategy}

Abstract type for the crossover parameters used within the Genetic Algorithm (GA).

Subtypes of this abstract type should define the following methods:
- `get_parameter(params::AbstractGACrossoverParameters, i)`: Returns the crossover parameter for the `i`-th candidate.
- `initialize!(params::AbstractGACrossoverParameters, num_dims, population_size)`: Initializes the crossover parameters.
- `adapt!(params::AbstractGACrossoverParameters, improved, global_best_improved)`: Adapts the crossover parameters based on the improvement status of the candidates.
- `crossover!(population::GAPopulation, crossover_params, search_space)`: Performs the crossover operation on the mating pool using the specified crossover parameters.
"""

abstract type AbstractGACrossoverParameters{AS<:AbstractAdaptationStrategy} end

function adapt!(
    params::AbstractGACrossoverParameters{NoAdaptation}, global_best_improved
)
    return nothing
end

function adapt!(
    params::AbstractGACrossoverParameters{RandomAdaptation}, global_best_improved
)

    if !global_best_improved
        params.CR = one_clamped_rand(params.dist)
    end

    return nothing

end

"""
    BLXAlphaCrossover{AS<:AbstractAdaptationStrategy}

The BLX-α crossover mating strategy for GA.
"""
mutable struct BLXAlphaCrossover{AS<:AbstractAdaptationStrategy, D} <: AbstractGACrossoverParameters{AS}

    CR::Float64
    dist::D # in case we desire an adaptive parameter
    alpha::Float64 # BKX-α parameter


    """
        function BLXAlphaCrossover(CR::Float64)
    Creates non-adaptive BLX-α crossover parameters with a static crossover ratio CR applied to the whole population.
    """
    function BLXAlphaCrossover(CR::Float64, alpha)
        return new{NoAdaptation,Nothing}(CR, nothing, alpha)
    end


    """
        function BLXAlphaCrossover(; dist=default_ga_crossover_dist)
    Creates an adaptive BLX-α crossover parameters where `CR` is determined by `dist`.
    """
    function BLXAlphaCrossover(alpha; dist=default_ga_crossover_dist)
        return new{RandomAdaptation, typeof(dist)}(0.0, dist, alpha) # intialize CR as 0.0, will be changed in initialize!
    end

end

function initialize!(params::BLXAlphaCrossover{NoAdaptation}) # For no adaptation, no initialization is needed
    return nothing
end

function initialize!(params::BLXAlphaCrossover{RandomAdaptation})
    params.CR = rand(params.dist)
    return nothing
end


"""
    function crossover!(population::GAPopulation, crossover_params::BLXAlphaCrossover, search space)

    Performs the BLX-α crossover on the `population`
"""
function crossover!(
    population::GAPopulation,
    crossover_params::BLXAlphaCrossover
)
    
    # Unpack the 3 sub populations: current generation, mating pool, children. We will be populating the children in this function
    @unpack current_generation, mating_pool, children = population
    
    # Point to the candidates of each
    current_generation = current_generation.candidates
    mating_pool = mating_pool.candidates
    children = children.candidates

    alpha = crossover_params.alpha
    m = length(current_generation[1])
    
    # first, need to see how many pairs we can make
    pop_len = length(population)

    extra_candidates = mod(pop_len, 2)
    pop_is_odd = extra_candidates == 1 # if pop is odd (why) we'll have one sad, lonely candidate we'll need to handle

    # shuffle the MP
    shuffle!(mating_pool)

    # shuffle was random, so we can iterate over pairs
    @inbounds for i=1:2:pop_len-1-extra_candidates # if any extra candidates, we'll leave it off at the end and add it to children
        
        if rand()<crossover_params.CR
            # create pointers to parents
            parent1 = mating_pool[i]
            parent2 = mating_pool[i+1]

            gamma_i = (1+2*alpha)*rand() - alpha
    
            # Assign children based on BLX-α crossover
            children[i][:] .= (1-gamma_i)*parent1 + gamma_i*parent2
            children[i+1] .= (1-gamma_i)*parent2 + gamma_i*parent1
        else # pass parents onto next gen
            children[i] .= mating_pool[i]
            children[i+1] .= mating_pool[i+1]
        end


    end

    if pop_is_odd
        children[end]= mating_pool[end] # the lonely one
    end

    # Saving the clamp to bounds to the mutation step so we dont have to do it more than once

    return nothing

end