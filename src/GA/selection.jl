"""
    AbstractGASelectionParameters
Abstract type for parameters of selection operators in genetic algorithms. Subtypes of this abstract type should define a selection! method.
Can also define an initialize! method if necessary. The abstract type will have a basic initialize!.
"""
abstract type AbstractGASelectionParameters end



mutable struct StochasticUniversalSampling <: AbstractGASelectionParameters
    pointer_cache::Vector{Float64}
    fitness_cache::Vector{Float64} # cache to store inverse of objective function values for selection, since we are minimizing and want to select candidates with probability proportional to the inverse of their fitness
    function StochasticUniversalSampling(pop_size)
        new(zeros(pop_size), zeros(pop_size))
    end

    function StochasticUniversalSampling() # we'll initialize to pop size anyway
        return new(Vector{Float64}(undef, 0), Vector{Float64}(undef, 0))
    end

end

"""
    TournamentSelection

Standard Tournament-style selection. `tournament_size` individuals are randomly selected from the population.
After tournament, individuals are re-entered into the population, so they can participate multiple times.

Constructor kwargs:
    -`size::Int`: Tournament size, defaults to 2
    -`selection_probability`: Probability `P` of selecting individual based on tournament rank: 1. P, 2. P*(1-P) 3. P*(1-P^2), etc. Defaults to 1.0
"""
struct TournamentSelection <: AbstractGASelectionParameters
    tournament_size::Int
    selection_probabilty::Float64
    participant_idx_cache::Vector{Int}
    function TournamentSelection(; size=2, selection_probability=1.0)
        return new(size, selection_probability, zeros(Int64, size))
    end

end

function initialize!(sel::AbstractGASelectionParameters, pop)
    return nothing
end

function initialize!(sel::StochasticUniversalSampling, pop)
    N = length(pop)
    sel.fitness_cache = zeros(N)
    sel.pointer_cache = zeros(N)
    return nothing
end

"""
    function selection!(pop::GAPopulation, selection_params::StochasticUniversalSampling)
Perform selection using SUS, where candidates are selected with probability proportional to their fitness. The mating pool is filled accordingly.
This function should be called after evaluating the fitness of the current generation, and before performing crossover and mutation to generate the children.
"""
function selection!(pop::GAPopulation, selection_params::StochasticUniversalSampling)
    # Unpack the current generation and mating pool (we will be filling the mating pool in this function)
    @unpack current_generation, mating_pool = pop

    # get candidates and fitness
    candidates = current_generation.candidates
    fitness = current_generation.candidates_fitness

    N = length(pop)

    # Fitness handling
    idx_arr = pop.parent_idx_array
    sortperm!(idx_arr, fitness)
    selection_fitness_cache = selection_params.fitness_cache # best needs highest score
    selection_fitness_cache[idx_arr] .= N:-1:1
    selection_fitness_cache .= selection_fitness_cache/sum(selection_fitness_cache)
    fitness_normed = selection_fitness_cache

    start = rand(); # pick start point
    pointers = selection_params.pointer_cache;
    for i = 1:N 
        pointers[i] = start+(i-1)/N;
        pointers[i] = mod(pointers[i], 1)
    end

    # With pointers generated, see where they fall on the 'wheel'
    # i.e., if its between 0 and fitness_normed(1), add member 1 to mating
    # pool
    for i in 1:N
        running_norm_fitness = 0.0;
        p_i = pointers[i];
        for j in 1:N
            rnfm1 = running_norm_fitness;
            running_norm_fitness = rnfm1 + fitness_normed[j];
            if p_i < running_norm_fitness # once rnf becomes greater than pi, add to mating pool and break loop
                mating_pool.candidates[i][:] = candidates[j][:]; 
                break
            end
        end
    end

    
end

function selection!(pop::GAPopulation, selection_params::TournamentSelection)

    @unpack current_generation, mating_pool = pop

    k = selection_params.tournament_size # size
    participants = selection_params.participant_idx_cache
    N = length(pop)

    SP = selection_params.selection_probabilty

    # do the normal tournaments
    for t=1:N
        # Sample from pop
        participants .= sample(1:N, k, replace=false)

        # Sort participant index list based on their fitness value, least to greatest
        sort!(participants, by=i->current_generation.candidates_fitness[i])
        
        # Assign best performer to mating pool
        # OPTIONAL: later, include some probability based selection here
        mating_pool.candidates[t] = current_generation.candidates[participants[1]]
        mating_pool.candidates_fitness[t] = current_generation.candidates_fitness[participants[1]]
    end
        
    return nothing

end