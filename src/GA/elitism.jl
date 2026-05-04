"""
    AbstractGAElitismParameters

Parameters to decide how to set the next generation, and whether or not to use elitism. Subtypes should have set_next_generation! method
"""

abstract type AbstractGAElitismParameters end

"""
    NoElitism 
Just set the next generation to be the children.
"""
struct NoElitism <: AbstractGAElitismParameters end

"""
    SimpleElitism
Set a percentage of the parent population that will replace the same amount of the worst performers in the child population.
Should be between 0.01 and 0.05. Rounds up if needed.
"""
struct SimpleElitism <: AbstractGAElitismParameters
    fraction::Float64
end

function set_next_generation!(population, elite::NoElitism)
    population.current_generation.candidates .= population.children.candidates
    population.current_generation.candidates_fitness .= population.children.candidates_fitness
    return nothing
end

function set_next_generation!(population, elite::SimpleElitism)
    frac = elite.fraction
    N = length(population)

    num_to_keep = Int(ceil(N*frac))

    # Sort parents smallest to largest
    parent_idx_arr = population.parent_idx_array
    sortperm!(parent_idx_arr, population.current_generation.candidates_fitness)

    # Sort children opposite
    child_idx_arr = population.child_idx_array
    sortperm!(child_idx_arr, population.children.candidates_fitness; rev=true) # reverse to put largest values at top

    # remove worst children and set parents
    best_parent_idxs = view(parent_idx_arr, 1:num_to_keep)
    worst_children_idxs  = view(child_idx_arr, 1:num_to_keep)

    population.children.candidates[worst_children_idxs] .= population.current_generation.candidates[best_parent_idxs]
    population.children.candidates_fitness[worst_children_idxs] .= population.current_generation.candidates_fitness[best_parent_idxs]
    
    # Now, overwrite the parents with the children
    population.current_generation.candidates .= population.children.candidates
    population.current_generation.candidates_fitness .= population.children.candidates_fitness
    return nothing
end