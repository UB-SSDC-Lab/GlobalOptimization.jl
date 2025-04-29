
# The following two bi-model cauchy distribution are proposed for use in adapting the
# mutation and crossover parameters in:
#
# Yong Wang, Han-Xiong Li, Tingwen Huang, and Long Li "Differential Evolution Based on
# Covariance Matrix Learning and Bimodal Distribution Parameter Setting" (2014).
#
# const default_mutation_dist = BimodalCauchy(
#     0.65, 0.1, 1.0, 0.1, clampBelow0 = false
# )
# const default_binomial_crossover_dist = BimodalCauchy(
#     0.1, 0.1, 0.95, 0.1, clampBelow0 = false
# )
const default_mutation_dist = MixtureModel(Cauchy, [(0.65, 0.1), (1.0, 0.1)], [0.5, 0.5])
const default_binomial_crossover_dist = MixtureModel(
    Cauchy, [(0.1, 0.1), (0.95, 0.1)], [0.5, 0.5]
)

# Types used to indicate adaptivity (modify to support new adaptation methods as necessary)
abstract type AbstractAdaptationStrategy end
struct RandomAdaptation <: AbstractAdaptationStrategy end
struct NoAdaptation <: AbstractAdaptationStrategy end

# Generic selection types
abstract type AbstractSelector end

"""
    SimpleSelector

A selector that simply *selects* all candidates in the population.
"""
struct SimpleSelector <: AbstractSelector end

"""
    RadiusLimitedSelector

A selector that selects candidates within a given radius of the target candidate.

For example, for population size of 10 and a radius of 2, the following will be selected
for the given target indices:

`target = 5` will select `[3, 4, 5, 6, 7]`

`target = 1` will select `[9, 10, 1, 2, 3]`

`target = 9` will select `[7, 8, 9, 10, 1]`
"""
struct RadiusLimitedSelector <: AbstractSelector
    radius::Int
    idxs::Vector{UInt16}

    function RadiusLimitedSelector(radius::Int)
        idxs = Vector{UInt16}(undef, 2 * radius + 1)
        return new(radius, idxs)
    end
end

select(s::SimpleSelector, target, pop_size) = 1:pop_size
function select(s::RadiusLimitedSelector, target, pop_size)
    # Pick target index
    #target = rand(1:pop_size)

    # Set indices
    for i in eachindex(s.idxs)
        s.idxs[i] = UInt16(mod1(target + i - s.radius - 1, pop_size))
    end

    return s.idxs
end

# Define function for sampling a distribution and clamping for realization > 1 and
# resampling for realization < 0
function one_clamped_rand(dist)
    iters = 0
    while true
        iters += 1
        value = min(rand(dist), 1.0)
        if value > 0.0
            return value
        end
        if iters > 1000
            throw(
                ArgumentError(
                    "one_clamped_rand failed to generate a valid value in 1000 attempts. " *
                    "Consider using a different distribution.",
                ),
            )
        end
    end
end
