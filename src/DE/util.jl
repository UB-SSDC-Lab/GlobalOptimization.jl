
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
const default_mutation_dist = MixtureModel(
   Cauchy, [(0.65, 0.1), (1.0, 0.1)], [0.5, 0.5]
)
const default_binomial_crossover_dist = MixtureModel(
   Cauchy, [(0.1, 0.1), (0.95, 0.1)], [0.5, 0.5]
)

# Types used to indicate adaptivity (modify to support new adaptation methods as necessary)
abstract type AbstractAdaptationStrategy end
struct RandomAdaptation <: AbstractAdaptationStrategy end
struct NoAdaptation <: AbstractAdaptationStrategy end

# Define function for sampling a distribution and clamping for realization > 1 and
# resampling for realization < 0
function semiclamped_rand(dist)
    iters = 0
    while true
        iters += 1
        value = min(rand(dist), 1.0)
        if value > 0.0
            return value
        end
        if iters > 1000
            throw(ArgumentError(""))
        end
    end
end
