# a basic default truncated distribution. not advised by anything except centered around 0.8, which seems to be best based on
# the MAE552 notes. The truncated points are where the distribution approaches zero, so this should be fine.
const default_ga_crossover_dist = Normal(0.9, 0.1)
const default_ga_mutation_dist = Normal(0.025, 0.005)