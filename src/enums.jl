
# Optimization Status
@enum(
    Status::Int8,
    IN_PROGRESS = 0,
    MINIMUM_COST_ACHIEVED = 1,
    MAXIMUM_TIME_EXCEEDED = 2,
    MAXIMUM_ITERATIONS_EXCEEDED = 3,
    MAXIMUM_STALL_ITERATIONS_EXCEEDED = 4,
    MAXIMUM_STALL_TIME_EXCEEDED = 5,
)
