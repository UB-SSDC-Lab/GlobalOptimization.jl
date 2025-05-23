"""
    AbstractOptions

Abstract type for multiple options
"""
abstract type AbstractOptions end

"""
    GeneralOptions

General options for all optimizers
"""
struct GeneralOptions{
    FVC <: Union{Val{false}, Val{true}},
    TR,
} <: AbstractOptions
    # Trace
    trace::TR

    # Check function value
    function_value_check::FVC

    # Minimum cost value
    min_cost::Float64

    # Maximum time (seconds)
    max_time::Float64

    # Maximum iterations
    max_iterations::Int

    # Stall
    function_tolerance::Float64
    max_stall_time::Float64
    max_stall_iterations::Int
end

"""
    AbstractAlgorithmSpecificOptions

Abstract type for algorithm specific options

All subtypes must define the following fields:
- `general`: The general options for an optimizer.
"""
abstract type AbstractAlgorithmSpecificOptions <: AbstractOptions end

"""
    get_general(opts::AbstractAlgorithmSpecificOptions)

Returns the general options from an algorithm options type.
"""
get_general(opts::AbstractAlgorithmSpecificOptions) = opts.general

"""
    get_trace(opts::AbstractOptions)

Returns the display option from an options type.
"""
get_trace(opts::GeneralOptions) = opts.trace
get_trace(opts::AbstractAlgorithmSpecificOptions) = get_trace(get_general(opts))

"""
    get_function_value_check(opts::AbstractAlgorithmSpecificOptions)

Returns the function value check option from an algorithm options type.
"""
get_function_value_check(opts::GeneralOptions) = opts.function_value_check
function get_function_value_check(opts::AbstractAlgorithmSpecificOptions)
    return get_function_value_check(get_general(opts))
end

"""
    get_min_cost(opts::AbstractAlgorithmSpecificOptions)

Returns the min cost option from an algorithm options type.
"""
get_min_cost(opts::GeneralOptions) = opts.min_cost
get_min_cost(opts::AbstractAlgorithmSpecificOptions) = get_min_cost(get_general(opts))

"""
    get_max_time(opts::AbstractAlgorithmSpecificOptions)

Returns the max time option from an algorithm options type.
"""
get_max_time(opts::GeneralOptions) = opts.max_time
get_max_time(opts::AbstractAlgorithmSpecificOptions) = get_max_time(get_general(opts))

"""
    get_max_iterations(opts::AbstractAlgorithmSpecificOptions)

Returns the max iterations option from an algorithm options type.
"""
get_max_iterations(opts::GeneralOptions) = opts.max_iterations
get_max_iterations(opts::AbstractAlgorithmSpecificOptions) = get_max_iterations(get_general(opts))

"""
    get_function_tolerance(opts::AbstractAlgorithmSpecificOptions)

Returns the function tolerance option from an algorithm options type.
"""
get_function_tolerance(opts::GeneralOptions) = opts.function_tolerance
get_function_tolerance(opts::AbstractAlgorithmSpecificOptions) = get_function_tolerance(get_general(opts))

"""
    get_max_stall_time(opts::AbstractAlgorithmSpecificOptions)

Returns the max stall time option from an algorithm options type.
"""
get_max_stall_time(opts::GeneralOptions) = opts.max_stall_time
get_max_stall_time(opts::AbstractAlgorithmSpecificOptions) = get_max_stall_time(get_general(opts))

"""
    get_max_stall_iterations(opts::AbstractAlgorithmSpecificOptions)

Returns the max stall iterations option from an algorithm options type.
"""
get_max_stall_iterations(opts::GeneralOptions) = opts.max_stall_iterations
function get_max_stall_iterations(opts::AbstractAlgorithmSpecificOptions)
    return get_max_stall_iterations(get_general(opts))
end
