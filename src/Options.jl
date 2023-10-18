"""
    AbstractOptions

Abstract type for options
"""
abstract type AbstractOptions end


"""
    GeneralOptions

General options for all optimizers
"""
struct GeneralOptions <: AbstractOptions
    # Display options
    display::Bool
    display_interval::Int

    # Check function value for NaN and Inf
    function_value_check::Bool

    # Maximum time (seconds)
    max_time::Float64
end


"""
    AbstractAlgorithmSpecificOptions

Abstract type for algorithm specific options
"""
abstract type AbstractAlgorithmSpecificOptions <: AbstractOptions end

"""
    get_general(opts::AbstractAlgorithmSpecificOptions)

Returns the general options from an algorithm options type.
"""
get_general(opts::AbstractAlgorithmSpecificOptions) = opts.general

"""
    get_display(opts::AbstractAlgorithmSpecificOptions)

Returns the display option from an algorithm options type.
"""
get_display(opts::AbstractAlgorithmSpecificOptions) = opts.general.display

"""
    get_display_interval(opts::AbstractAlgorithmSpecificOptions)

Returns the display interval from an algorithm options type.
"""
get_display_interval(opts::AbstractAlgorithmSpecificOptions) = opts.general.display_interval

"""
    get_function_value_check(opts::AbstractAlgorithmSpecificOptions)

Returns the function value check option from an algorithm options type.
"""
get_function_value_check(opts::AbstractAlgorithmSpecificOptions) = opts.general.function_value_check

"""
    get_max_time(opts::AbstractAlgorithmSpecificOptions)

Returns the max time option from an algorithm options type.
"""
get_max_time(opts::AbstractAlgorithmSpecificOptions) = opts.general.max_time
