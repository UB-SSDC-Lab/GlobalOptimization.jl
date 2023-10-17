"""
    AbstractOptions

Abstract type for options
"""
abstract type AbstractOptions end


"""
    GeneralOptions

General options for all optimizers
"""
struct GeneralOptions{T <: AbstractFloat} <: AbstractOptions
    # Display options
    display::Bool
    display_interval::Int

    # Check function value for NaN and Inf
    function_value_check::Bool

    # Maximum time
    max_time::T
end


"""
    AbstractAlgorithmOptions

Abstract type for algorithm specific options
"""
abstract type AbstractAlgorithmOptions end

"""
    get_general(opts::AbstractAlgorithmOptions)

Returns the general options from an algorithm options type.
"""
get_general(opts::AbstractAlgorithmOptions) = opts.general

"""
    get_display(opts::AbstractAlgorithmOptions)

Returns the display option from an algorithm options type.
"""
get_display(opts::AbstractAlgorithmOptions) = opts.general.display

"""
    get_display_interval(opts::AbstractAlgorithmOptions)

Returns the display interval from an algorithm options type.
"""
get_display_interval(opts::AbstractAlgorithmOptions) = opts.general.display_interval

"""
    get_function_value_check(opts::AbstractAlgorithmOptions)

Returns the function value check option from an algorithm options type.
"""
get_function_value_check(opts::AbstractAlgorithmOptions) = opts.general.function_value_check

"""
    get_max_time(opts::AbstractAlgorithmOptions)

Returns the max time option from an algorithm options type.
"""
get_max_time(opts::AbstractAlgorithmOptions) = opts.general.max_time
