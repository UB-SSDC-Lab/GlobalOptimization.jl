"""
    AbstractOptions

Abstract type for multiple options
"""
abstract type AbstractOptions end

"""
    GeneralOptions{display, function_value_check}

General options for all optimizers
"""
struct GeneralOptions{display,funciton_value_check} <: AbstractOptions
    # Display options
    display_interval::Int

    # Maximum time (seconds)
    max_time::Float64

    # Minimum cost value
    min_cost::Float64

    function GeneralOptions(
        function_value_check::Val{true},
        display::Val{true},
        display_interval::Int,
        max_time,
        min_cost,
    )
        return new{display,function_value_check}(display_interval, max_time, min_cost)
    end
    function GeneralOptions(
        function_value_check::Val{true},
        display::Val{false},
        display_interval::Int,
        max_time,
        min_cost,
    )
        return new{display,function_value_check}(display_interval, max_time, min_cost)
    end
    function GeneralOptions(
        function_value_check::Val{false},
        display::Val{true},
        display_interval::Int,
        max_time,
        min_cost,
    )
        return new{display,function_value_check}(display_interval, max_time, min_cost)
    end
    function GeneralOptions(
        function_value_check::Val{false},
        display::Val{false},
        display_interval::Int,
        max_time,
        min_cost,
    )
        return new{display,function_value_check}(display_interval, max_time, min_cost)
    end
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
    get_display(opts::AbstractOptions)

Returns the display option from an options type.
"""
get_display(opts::GeneralOptions{Val{true},fvc}) where {fvc} = true
get_display(opts::GeneralOptions{Val{false},fvc}) where {fvc} = false
get_display(opts::AbstractAlgorithmSpecificOptions) = get_display(get_general(opts))

"""
    get_display_interval(opts::AbstractAlgorithmSpecificOptions)

Returns the display interval from an algorithm options type.
"""
get_display_interval(opts::GeneralOptions) = opts.display_interval
get_display_interval(opts::AbstractAlgorithmSpecificOptions) = opts.general.display_interval

"""
    get_function_value_check(opts::AbstractAlgorithmSpecificOptions)

Returns the function value check option from an algorithm options type.
"""
get_function_value_check(opts::AbstractAlgorithmSpecificOptions) =
    opts.general.function_value_check

"""
    get_max_time(opts::AbstractAlgorithmSpecificOptions)

Returns the max time option from an algorithm options type.
"""
get_max_time(opts::AbstractAlgorithmSpecificOptions) = opts.general.max_time

"""
    get_min_cost(opts::AbstractAlgorithmSpecificOptions)

Returns the min cost option from an algorithm options type.
"""
get_min_cost(opts::AbstractAlgorithmSpecificOptions) = opts.general.min_cost
