"""
    TraceLevel{TM}

A structure to with information about the tracing of the global optimization process.
"""
struct TraceLevel{
    TM <: Union{Val{:minimal}, Val{:detailed}, Val{:all}}
}
    trace_mode::TM
    print_frequency::Int
    save_frequency::Int
end

"""
    TraceMinimal(freq)
    TraceMinimal(; print_frequency = 1, save_frequency = 1)

Trace Minimal information about the optimization process.

For example, this will set `PSO` or `DE` to print the elapsed time, iteration number, stall
iterations, and global best fitness.

# Returns
- `TraceLevel{Val{:minimal}}`: A trace level object with the minimal trace level.
"""
function TraceMinimal(;print_frequency = 1, save_frequency = 1)
    return TraceLevel(Val{:minimal}(), print_frequency, save_frequency)
end

"""
    TraceDetailed(freq)
    TraceDetailed(; print_frequency = 1, save_frequency = 1)

Trace Detailed information about the optimization process (including the information in
the minimal trace).

# Returns
- `TraceLevel{Val{:detailed}}`: A trace level object with the detailed trace level.
"""
function TraceDetailed(;print_frequency = 1, save_frequency = 1)
    return TraceLevel(Val{:detailed}(), print_frequency, save_frequency)
end

"""
    TraceAll(freq)
    TraceAll(; print_frequency = 1, save_frequency = 1)

Trace All information about the optimization process (including the information in the
detailed trace). This trace option should likely only be used for debugging purposes.

# Returns
- `TraceLevel{Val{:all}}`: A trace level object with the all trace level.
"""
function TraceAll(;print_frequency = 1, save_frequency = 1)
    return TraceLevel(Val{:all}(), print_frequency, save_frequency)
end

for Tr in (:TraceMinimal, :TraceDetailed, :TraceAll)
    @eval $(Tr)(freq) = $(Tr)(;print_frequency = freq, save_frequency = freq)
end

"""
    GlobalOptimizationTrace{SHT, SAT, TM}

A structure to hold the global optimization trace settings.

# Fields:
- `show_trace::SHT`: A flag indicating whether to show the trace in the console.
- `save_trace::SAT`: A flag indicating whether to save the trace to a file.
- `save_file::String`: The file path where the trace will be saved.
- `trace_level::TraceLevel{TM}`: The trace level settings, which can be `TraceMinimal`, `TraceDetailed`, or `TraceAll`.
"""
struct GlobalOptimizationTrace{
    SHT <: Union{Val{false}, Val{true}},
    SAT <: Union{Val{false}, Val{true}},
    TM
}
    show_trace::SHT
    save_trace::SAT
    save_file::String
    trace_level::TraceLevel{TM}
end

"""
    TraceElement{T}

A structure to hold a single trace element, which includes a label, flag, length, precision, and value.

An `AbstractOptimizer` should return a tuple of `TraceElement` objects when the
`get_show_trace_elements` or `get_save_trace_elements` functions are called.

# Fields:
- `label::String`: The label for the trace element.
- `flag::Char`: A character indicating the format of the value (e.g., 'd' for integer, 'f' for float).
- `length::Int`: The length of the formatted string representation of the value.
- `precision::Int`: The precision for floating-point values.
- `value::T`: The value of the trace element, which can be of any type `T`.
"""
struct TraceElement{T}
    label::String
    flag::Char
    length::Int
    precision::Int
    value::T
end

function get_label_str(te::TraceElement, fmt::Val{true})
    fmt_str = Format("%-$(te.length)s")
    return format(fmt_str, ' ' * te.label * ' ')
end
function get_label_str(te::TraceElement, fmt::Val{false})
    return te.label
end
function get_label_str(te::AbstractVector{TE}, fmt::Val) where {TE <: TraceElement}
    str = ""
    for i in eachindex(te)
        str *= get_label_str(te[i], fmt)
        if i < length(te)
            fmt isa Val{true} && (str *= ' ')
            fmt isa Val{false} && (str *= ',')
        end
    end
    return str
end

function get_line_str(te::TraceElement)
    fmt_str = Format("%-$(te.length)s")
    return format(fmt_str, '-'^(length(te.label) + 2))
end
function get_line_str(te::AbstractVector{TE}) where {TE <: TraceElement}
    str = ""
    for i in eachindex(te)
        str *= get_line_str(te[i])
        if i < length(te)
            str *= ' '
        end
    end
    return str
end

function get_str(te::TraceElement{T}, fmt::Val{true}) where {T <: Integer}
    fmt_str = Format("%-$(te.length)d")
    return format(fmt_str, te.value)
end
function get_str(te::TraceElement{T}, fmt::Val{true}) where {T <: AbstractFloat}
    fmt_str = Format("%-$(te.length).$(te.precision)$(te.flag)")
    return format(fmt_str, te.value)
end
function get_str(te::TraceElement{T}, fmt::Val{false}) where {T <: Integer}
    return string(te.value)
end
function get_str(te::TraceElement{T}, fmt::Val{false}) where {T <: AbstractFloat}
    fmt_str = Format("%$(te.flag)")
    return format(fmt_str, te.value)
end
function get_str(te::AbstractVector{TE}, fmt::Val) where {TE <: TraceElement}
    str = ""
    for i in eachindex(te)
        str *= get_str(te[i], fmt)
        if i < length(te)
            fmt isa Val{true} && (str *= ' ')
            fmt isa Val{false} && (str *= ',')
        end
    end
    return str
end

# ===== Top level trace functions

function get_top_level_show_trace(opt, trace_mode)
    trace_elements = get_show_trace_elements(opt, trace_mode)

    str = ""
    for i in eachindex(trace_elements)
        str *= get_line_str(trace_elements[i])
        i < length(trace_elements) && (str *= ' ')
    end
    str *= '\n'

    for i in eachindex(trace_elements)
        str *= get_label_str(trace_elements[i], Val{true}())
        i < length(trace_elements) && (str *= ' ')
    end
    str *= '\n'

    for i in eachindex(trace_elements)
        str *= get_line_str(trace_elements[i])
        i < length(trace_elements) && (str *= ' ')
    end
    str *= '\n'

    return str
end

function get_top_level_save_trace(opt, trace_mode)
    trace_elements = get_save_trace_elements(opt, trace_mode)

    str = ""
    for i in eachindex(trace_elements)
        str *= get_label_str(trace_elements[i], Val{false}())
        i < length(trace_elements) && (str *= ',')
    end
    str *= '\n'

    return str
end

function top_level_trace(opt)
    trace = get_trace(opt.options)
    trace.save_trace isa Val{false} && trace.show_trace isa Val{false} && return nothing

    if trace.show_trace isa Val{true}
        print(stdout, get_top_level_show_trace(opt, trace.trace_level.trace_mode))
    end

    if trace.save_trace isa Val{true}
        open(trace.save_file, "w") do io
            print(io, get_top_level_save_trace(opt, trace.trace_level.trace_mode))
        end
    end
    return nothing
end

# ===== Per iteration trace functions
function get_show_trace(opt, trace_mode)
    trace_elements = get_show_trace_elements(opt, trace_mode)
    str = ""
    for i in eachindex(trace_elements)
        str *= get_str(trace_elements[i], Val{true}())
        i < length(trace_elements) && (str *= ' ')
    end
    str *= '\n'
    return str
end
function get_save_trace(opt, trace_mode)
    trace_elements = get_save_trace_elements(opt, trace_mode)
    str = ""
    for i in eachindex(trace_elements)
        str *= get_str(trace_elements[i], Val{false}())
        i < length(trace_elements) && (str *= ',')
    end
    str *= '\n'
    return str
end

function trace(opt, final)
    # Do nothing if tracing is not enabled
    trace = get_trace(opt.options)
    trace.save_trace isa Val{false} && trace.show_trace isa Val{false} && return nothing

    iteration = get_iteration(opt)
    show_now = trace.show_trace isa Val{true} &&
        ((mod1(iteration, trace.trace_level.print_frequency) == 1) || final)
    save_now = trace.save_trace isa Val{true} &&
        ((mod1(iteration, trace.trace_level.save_frequency) == 1) || final)

    if show_now || save_now
        show_now && print(stdout, get_show_trace(opt, trace.trace_level.trace_mode))
        if save_now
            save_trace_str = get_save_trace(opt, trace.trace_level.trace_mode)
            open(trace.save_file, "a") do io
                print(io, save_trace_str)
            end
        end
    end
end

# Utility functions to help return trace elements from optimizers
function cat_elements(a::Tuple, b::Tuple)
    return (a..., b...)
end
function cat_elements(a::Tuple, b::AbstractVector)
    return (a..., b)
end
function cat_elements(a::AbstractVector, b::Tuple)
    return (a, b...)
end
function cat_elements(a::AbstractVector, b::AbstractVector)
    return (a, b)
end
