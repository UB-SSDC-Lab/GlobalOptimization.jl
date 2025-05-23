
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
"""
function TraceMinimal(;print_frequency = 1, save_frequency = 1)
    return TraceLevel(Val{:minimal}(), print_frequency, save_frequency)
end

"""
    TraceDetailed(freq)
    TraceDetailed(; print_frequency = 1, save_frequency = 1)

Trace Detailed information about the optimization process (including the information in
the minimal trace).
"""
function TraceDetailed(;print_frequency = 1, save_frequency = 1)
    return TraceLevel(Val{:detailed}(), print_frequency, save_frequency)
end

"""
    TraceAll(freq)
    TraceAll(; print_frequency = 1, save_frequency = 1)

Trace All information about the optimization process (including the information in the
detailed trace). This trace option should likely only be used for debugging purposes.
"""
function TraceAll(;print_frequency = 1, save_frequency = 1)
    return TraceLevel(Val{:all}(), print_frequency, save_frequency)
end

for Tr in (:TraceMinimal, :TraceDetailed, :TraceAll)
    @eval $(Tr)(freq) = $(Tr)(;print_frequency = freq, save_frequency = freq)
end

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

function trace(opt::AbstractOptimizer)
    # Do nothing if tracing is not enabled
    trace = get_trace(opt.options)
    trace.save_trace isa Val{false} && trace.show_trace isa Val{false} && return nothing

    iteration = get_iteration(opt)
    show_now = trace.show_trace isa Val{true} &&
        (mod1(iteration, trace.trace_level.print_frequency) == 0)
    save_now = trace.save_trace isa Val{true} &&
        (mod1(iteration, trace.trace_level.save_frequency) == 0)

    if show_now || save_now
        show_now && show_trace(opt, trace.trace_level.trace_mode)
        if save_now
            save_trace_str = get_save_trace(opt, trace.trace_level.trace_mode)
            open(trace.save_file, "a") do io
                println(io, save_trace_str)
            end
        end
    end
end

function show_trace(opt::AbstractOptimizer, trace_mode::Val{mode}) where mode
    throw(ArgumentError(
        "show_trace with trace mode " * String(mode) * " not implemented for $(typeof(opt))."
    ))
end

function get_save_trace(opt::AbstractOptimizer, trace_mode::Val{mode}) where mode
    throw(ArgumentError(
        "get_save_trace with trace mode " * String(mode) * " not implemented for $(typeof(opt))."
    ))
end
