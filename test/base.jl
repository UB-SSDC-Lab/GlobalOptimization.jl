using GlobalOptimization, Test
using ChunkSplitters: ChunkSplitters
using Suppressor

# Define a concrete simple population type
struct SimplePopulation <: GlobalOptimization.AbstractPopulation{Float64}
    candidates::Vector{Vector{Float64}}
    candidates_fitness::Vector{Float64}
end

# Create a dummy optimizer
struct DummyOptimizerOptions <: GlobalOptimization.AbstractAlgorithmSpecificOptions
    general::GlobalOptimization.GeneralOptions
end
struct DummyOptimizer <: GlobalOptimization.AbstractOptimizer
    options::DummyOptimizerOptions
    cache::GlobalOptimization.MinimalOptimizerCache{Float64}
end
GlobalOptimization.get_best_fitness(opt::DummyOptimizer) = 0.0
GlobalOptimization.get_best_candidate(opt::DummyOptimizer) = [0.0, 0.0]
GlobalOptimization.get_elapsed_time(opt::DummyOptimizer) = 0.0
function GlobalOptimization.get_show_trace_elements(
    opt::DummyOptimizer,
    trace_mode::Union{Val{:detailed}, Val{:all}},
)
    return GlobalOptimization.get_show_trace_elements(opt, Val{:minimal}())
end
function GlobalOptimization.get_save_trace_elements(
    opt::DummyOptimizer,
    trace_mode::Union{Val{:detailed}, Val{:all}},
)
    return GlobalOptimization.get_save_trace_elements(opt, Val{:minimal}())
end

@testset showtiming = true "SearchSpace" begin

    # Test construction and basic accessors
    LB = [-1.0, 0.0]
    UB = [1.0, 2.0]
    ss = ContinuousRectangularSearchSpace(LB, UB)
    @test GlobalOptimization.num_dims(ss) == 2
    @test GlobalOptimization.dim_min(ss) == LB
    @test GlobalOptimization.dim_max(ss) == UB
    @test GlobalOptimization.dim_delta(ss) == (UB .- LB)
    @test GlobalOptimization.dim_min(ss, 1) == LB[1]
    @test GlobalOptimization.dim_max(ss, 2) == UB[2]
    @test GlobalOptimization.dim_range(ss, 1) == (LB[1], UB[1])
    @test GlobalOptimization.dim_range(ss) == [(LB[i], UB[i]) for i in 1:2]

    # Test intersection
    ss2 = ContinuousRectangularSearchSpace([-0.5, 0.5], [0.5, 1.5])
    ss_int = GlobalOptimization.intersection(ss, ss2)
    @test GlobalOptimization.dim_min(ss_int) == [-0.5, 0.5]
    @test GlobalOptimization.dim_max(ss_int) == [0.5, 1.5]

    # Intersection with Nothing
    @test GlobalOptimization.intersection(ss, nothing) === ss
    @test GlobalOptimization.intersection(nothing, ss2) === ss2

    # Test feasibility
    @test GlobalOptimization.feasible([0.0, 1.0], ss)
    @test !GlobalOptimization.feasible([-2.0, 1.0], ss)
    @test !GlobalOptimization.feasible([0.0, 3.0], ss)

    # Test constructor error conditions
    @test_throws DimensionMismatch ContinuousRectangularSearchSpace([0.0, 1.0], [0.0])
    @test_throws ArgumentError ContinuousRectangularSearchSpace([1.0, 0.0], [0.0, 2.0])
end

@testset showtiming = true "Results" begin
    # Test constructor and field access
    fbest = 1.23
    xbest = [4.5, 6.7]
    r = GlobalOptimization.Results(fbest, xbest, 10, 0.456, GlobalOptimization.IN_PROGRESS)
    @test r.fbest == fbest
    @test r.xbest == xbest
    @test r.xbest !== xbest
    @test r.iters == 10
    @test r.time == 0.456
    @test r.exitFlag == GlobalOptimization.IN_PROGRESS

    # Test pretty-printing via Base.show
    buf = IOBuffer()
    show(buf, MIME"text/plain"(), r)
    output = String(take!(buf))
    @test occursin("Results:", output)
    @test occursin("Best function value: 1.23", output)
    @test occursin("Best candidate: [4.5, 6.7]", output)
    @test occursin("Iterations: 10", output)
    @test occursin("Time: 0.456 seconds", output)
    @test occursin("Exit flag: IN_PROGRESS", output)

    # Test type parameter propagation and copying for Float32
    f32 = Float32(2.5)
    x32 = Float32[1.0, 2.0]
    r32 = GlobalOptimization.Results(f32, x32, 3, 0.1, GlobalOptimization.IN_PROGRESS)
    @test typeof(r32.fbest) == Float32
    @test typeof(r32.xbest) == Vector{Float32}

    # Test constructor rejects mismatched xbest element types
    @test_throws MethodError GlobalOptimization.Results(1.0, [1, 2], 1, 0.1, 0)
end

@testset showtiming = true "Problems" begin
    # Setup a simple search space and test problem constructors
    LB = [-1.0, 0.0]
    UB = [1.0, 2.0]
    ss = ContinuousRectangularSearchSpace(LB, UB)

    # Test OptimizationProblem without penalty
    f = x -> sum(x)
    prob1 = GlobalOptimization.OptimizationProblem(f, ss)
    @test GlobalOptimization.search_space(prob1) === ss
    @test GlobalOptimization.num_dims(prob1) == 2
    x = [1.0, 2.0]
    @test GlobalOptimization.scalar_function(prob1, x) == 3.0
    val, pen = GlobalOptimization.scalar_function_with_penalty(prob1, x)
    @test val == 3.0
    @test pen == 0.0
    sf = GlobalOptimization.get_scalar_function(prob1)
    @test sf(x) == 3.0
    sfp = GlobalOptimization.get_scalar_function_with_penalty(prob1)
    @test sfp(x) == (3.0, 0.0)

    # Test OptimizationProblem with penalty
    fpen = x -> (sum(x), sum(x))
    prob2 = GlobalOptimization.OptimizationProblem(fpen, ss)
    @test GlobalOptimization.num_dims(prob2) == 2
    # scalar_function: f + max(0.0,g) = 3.0 + max(0.0,3.0) = 3.0 + 6.0 = 6.0
    @test isapprox(GlobalOptimization.scalar_function(prob2, x), 6.0; atol=1e-8)
    vp, gp = GlobalOptimization.scalar_function_with_penalty(prob2, x)
    @test vp == 3.0
    @test gp == 3.0

    # Test NonlinearProblem without penalty
    fres = x -> [x[1] - 2.0, x[2] - 3.0]
    nprob = GlobalOptimization.NonlinearProblem(fres, ss)
    # At x = [2.0,3.0], residuals zero -> cost zero
    xn = [2.0, 3.0]
    @test GlobalOptimization.scalar_function(nprob, xn) == 0.0
    cn, pn = GlobalOptimization.scalar_function_with_penalty(nprob, xn)
    @test cn == 0.0
    @test pn == 0.0

    # Test get_scalar_function for nonlinear problem
    nf = GlobalOptimization.get_scalar_function(nprob)
    @test nf(xn) == 0.0

    # Test NonlinearLeastSquaresProblem constructor and field
    nlsprob = GlobalOptimization.NonlinearLeastSquaresProblem(fres, ss, 2)
    @test nlsprob.n == 2
    # Same cost behavior
    @test GlobalOptimization.scalar_function(nlsprob, xn) == 0.0
    @test GlobalOptimization.scalar_function_with_penalty(nlsprob, xn) == (0.0, 0.0)
end

@testset showtiming = true "Trace" begin
    # Test TraceElement methods
    te_int = GlobalOptimization.TraceElement("Int", 'd', 8, 0, 10)
    te_flt = GlobalOptimization.TraceElement("Float", 'e', 16, 8, Float64(pi))
    te_vec = map(xx->GlobalOptimization.TraceElement("Vec", 'f', 6, 2, xx), [1.0, 2.0])
    @test GlobalOptimization.get_label_str(te_int, Val{true}()) == " Int    "
    @test GlobalOptimization.get_label_str(te_int, Val{false}()) == "Int"
    @test GlobalOptimization.get_label_str(te_flt, Val{true}()) == " Float          "
    @test GlobalOptimization.get_label_str(te_flt, Val{false}()) == "Float"
    @test GlobalOptimization.get_label_str(te_vec, Val{true}()) == " Vec    Vec  "
    @test GlobalOptimization.get_label_str(te_vec, Val{false}()) == "Vec,Vec"

    @test GlobalOptimization.get_line_str(te_int) == "-----   "
    @test GlobalOptimization.get_line_str(te_vec) == "-----  ----- "

    @test GlobalOptimization.get_str(te_int, Val{true}()) == "10      "
    @test GlobalOptimization.get_str(te_int, Val{false}()) == "10"
    @test GlobalOptimization.get_str(te_flt, Val{true}()) == "3.14159265e+00  "
    @test GlobalOptimization.get_str(te_flt, Val{false}()) == "3.141593e+00"
    @test GlobalOptimization.get_str(te_vec, Val{true}()) == "1.00   2.00  "

    # Utility methods
    a = (1, 2)
    b = [3, 4]
    @test GlobalOptimization.cat_elements(a, a) == (1, 2, 1, 2)
    @test GlobalOptimization.cat_elements(a, b) == (1, 2, [3, 4])
    @test GlobalOptimization.cat_elements(b, a) == ([3, 4], 1, 2)
    @test GlobalOptimization.cat_elements(b, b) == ([3, 4], [3, 4])

    # Test GlobalOptimizationTrace
    trace_file = "./trace.txt"
    TraceLevelConstructors = [:TraceMinimal, :TraceDetailed, :TraceAll]
    trace_modes = [:minimal, :detailed, :all]
    for i in eachindex(TraceLevelConstructors)
        do_no_trace = DummyOptimizer(
            DummyOptimizerOptions(
                GlobalOptimization.GeneralOptions(
                    GlobalOptimization.GlobalOptimizationTrace(
                        Val(false),
                        Val(false),
                        trace_file,
                        @eval $(TraceLevelConstructors[i])(1)
                    ),
                    Val(true),
                    0.0,
                    10.0,
                    100,
                    1e-4,
                    2.0,
                    5,
                )
            ),
            GlobalOptimization.MinimalOptimizerCache{Float64}(),
        )

        output = @capture_out begin
            GlobalOptimization.top_level_trace(do_no_trace)
        end
        @test isempty(output)

        output = @capture_out begin
            GlobalOptimization.trace(do_no_trace, false)
        end
        @test isempty(output)

        # Test only show trace
        do_only_show_trace = DummyOptimizer(
            DummyOptimizerOptions(
                GlobalOptimization.GeneralOptions(
                    GlobalOptimization.GlobalOptimizationTrace(
                        Val(true),
                        Val(false),
                        trace_file,
                        @eval $(TraceLevelConstructors[i])(2)
                    ),
                    Val(true),
                    0.0,
                    10.0,
                    100,
                    1e-4,
                    2.0,
                    5,
                )
            ),
            GlobalOptimization.MinimalOptimizerCache{Float64}(),
        )

        output = @capture_out begin
            GlobalOptimization.top_level_trace(do_only_show_trace)
        end
        labels = ["Iter","Time","S","Best Fitness"]
        for l in labels
            @test contains(output, l)
        end

        do_only_show_trace.cache.iteration = 1
        output = @capture_out begin
            GlobalOptimization.trace(do_only_show_trace, false)
        end
        @test output == "1        0.00     0    0.00000000e+00  \n"

        do_only_show_trace.cache.iteration = 2
        output = @capture_out begin
            GlobalOptimization.trace(do_only_show_trace, false)
        end
        @test isempty(output)

        do_only_show_trace.cache.iteration = 3
        output = @capture_out begin
            GlobalOptimization.trace(do_only_show_trace, false)
        end
        @test output == "3        0.00     0    0.00000000e+00  \n"

        do_only_save_trace = DummyOptimizer(
            DummyOptimizerOptions(
                GlobalOptimization.GeneralOptions(
                    GlobalOptimization.GlobalOptimizationTrace(
                        Val(false),
                        Val(true),
                        trace_file,
                        @eval $(TraceLevelConstructors[i])(2)
                    ),
                    Val(true),
                    0.0,
                    10.0,
                    100,
                    1e-4,
                    2.0,
                    5,
                )
            ),
            GlobalOptimization.MinimalOptimizerCache{Float64}(),
        )

        output = @capture_out begin
            GlobalOptimization.top_level_trace(do_only_save_trace)
        end
        @test isempty(output)
        lines = readlines(trace_file)
        @test length(lines) == 1
        @test lines[1] == "Iter,Time,S,Best Fitness"

        do_only_save_trace.cache.iteration = 1
        output = @capture_out begin
            GlobalOptimization.trace(do_only_save_trace, false)
        end
        @test isempty(output)
        lines = readlines(trace_file)
        @test length(lines) == 2
        @test lines[1] == "Iter,Time,S,Best Fitness"
        @test lines[2] == "1,0.000000,0,0.000000e+00"

        do_only_save_trace.cache.iteration = 2
        output = @capture_out begin
            GlobalOptimization.trace(do_only_save_trace, false)
        end
        @test isempty(output)
        lines = readlines(trace_file)
        @test length(lines) == 2
        @test lines[1] == "Iter,Time,S,Best Fitness"
        @test lines[2] == "1,0.000000,0,0.000000e+00"

        do_only_save_trace.cache.iteration = 3
        output = @capture_out begin
            GlobalOptimization.trace(do_only_save_trace, false)
        end
        @test isempty(output)
        lines = readlines(trace_file)
        @test length(lines) == 3
        @test lines[1] == "Iter,Time,S,Best Fitness"
        @test lines[2] == "1,0.000000,0,0.000000e+00"
        @test lines[3] == "3,0.000000,0,0.000000e+00"
        rm(trace_file)
    end

    # Test that errors are thrown if AbstractOptimizer trace methods are not implemented
    struct NotImplementedOptimizer <: GlobalOptimization.AbstractOptimizer end

    nio = NotImplementedOptimizer()
    vals = (Val{:detailed}(), Val{:all}())
    for val in vals
        @test_throws ArgumentError GlobalOptimization.get_show_trace_elements(nio, val)
        @test_throws ArgumentError GlobalOptimization.get_save_trace_elements(nio, val)
    end
end

@testset showtiming = true "Options" begin
    # Test GeneralOptions constructors and accessors
    go_tt = GlobalOptimization.GeneralOptions(
        GlobalOptimization.GlobalOptimizationTrace(
            Val(true),
            Val(false),
            "trace.txt",
            GlobalOptimization.TraceMinimal(1),
        ), Val(true), 0.0, 10.0, 100, 1e-4, 2.0, 5
    )
    @test GlobalOptimization.get_max_time(go_tt) == 10.0
    @test GlobalOptimization.get_min_cost(go_tt) == 0.0
    @test GlobalOptimization.get_function_value_check(go_tt) isa Val{true}
    @test GlobalOptimization.get_max_iterations(go_tt) == 100
    @test GlobalOptimization.get_function_tolerance(go_tt) == 1e-4
    @test GlobalOptimization.get_max_stall_time(go_tt) == 2.0
    @test GlobalOptimization.get_max_stall_iterations(go_tt) == 5

    go_tf = GlobalOptimization.GeneralOptions(
        GlobalOptimization.GlobalOptimizationTrace(
            Val(true),
            Val(false),
            "trace.txt",
            GlobalOptimization.TraceMinimal(1),
        ), Val(false), 0.0, 10.0, 100, 1e-4, 2.0, 5
    )
    @test GlobalOptimization.get_max_time(go_tf) == 10.0
    @test GlobalOptimization.get_min_cost(go_tf) == 0.0
    @test GlobalOptimization.get_function_value_check(go_tf) isa Val{false}
    @test GlobalOptimization.get_max_iterations(go_tf) == 100
    @test GlobalOptimization.get_function_tolerance(go_tf) == 1e-4
    @test GlobalOptimization.get_max_stall_time(go_tf) == 2.0
    @test GlobalOptimization.get_max_stall_iterations(go_tf) == 5

    # Define a dummy algorithm-specific options type
    struct DummyAlgoOpts{GO} <: GlobalOptimization.AbstractAlgorithmSpecificOptions
        general::GO
    end
    dummy = DummyAlgoOpts(go_tf)

    # Test get_general and delegation methods
    @test GlobalOptimization.get_max_time(dummy) == 10.0
    @test GlobalOptimization.get_min_cost(dummy) == 0.0
    @test GlobalOptimization.get_function_value_check(dummy) isa Val{false}
    @test GlobalOptimization.get_max_iterations(dummy) == 100
    @test GlobalOptimization.get_function_tolerance(dummy) == 1e-4
    @test GlobalOptimization.get_max_stall_time(dummy) == 2.0
    @test GlobalOptimization.get_max_stall_iterations(dummy) == 5
end

@testset showtiming = true "Optimizers" begin
    # Define a dummy optimizer subtype
    struct MissingMethodOptimizer <: GlobalOptimization.AbstractOptimizer end

    # Test default optimize! throws NotImplementedError
    opt = MissingMethodOptimizer()
    methods = [
        optimize!,
        GlobalOptimization.initialize!,
        GlobalOptimization.step!,
        GlobalOptimization.get_best_fitness,
        GlobalOptimization.get_best_candidate,
    ]
    for m in methods
        fun = @eval $(m)
        @test_throws ArgumentError fun(opt)
        try
            fun(opt)
        catch e
            @test isa(e, ArgumentError)
            @test occursin("MissingMethodOptimizer", e.msg)
        end
    end

    # Test stall check
    dummy_opt = DummyOptimizer(
        DummyOptimizerOptions(
            GlobalOptimization.GeneralOptions(
                GlobalOptimization.GlobalOptimizationTrace(
                    Val(false),
                    Val(false),
                    "trace.txt",
                    GlobalOptimization.TraceMinimal(1),
                ),
                Val(true),
                0.0,
                10.0,
                100,
                1e-4,
                2.0,
                5,
            )
        ),
        GlobalOptimization.MinimalOptimizerCache{Float64}(),
    )
    GlobalOptimization.initialize!(dummy_opt.cache, 1e-8)

    # Force stall
    dummy_opt.cache.stall_value = 1e-8
    GlobalOptimization.handle_stall!(dummy_opt)
    @test dummy_opt.cache.stall_iteration == 1
    GlobalOptimization.handle_stall!(dummy_opt)
    @test dummy_opt.cache.stall_iteration == 2

    # Test no stall
    dummy_opt.cache.stall_value = 2e-4
    GlobalOptimization.handle_stall!(dummy_opt)
    @test dummy_opt.cache.stall_iteration == 0
    @test dummy_opt.cache.stall_value == 0.0

    # Test minimum cost stopping criterion
    @test GlobalOptimization.check_stopping_criteria(dummy_opt) == GlobalOptimization.MINIMUM_COST_ACHIEVED

    # Test maximum time stopping criterion
    dummy_opt2 = DummyOptimizer(
        DummyOptimizerOptions(
            GlobalOptimization.GeneralOptions(
                GlobalOptimization.GlobalOptimizationTrace(
                    Val(false),
                    Val(false),
                    "trace.txt",
                    GlobalOptimization.TraceMinimal(1),
                ),
                Val(true),
                -Inf,
                0.0,
                100,
                1e-4,
                Inf,
                5,
            )
        ),
        GlobalOptimization.MinimalOptimizerCache{Float64}(),
    )
    GlobalOptimization.initialize!(dummy_opt2.cache, 1e-8)
    @test GlobalOptimization.check_stopping_criteria(dummy_opt2) == GlobalOptimization.MAXIMUM_TIME_EXCEEDED

    # Test maximum iterations stopping criterion
    dummy_opt3 = DummyOptimizer(
        DummyOptimizerOptions(
            GlobalOptimization.GeneralOptions(
                GlobalOptimization.GlobalOptimizationTrace(
                    Val(false),
                    Val(false),
                    "trace.txt",
                    GlobalOptimization.TraceMinimal(1),
                ),
                Val(true),
                -Inf,
                Inf,
                1,
                1e-4,
                Inf,
                5,
            )
        ),
        GlobalOptimization.MinimalOptimizerCache{Float64}(),
    )
    GlobalOptimization.initialize!(dummy_opt3.cache, 1e-8)
    dummy_opt3.cache.iteration = 1
    @test GlobalOptimization.check_stopping_criteria(dummy_opt3) == GlobalOptimization.MAXIMUM_ITERATIONS_EXCEEDED

    # Test maximum stall iterations exceeded
    dummy_opt4 = DummyOptimizer(
        DummyOptimizerOptions(
            GlobalOptimization.GeneralOptions(
                GlobalOptimization.GlobalOptimizationTrace(
                    Val(false),
                    Val(false),
                    "trace.txt",
                    GlobalOptimization.TraceMinimal(1),
                ),
                Val(true),
                -Inf,
                Inf,
                100,
                1e-4,
                Inf,
                1,
            )
        ),
        GlobalOptimization.MinimalOptimizerCache{Float64}(),
    )
    GlobalOptimization.initialize!(dummy_opt4.cache, 1e-8)
    dummy_opt4.cache.stall_iteration = 1
    @test GlobalOptimization.check_stopping_criteria(dummy_opt4) == GlobalOptimization.MAXIMUM_STALL_ITERATIONS_EXCEEDED

    # Test maximum stall time exceeded
    dummy_opt5 = DummyOptimizer(
        DummyOptimizerOptions(
            GlobalOptimization.GeneralOptions(
                GlobalOptimization.GlobalOptimizationTrace(
                    Val(false),
                    Val(false),
                    "trace.txt",
                    GlobalOptimization.TraceMinimal(1),
                ),
                Val(true),
                -Inf,
                Inf,
                100,
                1e-4,
                0.0,
                100,
            )
        ),
        GlobalOptimization.MinimalOptimizerCache{Float64}(),
    )
    GlobalOptimization.initialize!(dummy_opt5.cache, 1e-8)
    @test GlobalOptimization.check_stopping_criteria(dummy_opt5) == GlobalOptimization.MAXIMUM_STALL_TIME_EXCEEDED

    # Check no stopping criteria
    dummy_opt6 = DummyOptimizer(
        DummyOptimizerOptions(
            GlobalOptimization.GeneralOptions(
                GlobalOptimization.GlobalOptimizationTrace(
                    Val(false),
                    Val(false),
                    "trace.txt",
                    GlobalOptimization.TraceMinimal(1),
                ),
                Val(true),
                -Inf,
                Inf,
                100,
                Inf,
                Inf,
                100,
            )
        ),
        GlobalOptimization.MinimalOptimizerCache{Float64}(),
    )
    GlobalOptimization.initialize!(dummy_opt6.cache, 1e-8)
    @test GlobalOptimization.check_stopping_criteria(dummy_opt6) == GlobalOptimization.IN_PROGRESS
end

@testset showtiming = true "Candidate" begin
    # Define a concrete candidate type
    mutable struct DummyCandidate <: GlobalOptimization.AbstractCandidate{Float64}
        candidate::Vector{Float64}
        candidate_fitness::Float64
    end

    # Test candidate and fitness accessors
    dc = DummyCandidate([1.0, 2.0], 3.4)
    @test GlobalOptimization.candidate(dc) == [1.0, 2.0]
    @test GlobalOptimization.fitness(dc) == 3.4

    # Test set_fitness!
    GlobalOptimization.set_fitness!(dc, 5.6)
    @test GlobalOptimization.fitness(dc) == 5.6

    # Test check_fitness! with Val(true) on finite fitness
    @test GlobalOptimization.check_fitness!(dc, Val{true}()) === nothing

    # Test check_fitness! with Val(false) on infinite fitness does nothing
    GlobalOptimization.set_fitness!(dc, Inf)
    @test GlobalOptimization.check_fitness!(dc, Val{false}()) === nothing

    # Test check_fitness! with Val(true) on invalid fitness throws error
    @test_throws ErrorException GlobalOptimization.check_fitness!(dc, Val{true}())

    GlobalOptimization.set_fitness!(dc, 2.2)
    @test GlobalOptimization.check_fitness!(dc, Val(true)) === nothing
    @test GlobalOptimization.check_fitness!(dc, Val(false)) === nothing

    GlobalOptimization.set_fitness!(dc, -Inf)
    @test_throws ErrorException GlobalOptimization.check_fitness!(dc, Val(true))
    @test GlobalOptimization.check_fitness!(dc, Val(false)) === nothing
end

@testset showtiming = true "Population" begin
    # Create a sample population
    cand = [[1.0, 2.0], [3.0, 4.0]]
    fit = [0.0, 0.0]
    pop = SimplePopulation(cand, copy(fit))

    # Test length, size, and indexing
    @test length(pop) == 2
    @test size(pop) == (2,)
    @test collect(eachindex(pop)) == [1, 2]

    # Test candidates and fitness accessors
    @test GlobalOptimization.candidates(pop) === pop.candidates
    @test GlobalOptimization.candidates(pop, 2) == [3.0, 4.0]
    @test GlobalOptimization.fitness(pop) === pop.candidates_fitness
    @test GlobalOptimization.fitness(pop, 1) == 0.0

    # Test set_fitness! with a vector
    GlobalOptimization.set_fitness!(pop, [5.0, 6.0])
    @test pop.candidates_fitness == [5.0, 6.0]

    # Test set_fitness! with a scalar index
    GlobalOptimization.set_fitness!(pop, 7.5, 1)
    @test pop.candidates_fitness[1] == 7.5

    # Test dimension mismatch error for set_fitness!
    @test_throws DimensionMismatch GlobalOptimization.set_fitness!(pop, [1.0])

    # Test check_fitness! with Val(false) does nothing
    pop.candidates_fitness .= [1.0, 2.0]
    @test GlobalOptimization.check_fitness!(pop, Val(false)) === nothing

    # Test check_fitness! with Val(true) errors on invalid fitness
    pop.candidates_fitness[2] = Inf
    @test_throws ErrorException GlobalOptimization.check_fitness!(pop, Val(true))
end

@testset showtiming = true "Population Initialization" begin
    using Random

    # Test UniformInitialization
    min_vec = [0.0, 1.0]
    max_vec = [1.0, 2.0]
    popvec = [zeros(2) for _ in 1:5]
    uinit = GlobalOptimization.UniformInitialization()
    GlobalOptimization.initialize_population_vector!(popvec, min_vec, max_vec, uinit)
    for vec in popvec
        for j in eachindex(vec)
            @test vec[j] ≥ min_vec[j] && vec[j] ≤ max_vec[j]
        end
    end

    # Test LatinHypercubeInitialization default constructor
    lhcd = GlobalOptimization.LatinHypercubeInitialization()
    @test lhcd.gens == 10
    @test lhcd.pop_size == 100
    @test isa(lhcd.rng, Random.AbstractRNG)

    # Test LatinHypercubeInitialization sampling and bounds
    rng = MersenneTwister(1234)
    lhci = GlobalOptimization.LatinHypercubeInitialization(
        5;
        rng=rng,
        pop_size=5,
        n_tour=1,
        p_tour=0.5,
        inter_sample_weight=1.0,
        periodic_ae=false,
        ae_power=2.0,
    )
    popvec2 = [zeros(3) for _ in 1:5]
    min2 = [0.0, 0.0, 0.0]
    max2 = [1.0, 2.0, 3.0]
    GlobalOptimization.initialize_population_vector!(popvec2, min2, max2, lhci)

    # Check bounds
    for vec in popvec2
        for j in eachindex(vec)
            @test vec[j] ≥ min2[j] && vec[j] ≤ max2[j]
        end
    end

    # Check each dimension yields unique samples
    for j in 1:3
        col = [popvec2[i][j] for i in 1:length(popvec2)]
        @test length(unique(col)) == length(popvec2)
    end
end

@testset showtiming = true "Evaluator" begin
    # Define cost functions for testing serial and parallel evaluators
    cost(x) = Threads.threadid()

    # Define problem
    prob = GlobalOptimization.OptimizationProblem(cost, [1.0, 1.0], [2.0, 2.0])

    # Construct population and evaluator
    N = 1000
    spop = SimplePopulation([zeros(2) for _ in 1:N], zeros(N))
    tpop = deepcopy(spop)
    ppop = deepcopy(spop)
    seval = GlobalOptimization.SerialBatchEvaluator(prob)
    teval = GlobalOptimization.ThreadedBatchEvaluator(
        prob, Threads.nthreads(), ChunkSplitters.RoundRobin()
    )
    peval = GlobalOptimization.PolyesterBatchEvaluator(prob)

    GlobalOptimization.evaluate!(spop, seval)
    GlobalOptimization.evaluate!(tpop, teval)
    GlobalOptimization.evaluate!(ppop, peval)

    # Check serial evaluator results
    all_one = true
    for i in eachindex(spop.candidates_fitness)
        if spop.candidates_fitness[i] != 1.0
            all_one = false
            break
        end
    end
    @test all_one

    threads_used = [false for _ in 1:Threads.nthreads()]
    for i in eachindex(tpop.candidates_fitness)
        thread_id = Int(tpop.candidates_fitness[i])
        threads_used[thread_id] = true
        if all(threads_used)
            break
        end
    end
    @test all(threads_used)

    threads_used = [false for _ in 1:Threads.nthreads()]
    for i in eachindex(ppop.candidates_fitness)
        thread_id = Int(ppop.candidates_fitness[i])
        threads_used[thread_id] = true
        if all(threads_used)
            break
        end
    end
    @test all(threads_used)
end
