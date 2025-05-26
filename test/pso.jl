using GlobalOptimization, Random, Test
using Suppressor

# Define objective function
@inline function layeb_1(x)
    obj = 0.0
    @fastmath for val in x
        xm1sq = (val - 1)^2
        obj += 10000.0 * sqrt(abs(exp(xm1sq) - 1.0))
    end
    return obj
end

# Utility test function
function check_swarm_equality(pso1, pso2, pso3)
    sswarm = pso1.swarm
    tswarm = pso2.swarm
    pswarm = pso3.swarm
    for i in eachindex(sswarm.candidates)
        sc = sswarm.candidates[i]
        sbc = sswarm.best_candidates[i]
        tc = tswarm.candidates[i]
        tbc = tswarm.best_candidates[i]
        pc = pswarm.candidates[i]
        pbc = pswarm.best_candidates[i]
        @test sswarm.candidates_fitness[i] == tswarm.candidates_fitness[i]
        @test sswarm.candidates_fitness[i] == pswarm.candidates_fitness[i]
        @test sswarm.best_candidates_fitness[i] == tswarm.best_candidates_fitness[i]
        @test sswarm.best_candidates_fitness[i] == pswarm.best_candidates_fitness[i]
        @test sc == tc
        @test sc == pc
        @test sbc == tbc
        @test sbc == pbc
    end
end

# Define problem
N = 3
ss = ContinuousRectangularSearchSpace([-5.12 for i in 1:N], [5.12 for i in 1:N])
prob = OptimizationProblem(layeb_1, ss)

# Instantiate PSO
spso = PSO(prob; eval_method=SerialFunctionEvaluation())
tpso = PSO(prob; eval_method=ThreadedFunctionEvaluation())
ppso = PSO(prob; eval_method=PolyesterFunctionEvaluation())

# Check optimization is same
Random.seed!(1234)
sres = GlobalOptimization.optimize!(spso)
Random.seed!(1234)
tres = GlobalOptimization.optimize!(tpso)
Random.seed!(1234)
pres = GlobalOptimization.optimize!(ppso)

if VERSION >= v"1.10" # VERSION < v1.10 has bug that results different rng state when using Threads.@threads
    check_swarm_equality(spso, tpso, ppso)
    @test sres.exitFlag == tres.exitFlag
    @test sres.exitFlag == pres.exitFlag
    @test sres.iters == tres.iters
    @test sres.iters == pres.iters
end
@test sres.fbest ≈ tres.fbest atol = 1e-6
@test sres.xbest ≈ tres.xbest atol = 1e-6

# Check for correct answer
@test sres.fbest ≈ 0.0 atol = 1e-6
@test sres.xbest ≈ fill(1.0, N) atol = 1e-6

# Test with CSRNVelocityUpdate scheme
N = 10
ss2 = ContinuousRectangularSearchSpace([-5.12 for i in 1:N], [5.12 for i in 1:N])
prob2 = OptimizationProblem(layeb_1, ss2)
csrn_pso = PSO(prob2; velocity_update=CSRNVelocityUpdate())
Random.seed!(1234)
sres2 = optimize!(csrn_pso)
@test sres2.fbest ≈ 0.0 atol = 1e-6
@test sres2.xbest ≈ fill(1.0, N) atol = 1e-6

# Test tracing
pso = PSO(prob; show_trace=Val(true), trace_level=TraceDetailed(1))
GlobalOptimization.initialize!(pso)

te = GlobalOptimization.get_show_trace_elements(pso, Val{:detailed}())
@test length(te) == 7

pso2 = PSO(prob; show_trace=Val(true), trace_level=TraceDetailed(1), velocity_update=CSRNVelocityUpdate())
GlobalOptimization.initialize!(pso2)
te2 = GlobalOptimization.get_show_trace_elements(pso2, Val{:detailed}())
@test length(te2) == 5

# Check for expected errors
struct InvalidSearchSpace <: GlobalOptimization.SearchSpace{Float64} end
prob = OptimizationProblem(layeb_1, InvalidSearchSpace())
@test_throws ArgumentError PSO(prob)
