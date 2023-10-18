using GlobalOptimization, Random, Test

# Define objective function
@inline function layeb_1(x)
    obj = 0.0
    @fastmath for val in x
        xm1sq = (val - 1)^2
        obj += 10000.0*sqrt(abs(exp(xm1sq) - 1.0))
    end
    return obj
end

# Utility test function
function check_swarm_equality(pso1,pso2)
    sswarm = pso1.swarm
    tswarm = pso2.swarm
    for i in eachindex(sswarm.candidates)
        sc = sswarm.candidates[i]
        sbc = sswarm.best_candidates[i]
        tc = tswarm.candidates[i]
        tbc = tswarm.best_candidates[i]
        @test sswarm.candidates_fitness[i] == tswarm.candidates_fitness[i]
        @test sswarm.best_candidates_fitness[i] == tswarm.best_candidates_fitness[i]
        @test sc == tc
        @test sbc == tbc
    end
end

# Define problem
N  = 10
ss = GlobalOptimization.ContinuousRectangularSearchSpace(
    [-5.12 for i in 1:N],
    [5.12 for i in 1:N],
)
prob = GlobalOptimization.OptimizationProblem(layeb_1, ss)

# Instantiate PSO
spso = GlobalOptimization.SerialPSO(prob)
tpso = GlobalOptimization.ThreadedPSO(prob)

# Check optimization is same
Random.seed!(1234)
sres = GlobalOptimization.optimize!(spso)
Random.seed!(1234)
tres = GlobalOptimization.optimize!(tpso)

check_swarm_equality(spso,tpso) # THIS SHOULD NOT PASS TESTS!!! WTF?!?
@test sres.fbest == tres.fbest
@test sres.xbest == tres.xbest
@test sres.iters == tres.iters
@test sres.exitFlag == tres.exitFlag

# Check for correct answer
@test sres.fbest ≈ 0.0 atol=1e-6
@test sres.xbest ≈ fill(1.0, N) atol=1e-6