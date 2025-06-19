
using GlobalOptimization
using BenchmarkTools
using Random
using Distributions
#using BlackBoxOptim
#using LoopVectorization
#using PaddedViews
#using StaticArrays
using Profile
#using JET
using Infiltrator
#Random.seed!(1234)

# Schwefel Function
function schaffer(x)
    obj = 0.5 + (sin(x[1]^2 + x[2]^2)^2 - 0.5) / (1 + 0.001 * (x[1]^2 + x[2]^2))^2
    return obj, 0.0
end

function waveDrop(x)
    obj = -(1 + cos(12 * sqrt(x[1]^2 + x[2]^2))) / (0.5 * (x[1]^2 + x[2]^2) + 2.0)
    return obj, 0.0
end

@inline function layeb_1(x)
    obj = 0.0
    @fastmath for val in x
        xm1sq = (val - 1)^2
        obj += 10000.0 * sqrt(abs(exp(xm1sq) - 1.0))
    end
    return obj, 0.0
end

function rastrigin(x; A=10)
    obj = A * length(x)
    for val in x
        obj += val^2 - A * cos(2 * pi * val)
    end
    #sleep(rand()*2e-6)
    return obj
end

# Setup Problem
N = 7
ss = ContinuousRectangularSearchSpace([-100.0 for i in 1:N], [100.0 for i in 1:N])
prob = GlobalOptimization.OptimizationProblem(rastrigin, ss)

# Instantiate DE
mutation_strategy = SelfMutationParameters(
    Rand1();
    #dist=Uniform(0.0,1.0),
    sel=GlobalOptimization.RadiusLimitedSelector(8),
)
crossover_strategy = SelfBinomialCrossoverParameters(;
    dist=Uniform(0.0, 1.0),
    #transform = GlobalOptimization.NoTransformation()
    transform = GlobalOptimization.UncorrelatedCovarianceTransformation(0.5, 0.8, N; ps = 1.0),
    #transform = GlobalOptimization.CovarianceTransformation(0.1, 0.5, N)
)

de = DE(
    prob;
    eval_method=SerialFunctionEvaluation(),
    num_candidates=100,
    max_iterations=1000,
    max_stall_iterations=100,
    mutation_params=mutation_strategy,
    crossover_params=crossover_strategy,
    show_trace=Val(false),
)

res = optimize!(de)
#iters_per_solve = map(i->optimize!(deepcopy(de)).iters, 1:100);

# bb_res = bboptimize(
#     rastrigin;
#     Method = :adaptive_de_rand_1_bin_radiuslimited,
#     PopulationSize = 100,
#     SearchRange = (-5.0, 5.0),
#     NumDimensions = N,
#     TraceMode = :compact,
#     TraceInterval = 0.001,
#     #MaxSteps = 1000,
# )

# Instantiate PSO
# spso = SerialPSO(prob; max_time = 20.0)
#tpso = ThreadedPSO(prob; max_time = 20.0)
#ppso = PolyesterPSO(prob; max_time = 20.0)

# #res = optimize!(spso)
# res = optimize!(spso); display(res)
# res = optimize!(tpso); display(res)
# res = optimize!(ppso); display(res)

# ======== BENCHMARKING
# sres = @benchmark optimize!(_pso) setup=(_pso = SerialPSO(prob; max_iterations=20))
# tres = @benchmark optimize!(_pso) setup=(_pso = ThreadedPSO(prob; max_iterations=20))
# pres = @benchmark optimize!(_pso) setup=(_pso = PolyesterPSO(prob; max_iterations=20))
# display(sres)
# display(tres)
# display(pres)
# GlobalOptimization.initialize!(spso)
# GlobalOptimization.update_velocity!(spso.swarm, spso.cache, 10, 0.5, 0.49, 0.49)
# GlobalOptimization.step!(spso.swarm)
# GlobalOptimization.enforce_bounds!(spso.swarm, ss)

# go = @benchmark GlobalOptimization.evaluate_fitness!($spso.swarm, $spso.evaluator)
# display(go)

# ======== ALLOCATION TRACKING
# pso1 = SerialPSO(prob)
# pso2 = SerialPSO(prob)
# optimize!(pso1)
# Profile.clear_malloc_data()
# optimize!(pso2)

# ======== TYPES
#@report_call GlobalOptimization.optimize!(spso)
#report_package(GlobalOptimization)
