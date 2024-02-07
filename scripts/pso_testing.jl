
using GlobalOptimization
using BenchmarkTools
using Random
#using LoopVectorization
#using PaddedViews
#using StaticArrays
#using Profile
#using JET
using Infiltrator
Random.seed!(1234)

# Schwefel Function
function schaffer(x)
    obj = 0.5 + (sin(x[1]^2 + x[2]^2)^2 - 0.5)/(1 + 0.001*(x[1]^2+x[2]^2))^2
    return obj 
end

function waveDrop(x)
    obj = -(1 + cos(12*sqrt(x[1]^2 + x[2]^2)))/(0.5*(x[1]^2 + x[2]^2) + 2.0)
    return obj
end

@inline function layeb_1(x)
    obj = 0.0
    @fastmath for val in x
        xm1sq = (val - 1)^2
        obj += 10000.0*sqrt(abs(exp(xm1sq) - 1.0))
    end
    return obj
end

# Setup Problem
N = 10
ss = ContinuousRectangularSearchSpace(
    [-1.0 for i in 1:N],
    [1.0 for i in 1:N],
)
prob = OptimizationProblem(layeb_1, ss)

# Instantiate PSO
#spso = SerialPSO(prob)
#bmbh = GlobalOptimization.AdaptiveMBH(prob; display = true, display_interval = 10000)
#tpso = ThreadedPSO(prob)

#res = optimize!(spso)
#res = optimize!(bmbh)
#pso1 = StaticPSO(prob; numParticles = 100)
#pso2 = deepcopy(pso1)

# ======== BENCHMARKING
sres = @benchmark optimize!(_pso) setup=(_pso = SerialPSO(prob))
tres = @benchmark optimize!(_pso) setup=(_pso = ThreadedPSO(prob))
pres = @benchmark optimize!(_pso) setup=(_pso = PolyesterPSO(prob))
display(sres)
display(tres)
display(pres)
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

