
using GlobalOptimization
#using BenchmarkTools
using Random
#using LoopVectorization
#using PaddedViews
using StaticArrays
#using Profile
#using JET
using Infiltrator
#Random.seed!(1234)

# Schwefel Function
function schaffer(x::AbstractArray{T})::Tuple{T,T} where {T}
    obj = 0.5 + (sin(x[1]^2 + x[2]^2)^2 - 0.5) / (1 + 0.001 * (x[1]^2 + x[2]^2))^2
    return obj, zero(T)
end

function waveDrop(x)
    obj = -(1 + cos(12 * sqrt(x[1]^2 + x[2]^2))) / (0.5 * (x[1]^2 + x[2]^2) + 2.0)
    return obj, 0.0
end

function layeb_1(x)
    obj = zero(eltype(x))
    for val in x
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
    return obj, 0.0
end

function simple_nonlinear_equation(x)
    return SA[x[1] * x[1] - 2.0, x[2] * x[2] - 2.0]
end
function simple_nonlinearleastsquares_equation(x)
    return SA[x[1] * x[1] - x[3], x[2] * x[2] - x[3]]
end

# Setup Problem
N = 5
ss = ContinuousRectangularSearchSpace([-100.0 for i in 1:N], [100.0 for i in 1:N])
prob = OptimizationProblem(rastrigin, ss)

# Instantiate MBH
dist = GlobalOptimization.MBHAdaptiveDistribution{Float64}(
    1000, 5; a=0.97, b=0.1, c=1.0, λhat0=0.01
)
lsgb = GlobalOptimization.LBFGSLocalSearch{Float64}(;
    iters_per_solve=5, percent_decrease_tol=30.0, m=10, max_solve_time=0.1
)
lss = GlobalOptimization.LocalStochasticSearch{Float64}(1e-2, 100)
mbh = GlobalOptimization.PolyesterCMBH(
    prob;
    #hop_distribution=dist,
    #local_search=lss,
    num_hoppers=30,
    max_time=20.0,
    min_cost=1e-20,
    display=true,
    display_interval=10,
)

res = optimize!(mbh);
display(res);

# ======== BENCHMARKING
#sres = @benchmark optimize!(_pso) setup=(_pso = SerialPSO(prob))
#tres = @benchmark optimize!(_pso) setup=(_pso = ThreadedPSO(prob))
#display(sres)
#display(tres)
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
