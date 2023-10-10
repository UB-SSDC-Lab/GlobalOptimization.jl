
using GlobalOptimization
using BenchmarkTools
using Random
using StaticArrays
using Profile
#Random.seed!(1234)

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
d = 10
#LB = -5.12*ones(d)
#UB = 5.12*ones(d)
LB = @SVector [-5.12 for i in 1:d]
UB = @SVector [5.12 for i in 1:d]

prob = Problem(layeb_1, LB, UB)
res = optimize!(
    PSO(prob; numParticles = 1000),
    Options(; display = true, useParallel = true, maxStallIters = 50)
)
#pso1 = PSO(prob; numParticles = 100)
#pso2 = deepcopy(pso1)

# optimize
#opts_serial  = Options(;display = false, maxStallIters = 25, useParallel = false)
#opts_threads = Options(;display = false, maxStallIters = 25, useParallel = true)
#res_serial = @benchmark optimize!(_pso, $opts_serial) setup=(_pso = PSO(prob; numParticles = 100))
#res_threads = @benchmark optimize!(_pso, $opts_threads) setup=(_pso = PSO(prob; numParticles = 100))
#display(res_serial)
#display(res_threads)
#display(optimize!(PSO(prob; numParticles = 100), opts))

#res = GlobalOptimization.optimize!(pso1, opts)
#optimize!(pso1, opts)
#Profile.clear_malloc_data()
#optimize!(pso2, opts)


