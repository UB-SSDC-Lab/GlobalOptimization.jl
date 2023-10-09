
using GlobalOptimization
using BenchmarkTools
using Random
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

function layeb_1(x)
    obj = 0.0
    for val in x
        xm1sq = (val - 1)^2
        obj += 10000.0*sqrt(abs(exp(xm1sq) - 1.0))
    end
    return obj
end

# Setup Problem
d = 2
LB = -5.12*ones(d)
UB = 5.12*ones(d)

prob = Problem(layeb_1, LB, UB)
pso  = PSO(prob; numParticles = 100)

# optimize
opts = Options(;display = false, maxStallIters = 25)
res = @benchmark optimize!(_pso, $opts) setup=(_pso = PSO(prob; numParticles = 100))
display(res)
display(optimize!(PSO(prob; numParticles = 100), opts))
