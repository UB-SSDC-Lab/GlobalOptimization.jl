using GlobalOptimization, Test

# Define objective function
@inline function layeb_1(x)
    obj = 0.0
    @fastmath for val in x
        xm1sq = (val - 1)^2
        obj += 10000.0*sqrt(abs(exp(xm1sq) - 1.0))
    end
    return obj
end

# Define problem
N  = 10
ss = GlobalOptimization.ContinuousRectangularSearchSpace(
    [-5.12 for i in 1:N],
    [5.12 for i in 1:N],
)
prob = GlobalOptimization.OptimizationProblem(layeb_1, ss)

# Instantiate PSO
pso = GlobalOptimization.PSO(prob)
tpso = GlobalOptimization.ThreadedPSO(prob)