using GlobalOptimization, Test
using Random

#----------------------------------------------------------------------------
# Define a simple test function (Sphere) for use in optimizer tests.
#----------------------------------------------------------------------------
function sphere(x::AbstractVector)
    return sum(x .^ 2)
end

# Use a low-dimensional test problem.
const DIM = 3
const LOWER = fill(-5.0, DIM)
const UPPER = fill(5.0, DIM)
const TOLERANCE = 1e-3

#----------------------------------------------------------------------------
# Run tests
#----------------------------------------------------------------------------
@time begin
    #@testset showtiming=true "Evaluator" begin; include("evaluator_test.jl"); end
    #@testset showtiming=true "PSO" begin; include("pso_test.jl"); end
    @testset showtiming=true "Utility Function Tests" begin; include("utils_test.jl"); end
    @testset showtiming=true "Selector Tests" begin; include("selector_test.jl"); end
    @testset showtiming=true "DE Tests" begin; include("de_test.jl"); end
end
