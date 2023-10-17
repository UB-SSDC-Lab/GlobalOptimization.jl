using GlobalOptimization, Test, SafeTestsets
@time begin
    @safetestset "Evaluator" begin; include("evaluator_test.jl"); end
    @safetestset "PSO" begin; include("pso_test.jl"); end
end