using GlobalOptimization, Test
@time begin
    @testset showtiming = true "Evaluator" begin
        include("evaluator_test.jl")
    end
    @testset showtiming = true "PSO" begin
        include("pso_test.jl")
    end
end
