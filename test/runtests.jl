using GlobalOptimization, Test, SafeTestsets
@time begin
    @time @safetestset "Evaluator" begin
        include("evaluator_test.jl")
    end
end