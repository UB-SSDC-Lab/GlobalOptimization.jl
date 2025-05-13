using GlobalOptimization, Test
@time begin
    @testset showtiming = true "Aqua" begin
        include("aqua.jl")
    end
    @testset showtiming = true "Base" begin
        include("base.jl")
    end
    @testset showtiming = true "PSO" begin
        include("pso_test.jl")
    end
end
