using GlobalOptimization, Test
@time begin
    @testset showtiming = true "Aqua" begin
        include("aqua.jl")
    end
    @testset showtiming = true "Base" begin
        include("base.jl")
    end
    @testset showtiming = true "PSO" begin
        include("pso.jl")
    end
    @testset showtiming = true "DE" begin
        include("de.jl")
    end
    @testset showtiming = true "MBH" begin
        include("mbh.jl")
    end
end
