
@testset "one_clamped_rand" begin
    for _ in 1:100
        v = GlobalOptimization.one_clamped_rand(
            GlobalOptimization.default_mutation_dist
        )
        @test v > 0.0
        @test v ≤ 1.0
    end
end
