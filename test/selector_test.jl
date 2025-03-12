
pop_size = 20

@testset "SimpleSelector" begin
    simple = GlobalOptimization.SimpleSelector()
    idxs = GlobalOptimization.select(simple, rand(1:pop_size), pop_size)
    @test idxs == 1:pop_size
end

@testset "RadiusLimitedSelector (Trivial Geography)" begin
    radius = 2
    rls = GlobalOptimization.RadiusLimitedSelector(radius)
    Random.seed!(1234)
    idxs = GlobalOptimization.select(rls, rand(1:pop_size), pop_size)
    @test length(idxs) == 2 * radius + 1
    @test all(x -> 1 ≤ x ≤ pop_size, idxs)

    # Verify that indices form a contiguous block (with wrap-around)
    target = idxs[radius+1]  # the target index chosen inside select
    expected = [mod1(target + j - radius - 1, pop_size) for j in 1:(2*radius + 1)]
    @test collect(idxs) == expected
end
