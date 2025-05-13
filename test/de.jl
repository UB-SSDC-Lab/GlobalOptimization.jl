using GlobalOptimization, Test
using Distributions

@testset showtiming=true "Utilities" begin

    # Test default mutation distribution
    dmut = GlobalOptimization.default_mutation_dist
    @test isa(dmut, MixtureModel{Univariate,Continuous})

    # Test default binomial crossover distribution
    dbin = GlobalOptimization.default_binomial_crossover_dist
    @test isa(dbin, MixtureModel{Univariate,Continuous})

    # Test adaptation strategy types
    @test isa(GlobalOptimization.RandomAdaptation(), GlobalOptimization.AbstractAdaptationStrategy)
    @test isa(GlobalOptimization.NoAdaptation(), GlobalOptimization.AbstractAdaptationStrategy)

    # Test SimpleSelector
    s = GlobalOptimization.SimpleSelector()
    @test GlobalOptimization.initialize!(s, 5) === nothing
    sel = GlobalOptimization.select(s, 3, 5)
    @test sel == 1:5

    # Test RadiusLimitedSelector
    rs = GlobalOptimization.RadiusLimitedSelector(2)
    @test length(rs.idxs) == 5
    @test GlobalOptimization.initialize!(rs, 10) === nothing
    sel2 = GlobalOptimization.select(rs, 5, 10)
    @test collect(sel2) == UInt16[3, 4, 5, 6, 7]
    sel3 = GlobalOptimization.select(rs, 1, 10)
    @test collect(sel3) == UInt16[9, 10, 1, 2, 3]
    sel4 = GlobalOptimization.select(rs, 9, 10)
    @test collect(sel4) == UInt16[7, 8, 9, 10, 1]

    # Test RandomSubsetSelector
    rss = GlobalOptimization.RandomSubsetSelector(3)
    @test_throws ErrorException GlobalOptimization.initialize!(rss, 2)
    GlobalOptimization.initialize!(rss, 5)
    @test length(rss.idxs) == 5
    subs = GlobalOptimization.select(rss, 1, 5)
    @test length(subs) == 3
    @test all(x in UInt16(1):UInt16(5) for x in subs)
    @test length(unique(subs)) == 3

    # Test one_clamped_rand with default mutation distribution
    min_val = Inf
    max_val = -Inf
    for i in 1:1000
        val = GlobalOptimization.one_clamped_rand(dmut)
        min_val = min(min_val, val)
        max_val = max(max_val, val)
    end
    @test 0.0 < min_val <= 1.0
    @test 0.0 < max_val <= 1.0

    # Test one_clamped_rand error with always-zero distribution
    struct AlwaysZeroDist <: Distribution{Univariate, Continuous} end
    Base.rand(::AlwaysZeroDist) = 0.0
    @test_throws ArgumentError GlobalOptimization.one_clamped_rand(AlwaysZeroDist())
end
