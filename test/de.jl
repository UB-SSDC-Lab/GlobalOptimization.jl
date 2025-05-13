using GlobalOptimization, Test
using Distributions
using StaticArrays

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

@testset showtiming=true "Mutation" begin
    # Define a constant distribution for testing
    struct ConstDist{T} <: Distribution{Univariate, Continuous}
        val::T
    end
    Base.rand(d::ConstDist{T}) where {T} = d.val

    # Test get_parameters for each mutation strategy
    c = 0.3
    dist = ConstDist(c)
    # Rand1
    p = GlobalOptimization.get_parameters(GlobalOptimization.Rand1(), dist)
    @test isa(p, SVector{4,Float64})
    @test p == SVector(0.0, 1.0, c, 0.0)
    # Rand2
    p = GlobalOptimization.get_parameters(GlobalOptimization.Rand2(), dist)
    @test p == SVector(0.0, 1.0, c, c)
    # Best1
    p = GlobalOptimization.get_parameters(GlobalOptimization.Best1(), dist)
    @test p == SVector(1.0, 0.0, c, 0.0)
    # Best2
    p = GlobalOptimization.get_parameters(GlobalOptimization.Best2(), dist)
    @test p == SVector(1.0, 0.0, c, c)
    # CurrentToBest1
    p = GlobalOptimization.get_parameters(GlobalOptimization.CurrentToBest1(), dist)
    @test p == SVector(c, 0.0, c, 0.0)
    # CurrentToBest2
    p = GlobalOptimization.get_parameters(GlobalOptimization.CurrentToBest2(), dist)
    @test p == SVector(c, 0.0, c, c)
    # CurrentToRand1
    p = GlobalOptimization.get_parameters(GlobalOptimization.CurrentToRand1(), dist)
    @test p == SVector(0.0, c, c, 0.0)
    # CurrentToRand2
    p = GlobalOptimization.get_parameters(GlobalOptimization.CurrentToRand2(), dist)
    @test p == SVector(0.0, c, c, c)
    # RandToBest1
    p = GlobalOptimization.get_parameters(GlobalOptimization.RandToBest1(), dist)
    @test p == SVector(c, 1.0, c, 0.0)
    # RandToBest2
    p = GlobalOptimization.get_parameters(GlobalOptimization.RandToBest2(), dist)
    @test p == SVector(c, 1.0, c, c)
    # Unified
    p = GlobalOptimization.get_parameters(GlobalOptimization.Unified(), dist)
    @test p == SVector(c, c, c, c)

    # Test MutationParameters constructors
    mp = GlobalOptimization.MutationParameters(0.5, 0.6, 0.7, 0.8)
    @test mp.F1 == 0.5 && mp.F2 == 0.6 && mp.F3 == 0.7 && mp.F4 == 0.8
    @test isa(mp.sel, GlobalOptimization.SimpleSelector)
    @test mp.dist === nothing

    mp2 = GlobalOptimization.MutationParameters(GlobalOptimization.Rand1(); dist=dist)
    @test mp2.F1 == 0.0 && mp2.F2 == 1.0 && mp2.F3 == c && mp2.F4 == 0.0
    @test isa(mp2.sel, GlobalOptimization.SimpleSelector)
    @test mp2.dist === dist

    # Test initialize! and adapt! for MutationParameters
    # NoAdaptation: should do nothing to F fields
    GlobalOptimization.initialize!(mp, 4)
    @test mp.F3 == 0.7
    GlobalOptimization.adapt!(mp, [true], false)
    @test mp.F3 == 0.7
    GlobalOptimization.adapt!(mp, [false], false)
    @test mp.F3 == 0.7

    # RandomAdaptation: update when global_best_improved=false
    mp2.dist = ConstDist(0.9)
    GlobalOptimization.initialize!(mp2, 3)
    @test mp2.F3 == 0.9
    mp2.dist = ConstDist(0.8)
    GlobalOptimization.adapt!(mp2, [true], false)
    @test mp2.F3 == 0.8

    # No update when global_best_improved=true
    mp2.F3 = 0.123
    GlobalOptimization.adapt!(mp2, [false], true)
    @test mp2.F3 == 0.123

    # Test SelfMutationParameters
    sp = GlobalOptimization.SelfMutationParameters(GlobalOptimization.Rand2(); dist=ConstDist(0.4))
    @test isempty(sp.Fs)
    GlobalOptimization.initialize!(sp, 3)
    @test length(sp.Fs) == 3
    @test all(sp.Fs[i] == SVector(0.0, 1.0, 0.4, 0.4) for i in 1:3)

    # Test adapt!: only updates when improved[i] == false
    sp2 = GlobalOptimization.SelfMutationParameters(
        GlobalOptimization.Rand2();
        dist=Uniform(0.1, 0.9),
    )
    GlobalOptimization.initialize!(sp2, 3)
    original_Fs = deepcopy(sp2.Fs)

    improved = [true, false, true]
    GlobalOptimization.adapt!(sp2, improved, false)
    @test sp2.Fs[1] == original_Fs[1]
    @test sp2.Fs[2] != original_Fs[2]
    @test sp2.Fs[3] == original_Fs[3]

    # Test get_best_candidate_in_selection
    pop = GlobalOptimization.DEPopulation(4, 3)
    pop.current_generation.candidates .= [
        SVector(0.1, 0.2, 0.3),
        SVector(0.4, 0.5, 0.6),
        SVector(0.7, 0.8, 0.9),
        SVector(1.0, 1.1, 1.2)
    ]
    pop.current_generation.candidates_fitness .= [0.5, 0.2, 0.8, 0.1]
    best = GlobalOptimization.get_best_candidate_in_selection(pop, [1, 3, 4])
    @test best == SVector(1.0, 1.1, 1.2)

    # Test mutate! no-op behavior with zero parameters
    pop = GlobalOptimization.DEPopulation(5, 2)
    # Initialize current generation candidates and mutants
    for i in 1:5
        pop.current_generation.candidates[i] .= SVector(1.0, 2.0)
        pop.current_generation.candidates_fitness[i] = 0.0
        pop.mutants.candidates[i] = pop.current_generation.candidates[i]
    end
    mp_zero = GlobalOptimization.MutationParameters(0.0, 0.0, 0.0, 0.0; sel=GlobalOptimization.SimpleSelector())
    GlobalOptimization.initialize!(mp_zero, length(pop.current_generation))
    GlobalOptimization.mutate!(pop, mp_zero)
    @test pop.mutants.candidates == pop.current_generation.candidates

    # Test mutate! produces changes
    # Note: This isn't super thorough, but it should be enough to catch any major issues
    pop2 = GlobalOptimization.DEPopulation(10, 4)
    mp_f2 = GlobalOptimization.MutationParameters(1.0, 1.0, 1.0, 1.0)
    GlobalOptimization.initialize!(mp_f2, length(pop.current_generation))
    # Reset current and mutants
    for i in 1:10
        pop2.current_generation.candidates[i] .= randn(4)
        pop2.mutants.candidates[i] .= pop2.current_generation.candidates[i]
    end
    GlobalOptimization.mutate!(pop2, mp_f2)

    # Since F[i] != 0.0 ∀ i ∈ 1:4, all mutants should be different from the current generation
    for i in 1:10
        @test pop2.mutants.candidates[i] != pop2.current_generation.candidates[i]
    end

end
