using GlobalOptimization, Test
using Distributions
using StaticArrays
using Random

@testset showtiming=true "Utilities" begin

    # Test default mutation distribution
    dmut = GlobalOptimization.default_mutation_dist
    @test isa(dmut, MixtureModel{Univariate,Continuous})

    # Test default binomial crossover distribution
    dbin = GlobalOptimization.default_binomial_crossover_dist
    @test isa(dbin, MixtureModel{Univariate,Continuous})

    # Test adaptation strategy types
    @test isa(
        GlobalOptimization.RandomAdaptation(), GlobalOptimization.AbstractAdaptationStrategy
    )
    @test isa(
        GlobalOptimization.NoAdaptation(), GlobalOptimization.AbstractAdaptationStrategy
    )

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
    struct AlwaysZeroDist <: Distribution{Univariate,Continuous} end
    Base.rand(::AlwaysZeroDist) = 0.0
    @test_throws ArgumentError GlobalOptimization.one_clamped_rand(AlwaysZeroDist())
end

@testset showtiming=true "Population" begin

    # Test DEBasePopulation constructor
    bp = GlobalOptimization.DEBasePopulation{Float64}(3, 2)
    @test length(bp.candidates) == 3
    @test all(length(v) == 2 for v in bp.candidates)
    @test bp.candidates_fitness == zeros(3)

    # Test DEPopulation default constructor
    pop = GlobalOptimization.DEPopulation(4, 3)
    @test length(pop) == 4
    @test length(pop.current_generation.candidates) == 4
    @test length(pop.improved) == 4

    # Test error conditions for DEPopulation
    @test_throws ArgumentError GlobalOptimization.DEPopulation(0, 2)
    @test_throws ArgumentError GlobalOptimization.DEPopulation(2, 0)

    # Test initialize! populates candidates within bounds
    lb = [-1.0, 0.0, 2.0]
    ub = [1.0, 2.0, 3.0]
    ss = GlobalOptimization.ContinuousRectangularSearchSpace(lb, ub)
    init = GlobalOptimization.UniformInitialization()
    GlobalOptimization.initialize!(pop, init, ss)
    for v in pop.current_generation.candidates
        for j in eachindex(v)
            @test v[j] ≥ lb[j] && v[j] ≤ ub[j]
        end
    end

    # Test selection! replaces improved candidates and sets flags
    pop2 = GlobalOptimization.DEPopulation(3, 2)
    pop2.current_generation.candidates .= [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    pop2.current_generation.candidates_fitness .= [3.0, 1.0, 2.0]
    pop2.mutants.candidates .= [[1.0, 1.0], [0.0, 0.0], [3.0, 3.0]]
    pop2.mutants.candidates_fitness .= [2.0, 2.0, 1.0]
    GlobalOptimization.selection!(pop2)
    # Check replacements
    @test pop2.current_generation.candidates[1] == [1.0, 1.0]
    @test pop2.current_generation.candidates[2] == [1.0, 1.0]
    @test pop2.current_generation.candidates[3] == [3.0, 3.0]
    @test pop2.improved == [true, false, true]
end

@testset showtiming=true "Mutation" begin
    # Define a constant distribution for testing
    struct ConstDist{T} <: Distribution{Univariate,Continuous}
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
    sp = GlobalOptimization.SelfMutationParameters(
        GlobalOptimization.Rand2(); dist=ConstDist(0.4)
    )
    @test isempty(sp.Fs)
    GlobalOptimization.initialize!(sp, 3)
    @test length(sp.Fs) == 3
    @test all(sp.Fs[i] == SVector(0.0, 1.0, 0.4, 0.4) for i in 1:3)

    # Test adapt!: only updates when improved[i] == false
    sp2 = GlobalOptimization.SelfMutationParameters(
        GlobalOptimization.Rand2(); dist=Uniform(0.1, 0.9)
    )
    GlobalOptimization.initialize!(sp2, 3)
    original_Fs = deepcopy(sp2.Fs)

    improved = [true, false, true]
    GlobalOptimization.adapt!(sp2, improved, false)
    @test sp2.Fs[1] == original_Fs[1]
    @test sp2.Fs[2] != original_Fs[2]
    @test sp2.Fs[3] == original_Fs[3]

    # Test get_best_candidate_in_subset
    pop = GlobalOptimization.DEPopulation(4, 3)
    pop.current_generation.candidates .= [
        SVector(0.1, 0.2, 0.3),
        SVector(0.4, 0.5, 0.6),
        SVector(0.7, 0.8, 0.9),
        SVector(1.0, 1.1, 1.2),
    ]
    pop.current_generation.candidates_fitness .= [0.5, 0.2, 0.8, 0.1]
    best = GlobalOptimization.get_best_candidate_in_subset(pop, [1, 3, 4])
    @test best == SVector(1.0, 1.1, 1.2)

    # Test mutate! no-op behavior with zero parameters
    pop = GlobalOptimization.DEPopulation(5, 2)
    # Initialize current generation candidates and mutants
    for i in 1:5
        pop.current_generation.candidates[i] .= SVector(1.0, 2.0)
        pop.current_generation.candidates_fitness[i] = 0.0
        pop.mutants.candidates[i] = pop.current_generation.candidates[i]
    end
    mp_zero = GlobalOptimization.MutationParameters(
        0.0, 0.0, 0.0, 0.0; sel=GlobalOptimization.SimpleSelector()
    )
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

@testset showtiming=true "Crossover" begin

    # Test NoTransformation: to_transformed and from_transformed!
    nt = GlobalOptimization.NoTransformation()
    c = SVector(1.0, 2.0)
    m = SVector(3.0, 4.0)
    ct, mt, flag = GlobalOptimization.to_transformed(nt, c, m)
    @test ct === c && mt === m && flag == false
    mcopy = deepcopy(m)
    GlobalOptimization.from_transformed!(nt, mt, mcopy)
    @test mcopy == m

    # Test CovarianceTransformation constructor errors
    @test_throws ArgumentError GlobalOptimization.CovarianceTransformation(0.0, 0.5, 2)
    @test_throws ArgumentError GlobalOptimization.CovarianceTransformation(0.5, 0.0, 2)

    # Test valid CovarianceTransformation initialization and update
    ct2 = GlobalOptimization.CovarianceTransformation(1.0, 1.0, 2)
    GlobalOptimization.initialize!(ct2, 3)
    @test ct2.idxs == UInt16[1, 2, 3]
    pop = GlobalOptimization.DEPopulation(3, 2)
    # Make all candidates identical so covariance is zero matrix
    for i in 1:3
        pop.current_generation.candidates[i] = SVector(5.0, 6.0)
        pop.current_generation.candidates_fitness[i] = 0.0
        pop.mutants.candidates[i] = SVector(7.0, 8.0)
    end
    GlobalOptimization.update_transformation!(ct2, pop)
    @test any([
        isapprox(ct2.B, [1.0 0.0; 0.0 1.0]; atol=1e-8),
        isapprox(ct2.B, [1.0 0.0; 0.0 -1.0]; atol=1e-8),
        isapprox(ct2.B, [-1.0 0.0; 0.0 1.0]; atol=1e-8),
        isapprox(ct2.B, [-1.0 0.0; 0.0 -1.0]; atol=1e-8),
    ])

    # Test to_transformed always transforms when pb=1.0
    orig_c = pop.current_generation.candidates[1]
    orig_m = pop.mutants.candidates[1]
    ctvec, mtvec, flag2 = GlobalOptimization.to_transformed(ct2, orig_c, orig_m)
    @test flag2 == true
    @test ctvec == transpose(ct2.B)*orig_c
    @test mtvec == transpose(ct2.B)*orig_m
    # Test from_transformed! maps back correctly
    newm = copy(ctvec)
    GlobalOptimization.from_transformed!(ct2, ctvec, newm)
    @test newm == collect(orig_c)

    # Test BinomialCrossoverParameters constructors
    bp1 = GlobalOptimization.BinomialCrossoverParameters(0.25)
    @test bp1.CR == 0.25
    @test bp1.transform isa GlobalOptimization.NoTransformation
    @test bp1.dist === nothing
    bp2 = GlobalOptimization.BinomialCrossoverParameters()
    @test bp2.CR == 0.0
    @test bp2.transform isa GlobalOptimization.NoTransformation
    @test isa(bp2.dist, MixtureModel)

    # Test SelfBinomialCrossoverParameters default
    sbpd = GlobalOptimization.SelfBinomialCrossoverParameters()
    @test sbpd.CRs == Float64[]
    @test sbpd.transform isa GlobalOptimization.NoTransformation
    @test isa(sbpd.dist, MixtureModel)

    # Test initialize! and adapt! for BinomialCrossoverParameters
    bp3 = GlobalOptimization.BinomialCrossoverParameters(; dist=Uniform(0.0, 0.99))
    GlobalOptimization.initialize!(bp3, 2, 3)
    oldCR = bp3.CR
    GlobalOptimization.adapt!(bp3, [false, false], false)
    @test bp3.CR != oldCR
    bp3.CR = 0.123
    GlobalOptimization.adapt!(bp3, [true, true], true)
    @test bp3.CR == 0.123

    # Test initialize! and adapt! for SelfBinomialCrossoverParameters
    sbp = GlobalOptimization.SelfBinomialCrossoverParameters(; dist=Uniform(0.0, 0.99))
    GlobalOptimization.initialize!(sbp, 2, 3)
    @test length(sbp.CRs) == 3
    oldCRs = copy(sbp.CRs)
    improved = [true, false, true]
    GlobalOptimization.adapt!(sbp, improved, false)
    @test sbp.CRs[1] == oldCRs[1]
    @test sbp.CRs[2] != oldCRs[2]
    @test sbp.CRs[3] == oldCRs[3]

    # Test crossover! with CR=1.0 (no change)
    pop2 = GlobalOptimization.DEPopulation(4, 2)
    search_space = GlobalOptimization.ContinuousRectangularSearchSpace(
        [0.0, 0.0], [10.0, 10.0]
    )
    for i in 1:4
        pop2.current_generation.candidates[i] = SVector(1.0, 2.0)
        pop2.mutants.candidates[i] = SVector(3.0, 4.0)
    end
    bp_fixed = GlobalOptimization.BinomialCrossoverParameters(
        1.0; transform=GlobalOptimization.NoTransformation()
    )
    GlobalOptimization.crossover!(pop2, bp_fixed, search_space)
    @test pop2.mutants.candidates == [SVector(3.0, 4.0) for _ in 1:4]

    # Test crossover! with CR=0.0 (some crossover occurs)
    bp0 = GlobalOptimization.BinomialCrossoverParameters(
        0.0; transform=GlobalOptimization.NoTransformation()
    )
    # Reset mutants
    for i in 1:4
        pop2.mutants.candidates[i] = SVector(3.0, 4.0)
    end
    GlobalOptimization.crossover!(pop2, bp0, search_space)
    @test any(pop2.mutants.candidates[i] != SVector(3.0, 4.0) for i in 1:4)
end

@testset showtiming=true "Full DE Optimization" begin

    # Define a simple sphere objective with known minimum at x = 1
    sphere(x) = sum(xx -> (xx - 1.0)^2, x)

    # Problem setup
    N = 5
    ss = GlobalOptimization.ContinuousRectangularSearchSpace(fill(-5.0, N), fill(5.0, N))
    prob = GlobalOptimization.OptimizationProblem(sphere, ss)

    # Run DE variants with fixed seed for reproducibility
    seed = 1234
    Random.seed!(seed)
    res1 = GlobalOptimization.optimize!(
        GlobalOptimization.DE(prob; num_candidates=20, max_iterations=100)
    )
    Random.seed!(seed)
    res2 = GlobalOptimization.optimize!(
        GlobalOptimization.DE(
            prob;
            eval_method=GlobalOptimization.ThreadedFunctionEvaluation(),
            num_candidates=20,
            max_iterations=100,
        ),
    )
    Random.seed!(seed)
    res3 = GlobalOptimization.optimize!(
        GlobalOptimization.DE(
            prob;
            eval_method=GlobalOptimization.PolyesterFunctionEvaluation(),
            num_candidates=20,
            max_iterations=100,
        ),
    )

    # Ensure consistent behavior across implementations
    @test res1.exitFlag == res2.exitFlag == res3.exitFlag
    @test isapprox(res1.fbest, res2.fbest; atol=1e-6)
    @test isapprox(res1.fbest, res3.fbest; atol=1e-6)
    @test isapprox(res1.xbest, res2.xbest; atol=1e-6)
    @test isapprox(res1.xbest, res3.xbest; atol=1e-6)

    # Verify solution correctness
    @test isapprox(res1.fbest, 0.0; atol=5e-4)
    for i in 1:N
        @test isapprox(res1.xbest[i], 1.0; atol=1e-2)
    end
end
