using GlobalOptimization, Test
using Distributions, Random
using Optim, LineSearches

@testset showtiming = true "Hopper" begin

    # Test Hopper constructor with dimensions
    h = GlobalOptimization.Hopper{Float64}(3)
    @test length(h) == 3
    @test all(x -> x == 0.0, h.candidate)
    @test h.candidate_fitness == Inf
    @test h.best_candidate_fitness == Inf

    # Test Hopper undef initializer constructor
    hu = GlobalOptimization.Hopper{Float64}(undef)
    @test length(hu) == 0
    @test hu.candidate_fitness == Inf
    @test length(hu.best_candidate) == 0

    # Test draw_update! with dummy distribution
    struct DummyDist <: GlobalOptimization.AbstractMBHDistribution{Float64} end
    function GlobalOptimization.draw_step!(step::Vector{Float64}, ::DummyDist)
        for i in eachindex(step)
            step[i] = 1.5
        end
        nothing
    end

    hd = GlobalOptimization.Hopper{Float64}(2)
    # Set best_candidate to known values
    hd.best_candidate .= [2.0, 3.0]
    # Ensure candidate_step is zeroed
    hd.candidate_step .= zero.(hd.best_candidate)
    # Invoke draw_update!
    GlobalOptimization.draw_update!(hd, DummyDist())
    @test hd.candidate_step == [1.5, 1.5]
    @test hd.candidate == [3.5, 4.5]
end

@testset showtiming = true "Distributions" begin

    # Test MBHStepMemory constructor and properties
    mem = GlobalOptimization.MBHStepMemory{Float64}(2, 2)
    @test size(mem.data) == (4, 2)  # num_dims+2 by memory_len
    @test mem.steps_in_memory == 0

    # Test push! behavior and column sliding
    GlobalOptimization.push!(mem, [1.0, 10.0], 0.5, 1.5)
    @test mem.steps_in_memory == 1
    @test mem.data[1:2, 1] == [1.0, 10.0]
    @test mem.data[3, 1] == 0.5
    @test mem.data[4, 1] == 1.5

    GlobalOptimization.push!(mem, [2.0, 20.0], 0.6, 1.6)
    @test mem.steps_in_memory == 2
    @test mem.data[1:2, 2] == [1.0, 10.0]  # older step shifted
    @test mem.data[1:2, 1] == [2.0, 20.0]

    # Test step_std computation
    σ1 = GlobalOptimization.step_std(mem, 1)
    @test isapprox(σ1, 0.5; atol=1e-8)
    σ2 = GlobalOptimization.step_std(mem, 2)
    @test isapprox(σ2, 5.0; atol=1e-8)

    # Test MBHStaticDistribution defaults and draw_step!
    sd = GlobalOptimization.MBHStaticDistribution{Float64}()
    @test sd.a == 0.93 && sd.b == 0.05 && sd.c == 1.0 && sd.λ == 1.0
    vec = zeros(3)
    Random.seed!(42)
    GlobalOptimization.draw_step!(vec, sd)
    @test all(isfinite, vec)

    # Test MBHAdaptiveDistribution constructor and push_accepted_step!
    ad = GlobalOptimization.MBHAdaptiveDistribution{Float64}(21, 0; a=0.9, b=0.1, c=2.0, λhat0=1.0)
    GlobalOptimization.initialize!(ad, 2)
    @test ad.min_memory_update == 0
    @test length(ad.λhat) == 2
    # Push fewer than threshold => no λhat change
    for i in 1:19
        GlobalOptimization.push_accepted_step!(ad, [1.0, 2.0], 0.0, 0.0)
    end
    @test ad.step_memory.steps_in_memory == 19
    @test all(ad.λhat .== 1.0)
    # Push one more to reach threshold 20 => λhat should update
    GlobalOptimization.push_accepted_step!(ad, [1.0, 2.0], 0.0, 0.0)
    @test ad.step_memory.steps_in_memory == 20
    @test all(ad.λhat .!= 1.0)

    # Test draw_step! for adaptive distribution
    vec2 = zeros(3)
    # adjust λhat length for vec2 size
    ad2 = GlobalOptimization.MBHAdaptiveDistribution{Float64}(21, 0)
    GlobalOptimization.initialize!(ad2, 3)
    Random.seed!(123)
    GlobalOptimization.draw_step!(vec2, ad2)
    @test all(isfinite, vec2)
end

@testset showtiming = true "LocalSearch" begin
    # Test timeout success and failure
    f(x) = x + 1
    @test GlobalOptimization.timeout(f, 1, 0.1, -1) == 2
    slow(x) = (sleep(0.2); x)
    @test GlobalOptimization.timeout(slow, 1, 0.1, -1) == -1

    # Test LocalStochasticSearch constructor and draw_step!
    ls = GlobalOptimization.LocalStochasticSearch{Float64}(2.0, 5)
    GlobalOptimization.initialize!(ls, 3)
    @test ls.iters == 5
    @test length(ls.step) == 3
    vec = zeros(3)
    Random.seed!(42)
    GlobalOptimization.draw_step!(vec, ls)
    @test all(isfinite, vec)

    # Test OptimSolutionCache initialization
    cache = GlobalOptimization.OptimSolutionCache{Float64}()
    GlobalOptimization.initialize!(cache, 4)
    @test length(cache.x) == 4
    @test cache.cost == 0.0

    # Test optim_solve!
    sphere(x) = sum(xx -> (xx - 1.0)^2, x)
    N = 2
    ss = GlobalOptimization.ContinuousRectangularSearchSpace(fill(-5.0, N), fill(5.0, N))
    prob = GlobalOptimization.OptimizationProblem(sphere, ss)
    cache2 = GlobalOptimization.OptimSolutionCache{Float64}()
    GlobalOptimization.initialize!(cache2, N)
    x0 = fill(0.0, N)
    res = GlobalOptimization.optim_solve!(cache2, prob, x0, Optim.Fminbox(Optim.LBFGS()), Optim.Options(iterations=2))
    @test res == true
    @test isapprox(cache2.cost, sphere(cache2.x); atol=1e-6)
end

@testset showtiming = true "Full MBH Optimization" begin

    # Define a trivial objective (constant zero)
    constzero(x) = zero(eltype(x[1])) + zero(eltype(x[1])) # always returns 0.0
    # Use a 2-dimensional search space
    ss = GlobalOptimization.ContinuousRectangularSearchSpace([0.0, 0.0], [1.0, 1.0])
    prob = GlobalOptimization.OptimizationProblem(constzero, ss)

    # Dummy distribution that produces zero steps
    struct ZeroDist <: GlobalOptimization.AbstractMBHDistribution{Float64} end
    function GlobalOptimization.draw_step!(step::Vector{Float64}, ::ZeroDist)
        # no change
        nothing
    end
    function GlobalOptimization.update_fitness!(
        hopper::GlobalOptimization.Hopper{Float64}, ::ZeroDist
    )
        # The same as for MBHStaticDistribution
        if hopper.candidate_fitness < hopper.best_candidate_fitness
            # Update hopper
            hopper.best_candidate .= hopper.candidate
            hopper.best_candidate_fitness = hopper.candidate_fitness
        end
        return nothing
    end

    # Dummy local search that does nothing
    struct DummyLS <: GlobalOptimization.AbstractLocalSearch{Float64} end
    GlobalOptimization.initialize!(::DummyLS, num_dims) = nothing
    function GlobalOptimization.local_search!(hopper, evaluator, ::DummyLS)
        # no change
        nothing
    end

    # Test MBH with ZeroDist and DummyLS
    mbh1 = GlobalOptimization.MBH(
        prob;
        hop_distribution=ZeroDist(),
        local_search=DummyLS(),
        max_time = 0.0,
    )
    res1 = GlobalOptimization.optimize!(mbh1)
    @test res1.fbest == 0.0
    @test length(res1.xbest) == 2
    @test res1.exitFlag == 1

    # Test MBH with static distribution
    stat = GlobalOptimization.MBHStaticDistribution{Float64}()
    mbh2 = GlobalOptimization.MBH(
        prob;
        hop_distribution = stat,
        local_search = DummyLS(),
        max_time = 0.0,
    )
    res2 = GlobalOptimization.optimize!(mbh2)
    @test res2.fbest == 0.0
    @test length(res2.xbest) == 2
    @test res2.exitFlag == 1

    # Test MBH with adaptive distribution
    ad = GlobalOptimization.MBHAdaptiveDistribution{Float64}(1, 0)
    mbh3 = GlobalOptimization.MBH(
        prob;
        hop_distribution = ad,
        local_search = DummyLS(),
        max_time = 0.0,
    )
    res3 = GlobalOptimization.optimize!(mbh3)
    @test res3.fbest == 0.0
    @test length(res3.xbest) == 2
    @test res3.exitFlag == 1

    # Test MBH with ZeroDist and LocalStochasticSearch
    ls = GlobalOptimization.LocalStochasticSearch{Float64}(0.1, 1)
    mbh4 = GlobalOptimization.MBH(
        prob;
        hop_distribution = ZeroDist(),
        local_search = ls,
        max_time = 0.0,
    )
    res4 = GlobalOptimization.optimize!(mbh4)
    @test res4.fbest == 0.0
    @test length(res4.xbest) == 2
    @test res4.exitFlag == 1

    # Test MBH with static distribution and LocalStochasticSearch
    ls2 = GlobalOptimization.LocalStochasticSearch{Float64}(0.1, 2)
    mbh5 = GlobalOptimization.MBH(
        prob;
        hop_distribution = GlobalOptimization.MBHStaticDistribution{Float64}(),
        local_search = ls2,
        max_time = 2.0,
    )
    res5 = GlobalOptimization.optimize!(mbh5)
    @test res5.fbest == 0.0
    @test length(res5.xbest) == 2
    @test res5.exitFlag == 1

    # Test MBH with adaptive distribution and LocalStochasticSearch
    ls3 = GlobalOptimization.LocalStochasticSearch{Float64}(0.1, 2)
    mbh6 = GlobalOptimization.MBH(
        prob;
        hop_distribution = GlobalOptimization.MBHAdaptiveDistribution{Float64}(1, 0),
        local_search = ls3,
        max_time = 2.0,
    )
    res6 = GlobalOptimization.optimize!(mbh6)
    @test res6.fbest == 0.0
    @test length(res6.xbest) == 2
    @test res6.exitFlag == 1

    # Test MBH with static distribution and LBFGSLocalSearch
    ls4 = GlobalOptimization.LBFGSLocalSearch{Float64}()
    mbh7 = GlobalOptimization.MBH(
        prob;
        hop_distribution = GlobalOptimization.MBHStaticDistribution{Float64}(),
        local_search = ls4,
        max_time = 2.0,
    )
    res7 = GlobalOptimization.optimize!(mbh7)
    @test res7.fbest == 0.0
    @test length(res7.xbest) == 2
    @test res7.exitFlag == 1

    # Test MBH with adaptive distribution and LocalStochasticSearch
    ls5 = GlobalOptimization.LBFGSLocalSearch{Float64}()
    mbh8 = GlobalOptimization.MBH(
        prob;
        hop_distribution = GlobalOptimization.MBHAdaptiveDistribution{Float64}(1, 0),
        local_search = ls5,
        max_time = 2.0,
    )
    res8 = GlobalOptimization.optimize!(mbh8)
    @test res8.fbest == 0.0
    @test length(res8.xbest) == 2
    @test res8.exitFlag == 1
end
