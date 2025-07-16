using GlobalOptimization, Test
using Distributions, Random
using Optim, LineSearches, ADTypes
using NonlinearSolve: NonlinearSolve
using ReverseDiff: ReverseDiff

@testset showtiming = true "Hopper" begin

    # Test Hopper constructor
    h = GlobalOptimization.Hopper{Float64}(3)
    @test length(h) == 3
    @test GlobalOptimization.num_dims(h) == 3
    @test all(x -> x == 0.0, GlobalOptimization.candidate(h))
    @test GlobalOptimization.fitness(h) == Inf

    # Test SingleHopperSet constructor
    sh = GlobalOptimization.SingleHopperSet{Float64}(3)
    @test length(sh) == 3
    @test GlobalOptimization.num_dims(sh) == 3
    @test all(x -> x == 0.0, GlobalOptimization.candidate(sh.hopper))
    @test GlobalOptimization.fitness(sh.hopper) == Inf

    # Test MCHSet constructor
    mh = GlobalOptimization.MCHSet{Float64}(3, 5)
    @test length(mh) == 5
    @test eachindex(mh) == 1:5
    @test GlobalOptimization.num_dims(mh) == 3

    # Test trace element getters
    shte = GlobalOptimization.get_show_trace_elements(sh, Val{:detailed}())
    @test length(shte) == 1
    @test shte[1].label == "Hops"
    @test shte == GlobalOptimization.get_save_trace_elements(sh, Val{:detailed}())

    mhte = GlobalOptimization.get_show_trace_elements(mh, Val{:detailed}())
    @test length(mhte) == 2
    @test mhte[1].label == "Min Hops"
    @test mhte[2].label == "Max Hops"
    @test mhte == GlobalOptimization.get_save_trace_elements(mh, Val{:detailed}())

    # Test initialization of SingleHopperSet and MCHSet
    search_space = GlobalOptimization.ContinuousRectangularSearchSpace(
        [-1.0, -1.0], [1.0, 1.0]
    )
    initial_space = GlobalOptimization.ContinuousRectangularSearchSpace(
        [0.0, 0.0], [2.0, 2.0]
    )
    evaluator = GlobalOptimization.FeasibilityHandlingEvaluator(
        GlobalOptimization.OptimizationProblem(sum, search_space)
    )

    # SingleHopperSet
    sh2 = GlobalOptimization.SingleHopperSet{Float64}(2)
    GlobalOptimization.initialize!(
        sh2,
        GlobalOptimization.intersection(search_space, initial_space),
        evaluator,
        nothing,
    )
    @test GlobalOptimization.candidate(sh2.hopper) != [0.0, 0.0]
    @test GlobalOptimization.candidate(sh2.hopper) == sh2.best_candidate
    @test isfinite(GlobalOptimization.fitness(sh2.hopper))
    @test GlobalOptimization.fitness(sh2.hopper) == sh2.best_candidate_fitness
    @test GlobalOptimization.feasible(
        GlobalOptimization.candidate(sh2.hopper), search_space
    )
    @test GlobalOptimization.feasible(
        GlobalOptimization.candidate(sh2.hopper), initial_space
    )

    # MCHSet
    mh2 = GlobalOptimization.MCHSet{Float64}(2, 3)
    GlobalOptimization.initialize!(
        mh2,
        GlobalOptimization.intersection(search_space, initial_space),
        evaluator,
        GlobalOptimization.SerialBatchJobEvaluator(),
    )
    @test all(h -> GlobalOptimization.candidate(h) == mh2.best_candidate, mh2.hoppers)
    @test mh2.best_candidate != [0.0, 0.0]
    @test all(h -> GlobalOptimization.fitness(h) == mh2.best_candidate_fitness, mh2.hoppers)
    @test isfinite(mh2.best_candidate_fitness)
    @test GlobalOptimization.feasible(mh2.best_candidate, search_space)
    @test GlobalOptimization.feasible(mh2.best_candidate, initial_space)

    # Test draw_update! with dummy distribution
    struct DummyDist <: GlobalOptimization.AbstractMBHDistribution{Float64} end
    function GlobalOptimization.draw_step!(step::Vector{Float64}, ::DummyDist)
        for i in eachindex(step)
            step[i] = 1.5
        end
        nothing
    end

    # draw_update! for Hopper
    hd = GlobalOptimization.Hopper{Float64}(2)
    GlobalOptimization.set_candidate!(hd, [2.0, 3.0])
    GlobalOptimization.draw_update!(hd, DummyDist())
    @test hd.candidate_step == [1.5, 1.5]
    @test GlobalOptimization.candidate(hd) == [3.5, 4.5]

    # Test update_fitness! with SingleHopperSet and MBHStaticDistribution
    sd = MBHStaticDistribution{Float64}()
    sh3 = GlobalOptimization.SingleHopperSet{Float64}(2)

    # Test for improved fitness
    GlobalOptimization.set_candidate!(sh3.hopper, [2.0, 3.0])
    GlobalOptimization.set_fitness!(sh3.hopper, 5.0)
    GlobalOptimization.update_fitness!(sh3, sd)
    @test sh3.best_candidate == [2.0, 3.0]
    @test sh3.best_candidate_fitness == 5.0

    # Test for worse fitness
    GlobalOptimization.set_candidate!(sh3.hopper, [1.0, 2.0])
    GlobalOptimization.set_fitness!(sh3.hopper, 10.0)
    GlobalOptimization.update_fitness!(sh3, sd)
    @test sh3.best_candidate == [2.0, 3.0]
    @test sh3.best_candidate_fitness == 5.0

    # Test update_fitness! with MCHSet and MBHStaticDistribution
    mh3 = GlobalOptimization.MCHSet{Float64}(2, 3)

    # Test for improved fitness
    for i in 1:3
        GlobalOptimization.set_candidate!(mh3.hoppers[i], [i, i])
        GlobalOptimization.set_fitness!(mh3.hoppers[i], i * 10.0)
    end
    GlobalOptimization.update_fitness!(mh3, sd)
    @test mh3.best_candidate == [1.0, 1.0]
    @test mh3.best_candidate_fitness == 10.0
    @test all(h -> GlobalOptimization.candidate(h) == mh3.best_candidate, mh3.hoppers)
    @test all(h -> GlobalOptimization.fitness(h) == mh3.best_candidate_fitness, mh3.hoppers)

    # Test for worse fitness
    for i in 1:3
        GlobalOptimization.set_candidate!(mh3.hoppers[i], [-i, -i])
        GlobalOptimization.set_fitness!(mh3.hoppers[i], i * 100.0)
    end
    GlobalOptimization.update_fitness!(mh3, sd)
    @test mh3.best_candidate == [1.0, 1.0]
    @test mh3.best_candidate_fitness == 10.0
    @test all(h -> GlobalOptimization.candidate(h) == mh3.best_candidate, mh3.hoppers)
    @test all(h -> GlobalOptimization.fitness(h) == mh3.best_candidate_fitness, mh3.hoppers)

    # Test update_fitness! with SingleHopperSet and MBHAdaptiveDistribution
    ad = MBHAdaptiveDistribution{Float64}(4, 0)
    ss = ContinuousRectangularSearchSpace([-100.0, -100.0], [100.0, 100.0])
    sh4 = GlobalOptimization.SingleHopperSet{Float64}(2)
    GlobalOptimization.initialize!(ad, ss)

    # Test for improved fitness
    GlobalOptimization.set_candidate!(sh4.hopper, [2.0, 3.0])
    GlobalOptimization.set_fitness!(sh4.hopper, 5.0)
    sh4.hopper.candidate_step .= [1.0, 1.0]
    sh4.best_candidate_fitness = 100.0
    GlobalOptimization.update_fitness!(sh4, ad)
    @test sh3.best_candidate == [2.0, 3.0]
    @test sh3.best_candidate_fitness == 5.0
    @test ad.step_memory.steps_in_memory == 1
    @test ad.step_memory.data[1:2, 1] == [1.0, 1.0]
    @test ad.step_memory.data[3, 1] == 100.0
    @test ad.step_memory.data[4, 1] == 5.0

    # Test for worse fitness
    GlobalOptimization.set_candidate!(sh4.hopper, [1.0, 2.0])
    GlobalOptimization.set_fitness!(sh4.hopper, 10.0)
    GlobalOptimization.update_fitness!(sh4, ad)
    @test sh4.best_candidate == [2.0, 3.0]
    @test sh4.best_candidate_fitness == 5.0
    @test ad.step_memory.steps_in_memory == 1
    @test ad.step_memory.data[1:2, 1] == [1.0, 1.0]
    @test ad.step_memory.data[3, 1] == 100.0
    @test ad.step_memory.data[4, 1] == 5.0

    # Test update_fitness! with MCHSet and MBHStaticDistribution
    ad2 = MBHAdaptiveDistribution{Float64}(4, 0)
    mh4 = GlobalOptimization.MCHSet{Float64}(2, 3)
    ss2 = ContinuousRectangularSearchSpace([-100.0, -100.0], [100.0, 100.0])
    GlobalOptimization.initialize!(ad2, ss2)

    # Test for improved fitness
    for i in 1:3
        GlobalOptimization.set_candidate!(mh4.hoppers[i], [i, i])
        GlobalOptimization.set_fitness!(mh4.hoppers[i], i * 10.0)
        mh4.hoppers[i].candidate_step .= [2.0*i, 2.0*i]
    end
    mh4.best_candidate_fitness = 100.0
    GlobalOptimization.update_fitness!(mh4, ad2)
    @test mh4.best_candidate == [1.0, 1.0]
    @test mh4.best_candidate_fitness == 10.0
    @test all(h -> GlobalOptimization.candidate(h) == mh4.best_candidate, mh4.hoppers)
    @test all(h -> GlobalOptimization.fitness(h) == mh4.best_candidate_fitness, mh4.hoppers)
    @test ad2.step_memory.steps_in_memory == 1
    @test ad2.step_memory.data[1:2, 1] == [2.0, 2.0]
    @test ad2.step_memory.data[3, 1] == 100.0
    @test ad2.step_memory.data[4, 1] == 10.0

    # Test for worse fitness
    for i in 1:3
        GlobalOptimization.set_candidate!(mh3.hoppers[i], [-i, -i])
        GlobalOptimization.set_fitness!(mh3.hoppers[i], i * 100.0)
    end
    GlobalOptimization.update_fitness!(mh3, sd)
    @test mh3.best_candidate == [1.0, 1.0]
    @test mh3.best_candidate_fitness == 10.0
    @test all(h -> GlobalOptimization.candidate(h) == mh3.best_candidate, mh3.hoppers)
    @test all(h -> GlobalOptimization.fitness(h) == mh3.best_candidate_fitness, mh3.hoppers)
    @test ad2.step_memory.steps_in_memory == 1
    @test ad2.step_memory.data[1:2, 1] == [2.0, 2.0]
    @test ad2.step_memory.data[3, 1] == 100.0
    @test ad2.step_memory.data[4, 1] == 10.0
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
    @test isapprox(σ1, 0.7071067811865476; atol=1e-8)
    σ2 = GlobalOptimization.step_std(mem, 2)
    @test isapprox(σ2, 7.0710678118654755; atol=1e-8)
    mad1 = GlobalOptimization.step_MAD_median(mem, 1)
    @test isapprox(mad1, 0.5; atol=1e-8)
    mad2 = GlobalOptimization.step_MAD_median(mem, 2)
    @test isapprox(mad2, 5.0; atol=1e-8)

    # Test MBHStaticDistribution defaults and draw_step!
    sd = MBHStaticDistribution{Float64}()
    GlobalOptimization.initialize!(
        sd, ContinuousRectangularSearchSpace(fill(-1.0, 3), fill(1.0, 3))
    )
    @test sd.b == 0.05 && sd.λ == 0.7
    vec = zeros(3)
    Random.seed!(42)
    GlobalOptimization.draw_step!(vec, sd)
    @test all(isfinite, vec)

    # Trace...
    @test GlobalOptimization.get_show_trace_elements(sd, Val{:detailed}()) isa Tuple{}
    @test GlobalOptimization.get_save_trace_elements(sd, Val{:detailed}()) isa Tuple{}

    # Test MBHAdaptiveDistribution constructor and push_accepted_step!
    ad = MBHAdaptiveDistribution{Float64}(21, 20; a=0.9, b=0.1, c=2.0, λhat0=1.0)
    ss = ContinuousRectangularSearchSpace([-1.0, -1.0], [1.0, 1.0])
    GlobalOptimization.initialize!(ad, ss)
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
    ad2 = MBHAdaptiveDistribution{Float64}(21, 0)
    ss2 = ContinuousRectangularSearchSpace([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    GlobalOptimization.initialize!(ad2, ss2)
    Random.seed!(123)
    GlobalOptimization.draw_step!(vec2, ad2)
    @test all(isfinite, vec2)

    # Trace
    adte = GlobalOptimization.get_show_trace_elements(ad, Val{:detailed}())
    @test length(adte) == 4
    adte_all = GlobalOptimization.get_save_trace_elements(ad, Val{:all}())
    @test length(adte_all) == 2
    @test adte == GlobalOptimization.get_save_trace_elements(ad, Val{:detailed}())
end

@testset showtiming = true "LocalSearch" begin
    # Test timeout success and failure
    f(x) = x + 1
    @test GlobalOptimization.timeout(f, 1, 0.1, -1) == 2
    slow(x) = (sleep(0.2); x)
    @test GlobalOptimization.timeout(slow, 1, 0.1, -1) == -1
    err(x) = (sleep(0.01); throw(InterruptException()))
    @test_throws InterruptException GlobalOptimization.timeout(err, 1, 0.1, -1)

    # Test LocalStochasticSearch constructor and draw_step!
    ls = LocalStochasticSearch{Float64}(2.0, 5)
    GlobalOptimization.initialize!(ls, 3)
    @test ls.iters == 5
    @test length(ls.step) == 3
    vec = zeros(3)
    Random.seed!(42)
    GlobalOptimization.draw_step!(vec, ls)
    @test all(isfinite, vec)

    # Test LocalSearchSolutionCache initialization
    cache = GlobalOptimization.LocalSearchSolutionCache{Float64}()
    GlobalOptimization.initialize!(cache, 4)
    @test length(cache.x) == 4
    @test cache.cost == 0.0

    # Test optim_solve!
    optimls_ext = Base.get_extension(GlobalOptimization, :OptimLocalSearchExt)
    sphere(x) = sum(xx -> (xx - 1.0)^2, x)
    N = 2
    ss = GlobalOptimization.ContinuousRectangularSearchSpace(fill(-5.0, N), fill(5.0, N))
    prob = GlobalOptimization.OptimizationProblem(sphere, ss)
    cache2 = GlobalOptimization.LocalSearchSolutionCache{Float64}()
    GlobalOptimization.initialize!(cache2, N)
    x0 = fill(0.0, N)
    res = optimls_ext.optim_solve!(
        cache2, prob, x0, Optim.Fminbox(Optim.LBFGS()), Optim.Options(iterations=2)
    )
    @test res == true
    @test isapprox(cache2.cost, sphere(cache2.x); atol=1e-6)

    cache3 = GlobalOptimization.LocalSearchSolutionCache{Float64}()
    GlobalOptimization.initialize!(cache3, N)
    x0 = fill(0.0, N)
    res = optimls_ext.optim_solve!(
        cache3,
        prob,
        x0,
        Optim.Fminbox(Optim.LBFGS()),
        ADTypes.AutoReverseDiff(),
        Optim.Options(iterations=2),
    )
    @test res == true
    @test isapprox(cache3.cost, sphere(cache3.x); atol=1e-6)

    # Test nonlinear problem solve
    nlls_ext = Base.get_extension(GlobalOptimization, :NonlinearSolveLocalSearchExt)
    nonlinear_eq(x) = [x[1]^2 - 1.0, x[2]^2 - 1.0]
    prob2 = GlobalOptimization.NonlinearProblem(nonlinear_eq, ss)
    cache4 = GlobalOptimization.LocalSearchSolutionCache{Float64}()
    GlobalOptimization.initialize!(cache4, N)
    x0 = fill(2.0, N)
    res = nlls_ext.nonlinear_solve!(
        cache4, prob2, x0, NonlinearSolve.NewtonRaphson(), 1e-8, 10
    )
    @test res == true
    @test isapprox(
        cache4.cost, GlobalOptimization.scalar_function(prob2, cache4.x), atol=1e-6
    )

    # Test local_search!
    fhe = GlobalOptimization.FeasibilityHandlingEvaluator(prob)

    # With LocalStochasticSearch
    h2 = GlobalOptimization.Hopper{Float64}(2)
    ls2 = LocalStochasticSearch{Float64}(1e-1, 1000)
    GlobalOptimization.initialize!(ls2, GlobalOptimization.num_dims(h2))
    start_point = [-4.0, -4.0]
    after_large_hop_point = [-4.5, -4.5]
    after_large_hop_fitness = sphere(after_large_hop_point)
    h2.candidate .= after_large_hop_point
    h2.candidate_fitness = after_large_hop_fitness
    h2.candidate_step .= after_large_hop_point .- start_point
    GlobalOptimization.local_search!(h2, fhe, ls2)
    @test h2.candidate != after_large_hop_point
    @test h2.candidate_fitness < after_large_hop_fitness
    for i in 1:GlobalOptimization.num_dims(h2)
        @test isapprox(h2.candidate_step[i], h2.candidate[i] - start_point[i]; atol=1e-12)
    end

    # With LBFGSLocalSearch and default AD
    h3 = GlobalOptimization.Hopper{Float64}(2)
    ls3 = LBFGSLocalSearch{Float64}()
    GlobalOptimization.initialize!(ls3, GlobalOptimization.num_dims(h3))
    h3.candidate .= after_large_hop_point
    h3.candidate_fitness = after_large_hop_fitness
    h3.candidate_step .= after_large_hop_point .- start_point
    GlobalOptimization.local_search!(h3, fhe, ls3)
    @test h3.candidate != after_large_hop_point
    @test h3.candidate_fitness < after_large_hop_fitness
    for i in 1:GlobalOptimization.num_dims(h3)
        @test isapprox(h3.candidate_step[i], h3.candidate[i] - start_point[i]; atol=1e-12)
    end

    # With LBFGSLocalSearch and ReverseDiff
    h4 = GlobalOptimization.Hopper{Float64}(2)
    ls4 = LBFGSLocalSearch{Float64}(; ad=ADTypes.AutoReverseDiff())
    GlobalOptimization.initialize!(ls4, GlobalOptimization.num_dims(h4))
    h4.candidate .= after_large_hop_point
    h4.candidate_fitness = after_large_hop_fitness
    h4.candidate_step .= after_large_hop_point .- start_point
    GlobalOptimization.local_search!(h4, fhe, ls4)
    @test h4.candidate != after_large_hop_point
    @test h4.candidate_fitness < after_large_hop_fitness
    for i in 1:GlobalOptimization.num_dims(h4)
        @test isapprox(h4.candidate_step[i], h4.candidate[i] - start_point[i]; atol=1e-12)
    end

    # With NonlinearSolveLocalSearch
    fhe2 = GlobalOptimization.FeasibilityHandlingEvaluator(prob2)
    h5 = GlobalOptimization.Hopper{Float64}(2)
    nls = NonlinearSolveLocalSearch{Float64}(NonlinearSolve.NewtonRaphson())
    GlobalOptimization.initialize!(nls, GlobalOptimization.num_dims(h5))
    h5.candidate .= after_large_hop_point
    h5.candidate_fitness = after_large_hop_fitness
    h5.candidate_step .= after_large_hop_point .- start_point
    GlobalOptimization.local_search!(h5, fhe2, nls)
    @test h5.candidate != after_large_hop_point
    @test h5.candidate_fitness < after_large_hop_fitness
    for i in 1:GlobalOptimization.num_dims(h5)
        @test isapprox(h5.candidate_step[i], h5.candidate[i] - start_point[i]; atol=1e-12)
    end
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
        hopper_set::GlobalOptimization.SingleHopperSet{Float64}, ::ZeroDist
    )
        # The same as for MBHStaticDistribution
        if GlobalOptimization.fitness(hopper_set.hopper) < hopper_set.best_candidate_fitness
            # Update hopper set
            hopper_set.best_candidate .= GlobalOptimization.candidate(hopper_set.hopper)
            hopper_set.best_candidate_fitness = GlobalOptimization.fitness(
                hopper_set.hopper
            )
        else
            # Reset hopper
            GlobalOptimization.set_candidate!(hopper_set.hopper, hopper_set.best_candidate)
            GlobalOptimization.set_fitness!(
                hopper_set.hopper, hopper_set.best_candidate_fitness
            )
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
    mbh1 = MBH(prob; hop_distribution=ZeroDist(), local_search=DummyLS(), max_time=0.0)
    res1 = GlobalOptimization.optimize!(mbh1)
    @test res1.fbest == 0.0
    @test length(res1.xbest) == 2
    @test res1.exitFlag == GlobalOptimization.MAXIMUM_TIME_EXCEEDED

    # Test MBH with static distribution
    stat = MBHStaticDistribution{Float64}()
    mbh2 = MBH(prob; hop_distribution=stat, local_search=DummyLS(), max_time=0.0)
    res2 = GlobalOptimization.optimize!(mbh2)
    @test res2.fbest == 0.0
    @test length(res2.xbest) == 2
    @test res2.exitFlag == GlobalOptimization.MAXIMUM_TIME_EXCEEDED

    # Test MBH with adaptive distribution
    ad = MBHAdaptiveDistribution{Float64}(1, 0)
    mbh3 = MBH(prob; hop_distribution=ad, local_search=DummyLS(), max_time=0.0)
    res3 = GlobalOptimization.optimize!(mbh3)
    @test res3.fbest == 0.0
    @test length(res3.xbest) == 2
    @test res3.exitFlag == GlobalOptimization.MAXIMUM_TIME_EXCEEDED

    # Test MBH with ZeroDist and LocalStochasticSearch
    ls = LocalStochasticSearch{Float64}(0.1, 1)
    mbh4 = MBH(prob; hop_distribution=ZeroDist(), local_search=ls, max_time=0.0)
    res4 = GlobalOptimization.optimize!(mbh4)
    @test res4.fbest == 0.0
    @test length(res4.xbest) == 2
    @test res4.exitFlag == GlobalOptimization.MAXIMUM_TIME_EXCEEDED

    # Test MBH with static distribution and LocalStochasticSearch
    ls2 = LocalStochasticSearch{Float64}(0.1, 2)
    mbh5 = MBH(
        prob;
        hop_distribution=MBHStaticDistribution{Float64}(),
        local_search=ls2,
        max_time=2.0,
    )
    res5 = GlobalOptimization.optimize!(mbh5)
    @test res5.fbest == 0.0
    @test length(res5.xbest) == 2

    # Test MBH with adaptive distribution and LocalStochasticSearch
    ls3 = LocalStochasticSearch{Float64}(0.1, 2)
    mbh6 = MBH(
        prob;
        hop_distribution=MBHAdaptiveDistribution{Float64}(1, 0),
        local_search=ls3,
        max_time=2.0,
    )
    res6 = GlobalOptimization.optimize!(mbh6)
    @test res6.fbest == 0.0
    @test length(res6.xbest) == 2

    # Test MBH with static distribution and LBFGSLocalSearch
    ls4 = LBFGSLocalSearch{Float64}()
    mbh7 = MBH(
        prob;
        hop_distribution=MBHStaticDistribution{Float64}(),
        local_search=ls4,
        max_time=2.0,
    )
    res7 = GlobalOptimization.optimize!(mbh7)
    @test res7.fbest == 0.0
    @test length(res7.xbest) == 2

    # Test MBH with adaptive distribution and LocalStochasticSearch
    ls5 = LBFGSLocalSearch{Float64}()
    mbh8 = MBH(
        prob;
        hop_distribution=MBHAdaptiveDistribution{Float64}(1, 0),
        local_search=ls5,
        max_time=2.0,
    )
    res8 = GlobalOptimization.optimize!(mbh8)
    @test res8.fbest == 0.0
    @test length(res8.xbest) == 2

    # Test MBH with MCH and each eval method for static distribution and LocalStochasticSearch
    eval_methods = [
        SerialFunctionEvaluation(),
        ThreadedFunctionEvaluation(),
        PolyesterFunctionEvaluation(),
    ]
    for eval_method in eval_methods
        cmbh1 = MBH(
            prob;
            hopper_type=MCH(; eval_method=eval_method),
            hop_distribution=MBHStaticDistribution{Float64}(),
            local_search=LocalStochasticSearch{Float64}(0.1, 2),
            max_time=2.0,
        )
        cres1 = GlobalOptimization.optimize!(cmbh1)
        @test cres1.fbest == 0.0
        @test length(cres1.xbest) == 2

        # Test MBH with adaptive distribution and LocalStochasticSearch
        cmbh2 = MBH(
            prob;
            hopper_type=MCH(; eval_method=eval_method),
            hop_distribution=MBHAdaptiveDistribution{Float64}(1, 0),
            local_search=LocalStochasticSearch{Float64}(0.1, 2),
            max_time=2.0,
        )
        cres2 = GlobalOptimization.optimize!(cmbh2)
        @test cres2.fbest == 0.0
        @test length(cres2.xbest) == 2

        # Test MBH with static distribution and LBFGSLocalSearch
        cmbh3 = MBH(
            prob;
            hopper_type=MCH(; eval_method=eval_method),
            hop_distribution=MBHStaticDistribution{Float64}(),
            local_search=LBFGSLocalSearch{Float64}(),
            max_time=2.0,
        )
        cres3 = GlobalOptimization.optimize!(cmbh3)
        @test cres3.fbest == 0.0
        @test length(cres3.xbest) == 2

        # Test MBH with adaptive distribution and LocalStochasticSearch
        cmbh4 = MBH(
            prob;
            hopper_type=MCH(; eval_method=eval_method),
            hop_distribution=MBHAdaptiveDistribution{Float64}(1, 0),
            local_search=LBFGSLocalSearch{Float64}(),
            max_time=2.0,
        )
        cres4 = GlobalOptimization.optimize!(cmbh4)
        @test cres4.fbest == 0.0
        @test length(cres4.xbest) == 2
    end

    # Test that MBH throws an error if trying to solve an OptimizationProblem with
    # a NonlinearSolveLocalSearch
    nl_ls = NonlinearSolveLocalSearch{Float64}(NonlinearSolve.NewtonRaphson())
    @test_throws ArgumentError MBH(prob; local_search=nl_ls)
end
