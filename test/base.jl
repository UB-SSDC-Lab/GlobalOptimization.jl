using GlobalOptimization, Test

# Define a concrete simple population type
struct SimplePopulation <: GlobalOptimization.AbstractPopulation{Float64}
    candidates::Vector{Vector{Float64}}
    candidates_fitness::Vector{Float64}
end

@testset showtiming = true "SearchSpace" begin

    # Test construction and basic accessors
    LB = [-1.0, 0.0]
    UB = [1.0, 2.0]
    ss = ContinuousRectangularSearchSpace(LB, UB)
    @test GlobalOptimization.num_dims(ss) == 2
    @test GlobalOptimization.dim_min(ss) == LB
    @test GlobalOptimization.dim_max(ss) == UB
    @test GlobalOptimization.dim_delta(ss) == (UB .- LB)
    @test GlobalOptimization.dim_min(ss, 1) == LB[1]
    @test GlobalOptimization.dim_max(ss, 2) == UB[2]
    @test GlobalOptimization.dim_range(ss, 1) == (LB[1], UB[1])
    @test GlobalOptimization.dim_range(ss) == [(LB[i], UB[i]) for i in 1:2]

    # Test intersection
    ss2 = ContinuousRectangularSearchSpace([-0.5, 0.5], [0.5, 1.5])
    ss_int = GlobalOptimization.intersection(ss, ss2)
    @test GlobalOptimization.dim_min(ss_int) == [-0.5, 0.5]
    @test GlobalOptimization.dim_max(ss_int) == [0.5, 1.5]

    # Intersection with Nothing
    @test GlobalOptimization.intersection(ss, nothing) === ss
    @test GlobalOptimization.intersection(nothing, ss2) === ss2

    # Test feasibility
    @test GlobalOptimization.feasible([0.0, 1.0], ss)
    @test !GlobalOptimization.feasible([-2.0, 1.0], ss)
    @test !GlobalOptimization.feasible([0.0, 3.0], ss)

    # Test constructor error conditions
    @test_throws DimensionMismatch ContinuousRectangularSearchSpace([0.0, 1.0], [0.0])
    @test_throws ArgumentError ContinuousRectangularSearchSpace([1.0, 0.0], [0.0, 2.0])

end

@testset showtiming = true "Results" begin
    # Test constructor and field access
    fbest = 1.23
    xbest = [4.5, 6.7]
    r = GlobalOptimization.Results(fbest, xbest, 10, 0.456, 0)
    @test r.fbest == fbest
    @test r.xbest == xbest
    @test r.xbest !== xbest
    @test r.iters == 10
    @test r.time == 0.456
    @test r.exitFlag == 0

    # Test pretty-printing via Base.show
    buf = IOBuffer()
    show(buf, MIME"text/plain"(), r)
    output = String(take!(buf))
    @test occursin("Results:", output)
    @test occursin("Best function value: 1.23", output)
    @test occursin("Best candidate: [4.5, 6.7]", output)
    @test occursin("Iterations: 10", output)
    @test occursin("Time: 0.456 seconds", output)
    @test occursin("Exit flag: 0", output)

    # Test type parameter propagation and copying for Float32
    f32 = Float32(2.5)
    x32 = Float32[1.0, 2.0]
    r32 = GlobalOptimization.Results(f32, x32, 3, 0.1, -1)
    @test typeof(r32.fbest) == Float32
    @test typeof(r32.xbest) == Vector{Float32}

    # Test constructor rejects mismatched xbest element types
    @test_throws MethodError GlobalOptimization.Results(1.0, [1, 2], 1, 0.1, 0)
end

@testset showtiming = true "Problems" begin
    # Setup a simple search space and test problem constructors
    LB = [-1.0, 0.0]
    UB = [1.0, 2.0]
    ss = ContinuousRectangularSearchSpace(LB, UB)

    # Test OptimizationProblem without penalty
    f = x -> sum(x)
    prob1 = GlobalOptimization.OptimizationProblem(f, ss)
    @test GlobalOptimization.search_space(prob1) === ss
    @test GlobalOptimization.num_dims(prob1) == 2
    x = [1.0, 2.0]
    @test GlobalOptimization.scalar_function(prob1, x) == 3.0
    val, pen = GlobalOptimization.scalar_function_with_penalty(prob1, x)
    @test val == 3.0
    @test pen == 0.0
    sf = GlobalOptimization.get_scalar_function(prob1)
    @test sf(x) == 3.0
    sfp = GlobalOptimization.get_scalar_function_with_penalty(prob1)
    @test sfp(x) == (3.0, 0.0)

    # Test OptimizationProblem with penalty
    fpen = x -> (sum(x), sum(x))
    prob2 = GlobalOptimization.OptimizationProblem(fpen, ss)
    @test GlobalOptimization.num_dims(prob2) == 2
    # scalar_function: f + 0.5*g^2 = 3.0 + 0.5*(3.0^2) = 3.0 + 4.5 = 7.5
    @test isapprox(GlobalOptimization.scalar_function(prob2, x), 7.5; atol=1e-8)
    vp, gp = GlobalOptimization.scalar_function_with_penalty(prob2, x)
    @test vp == 3.0
    @test gp == 3.0

    # Test NonlinearProblem without penalty
    fres = x -> [x[1] - 2.0, x[2] - 3.0]
    nprob = GlobalOptimization.NonlinearProblem(fres, ss)
    # At x = [2.0,3.0], residuals zero -> cost zero
    xn = [2.0, 3.0]
    @test GlobalOptimization.scalar_function(nprob, xn) == 0.0
    cn, pn = GlobalOptimization.scalar_function_with_penalty(nprob, xn)
    @test cn == 0.0
    @test pn == 0.0

    # Test get_scalar_function for nonlinear problem
    nf = GlobalOptimization.get_scalar_function(nprob)
    @test nf(xn) == 0.0

    # Test NonlinearLeastSquaresProblem constructor and field
    nlsprob = GlobalOptimization.NonlinearLeastSquaresProblem(fres, ss, 2)
    @test nlsprob.n == 2
    # Same cost behavior
    @test GlobalOptimization.scalar_function(nlsprob, xn) == 0.0
    @test GlobalOptimization.scalar_function_with_penalty(nlsprob, xn) == (0.0, 0.0)
end

@testset showtiming = true "Population" begin
    # Create a sample population
    cand = [[1.0, 2.0], [3.0, 4.0]]
    fit = [0.0, 0.0]
    pop = SimplePopulation(cand, copy(fit))

    # Test length, size, and indexing
    @test length(pop) == 2
    @test size(pop) == (2,)
    @test collect(eachindex(pop)) == [1, 2]

    # Test candidates and fitness accessors
    @test GlobalOptimization.candidates(pop) === pop.candidates
    @test GlobalOptimization.candidates(pop, 2) == [3.0, 4.0]
    @test GlobalOptimization.fitness(pop) === pop.candidates_fitness
    @test GlobalOptimization.fitness(pop, 1) == 0.0

    # Test set_fitness! with a vector
    GlobalOptimization.set_fitness!(pop, [5.0, 6.0])
    @test pop.candidates_fitness == [5.0, 6.0]

    # Test set_fitness! with a scalar index
    GlobalOptimization.set_fitness!(pop, 7.5, 1)
    @test pop.candidates_fitness[1] == 7.5

    # Test dimension mismatch error for set_fitness!
    @test_throws DimensionMismatch GlobalOptimization.set_fitness!(pop, [1.0])

    # Test check_fitness! with Val(false) does nothing
    pop.candidates_fitness .= [1.0, 2.0]
    @test GlobalOptimization.check_fitness!(pop, Val(false)) === nothing

    # Test check_fitness! with Val(true) errors on invalid fitness
    pop.candidates_fitness[2] = Inf
    @test_throws ErrorException GlobalOptimization.check_fitness!(pop, Val(true))
end

@testset showtiming = true "Population Initialization" begin
    using Random

    # Test UniformInitialization
    min_vec = [0.0, 1.0]
    max_vec = [1.0, 2.0]
    popvec = [zeros(2) for _ in 1:5]
    uinit = GlobalOptimization.UniformInitialization()
    GlobalOptimization.initialize_population_vector!(popvec, min_vec, max_vec, uinit)
    for vec in popvec
        for j in eachindex(vec)
            @test vec[j] ≥ min_vec[j] && vec[j] ≤ max_vec[j]
        end
    end

    # Test LatinHypercubeInitialization default constructor
    lhcd = GlobalOptimization.LatinHypercubeInitialization()
    @test lhcd.gens == 10
    @test lhcd.pop_size == 100
    @test isa(lhcd.rng, Random.AbstractRNG)

    # Test LatinHypercubeInitialization sampling and bounds
    rng = MersenneTwister(1234)
    lhci = GlobalOptimization.LatinHypercubeInitialization(
        5;
        rng=rng,
        pop_size=5,
        n_tour=1,
        p_tour=0.5,
        inter_sample_weight=1.0,
        periodic_ae=false,
        ae_power=2.0,
    )
    popvec2 = [zeros(3) for _ in 1:5]
    min2 = [0.0, 0.0, 0.0]
    max2 = [1.0, 2.0, 3.0]
    GlobalOptimization.initialize_population_vector!(popvec2, min2, max2, lhci)

    # Check bounds
    for vec in popvec2
        for j in eachindex(vec)
            @test vec[j] ≥ min2[j] && vec[j] ≤ max2[j]
        end
    end

    # Check each dimension yields unique samples
    for j in 1:3
        col = [popvec2[i][j] for i in 1:length(popvec2)]
        @test length(unique(col)) == length(popvec2)
    end
end

@testset showtiming = true "Evaluator" begin
    # Define cost functions for testing serial and parallel evaluators
    cost(x) = Threads.threadid()

    # Define problem
    prob = GlobalOptimization.OptimizationProblem(cost, [1.0, 1.0], [2.0, 2.0])

    # Construct population and evaluator
    N = 1000
    spop = SimplePopulation([zeros(2) for _ in 1:N], zeros(N))
    tpop = deepcopy(spop)
    ppop = deepcopy(spop)
    seval = GlobalOptimization.SerialBatchEvaluator(prob)
    teval = GlobalOptimization.ThreadedBatchEvaluator(prob)
    peval = GlobalOptimization.PolyesterBatchEvaluator(prob)

    GlobalOptimization.evaluate!(spop, seval)
    GlobalOptimization.evaluate!(tpop, teval)
    GlobalOptimization.evaluate!(ppop, peval)

    # Check serial evaluator results
    all_one = true
    for i in eachindex(spop.candidates_fitness)
        if spop.candidates_fitness[i] != 1.0
            all_one = false
            break
        end
    end
    @test all_one

    threads_used = [false for _ in 1:Threads.nthreads()]
    for i in eachindex(tpop.candidates_fitness)
        thread_id = Int(tpop.candidates_fitness[i])
        threads_used[thread_id] = true
        if all(threads_used)
            break
        end
    end
    @test all(threads_used)

    threads_used = [false for _ in 1:Threads.nthreads()]
    for i in eachindex(ppop.candidates_fitness)
        thread_id = Int(ppop.candidates_fitness[i])
        threads_used[thread_id] = true
        if all(threads_used)
            break
        end
    end
    @test all(threads_used)
end
