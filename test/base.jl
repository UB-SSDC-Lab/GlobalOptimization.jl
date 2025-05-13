using GlobalOptimization, Test

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

@testset showtiming = true "Evaluator" begin
    # Define a concrete simple population type
    struct SimplePopulation <: GlobalOptimization.AbstractPopulation{Float64}
        candidates::Vector{Vector{Float64}}
        candidates_fitness::Vector{Float64}
    end

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
