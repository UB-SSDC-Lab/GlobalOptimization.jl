
using GlobalOptimization, Test

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
N   = 1000
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
        global all_one = false
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
