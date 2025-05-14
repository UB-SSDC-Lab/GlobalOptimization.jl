
function get_problem_sets()
    # Define problem sets
    ProblemSets = Dict{String,Vector{Tuple{String,Int,Int,Int}}}(
        "easy" => [
            # ProblemName, NumDims, PopSize, MaxIters
            ("Sphere", 5, 20, 250),
            ("Sphere", 10, 20, 500),
            ("Sphere", 30, 20, 1500),
            ("Schwefel 2.22", 5, 20, 250),
            ("Schwefel 2.22", 10, 20, 500),
            ("Schwefel 2.22", 30, 20, 1500),
            ("Schwefel 2.21", 5, 20, 250),
            ("Schwefel 2.21", 10, 20, 500),
            ("Schwefel 2.21", 30, 20, 1500),
        ],
        "harder" => [
            # Harder problems
            ("Schwefel 1.2", 5, 20, 250),
            ("Schwefel 1.2", 10, 50, 1000),
            ("Schwefel 1.2", 30, 50, 4000),
            ("Schwefel 1.2", 50, 50, 6000),
            ("Rosenbrock", 5, 20, 500),
            ("Rosenbrock", 10, 50, 1000),
            ("Rosenbrock", 30, 50, 4000),
            ("Rosenbrock", 50, 40, 7500),
            ("Rastrigin", 50, 50, 10000),
            ("Rastrigin", 100, 90, 8889),
            ("Ackley", 50, 50, 10000),
            ("Ackley", 100, 90, 8889),
            ("Griewank", 50, 50, 10000),
            ("Griewank", 100, 90, 8889),
        ],
        "lowdim" => [
            ("Schwefel 1.2", 2, 25, 400),
            ("Rosenbrock", 2, 25, 400),
            ("Rastrigin", 2, 25, 400),
            ("Ackley", 2, 25, 400),
            ("Griewank", 2, 25, 400),
        ],
        "highpopsize" => [
            ("Schwefel 2.22", 5, 50, 100),
            ("Schwefel 2.22", 10, 100, 100),
            ("Schwefel 2.22", 30, 300, 100),
            ("Schwefel 2.21", 5, 50, 100),
            ("Schwefel 2.21", 10, 100, 100),
            ("Schwefel 2.21", 30, 300, 100),
            ("Schwefel 1.2", 5, 50, 100),
            ("Schwefel 1.2", 10, 250, 200),
            ("Schwefel 1.2", 30, 250, 800),
            ("Schwefel 1.2", 50, 250, 1200),
            ("Rosenbrock", 5, 50, 200),
            ("Rosenbrock", 10, 250, 200),
            ("Rosenbrock", 30, 250, 800),
            ("Rosenbrock", 50, 250, 1200),
            ("Rastrigin", 50, 250, 2000),
            ("Rastrigin", 100, 500, 1600),
            ("Ackley", 50, 250, 2000),
            ("Ackley", 100, 500, 1600),
            ("Griewank", 50, 250, 2000),
            ("Griewank", 100, 500, 1600),
        ],
    )
    ProblemSets["all"] = vcat(
        ProblemSets["easy"], ProblemSets["harder"], ProblemSets["highpopsize"]
    )
    ProblemSets["lowpopsize"] = vcat(ProblemSets["easy"], ProblemSets["harder"])
    return ProblemSets
end

function get_algorithm_sets()
    AlgorithmSets = Dict{String,Vector{Tuple{String,Function}}}(
        "pso" => [("pso", construct_pso),],
        "adaptive_de_bin" => [
            (
                "default_adaptive_de_rand_1_bin",
                (p, s, i) ->
                    construct_default_adaptive_de_mutstrat_bin(p, Rand1(), s, i),
            ),
            (
                "default_adaptive_de_best_1_bin",
                (p, s, i) ->
                    construct_default_adaptive_de_mutstrat_bin(p, Best1(), s, i),
            ),
            (
                "default_adaptive_de_current_to_best_1_bin",
                (p, s, i) -> construct_default_adaptive_de_mutstrat_bin(
                    p, CurrentToBest1(), s, i
                ),
            ),
            (
                "default_adaptive_de_current_to_rand_1_bin",
                (p, s, i) -> construct_default_adaptive_de_mutstrat_bin(
                    p, CurrentToRand1(), s, i
                ),
            ),
            (
                "default_adaptive_de_rand_to_best_1_bin",
                (p, s, i) ->
                    construct_default_adaptive_de_mutstrat_bin(p, RandToBest1(), s, i),
            ),
            (
                "default_adaptive_de_unified_bin",
                (p, s, i) ->
                    construct_default_adaptive_de_mutstrat_bin(p, Unified(), s, i),
            ),
            (
                "uniform_adaptive_de_rand_1_bin",
                (p, s, i) ->
                    construct_uniform_adaptive_de_mutstrat_bin(p, Rand1(), s, i),
            ),
            (
                "uniform_adaptive_de_best_1_bin",
                (p, s, i) ->
                    construct_uniform_adaptive_de_mutstrat_bin(p, Best1(), s, i),
            ),
            (
                "uniform_adaptive_de_current_to_best_1_bin",
                (p, s, i) -> construct_uniform_adaptive_de_mutstrat_bin(
                    p, CurrentToBest1(), s, i
                ),
            ),
            (
                "uniform_adaptive_de_current_to_rand_1_bin",
                (p, s, i) -> construct_uniform_adaptive_de_mutstrat_bin(
                    p, CurrentToRand1(), s, i
                ),
            ),
            (
                "uniform_adaptive_de_rand_to_best_1_bin",
                (p, s, i) ->
                    construct_uniform_adaptive_de_mutstrat_bin(p, RandToBest1(), s, i),
            ),
            (
                "uniform_adaptive_de_unified_bin",
                (p, s, i) ->
                    construct_uniform_adaptive_de_mutstrat_bin(p, Unified(), s, i),
            ),
        ],
        "rl_adaptive_de_bin" => [
            (
                "rl_default_adaptive_de_rand_1_bin",
                (p, s, i) ->
                    construct_rl_default_adaptive_de_mutstrat_bin(p, Rand1(), s, i),
            ),
            (
                "rl_default_adaptive_de_best_1_bin",
                (p, s, i) ->
                    construct_rl_default_adaptive_de_mutstrat_bin(p, Best1(), s, i),
            ),
            (
                "rl_default_adaptive_de_current_to_best_1_bin",
                (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_bin(
                    p, CurrentToBest1(), s, i
                ),
            ),
            (
                "rl_default_adaptive_de_current_to_rand_1_bin",
                (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_bin(
                    p, CurrentToRand1(), s, i
                ),
            ),
            (
                "rl_default_adaptive_de_rand_to_best_1_bin",
                (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_bin(
                    p, RandToBest1(), s, i
                ),
            ),
            (
                "rl_default_adaptive_de_unified_bin",
                (p, s, i) ->
                    construct_rl_default_adaptive_de_mutstrat_bin(p, Unified(), s, i),
            ),
            (
                "rl_uniform_adaptive_de_rand_1_bin",
                (p, s, i) ->
                    construct_rl_uniform_adaptive_de_mutstrat_bin(p, Rand1(), s, i),
            ),
            (
                "rl_uniform_adaptive_de_best_1_bin",
                (p, s, i) ->
                    construct_rl_uniform_adaptive_de_mutstrat_bin(p, Best1(), s, i),
            ),
            (
                "rl_uniform_adaptive_de_current_to_best_1_bin",
                (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_bin(
                    p, CurrentToBest1(), s, i
                ),
            ),
            (
                "rl_uniform_adaptive_de_current_to_rand_1_bin",
                (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_bin(
                    p, CurrentToRand1(), s, i
                ),
            ),
            (
                "rl_uniform_adaptive_de_rand_to_best_1_bin",
                (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_bin(
                    p, RandToBest1(), s, i
                ),
            ),
            (
                "rl_uniform_adaptive_de_unified_bin",
                (p, s, i) ->
                    construct_rl_uniform_adaptive_de_mutstrat_bin(p, Unified(), s, i),
            ),
        ],
        "rs_adaptive_de_bin" => [
            (
                "rs_default_adaptive_de_rand_1_bin",
                (p, s, i) ->
                    construct_rs_default_adaptive_de_mutstrat_bin(p, Rand1(), s, i),
            ),
            (
                "rs_default_adaptive_de_best_1_bin",
                (p, s, i) ->
                    construct_rs_default_adaptive_de_mutstrat_bin(p, Best1(), s, i),
            ),
            (
                "rs_default_adaptive_de_current_to_best_1_bin",
                (p, s, i) -> construct_rs_default_adaptive_de_mutstrat_bin(
                    p, CurrentToBest1(), s, i
                ),
            ),
            (
                "rs_default_adaptive_de_current_to_rand_1_bin",
                (p, s, i) -> construct_rs_default_adaptive_de_mutstrat_bin(
                    p, CurrentToRand1(), s, i
                ),
            ),
            (
                "rs_default_adaptive_de_rand_to_best_1_bin",
                (p, s, i) -> construct_rs_default_adaptive_de_mutstrat_bin(
                    p, RandToBest1(), s, i
                ),
            ),
            (
                "rs_default_adaptive_de_unified_bin",
                (p, s, i) ->
                    construct_rs_default_adaptive_de_mutstrat_bin(p, Unified(), s, i),
            ),
            (
                "rs_uniform_adaptive_de_rand_1_bin",
                (p, s, i) ->
                    construct_rs_uniform_adaptive_de_mutstrat_bin(p, Rand1(), s, i),
            ),
            (
                "rs_uniform_adaptive_de_best_1_bin",
                (p, s, i) ->
                    construct_rs_uniform_adaptive_de_mutstrat_bin(p, Best1(), s, i),
            ),
            (
                "rs_uniform_adaptive_de_current_to_best_1_bin",
                (p, s, i) -> construct_rs_uniform_adaptive_de_mutstrat_bin(
                    p, CurrentToBest1(), s, i
                ),
            ),
            (
                "rs_uniform_adaptive_de_current_to_rand_1_bin",
                (p, s, i) -> construct_rs_uniform_adaptive_de_mutstrat_bin(
                    p, CurrentToRand1(), s, i
                ),
            ),
            (
                "rs_uniform_adaptive_de_rand_to_best_1_bin",
                (p, s, i) -> construct_rs_uniform_adaptive_de_mutstrat_bin(
                    p, RandToBest1(), s, i
                ),
            ),
            (
                "rs_uniform_adaptive_de_unified_bin",
                (p, s, i) ->
                    construct_rs_uniform_adaptive_de_mutstrat_bin(p, Unified(), s, i),
            ),
        ],
        "adaptive_de_covbin" => [
            (
                "default_adaptive_de_rand_1_covbin",
                (p, s, i) ->
                    construct_default_adaptive_de_mutstrat_covbin(p, Rand1(), s, i),
            ),
            (
                "default_adaptive_de_best_1_covbin",
                (p, s, i) ->
                    construct_default_adaptive_de_mutstrat_covbin(p, Best1(), s, i),
            ),
            (
                "default_adaptive_de_current_to_best_1_covbin",
                (p, s, i) -> construct_default_adaptive_de_mutstrat_covbin(
                    p, CurrentToBest1(), s, i
                ),
            ),
            (
                "default_adaptive_de_current_to_rand_1_covbin",
                (p, s, i) -> construct_default_adaptive_de_mutstrat_covbin(
                    p, CurrentToRand1(), s, i
                ),
            ),
            (
                "default_adaptive_de_rand_to_best_1_covbin",
                (p, s, i) -> construct_default_adaptive_de_mutstrat_covbin(
                    p, RandToBest1(), s, i
                ),
            ),
            (
                "default_adaptive_de_unified_covbin",
                (p, s, i) ->
                    construct_default_adaptive_de_mutstrat_covbin(p, Unified(), s, i),
            ),
            (
                "uniform_adaptive_de_rand_1_covbin",
                (p, s, i) ->
                    construct_uniform_adaptive_de_mutstrat_covbin(p, Rand1(), s, i),
            ),
            (
                "uniform_adaptive_de_best_1_covbin",
                (p, s, i) ->
                    construct_uniform_adaptive_de_mutstrat_covbin(p, Best1(), s, i),
            ),
            (
                "uniform_adaptive_de_current_to_best_1_covbin",
                (p, s, i) -> construct_uniform_adaptive_de_mutstrat_covbin(
                    p, CurrentToBest1(), s, i
                ),
            ),
            (
                "uniform_adaptive_de_current_to_rand_1_covbin",
                (p, s, i) -> construct_uniform_adaptive_de_mutstrat_covbin(
                    p, CurrentToRand1(), s, i
                ),
            ),
            (
                "uniform_adaptive_de_rand_to_best_1_covbin",
                (p, s, i) -> construct_uniform_adaptive_de_mutstrat_covbin(
                    p, RandToBest1(), s, i
                ),
            ),
            (
                "uniform_adaptive_de_unified_covbin",
                (p, s, i) ->
                    construct_uniform_adaptive_de_mutstrat_covbin(p, Unified(), s, i),
            ),
        ],
        "rl_adaptive_de_covbin" => [
            (
                "rl_default_adaptive_de_rand_1_covbin",
                (p, s, i) ->
                    construct_rl_default_adaptive_de_mutstrat_covbin(p, Rand1(), s, i),
            ),
            (
                "rl_default_adaptive_de_best_1_covbin",
                (p, s, i) ->
                    construct_rl_default_adaptive_de_mutstrat_covbin(p, Best1(), s, i),
            ),
            (
                "rl_default_adaptive_de_current_to_best_1_covbin",
                (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_covbin(
                    p, CurrentToBest1(), s, i
                ),
            ),
            (
                "rl_default_adaptive_de_current_to_rand_1_covbin",
                (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_covbin(
                    p, CurrentToRand1(), s, i
                ),
            ),
            (
                "rl_default_adaptive_de_rand_to_best_1_covbin",
                (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_covbin(
                    p, RandToBest1(), s, i
                ),
            ),
            (
                "rl_default_adaptive_de_unified_covbin",
                (p, s, i) -> construct_rl_default_adaptive_de_mutstrat_covbin(
                    p, Unified(), s, i
                ),
            ),
            (
                "rl_uniform_adaptive_de_rand_1_covbin",
                (p, s, i) ->
                    construct_rl_uniform_adaptive_de_mutstrat_covbin(p, Rand1(), s, i),
            ),
            (
                "rl_uniform_adaptive_de_best_1_covbin",
                (p, s, i) ->
                    construct_rl_uniform_adaptive_de_mutstrat_covbin(p, Best1(), s, i),
            ),
            (
                "rl_uniform_adaptive_de_current_to_best_1_covbin",
                (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_covbin(
                    p, CurrentToBest1(), s, i
                ),
            ),
            (
                "rl_uniform_adaptive_de_current_to_rand_1_covbin",
                (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_covbin(
                    p, CurrentToRand1(), s, i
                ),
            ),
            (
                "rl_uniform_adaptive_de_rand_to_best_1_covbin",
                (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_covbin(
                    p, RandToBest1(), s, i
                ),
            ),
            (
                "rl_uniform_adaptive_de_unified_covbin",
                (p, s, i) -> construct_rl_uniform_adaptive_de_mutstrat_covbin(
                    p, Unified(), s, i
                ),
            ),
        ],
        "rs_adaptive_de_covbin" => [
            (
                "rs_default_adaptive_de_rand_1_covbin",
                (p, s, i) ->
                    construct_rs_default_adaptive_de_mutstrat_covbin(p, Rand1(), s, i),
            ),
            (
                "rs_default_adaptive_de_best_1_covbin",
                (p, s, i) ->
                    construct_rs_default_adaptive_de_mutstrat_covbin(p, Best1(), s, i),
            ),
            (
                "rs_default_adaptive_de_current_to_best_1_covbin",
                (p, s, i) -> construct_rs_default_adaptive_de_mutstrat_covbin(
                    p, CurrentToBest1(), s, i
                ),
            ),
            (
                "rs_default_adaptive_de_current_to_rand_1_covbin",
                (p, s, i) -> construct_rs_default_adaptive_de_mutstrat_covbin(
                    p, CurrentToRand1(), s, i
                ),
            ),
            (
                "rs_default_adaptive_de_rand_to_best_1_covbin",
                (p, s, i) -> construct_rs_default_adaptive_de_mutstrat_covbin(
                    p, RandToBest1(), s, i
                ),
            ),
            (
                "rs_default_adaptive_de_unified_covbin",
                (p, s, i) -> construct_rs_default_adaptive_de_mutstrat_covbin(
                    p, Unified(), s, i
                ),
            ),
            (
                "rs_uniform_adaptive_de_rand_1_covbin",
                (p, s, i) ->
                    construct_rs_uniform_adaptive_de_mutstrat_covbin(p, Rand1(), s, i),
            ),
            (
                "rs_uniform_adaptive_de_best_1_covbin",
                (p, s, i) ->
                    construct_rs_uniform_adaptive_de_mutstrat_covbin(p, Best1(), s, i),
            ),
            (
                "rs_uniform_adaptive_de_current_to_best_1_covbin",
                (p, s, i) -> construct_rs_uniform_adaptive_de_mutstrat_covbin(
                    p, CurrentToBest1(), s, i
                ),
            ),
            (
                "rs_uniform_adaptive_de_current_to_rand_1_covbin",
                (p, s, i) -> construct_rs_uniform_adaptive_de_mutstrat_covbin(
                    p, CurrentToRand1(), s, i
                ),
            ),
            (
                "rs_uniform_adaptive_de_rand_to_best_1_covbin",
                (p, s, i) -> construct_rs_uniform_adaptive_de_mutstrat_covbin(
                    p, RandToBest1(), s, i
                ),
            ),
            (
                "rs_uniform_adaptive_de_unified_covbin",
                (p, s, i) -> construct_rs_uniform_adaptive_de_mutstrat_covbin(
                    p, Unified(), s, i
                ),
            ),
        ],
    )
    AlgorithmSets["all"] = vcat(
        AlgorithmSets["pso"],
        AlgorithmSets["adaptive_de_bin"],
        AlgorithmSets["rl_adaptive_de_bin"],
        AlgorithmSets["rs_adaptive_de_bin"],
        AlgorithmSets["adaptive_de_covbin"],
        AlgorithmSets["rl_adaptive_de_covbin"],
        AlgorithmSets["rs_adaptive_de_covbin"],
    )
    return AlgorithmSets
end
