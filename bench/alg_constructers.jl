
function construct_pso(prob, pop_size, max_iters)
    return SerialPSO(
        prob;
        num_particles=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        max_time=Inf,
    )
end

function construct_default_adaptive_de_mutstrat_bin(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(mut_strat),
        crossover_params=SelfBinomialCrossoverParameters(),
    )
end

function construct_uniform_adaptive_de_mutstrat_bin(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(mut_strat; dist=Uniform(0.0, 1.0)),
        crossover_params=SelfBinomialCrossoverParameters(; dist=Uniform(0.0, 1.0)),
    )
end

function construct_rl_default_adaptive_de_mutstrat_bin(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(
            mut_strat;
            sel=GlobalOptimization.RadiusLimitedSelector(8),
        ),
        crossover_params=SelfBinomialCrossoverParameters(),
    )
end

function construct_rl_uniform_adaptive_de_mutstrat_bin(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(
            mut_strat;
            dist=Uniform(0.0, 1.0),
            sel=GlobalOptimization.RadiusLimitedSelector(8),
        ),
        crossover_params=SelfBinomialCrossoverParameters(; dist=Uniform(0.0, 1.0)),
    )
end

function construct_rs_default_adaptive_de_mutstrat_bin(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(
            mut_strat;
            sel=GlobalOptimization.RandomSubsetSelector(8),
        ),
        crossover_params=SelfBinomialCrossoverParameters(),
    )
end

function construct_rs_uniform_adaptive_de_mutstrat_bin(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(
            mut_strat;
            dist=Uniform(0.0, 1.0),
            sel=GlobalOptimization.RandomSubsetSelector(8),
        ),
        crossover_params=SelfBinomialCrossoverParameters(; dist=Uniform(0.0, 1.0)),
    )
end

function construct_default_adaptive_de_mutstrat_covbin(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(mut_strat),
        crossover_params=SelfBinomialCrossoverParameters(;
            transform=GlobalOptimization.CovarianceTransformation(
                0.1, 0.5, GlobalOptimization.numdims(prob.ss)
            ),
        ),
    )
end

function construct_uniform_adaptive_de_mutstrat_covbin(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(mut_strat; dist=Uniform(0.0, 1.0)),
        crossover_params=SelfBinomialCrossoverParameters(;
            dist=Uniform(0.0, 1.0),
            transform=GlobalOptimization.CovarianceTransformation(
                0.1, 0.5, GlobalOptimization.numdims(prob.ss)
            ),
        ),
    )
end

function construct_rl_default_adaptive_de_mutstrat_covbin(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(
            mut_strat;
            sel=GlobalOptimization.RadiusLimitedSelector(8),
        ),
        crossover_params=SelfBinomialCrossoverParameters(;
            transform=GlobalOptimization.CovarianceTransformation(
                0.1, 0.5, GlobalOptimization.numdims(prob.ss)
            ),
        ),
    )
end

function construct_rl_uniform_adaptive_de_mutstrat_covbin(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(
            mut_strat;
            dist=Uniform(0.0, 1.0),
            sel=GlobalOptimization.RadiusLimitedSelector(8),
        ),
        crossover_params=SelfBinomialCrossoverParameters(;
            dist=Uniform(0.0, 1.0),
            transform=GlobalOptimization.CovarianceTransformation(
                0.1, 0.5, GlobalOptimization.numdims(prob.ss)
            ),
        ),
    )
end

function construct_rs_default_adaptive_de_mutstrat_covbin(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(
            mut_strat;
            sel=GlobalOptimization.RandomSubsetSelector(8),
        ),
        crossover_params=SelfBinomialCrossoverParameters(;
            transform=GlobalOptimization.CovarianceTransformation(
                0.1, 0.5, GlobalOptimization.numdims(prob.ss)
            )
        ),
    )
end

function construct_rs_uniform_adaptive_de_mutstrat_covbin(prob, mut_strat, pop_size, max_iters)
    return SerialDE(
        prob;
        display=false,
        num_candidates=pop_size,
        max_iterations=max_iters,
        max_stall_iterations=max_iters,
        mutation_params=SelfMutationParameters(
            mut_strat;
            dist=Uniform(0.0, 1.0),
            sel=GlobalOptimization.RandomSubsetSelector(8),
        ),
        crossover_params=SelfBinomialCrossoverParameters(;
            dist=Uniform(0.0, 1.0),
            transform=GlobalOptimization.CovarianceTransformation(
                0.1, 0.5, GlobalOptimization.numdims(prob.ss)
            )
        ),
    )
end
