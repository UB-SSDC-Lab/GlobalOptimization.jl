
# Define a FixedSelector for testing
# This selector ignores the population size and returns a fixed vector.
struct FixedSelector <: GlobalOptimization.AbstractSelector
    fixed::Vector{Int}
end
GlobalOptimization.select(s::FixedSelector, target, pop_size::Int) = s.fixed

# To get predictable donor selections we override rand for Vector{Int} temporarily.
# We define a helper that installs a sequential rand method for vectors.
# This method returns successive elements of the vector (cycling if necessary).
module TestRandOverride
    export with_sequential_rand

    # Create a local counter (ref) so that each call returns the next element.
    function with_sequential_rand(f::Function)
        # Create a mutable counter (local to this closure).
        counter = Ref(0)

        # Define our sequential rand: each call returns the element at position:
        #    mod1(counter+1, length(v))
        sequential_rand(v::Vector{Int}) = begin
            counter[] += 1
            v[mod1(counter[], length(v))]
        end

        f(sequential_rand)  # run the passed function under the override
    end

end # module TestRandOverride

@testset "Mutation Operator Detailed Tests" begin
    # For these tests we use a small population with known candidate vectors.
    # Let population size be 6
    pop_size = 10

    # Define candidates: candidate i is [i, i, i]
    candidates = [fill(Float64(i), DIM) for i in 1:pop_size]

    # Create current generation and mutants generation.
    population = GlobalOptimization.DEPopulation(pop_size, DIM)
    population.current_generation.candidates .= candidates

    # We choose an arbitrary best candidate (used in the F₁ branch).
    best_candidate = fill(10.0, DIM)

    # Test F₁ branch only: mutant = candidate + F₁*(best - candidate)
    @testset "F1 branch only" begin
        # Set parameters: F1 nonzero, others zero.
        mp = MutationParameters(
            Rand1();
            dist=GlobalOptimization.default_mutation_dist,
            sel=GlobalOptimization.SimpleSelector(),
        )
        mp.F1 = 0.5; mp.F2 = 0.0; mp.F3 = 0.0; mp.F4 = 0.0

        # Reset mutants to a copy of candidates.
        GlobalOptimization.mutate!(population, mp, best_candidate)
        for i in 1:pop_size
            expected = candidates[i] .+ 0.5 .* (best_candidate .- candidates[i])
            @test population.mutants.candidates[i] ≈ expected
        end
    end

    # Test F₂ branch only: mutant = candidate + F₂*(candidate[r1] - candidate)
    # We want to force r1 to be predictable. For each candidate,
    # we use a FixedSelector that returns a fixed vector that does not contain the candidate’s index.
    # For simplicity, we test only candidate 1.
    @testset "F2 branch only" begin
        # Use a FixedSelector that returns [2, 3, 4, 5, 6].
        fixed_sel = FixedSelector([2, 3, 4, 5, 6])
        mp = MutationParameters(
            Rand1();
            dist=GlobalOptimization.default_mutation_dist,
            sel=fixed_sel,
        )
        mp.F1 = 0.0; mp.F2 = 0.3; mp.F3 = 0.0; mp.F4 = 0.0

        # Override rand so that for any vector input it returns the first element.
        TestRandOverride.with_sequential_rand(
            rng_idx -> GlobalOptimization.mutate!(population, mp, best_candidate, rng_idx)
        )

        for i in 1:pop_size
            expected = candidates[i] .+ 0.3 .* (candidates[fixed_sel.fixed[mod1(i,5)]] .- candidates[i])
            @test population.mutants.candidates[i] ≈ expected
        end
    end

    # Test F₃ branch only: mutant = candidate + F₃*(candidate[r2] - candidate[r3])
    # We test candidate 1 again.
    @testset "F3 branch only" begin
        fixed_sel = FixedSelector([2, 3, 4, 5, 6])
        mp = MutationParameters(
            Rand1();
            dist=GlobalOptimization.default_mutation_dist,
            sel=fixed_sel,
        )
        mp.F1 = 0.0; mp.F2 = 0.0; mp.F3 = 0.4; mp.F4 = 0.0

        # Now, the F₃ branch calls:
        #   r2 = rand(idxs) while r2 in (i, r1)
        #   r3 = rand(idxs) while r3 in (i, r1, r2)
        # Because F₂ is off, r1 remains 0.
        # With our sequential rand override, for candidate 1:
        #   r2 becomes fixed_sel[1] = 2.
        #   Then r3 becomes fixed_sel[2] = 3.
        # So the contribution is 0.4*(candidate[2] - candidate[3]).
        TestRandOverride.with_sequential_rand(
            rng_idx -> GlobalOptimization.mutate!(population, mp, best_candidate, rng_idx)
        )

        rng_count = 0
        for i in 1:pop_size
            rng_count += 1
            r2 = fixed_sel.fixed[mod1(rng_count,5)]
            if r2 == i
                rng_count += 1
                r2 = fixed_sel.fixed[mod1(rng_count,5)]
            end
            rng_count += 1
            r3 = fixed_sel.fixed[mod1(rng_count,5)]
            if r3 == i
                rng_count += 1
                r3 = fixed_sel.fixed[mod1(rng_count,5)]
            end

            expected = candidates[i] .+ 0.4 .* (candidates[r2] .- candidates[r3])
            @test population.mutants.candidates[i] ≈ expected
        end
    end

    # Test F₄ branch only: mutant = candidate + F₄*(candidate[r4] - candidate[r5])
    @testset "F4 branch only" begin
        fixed_sel = FixedSelector([2, 3, 4, 5, 6])
        mp = MutationParameters(
            Rand1();
            dist=GlobalOptimization.default_mutation_dist,
            sel=fixed_sel,
        )
        mp.F1 = 0.0; mp.F2 = 0.0; mp.F3 = 0.0; mp.F4 = 0.7

        # In the F₄ branch:
        #   r4 = rand(idxs) while r4 in (i, r1, r2, r3)
        #   r5 = rand(idxs) while r5 in (i, r1, r2, r3, r4)
        # With our sequential override (and with F₂ and F₃ off, r1, r2, r3 are 0 or not used),
        # for candidate 1:
        #   r4 becomes fixed_sel[1] = 2.
        #   r5 becomes fixed_sel[2] = 3.
        # So expected contribution is 0.7*(candidate[2] - candidate[3]).
        TestRandOverride.with_sequential_rand(
            rng_idx -> GlobalOptimization.mutate!(population, mp, best_candidate, rng_idx)
        )

        rng_count = 0
        for i in 1:pop_size
            rng_count += 1
            r2 = fixed_sel.fixed[mod1(rng_count,5)]
            if r2 == i
                rng_count += 1
                r2 = fixed_sel.fixed[mod1(rng_count,5)]
            end
            rng_count += 1
            r3 = fixed_sel.fixed[mod1(rng_count,5)]
            if r3 == i
                rng_count += 1
                r3 = fixed_sel.fixed[mod1(rng_count,5)]
            end

            expected = candidates[i] .+ 0.7 .* (candidates[r2] .- candidates[r3])
            @test population.mutants.candidates[i] ≈ expected
        end
    end

    # Test Combined Branches: All F parameters nonzero.
    @testset "Combined Branches" begin
        fixed_sel = FixedSelector([2, 3, 4, 5, 6, 7, 8])
        mp = MutationParameters(
            Rand1();
            dist=GlobalOptimization.default_mutation_dist,
            sel=fixed_sel,
        )
        mp.F1 = 0.2; mp.F2 = 0.3; mp.F3 = 0.4; mp.F4 = 0.5

        # With our sequential override, the branches for candidate 1 will be:
        # F1: contributes 0.2*(best - candidate[1])
        # F2: r1 becomes fixed_sel[1] = 2 → contributes 0.3*(candidate[2] - candidate[1])
        # F3: r2 becomes fixed_sel[1] = 2, r3 becomes fixed_sel[2] = 3 → contributes 0.4*(candidate[2] - candidate[3])
        # F4: r4 becomes fixed_sel[1] = 2, r5 becomes fixed_sel[2] = 3 → contributes 0.5*(candidate[2] - candidate[3])
        TestRandOverride.with_sequential_rand(
            rng_idx -> GlobalOptimization.mutate!(population, mp, best_candidate, rng_idx)
        )
        expected = candidates[1] .+ 0.2 .* (best_candidate .- candidates[1]) .+
                   0.3 .* (candidates[2] .- candidates[1]) .+
                   0.4 .* (candidates[2] .- candidates[3]) .+
                   0.5 .* (candidates[2] .- candidates[3])
        @test population.mutants.candidates[1] ≈ expected

        rng_count = 0
        for i in 1:pop_size
            r1 = 0; r2 = 0; r3 = 0; r4 = 0; r5 = 0
            while r1 == 0 || r1 == i
                rng_count += 1
                r1 = fixed_sel.fixed[mod1(rng_count,7)]
            end

            while r2 == 0 || r2 in (i, r1)
                rng_count += 1
                r2 = fixed_sel.fixed[mod1(rng_count,7)]
            end

            while r3 == 0 || r3 in (i, r1, r2)
                rng_count += 1
                r3 = fixed_sel.fixed[mod1(rng_count,7)]
            end

            while r4 == 0 || r4 in (i, r1, r2, r3)
                rng_count += 1
                r4 = fixed_sel.fixed[mod1(rng_count,7)]
            end

            while r5 == 0 || r5 in (i, r1, r2, r3, r4)
                rng_count += 1
                r5 = fixed_sel.fixed[mod1(rng_count,7)]
            end

            expected = candidates[i] .+ 0.2 .* (best_candidate .- candidates[i]) .+
                   0.3 .* (candidates[r1] .- candidates[i]) .+
                   0.4 .* (candidates[r2] .- candidates[r3]) .+
                   0.5 .* (candidates[r4] .- candidates[r5])
            @test population.mutants.candidates[i] ≈ expected
        end
    end
end
