
"""
    AbstractCrossoverParameters

An abstract type representing the parameters for a crossover strategy in Differential Evolution (DE).

Subtypes of this abstract type should define the following methods:
- `get_parameter(params::AbstractCrossoverParameters, i)`: Returns the crossover parameter for the `i`-th candidate.
- `initialize!(params::AbstractCrossoverParameters, num_dims, population_size)`: Initializes the crossover parameters.
- `adapt!(params::AbstractCrossoverParameters, improved, global_best_improved)`: Adapts the crossover parameters based on the improvement status of the candidates.
- `crossover!(population::DEPopulation, crossover_params, search_space)`: Performs the crossover operation on the population using the specified crossover parameters.
"""
abstract type AbstractCrossoverParameters{AS<:AbstractAdaptationStrategy} end

"""
    AbstractBinomialCrossoverParameters

An abstract type representing the parameters for a binomial crossover strategy in Differential Evolution (DE).
The `crossover!` method is provided for subtypes of this abstract type, however, the
`get_parameter`, `initialize!`, and `adapt!` methods are must still be defined.
"""
abstract type AbstractBinomialCrossoverParameters{AS} <: AbstractCrossoverParameters{AS} end

"""
    AbstractCrossoverTransformation

An abstract type representing a transformation applied to a candidate prior to applying the
crossover operator.

Subtypes of this abstract type should define the following methods:
- `initialize!(transformation::AbstractCrossoverTransformation, population_size)`: Initializes the transformation with the population
- `update_transformation!(transformation::AbstractCrossoverTransformation, population)`: Updates the transformation based on the current population
- `to_transformed(transformation::AbstractCrossoverTransformation, c, m)`: Transforms the candidate `c` and mutant `m` to an
    alternative representation, returning the transformed candidate, transformed mutant, and a boolean indicating whether the transformation was applied.
- `from_transformed!(transformation::AbstractCrossoverTransformation, mt, m)`: Transforms the mutant `mt` back to the original representation `m`.
"""
abstract type AbstractCrossoverTransformation end

"""
    NoTransformation

A transformation that does not apply any transformation to the candidate or mutant.
"""
struct NoTransformation <: AbstractCrossoverTransformation end

"""
    CovarianceTransformation{T<:AbstractCrossoverTransformation}

A transformation for performing crossover in the eigen-space of the covariance matrix of the
best candidates in the population.

This is an implementation of the method proposed by Wang and Li in "Differential Evolution
Based on Covariance Matrix Learning and Bimodal Distribution Parameter Setting, " 2014,
DOI: [10.1016/j.asoc.2014.01.038](https://doi.org/10.1016/j.asoc.2014.01.038).

# Fields
- `ps::Float64`: The proportion of candidates to consider in the covariance matrix. That is,
    for a population size of `N`, the covariance matrix is calculated using the
    `clamp(ceil(ps * N), 2, N)` best candidates.
- `pb::Float64`: The probability of applying the transformation.
- `B::Matrix{Float64}`: The real part of the eigenvectors of the covariance matrix.
- `ct::Vector{Float64}`: The transformed candidate.
- `mt::Vector{Float64}`: The transformed mutant.
"""
struct CovarianceTransformation <: AbstractCrossoverTransformation
    ps::Float64
    pb::Float64
    B::Matrix{Float64}

    # Preallocate storage for transformed candidate and mutant
    ct::Vector{Float64}
    mt::Vector{Float64}

    # Preallocate storage for calculating transformation
    idxs::Vector{UInt16}

    @doc """
        CovarianceTransformation(ps::Float64, pb::Float64, num_dims::Int)

    Creates a `CovarianceTransformation` object with the specified proportion of candidates
    to consider in the covariance matrix `ps`, the probability of applying the transformation
    `pb`, and the number of dimensions `num_dims`.

    This is an implementation of the method proposed by Wang and Li in "Differential Evolution
    Based on Covariance Matrix Learning and Bimodal Distribution Parameter Setting, " 2014,
    DOI: [10.1016/j.asoc.2014.01.038](https://doi.org/10.1016/j.asoc.2014.01.038).

    # Arguments
    - `ps::Float64`: The proportion of candidates to consider in the covariance matrix.
    - `pb::Float64`: The probability of applying the transformation.
    - `num_dims::Int`: The number of dimensions in the search space.

    # Returns
    - `CovarianceTransformation`: A `CovarianceTransformation` object with the specified
        parameters.

    # Examples
    ```julia-repl
    julia> using GlobalOptimization
    julia> transformation = CovarianceTransformation(0.5, 0.5, 10)
    CovarianceTransformation(0.5, 0.5, [2.3352254645e-314 6.3877104275e-314 … 1.0e-323 5.0e-324; 6.3877051114e-314 6.3877104196e-314 … 6.3877054276e-314 6.387705455e-314; … ; 2.3352254645e-314 2.333217732e-314 … 0.0 6.3877095184e-314; 6.387705143e-314 2.130067282e-314 … 6.387705459e-314 6.387705463e-314], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ```
    """
    function CovarianceTransformation(ps, pb, num_dims)
        if ps <= 0.0 || ps > 1.0
            throw(ArgumentError("ps must be in the range (0, 1]."))
        end
        if pb <= 0.0 || pb > 1.0
            throw(ArgumentError("pb must be in the range (0, 1]."))
        end
        B = Matrix{Float64}(undef, num_dims, num_dims)
        return new(ps, pb, B, zeros(num_dims), zeros(num_dims), Vector{UInt16}(undef, 0))
    end
end

initialize!(transformation::NoTransformation, population_size) = nothing
function initialize!(transformation::CovarianceTransformation, population_size)
    resize!(transformation.idxs, population_size)
    transformation.idxs .= 1:population_size
    return nothing
end

update_transformation!(transformation::NoTransformation, population) = nothing
function update_transformation!(transformation::CovarianceTransformation, population)

    # Get number of candidates to consider in the covariance matrix
    n = clamp(ceil(Int, transformation.ps * length(population)), 2, length(population))

    # Get indices of n best candidates
    sortperm!(transformation.idxs, population.current_generation.candidates_fitness)
    idxs = view(transformation.idxs, 1:n)

    # Calculate the covariance matrix for n best candidates
    C = cov(view(population.current_generation.candidates, idxs))

    # Compute eigen decomposition
    E = eigen!(C)
    transformation.B .= real.(E.vectors)

    return nothing
end

to_transformed(transformation::NoTransformation, c, m) = c, m, false
function to_transformed(transformation::CovarianceTransformation, c, m)
    if rand() < transformation.pb
        mul!(transformation.ct, transpose(transformation.B), c)
        mul!(transformation.mt, transpose(transformation.B), m)
        return transformation.ct, transformation.mt, true
    else
        return c, m, false
    end
end

from_transformed!(transformation::NoTransformation, mt, m) = nothing
function from_transformed!(transformation::CovarianceTransformation, mt, m)
    mul!(m, transformation.B, mt)
    return nothing
end

"""
    CorrelatedCovarianceTransformation{T<:AbstractCrossoverTransformation}

A transformation for performing crossover in the eigen-space of the covariance matrix of the
best candidates in the population which are also not too closely correlated.

This is an implementation of the method proposed by Wang and Li in "Differential Evolution
Based on Covariance Matrix Learning and Bimodal Distribution Parameter Setting, " 2014,
DOI: [10.1016/j.asoc.2014.01.038](https://doi.org/10.1016/j.asoc.2014.01.038).

Correlation alteration based on "Covariance Matrix Learning Differential Evolution Algorithm Based on Correlation"
DOI: https://doi.org/10.4236/ijis.2021.111002

# Fields
- `ps::Float64`: The proportion of candidates to consider in the covariance matrix. That is,
    for a population size of `N` with `M` candidates remaining after the correlation check, the covariance matrix is calculated using the
    `clamp(ceil(ps * M), 2, M)` best candidates.
- `pb::Float64`: The probability of applying the transformation.
- `a::Float64`: The correlation threshold for which two candidates are considered 'too close' to both be used in the covariance matrix construction. 
- `B::Matrix{Float64}`: The real part of the eigenvectors of the covariance matrix.
- `ct::Vector{Float64}`: The transformed candidate.
- `mt::Vector{Float64}`: The transformed mutant.
"""
struct CorrelatedCovarianceTransformation <: AbstractCrossoverTransformation
    ps::Float64
    pb::Float64
    a::Float64
    B::Matrix{Float64}

    # Preallocate storage for transformed candidate and mutant
    ct::Vector{Float64}
    mt::Vector{Float64}

    # Preallocate storage for calculating transformation
    idxs::Vector{UInt16}

    
        @doc """
        CorrelatedCovarianceTransformation{T<:AbstractCrossoverTransformation}

    A transformation for performing crossover in the eigen-space of the covariance matrix of the
    best candidates in the population which are also not too closely correlated.

    This is an implementation of the method proposed by Wang and Li in "Differential Evolution
    Based on Covariance Matrix Learning and Bimodal Distribution Parameter Setting, " 2014,
    DOI: [10.1016/j.asoc.2014.01.038](https://doi.org/10.1016/j.asoc.2014.01.038).

    Correlation alteration based on "Covariance Matrix Learning Differential Evolution Algorithm Based on Correlation"
    DOI: https://doi.org/10.4236/ijis.2021.111002

    # Arguments
    - `ps::Float64`: The proportion of candidates to consider in the covariance matrix.
    - `pb::Float64`: The probability of applying the transformation.
    - `a::Float64`: The correlation threshold for two candidates being 'too close'.
    - `num_dims::Int`: The number of dimensions in the search space.

    # Returns
    - `CorrelatedCovarianceTransformation`: A `CorrelatedCovarianceTransformation` object with the specified
        parameters.

    # Examples
    ```julia-repl
    julia> using GlobalOptimization
    julia> transformation = CorrelatedCovarianceTransformation(0.5, 0.5, .95, 10)
    CorrelatedCovarianceTransformation(0.5, 0.5, .95, [2.3352254645e-314 6.3877104275e-314 … 1.0e-323 5.0e-324; 6.3877051114e-314 6.3877104196e-314 … 6.3877054276e-314 6.387705455e-314; … ; 2.3352254645e-314 2.333217732e-314 … 0.0 6.3877095184e-314; 6.387705143e-314 2.130067282e-314 … 6.387705459e-314 6.387705463e-314], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ```
    """
    function CorrelatedCovarianceTransformation(ps, pb, a, num_dims)
        if ps <= 0.0 || ps > 1.0
            throw(ArgumentError("ps must be in the range (0, 1]."))
        end
        if pb <= 0.0 || pb > 1.0
            throw(ArgumentError("pb must be in the range (0, 1]."))
        end
        if a <= 0.0 || a > 1.0
            throw(ArgumentError("a must be in the range (0, 1]."))
        end
        B = Matrix{Float64}(undef, num_dims, num_dims)
        return new(ps, pb, a, B, zeros(num_dims), zeros(num_dims), Vector{UInt16}(undef, 0))
    end

end

function initialize!(transformation::CorrelatedCovarianceTransformation, population_size)
    resize!(transformation.idxs, population_size)
    transformation.idxs .= 1:population_size
    return nothing
end

function update_transformation!(transformation::CorrelatedCovarianceTransformation, population)
    # get correlation for each pair of vectors in population
    cor_mat = cor(stack(population.current_generation.candidates))

    if all(x -> x >= transformation.a, cor_mat) # if all candidates are sufficiently correlated, use all in the cov matrix
        # Calculate the covariance matrix for n best candidates
        C = cov(view(population.current_generation.candidates, transformation.idxs))

        # Compute eigen decomposition
        E = eigen!(C)
        transformation.B .= real.(E.vectors)
    else

        # lower triangular so that we only have unique pairs (excluding diagonal, since all elements are 1.0)
        tril!(cor_mat, -1)

        #  find points where two candidates are strongly correlated
        idxs_cart = findall(x->x >= transformation.a, cor_mat); #list of cartesian indexes of highly-correlated pairs
        
        idxs_to_remove = Vector{Int}(undef, 0)
        for pair in idxs_cart
            if !in(pair.I[1], idxs_to_remove)
                Base.push!(idxs_to_remove, pair[1]) # for each pair, remove the element in the first position
            end
        end

        # remove candidates
        remaining_idxs = setdiff(transformation.idxs, idxs_to_remove)
        remaining_pop_fitness = view(population.current_generation.candidates_fitness, remaining_idxs)

        # get number of candidates to use for covariance based on remaining candidates
        n = clamp(ceil(Int, transformation.ps * length(remaining_idxs)), 2, length(remaining_idxs))
        
        # Get indices of n best remaining candidates
        sortperm!(remaining_idxs, remaining_pop_fitness)
        idxs = view(transformation.idxs, remaining_idxs[1:n])
        # Calculate the covariance matrix for n best candidates
        C = cov(view(population.current_generation.candidates, idxs))

        # Compute eigen decomposition
        E = eigen!(C)
        transformation.B .= real.(E.vectors)
    end
    
    
    return nothing
end

function to_transformed(transformation::CorrelatedCovarianceTransformation, c, m)
        if rand() < transformation.pb
        mul!(transformation.ct, transpose(transformation.B), c)
        mul!(transformation.mt, transpose(transformation.B), m)
        return transformation.ct, transformation.mt, true
    else
        return c, m, false
    end
end

function from_transformed!(transformation::CorrelatedCovarianceTransformation, mt, m)
    mul!(m, transformation.B, mt)
    return nothing
end
"""
    BinomialCrossoverParameters{
        AS<:AbstractAdaptationStrategy,
        T<:AbstractCrossoverTransformation,
        D,
    }

The parameters for a DE binomial crossover strategy.

# Fields
- `CR::Float64`: The crossover rate.
- `transform::T`: The transformation to apply to the candidate and mutant.
- `dist<:Distribution{Univariate,Continuous}`: The distribution used to adapt the crossover
    rate parameter. Note that this should generally be a distribution from
    [Distributions.jl](https://juliastats.org/Distributions.jl/stable/), but the only
    strict requirement is that rand(dist) returns a floating point value.
"""
mutable struct BinomialCrossoverParameters{
    AS<:AbstractAdaptationStrategy,T<:AbstractCrossoverTransformation,D
} <: AbstractBinomialCrossoverParameters{AS}
    CR::Float64
    transform::T
    dist::D

    @doc """
        BinomialCrossoverParameters(CR::Float64; transform=NoTransformation())

    Creates a `BinomialCrossoverParameters` object with a fixed crossover rate `CR` and
    optional transformation `transform`.

    # Arguments
    - `CR::Float64`: The crossover rate.

    # Keyword Arguments
    - `transform::AbstractCrossoverTransformation`: The transformation to apply to the
        candidate and mutant. Defaults to `NoTransformation()`.

    # Returns
    - `BinomialCrossoverParameters{NoAdaptation,typeof(transform),Nothing}`: A
        `BinomialCrossoverParameters` object with a fixed crossover rate and the optionally
        specified transformation.

    # Examples
    ```julia-repl
    julia> using GlobalOptimization
    julia> params = BinomialCrossoverParameters(0.5)
    BinomialCrossoverParameters{GlobalOptimization.NoAdaptation, GlobalOptimization.NoTransformation, Nothing}(0.5, GlobalOptimization.NoTransformation(), nothing)
    ```
    ```julia-repl
    julia> using GlobalOptimization
    julia> params = BinomialCrossoverParameters(0.5, transform=CovarianceTransformation(0.5, 0.5, 10))
    BinomialCrossoverParameters{GlobalOptimization.NoAdaptation, CovarianceTransformation, Nothing}(0.5, CovarianceTransformation(0.5, 0.5, [0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), nothing)
    ```
    """
    function BinomialCrossoverParameters(CR::Float64; transform=NoTransformation())
        return new{NoAdaptation,typeof(transform),Nothing}(CR, transform, nothing)
    end

    @doc """
        BinomialCrossoverParameters(; dist=default_binomial_crossover_dist, transform=NoTransformation())

    Creates a `BinomialCrossoverParameters` object with an adaptive crossover rate and
    optional transformation `transform`.

    # Keyword Arguments
    - `dist::Distribution{Univariate,Continuous}`: The distribution used to adapt the
        crossover rate parameter. Note that this should generally be a distribution from
        [Distributions.jl](https://juliastats.org/Distributions.jl/stable/), but the only
        strict requirement is that rand(dist) returns a floating point value. Defaults to
        `default_binomial_crossover_dist`, which is a mixture model comprised of two Cauchy
        distributions, with probability density given by:

        ``f_{mix}(x; \\mu, \\sigma) = 0.5 f(x;\\mu_1,\\sigma_1) + 0.5 f(x;\\mu_2,\\sigma_2)``

        where ``\\mu = \\{0.1, 0.95\\}`` and ``\\sigma = \\{0.1, 0.1\\}``.

    - `transform::AbstractCrossoverTransformation`: The transformation to apply to the
        candidate and mutant. Defaults to `NoTransformation()`.

    # Returns
    - `BinomialCrossoverParameters{RandomAdaptation,typeof(transform),typeof(dist)}`: A
        `BinomialCrossoverParameters` object with an adaptive crossover rate and the
        optionally specified transformation.

    # Examples
    ```julia-repl
    julia> using GlobalOptimization
    julia> params = BinomialCrossoverParameters()
    BinomialCrossoverParameters{GlobalOptimization.RandomAdaptation, GlobalOptimization.NoTransformation, Distributions.MixtureModel{Distributions.Univariate, Distributions.Continuous, Distributions.Cauchy, Distributions.Categorical{Float64, Vector{Float64}}}}(0.0, GlobalOptimization.NoTransformation(), MixtureModel{Distributions.Cauchy}(K = 2)
    components[1] (prior = 0.5000): Distributions.Cauchy{Float64}(μ=0.1, σ=0.1)
    components[2] (prior = 0.5000): Distributions.Cauchy{Float64}(μ=0.95, σ=0.1)
    )
    ```
    ```julia-repl
    julia> using GlobalOptimization
    julia> params = BinomialCrossoverParameters(transform=CovarianceTransformation(0.5, 0.5, 10))
    BinomialCrossoverParameters{GlobalOptimization.RandomAdaptation, CovarianceTransformation, Distributions.MixtureModel{Distributions.Univariate, Distributions.Continuous, Distributions.Cauchy, Distributions.Categorical{Float64, Vector{Float64}}}}(0.0, CovarianceTransformation(0.5, 0.5, [2.195780764e-314 2.2117846174e-314 … 2.1293782266e-314 1.5617889024864e-311; 2.366805627e-314 2.316670011e-314 … 2.3355803934e-314 1.4259811738567e-311; … ; 2.195781025e-314 2.195781096e-314 … 1.4531427176862e-310 1.27319747493e-313; 2.366805627e-314 2.366805627e-314 … 1.0270459628367e-310 2.121995795e-314], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), MixtureModel{Distributions.Cauchy}(K = 2)
    components[1] (prior = 0.5000): Distributions.Cauchy{Float64}(μ=0.1, σ=0.1)
    components[2] (prior = 0.5000): Distributions.Cauchy{Float64}(μ=0.95, σ=0.1)
    )
    ```
    ```julia-repl
    julia> using GlobalOptimization
    julia> using Distributions
    julia> params = BinomialCrossoverParameters(dist=Uniform(0.0, 1.0))
    BinomialCrossoverParameters{GlobalOptimization.RandomAdaptation, GlobalOptimization.NoTransformation, Uniform{Float64}}(0.0, GlobalOptimization.NoTransformation(), Uniform{Float64}(a=0.0, b=1.0))
    ```
    """
    function BinomialCrossoverParameters(;
        dist=default_binomial_crossover_dist, transform=NoTransformation()
    )
        return new{RandomAdaptation,typeof(transform),typeof(dist)}(0.0, transform, dist)
    end
end

"""
    SelfBinomialCrossoverParameters{
        AS<:AbstractAdaptationStrategy,
        T<:AbstractCrossoverTransformation,
        D,
    }

The parameters for a DE self-adaptive binomial crossover strategy.

# Fields
- `CRs::Vector{Float64}`: The crossover rates for each candidate in the population.
- `transform::T`: The transformation to apply to the candidate and mutant.
- `dist<:Distribution{Univariate,Continuous}`: The distribution used to adapt the crossover
    rate parameter. Note that this should generally be a distribution from
    [Distributions.jl](https://juliastats.org/Distributions.jl/stable/), but the only
    strict requirement is that rand(dist) returns a floating point value.
"""
struct SelfBinomialCrossoverParameters{
    AS<:AbstractAdaptationStrategy,T<:AbstractCrossoverTransformation,D
} <: AbstractBinomialCrossoverParameters{AS}
    CRs::Vector{Float64}
    transform::T
    dist::D

    @doc """
        SelfBinomialCrossoverParameters(;
            dist=default_binomial_crossover_dist,
            transform=NoTransformation()
        )

    Creates a `SelfBinomialCrossoverParameters` object with an adaptive crossover rate for each
    candidate in the population and an optional transformation `transform`.

    # Keyword Arguments
    - `dist::Distribution{Univariate,Continuous}`: The distribution used to adapt the
        crossover rate parameter. Note that this should generally be a distribution from
        [Distributions.jl](https://juliastats.org/Distributions.jl/stable/), but the only
        strict requirement is that rand(dist) returns a floating point value. Defaults to
        `default_binomial_crossover_dist`, which is a mixture model comprised of two Cauchy
        distributions, with probability density given by:

        ``f_{mix}(x; \\mu, \\sigma) = 0.5 f(x;\\mu_1,\\sigma_1) + 0.5 f(x;\\mu_2,\\sigma_2)``

        where ``\\mu = \\{0.1, 0.95\\}`` and ``\\sigma = \\{0.1, 0.1\\}``.
    - `transform::AbstractCrossoverTransformation`: The transformation to apply to the
        candidate and mutant. Defaults to `NoTransformation()`.

    # Returns
    - `SelfBinomialCrossoverParameters{RandomAdaptation,typeof(transform),typeof(dist)}`: A
        `SelfBinomialCrossoverParameters` object with an adaptive crossover rate for each
        candidate and the optionally specified transformation.

    # Examples
    ```julia-repl
    julia> using GlobalOptimization
    julia> params = SelfBinomialCrossoverParameters()
    SelfBinomialCrossoverParameters{GlobalOptimization.RandomAdaptation, GlobalOptimization.NoTransformation, MixtureModel{Univariate, Continuous, Cauchy, Categorical{Float64, Vector{Float64}}}}(Float64[], GlobalOptimization.NoTransformation(), MixtureModel{Cauchy}(K = 2)
    components[1] (prior = 0.5000): Cauchy{Float64}(μ=0.1, σ=0.1)
    components[2] (prior = 0.5000): Cauchy{Float64}(μ=0.95, σ=0.1)
    )
    ```
    """
    function SelfBinomialCrossoverParameters(;
        dist=default_binomial_crossover_dist, transform=NoTransformation()
    )
        CRs = Vector{Float64}(undef, 0)
        return new{RandomAdaptation,typeof(transform),typeof(dist)}(CRs, transform, dist)
    end
end

get_parameter(params::BinomialCrossoverParameters, i) = params.CR
get_parameter(params::SelfBinomialCrossoverParameters, i) = params.CRs[i]

function initialize!(
    params::AbstractCrossoverParameters{NoAdaptation}, num_dims, population_size
)
    initialize!(params.transform, population_size)
    return nothing
end
function initialize!(
    params::BinomialCrossoverParameters{RandomAdaptation}, num_dims, population_size
)
    initialize!(params.transform, population_size)
    params.CR = one_clamped_rand(params.dist)
    return nothing
end
function initialize!(
    params::SelfBinomialCrossoverParameters{RandomAdaptation}, num_dims, population_size
)
    initialize!(params.transform, population_size)
    resize!(params.CRs, population_size)
    @inbounds for i in eachindex(params.CRs)
        params.CRs[i] = one_clamped_rand(params.dist)
    end
    return nothing
end

function adapt!(
    params::AbstractCrossoverParameters{NoAdaptation}, improved, global_best_improved
)
    return nothing
end
function adapt!(
    params::BinomialCrossoverParameters{RandomAdaptation}, improved, global_best_improved
)
    if !global_best_improved
        params.CR = one_clamped_rand(params.dist)
    end
    return nothing
end
function adapt!(
    params::SelfBinomialCrossoverParameters{RandomAdaptation},
    improved,
    global_best_improved,
)
    @inbounds for i in eachindex(params.CRs)
        if !improved[i]
            params.CRs[i] = one_clamped_rand(params.dist)
        end
    end
    return nothing
end

"""
    crossover!(population::DEPopulation{T}, crossover_params, search_space)

Performs the crossover operation on the population `population` using the DE crossover strategy.

This function also ensures that, after crossover, the mutants are within the search space.
"""
function crossover!(
    population::DEPopulation,
    crossover_params::AbstractBinomialCrossoverParameters,
    search_space,
)
    @unpack current_generation, mutants = population
    @unpack transform = crossover_params

    # Update and get transformation
    update_transformation!(crossover_params.transform, population)

    @inbounds for i in eachindex(mutants)
        CR = get_parameter(crossover_params, i)

        # Get candidate and mutant
        candidate = current_generation.candidates[i]
        mutant = mutants.candidates[i]

        # Get transformed candidate and mutant
        candidate_t, mutant_t, transformed = to_transformed(transform, candidate, mutant)

        # Perform crossover
        mbr_i = rand(1:length(mutant_t))
        for j in eachindex(mutant_t)
            if rand() > CR && j != mbr_i
                mutant_t[j] = candidate_t[j]
            end
        end

        # Transform back to original space
        if transform isa CovarianceTransformation && transformed
            from_transformed!(crossover_params.transform, mutant_t, mutant)
        end

        for j in eachindex(mutant)
            # Ensure mutant is within search space
            if mutant[j] < dim_min(search_space, j)
                mutant[j] = dim_min(search_space, j)
            elseif mutant[j] > dim_max(search_space, j)
                mutant[j] = dim_max(search_space, j)
            end
        end
    end

    return nothing
end
