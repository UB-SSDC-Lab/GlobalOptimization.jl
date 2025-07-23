
module OptimLocalSearchExt

# ADTypes and LineSearches are only used by OptimLocalSearchExt. However, Julia's extensions currently cannot
# have their own dependencies, see:
#   https://github.com/JuliaLang/Pkg.jl/issues/3641
#   https://github.com/JuliaLang/julia/issues/52663
# So they're currently still dependencies of GlobalOptimization.jl
using ADTypes: AbstractADType
using LineSearches: HagerZhang, InitialStatic

using GlobalOptimization, UnPack
using Optim: Optim

"""
    LBFGSLocalSearch{T,AT,OT,AD<:Union{ADTypes.AbstractADType, Nothing}}

A local search algorithm that uses the LBFGS algorithm with box constraints to locally
improve the candidate solution.

Note that this method employs the `LBFGS` algorithm with the `Fminbox` wrapper from
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

# Fields
- `percent_decrease_tolerance::T`: The tolerance on the percent decrease of the objective
    function for performing another local search. I.e., if after a local search involving
    `iters_per_solve` iterations, the objective function value is reduced by more than
    `percent_decrease_tolerance` percent, then another local search is performed.
- `alg::AT`: The `LBFGS` algorithm with the `Fminbox` wrapper.
- `options::OT`: The Optim.jl options. Only used to enforce the number of iterations
    performed in each local search.
- `max_solve_time::Float64`: The maximum time per solve in seconds. If a solve does not
    finish in this time, the solve process is terminated.
- `cache::LocalSearchSolutionCache{T}`: The solution cache for storing the solution from
    optimization with Optim.jl.
- `ad::AD`: The autodiff method to use. If `nothing`, then the default of ForwardDiff.jl is
    used. Can be any of the autodiff methods from
    [ADTypes.jl](https://github.com/SciML/ADTypes.jl).
"""
struct LBFGSLocalSearch{T,AT,OT,AD<:Union{AbstractADType,Nothing},UTO<:Union{Val{true},Val{false}}} <: GlobalOptimization.LBFGSLocalSearch{T}

    # Tollerance on percent decrease of objective function for performing another local search
    percent_decrease_tolerance::T

    # The LBFGS algorithm
    alg::AT

    # The Optim.jl options
    options::OT

    # Max time per solve
    use_timeout::UTO
    max_solve_time::Float64

    # Solution cache
    cache::GlobalOptimization.LocalSearchSolutionCache{T}

    # Autodiff method
    ad::AD

    @doc """
        LBFGSLocalSearch{T}(;
            iters_per_solve::Int=5,
            percent_decrease_tol::Number=50.0,
            m::Int=10,
            alphaguess=LineSearches.InitialStatic(),
            linesearch=LineSearches.HagerZhang(),
            manifold=Optim.Flat(),
            max_solve_time::Float64=0.1,
            ad=nothing,
        )

    Create a new `LBFGSLocalSearch` object with the given parameters.

    # Keyword Arguments
    - `iters_per_solve::Int`: The number of iterations to perform in each local search.
    - `percent_decrease_tol::Number`: The tolerance on the percent decrease of the objective
        function for performing another local search. I.e., if after a local search involving
        `iters_per_solve` iterations, the objective function value is reduced by more than
        `percent_decrease_tol` percent, then another local search is performed.
    - `m::Int`: The number of recent steps to employ in approximating the Hessian.
    - `alphaguess`: The initial guess for the step length. Default is
        `LineSearches.InitialStatic()`.
    - `linesearch`: The line search method to use. Default is `LineSearches.HagerZhang()`.
    - `manifold`: The manifold to use. Default is `Optim.Flat()`.
    - `max_solve_time::Float64`: The maximum time per solve in seconds. If a solve does not
        finish in this time, the solve process is terminated.
    - `ad`: The autodiff method to use. If `nothing`, then the default of ForwardDiff.jl is
        used. Can be any of the autodiff methods from
        [ADTypes.jl](https://github.com/SciML/ADTypes.jl).
    """
    function LBFGSLocalSearch{T}(;
        iters_per_solve::Int=5,
        percent_decrease_tol::Number=50.0,
        m::Int=10,
        alphaguess=InitialStatic(),
        linesearch=HagerZhang(),
        manifold=Optim.Flat(),
        use_timeout::VT=Val{true}(),
        max_solve_time=0.1,
        ad=nothing,
    ) where {T<:AbstractFloat,VT<:Union{Val{true},Val{false}}}
        alg = Optim.Fminbox(
            Optim.LBFGS(;
                m=m, alphaguess=alphaguess, linesearch=linesearch, manifold=manifold
            ),
        )
        opts = Optim.Options(; iterations=iters_per_solve)
        return new{T,typeof(alg),typeof(opts),typeof(ad),VT}(
            T(percent_decrease_tol),
            alg,
            opts,
            use_timeout,
            max_solve_time,
            GlobalOptimization.LocalSearchSolutionCache{T}(),
            ad,
        )
    end
end
function GlobalOptimization.LBFGSLocalSearch{T}(args...; kwargs...) where T
    return LBFGSLocalSearch{T}(args...; kwargs...)
end

function GlobalOptimization.initialize!(ls::LBFGSLocalSearch, num_dims)
    GlobalOptimization.initialize!(ls.cache, num_dims)
    return nothing
end

function optim_solve!(cache::GlobalOptimization.LocalSearchSolutionCache, prob, x0, alg, options)
    res = Optim.optimize(
        GlobalOptimization.get_scalar_function(prob), prob.ss.dim_min, prob.ss.dim_max, x0, alg, options;
    )
    cache.x .= Optim.minimizer(res)
    cache.cost = Optim.minimum(res)
    return true
end
function optim_solve!(cache::GlobalOptimization.LocalSearchSolutionCache, prob, x0, alg, ad, options)
    res = Optim.optimize(
        GlobalOptimization.get_scalar_function(prob),
        prob.ss.dim_min,
        prob.ss.dim_max,
        x0,
        alg,
        options;
        autodiff=ad,
    )
    cache.x .= Optim.minimizer(res)
    cache.cost = Optim.minimum(res)
    return true
end

HasAD(::LBFGSLocalSearch{T,AT,OT,Nothing}) where {T,AT,OT} = Val{false}()
HasAD(::LBFGSLocalSearch{T,AT,OT,<:AbstractADType}) where {T,AT,OT} = Val{true}()

GlobalOptimization.get_solve_fun(eval,ls::LBFGSLocalSearch) = get_solve_fun(eval, ls, HasAD(ls))
function get_solve_fun(evaluator, ls::LBFGSLocalSearch, ::Val{false})
    @unpack prob = evaluator
    @unpack alg, options, cache = ls
    solve! = let cache = cache, prob = prob, alg = alg, options = options
        x -> optim_solve!(cache, prob, x, alg, options)
    end
    return solve!
end
function get_solve_fun(evaluator, ls::LBFGSLocalSearch, ::Val{true})
    @unpack prob = evaluator
    @unpack alg, options, cache, ad = ls
    solve! = let cache = cache, prob = prob, alg = alg, options = options, ad = ad
        x -> optim_solve!(cache, prob, x, alg, ad, options)
    end
    return solve!
end

end
