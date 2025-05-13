import LibGit2

# ==== Base Test Problems ====

sphere(x) = sum(abs2, x)

function rastrigin(x)
    val = 10.0 * length(x)
    return val + sum(abs2,x) - 10.0*sum(xx->cos(2.0 * pi * xx), x)
end

function ackley(x::AbstractVector{T}) where T
    d_inv = 1.0 / length(x)
    t1 = -20.0*exp(-0.2*sqrt(d_inv*sum(abs2,x)))
    t2 = -exp(d_inv*sum(xx->cos(2.0*pi*xx), x)) 
    return t1 + t2 + 20.0 + 2.718281828459045
end

function rosenbrock(x)
    val = zero(eltype(x))
    @inbounds for i in 1:(length(x)-1)
        xi = x[i]
        t1 = x[i+1] - xi*xi
        t2 = xi - 1.0
        val += 100.0 * t1*t1 + t2*t2
    end
    return val
end

function griewank(x)
    sum_term = sum(abs2, x) / 4000.0
    prod_term = one(eltype(x)) 
    @inbounds for i in eachindex(x)
        prod_term *= cos(x[i] / sqrt(i))
    end
    return sum_term - prod_term + 1.0
end

function schwefel1_2(x)
    val = zero(eltype(x))
    inner_sum = zero(eltype(x))
    @inbounds for i in eachindex(x)
        inner_sum += x[i]
        val += inner_sum*inner_sum
    end
    return val
end

schwefel2_21(x) = maximum(abs, x)

schwefel2_22(x) = sum(abs, x) + prod(abs, x)

# ===== Test Problem Set =====
struct TestProblem{F <: Function}
    name::String
    fun::F
    lb_per_dim::Float64
    ub_per_dim::Float64
    min::Float64
end

base_test_problems = Dict{String, TestProblem}(
    "Sphere" => TestProblem("Sphere", sphere, -100.0, 100.0, 0.0),
    "Rastrigin" => TestProblem("Rastrigin", rastrigin, -5.12, 5.12, 0.0),
    "Ackley" => TestProblem("Ackley", ackley, -32.768, 32.768, 0.0),
    "Rosenbrock" => TestProblem("Rosenbrock", rosenbrock, -5.0, 10.0, 0.0),
    "Griewank" => TestProblem("Griewank", griewank, -600.0, 600.0, 0.0),
    "Schwefel 1.2" => TestProblem("Schwefel 1.2", schwefel1_2, -100.0, 100.0, 0.0),
    "Schwefel 2.21" => TestProblem("Schwefel 2.21", schwefel2_21, -100.0, 100.0, 0.0),
    "Schwefel 2.22" => TestProblem("Schwefel 2.22", schwefel2_22, -10.0, 10.0, 0.0),
)

# ==== Git Utility ===
function get_git_commit_hash(; abbrev::Bool = false)
    repo = LibGit2.GitRepoExt(joinpath(@__DIR__, ".."))

    suffix = ""
    if LibGit2.isdirty(repo)
        suffix = "-dirty"
        @warn "Repository is dirty, commit hash may not be accurate."
    end

    commit_hash = LibGit2.head_oid(repo)
    if abbrev
        return string(LibGit2.GitShortHash(commit_hash, 7)) * suffix
    else
        return string(commit_hash) * suffix
    end
end