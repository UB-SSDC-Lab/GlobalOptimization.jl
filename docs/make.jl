using Revise;
Revise.revise()
using GlobalOptimization
using Documenter

DocMeta.setdocmeta!(
    GlobalOptimization, :DocTestSetup, :(using GlobalOptimization); recursive=true
)

makedocs(;
    modules=[GlobalOptimization],
    authors="Grant Hecht",
    repo=Documenter.Remotes.GitHub(
        "GrantHecht",
        "https://github.com/UB-SSDC-Lab/GlobalOptimization.jl/blob/{commit}{path}#{line}",
    ),
    sitename="GlobalOptimization.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://UB-SSDC-Lab.github.io/GlobalOptimization.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Reference" => ["Public API" => "lib/public.md"],
        "Developers" => [
            "Contributing" => "dev/contributing.md",
            "Internals" => map(
                s -> "lib/internal/$(s)",
                sort(readdir(joinpath(@__DIR__, "src/lib/internal"))),
            ),
        ],
    ],
)

deploydocs(; repo="github.com/UB-SSDC-Lab/GlobalOptimization.jl", devbranch="main")
