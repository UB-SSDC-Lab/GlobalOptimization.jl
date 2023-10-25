using GlobalOptimization
using Documenter

DocMeta.setdocmeta!(GlobalOptimization, :DocTestSetup, :(using GlobalOptimization); recursive=true)

makedocs(;
    modules=[GlobalOptimization],
    authors="Grant Hecht",
    repo=Documenter.Remotes.GitHub("GrantHecht","https://github.com/UB-SSDC-Lab/GlobalOptimization.jl/blob/{commit}{path}#{line}"),
    sitename="GlobalOptimization.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://UB-SSDC-Lab.github.io/GlobalOptimization.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    #repo="github.com/UB-SSDC-Lab/GlobalOptimization.jl",
    repo=Documenter.Remotes.GitHub("GrantHecht","https://github.com/UB-SSDC-Lab/GlobalOptimization.jl/blob/{commit}{path}#{line}"),
    devbranch="main",
)
