using ParametricDFT
using Documenter

DocMeta.setdocmeta!(ParametricDFT, :DocTestSetup, :(using ParametricDFT); recursive=true)

makedocs(;
    modules=[ParametricDFT],
    authors="nzy1997",
    sitename="ParametricDFT.jl",
    format=Documenter.HTML(;
        canonical="https://nzy1997.github.io/ParametricDFT.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nzy1997/ParametricDFT.jl",
    devbranch="main",
)
