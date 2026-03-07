using ParametricDFT
using Documenter

DocMeta.setdocmeta!(ParametricDFT, :DocTestSetup, :(using ParametricDFT); recursive=true)

makedocs(;
    modules=[ParametricDFT],
    authors="nzy1997, zazabap",
    sitename="ParametricDFT.jl",
    format=Documenter.HTML(;
        canonical="https://nzy1997.github.io/ParametricDFT.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Basis Types" => "basis.md",
        "Training" => "training.md",
        "Compression & Serialization" => "compression.md",
        "Visualization" => "visualization.md",
    ],
)

deploydocs(;
    repo="github.com/nzy1997/ParametricDFT.jl",
    devbranch="main",
)
