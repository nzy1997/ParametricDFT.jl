using QDFT
using Documenter

DocMeta.setdocmeta!(QDFT, :DocTestSetup, :(using QDFT); recursive=true)

makedocs(;
    modules=[QDFT],
    authors="nzy1997",
    sitename="QDFT.jl",
    format=Documenter.HTML(;
        canonical="https://nzy1997.github.io/QDFT.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nzy1997/QDFT.jl",
    devbranch="main",
)
