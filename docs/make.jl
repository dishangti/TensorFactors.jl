using TensorFactors
using Documenter

DocMeta.setdocmeta!(TensorFactors, :DocTestSetup, :(using TensorFactors); recursive=true)

makedocs(;
    modules=[TensorFactors],
    authors="dishangti <16698219+dishangti@users.noreply.github.com> and contributors",
    sitename="TensorFactors.jl",
    format=Documenter.HTML(;
        canonical="https://dishangti.github.io/TensorFactors.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dishangti/TensorFactors.jl",
    devbranch="main",
)
