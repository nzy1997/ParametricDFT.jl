JL = julia --project

init:
	$(JL) -e 'using Pkg; Pkg.instantiate(); Pkg.activate("docs"); Pkg.develop(path="."); Pkg.instantiate(); Pkg.activate("examples"); Pkg.develop(path="."); Pkg.instantiate()'

update:
	$(JL) -e 'using Pkg; Pkg.update(); Pkg.instantiate(); Pkg.activate("docs"); Pkg.update(); Pkg.instantiate(); Pkg.activate("examples"); Pkg.update(); Pkg.instantiate()'

test:
	$(JL) -e 'using Pkg; Pkg.test()'

example:
	julia --project=examples examples/img_process.jl

.PHONY: init test update example

