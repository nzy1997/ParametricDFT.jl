#show link: set text(blue)

= Prerequisite

1. Understand image processing in Julia, please check YouTube Video: #link("https://www.youtube.com/watch?app=desktop&v=DGojI9xcCfg", [Working With Images in Julia | Week 1, lecture 3 | 18.S191 MIT Fall 2020 | Grant Sanderson]).
2. Undertand fast Fourier transformation for image processing, please check the following videos by Steve Brunton:
  - #link("https://www.youtube.com/watch?v=E8HeD-MUrjY", [The Fast Fourier Transform (FFT)]). It may require some knowledge about complex numbers.
  - #link("https://www.youtube.com/watch?v=gGEBUdM0PVc", [Image Compression and the FFT])
3. Understand tensor network, please check the following repository: #link("https://github.com/GiggleLiu/tutorial-tensornetwork", [Tutorial on Tensor Networks]).
4. Understand basic optimization theory, please check:
  - The 3blue1brown video: #link("https://youtu.be/IHZwWFHWa-w?si=8MWIX_0JHnDYkCSE")[Gradient descent, how neural networks learn | Deep Learning Chapter 2]
  - Manifold optimization: YouTube video #link("https://www.youtube.com/watch?v=dJz1klEutRY", [Manopt.jl: Optimisation on Riemannian Manifolds | Ronny Bergmann | JuliaCon 2022]) and Julia package #link("https://github.com/JuliaManifolds/Manopt.jl", [Manopt.jl]).

= Get started
1. Go through the code in `examples/img_process.jl`. It may require some knowledge about manifold optimization, please check the documentation page of #link("https://github.com/JuliaManifolds/Manifolds.jl", [Manifolds.jl]). Manifold optimization is very similar to gradient based optimization in machine learning, but with some additional constraints.
2. Read the note `note/main.typ` to understand they theory underlying the code.

= Tasks
1. Use GPU to speed up the code, please check the documentation page of #link("https://cuda.juliagpu.org/stable/", [CUDA.jl]).
2. Setup some image datasets and train the tensor network on the datasets.
3. Compare the performance with the Fourier basis.