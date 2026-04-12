#import "@preview/cetz:0.4.2": canvas, draw
#show link: set text(blue)

// Tensor box helper for diagrams (same as main.typ / stepbystep.typ)
#let ngate(pos, n, name, text: none, width: 1, gap-y: 1, padding-y: 0.25) = {
  import draw: *
  let height = gap-y * (n - 1) + 2 * padding-y
  group(name: name, {
    rect((rel: (- width/2, -height/2), to: pos), (rel: (width/2, height/2), to: pos), fill: white, name: "body")
    if text != none { content("body", text) }
    for i in range(n){
      let y = height/2 - padding-y - i * gap-y
      anchor("i" + str(i), (rel: (-width/2, y), to: pos))
      anchor("o" + str(i), (rel: (width/2, y), to: pos))
    }
    anchor("b", (rel: (0, -height/2), to: pos))
    anchor("t", (rel: (0, height/2), to: pos))
  })
}

= Batched Riemannian Optimization and GPU Acceleration

This document describes how ParametricDFT.jl implements efficient Riemannian optimization for quantum circuit parameters. Three problems with the Manopt.jl baseline motivate the design: (1) `ProductManifold` iterates over tensors one-by-one, preventing batched operations; (2) Manopt lacks Riemannian Adam or any adaptive optimizer; (3) `ArrayPartition` is incompatible with CUDA, blocking GPU acceleration. The solutions --- batched manifold operations, custom `RiemannianGD`/`RiemannianAdam` optimizers, batched einsum contractions, and GPU dispatch via `CUDAExt.jl` --- are presented below.

= Notation

We use the following notation consistently:
- $bold(theta)_k in CC^(d times d)$: the $k$-th gate tensor (circuit parameter), $k = 1, dots, K$
- $bold(X) in CC^(2 times dots.c times 2)$: input image reshaped to $m+n$ qubit dimensions (each of size 2)
- $bold(Y)$: output of the forward transform
- $cal(L)$: loss function value
- $bold(g)_k$: Euclidean gradient of $cal(L)$ with respect to $bold(theta)_k$
- $tilde(bold(g))_k = "proj"(bold(theta)_k, bold(g)_k)$: Riemannian gradient (projection of $bold(g)_k$ onto the tangent space at $bold(theta)_k$)
- $"Retr"_(bold(theta))(bold(xi))$: retraction from $bold(theta)$ along tangent vector $bold(xi)$ (maps back to the manifold)
- $Gamma_(bold(theta) arrow.r bold(theta)')$: parallel transport from tangent space at $bold(theta)$ to tangent space at $bold(theta)'$
- $alpha$: step size (learning rate)
- $K$: total number of gate tensors in the circuit
- $B$: batch size (number of images processed in one einsum call)
- $d$: gate matrix dimension (typically $d = 2$ for qubit gates)
