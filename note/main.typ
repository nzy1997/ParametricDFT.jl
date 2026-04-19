#import "@preview/cetz:0.4.2": canvas, draw
#import "@preview/quill:0.7.1": *
#show link: set text(blue)
#set math.equation(numbering: "(1)")
#set page(numbering: "1")
#set heading(numbering: "1.1")

// Title page
#align(center)[
  #v(3em)
  #text(size: 22pt, weight: "bold")[Parametric Tensor Network Bases for\ Sparse Image Representation]
  #v(2em)
  #text(size: 12pt, style: "italic")[Technical Notes]
  #v(4em)
]

#outline(indent: 1.5em, depth: 2)
#pagebreak()

#let ngate(pos, n, name, text: none, width: 1, gap-y: 1, padding-y: 0.25) = {
  import draw: *
  let height = gap-y * (n - 1) + 2 * padding-y
  
  group(name: name, {
    rect((rel: (- width/2, -height/2), to: pos), (rel: (width/2, height/2), to: pos), fill: white, name: "body")
    if text != none{
      content("body", text)
    }
 
    // Define custom anchors
    for i in range(n){
      let y = height/2 - padding-y - i * gap-y
      anchor("i" + str(i), (rel: (-width/2, y), to: pos))
      anchor("o" + str(i), (rel: (width/2, y), to: pos))
    }
    anchor("b", (rel: (0, -height/2), to: pos))
    anchor("t", (rel: (0, height/2), to: pos))
  })
}

= Introduction

A *sparse basis* is a transform basis in which a signal can be represented with few non-zero coefficients. For compression, the goal is to find a basis $cal(T)$ such that $bold(y) = cal(T)(bold(x))$ has most energy concentrated in a small number of components: $||bold(y)||_0 << dim(bold(y))$.

Classical transforms (FFT, DCT) provide fixed bases optimized for specific signal classes. However, the Fourier basis is not optimal for image processing because:
- It assumes periodic boundary conditions, which do not hold for natural images.
- The 2D Fourier basis assumes the $X$ and $Y$ coordinates are independent, ignoring spatial correlations common in natural images.

Observing that the tensor network representation of the FFT contains parameters (Hadamard gates $H$ and controlled-phase gates $M_k$) that can be tuned without affecting the computational complexity, we ask: can we find a transformation _better_ than the Fourier basis by learning these parameters from data?

This document presents _parametric tensor network bases_ --- a family of data-adaptive unitary transforms parameterized by quantum circuit structures. The circuit parameters live on Riemannian manifolds ($U(2)$ for Hadamard gates, $U(1)^4$ for phase gates) and are optimized via Riemannian gradient descent to find a sparse basis tailored to specific image data. The standard QFT with fixed parameters is one point on this manifold; learning moves us to a better point.

We describe four circuit topologies (QFT, Entangled QFT, TEBD, MERA), the loss functions that drive sparsity, and the Riemannian optimization machinery that makes training possible while preserving unitarity.

= Background

== Cooley-Tukey FFT

The _Fourier basis_ in $RR^n$ is given by the DFT matrix $(F_n)_(i,j) = omega^((i-1)(j-1))$, where $omega = e^(-2pi i\/n)$ is the primitive $n$-th root of unity. $F_n$ is a Vandermonde matrix:
#figure(canvas({
  import draw: *
  let dx = -1.0
   rect((-2.8 + dx, -2), (-1.6 + dx, 2), fill: blue.lighten(70%), stroke: none)
   rect((-4.2 + dx, -2), (-3.8 + dx, 2), fill: blue.lighten(70%), stroke: none)
   rect((1 + dx, -2), (2.4 + dx, 2), fill: blue.lighten(70%), stroke: none)
   rect((3 + dx, -2), (5 + dx, 2), fill: blue.lighten(70%), stroke: none)
  content((0, 0), [$
F_n = mat(augment: #(hline: 5),
1 , 1 , 1 , dots , 1, 1, dots, 1, 1 ;
1 , omega , omega^2 , dots , omega^(n/2-1), omega^(n/2), dots, omega^(n-2), omega^(n-1);
1 , omega^2 , omega^4 , dots , omega^(n-2), omega^(n), dots, omega^(2n-4), omega^(2n-2);
dots.v , dots.v , dots.v , dots.down , dots.v, dots.v, dots.down, dots.v, dots.v;
1 , omega^(n/2-1) , omega^(n-2) , dots , omega^((n/2-1)^2), omega^((n/2-1)n/2), dots, omega^((n/2-1)(n-2)), omega^((n/2-1)(n-1));
1 , omega^(n/2) , omega^(n) , dots , omega^(n/2(n/2-1)), omega^(n/2)^2, dots, omega^(n/2(n-2)), omega^(n/2(n-1));
dots.v , dots.v , dots.v , dots.down , dots.v, dots.v, dots.down, dots.v, dots.v;
1 , omega^(n-1) , omega^(2n-2) , dots , omega^((n-1)(n/2-1)), omega^((n-1)n/2), dots, omega^((n-1)(n-2)), omega^((n-1)^2)
). $])
}))
The Cooley-Tukey FFT is a divide-and-conquer algorithm for computing the DFT. For simplicity, we assume $n$ is a power of 2, such that we can divide the matrix into 4 blocks:
- The odd columns (blue background), top half:
  $
    F_("odd", "top") = mat(1,1, dots, 1; 1,omega^2, dots, omega^(n-2); dots.v, dots.v, dots.down, dots.v; 1, omega^(n-2), dots, omega^((n/2-1)(n-2))) = mat(1,1, dots, 1; 1,(omega^2), dots, (omega^2)^(n/2-1); dots.v, dots.v, dots.down, dots.v; 1, (omega^2)^(n/2-1), dots, (omega^2)^((n/2-1)(n/2-1)))
  = F_(n/2)
  $ 
- The even columns (white background), top half:
  $
    F_("even", "top") = D_(n/2) F_(n/2)
  $
  where $D_n = "diag"(1, omega, omega^2, ..., omega^(n-1))$.
- The odd columns (blue background), bottom half:
  $
    F_("odd", "bottom") = F_(n/2).
  $
  Note $omega^n = 1$ is ignored.
- The even columns (white background), bottom half:
  $
    F_("even", "bottom") = -D_(n/2) F_(n/2),
  $
  where the minus sign comes from $omega^(n/2) = -1$.

Finally, we arrive at the Cooley-Tukey FFT given by:
$ F_n bold(x) = mat(
  I_(n/2), D_(n/2);
  I_(n/2), -D_(n/2)
) mat(
  F_(n/2), 0;
  0, F_(n/2)
) vec(bold(x)_("odd"), bold(x)_("even"))
$ <eq:fft>
where $bold(x)_("odd")$ and $bold(x)_("even")$ contain the odd and even indexed elements of $bold(x)$, respectively.
This shows that the DFT in $RR^n$ decomposes into two smaller DFTs in $RR^(n/2)$ with a diagonal twiddle factor matrix $D_(n\/2)$ in between.
Since diagonal matrices can be applied in $O(n)$ operations, this decomposition yields the recurrence $T(n) = 2T(n\/2) + O(n)$, which solves to $O(n log n)$ total operations.

The inverse transformation is given by $F_n^dagger bold(x)\/n$. The DFT matrix is unitary up to a scale factor: $F_n F_n^dagger = n I$.

== Tensor Network Representation of the FFT <bg-tn>

We now reformulate the FFT as a tensor network, which reveals the parametric structure exploited throughout this work. In tensor network notation, a vector of size $n=2^k$ is represented as a rank-$k$ tensor with binary indices, where the basis index decomposes as $i = 2^0 q_0 + 2^1 q_1 + ... + 2^(k-1) q_(k-1)$.
#figure(canvas({
  import draw: *
  let n = 4
  ngate((0, 0), n, "F_n", text:[$bold(x)$], gap-y: 0.8, width: 0.7)
  for i in range(n){
    line("F_n.o" + str(i), (rel: (0.5, 0)))
    content((rel: (0.7, 0), to: "F_n.o" + str(i)), [$q_#i$])
  }
}))

In the following, we aim to find a tensor network decomposition for the linear map $F_n$:
#figure(canvas({
  import draw: *
  let n = 4
  ngate((0, 0), n, "x_n", text:[$bold(x)$], gap-y: 0.8, width: 0.7)
  ngate((1.5, 0), n, "F_n", text:[$F_n$], gap-y: 0.8, width: 1)
  for i in range(n){
    line("x_n.o" + str(i), "F_n.i" + str(i))
    line("F_n.o" + str(i), (rel: (0.5, 0)))
  }
}))

Step 1: To start, the equation @eq:fft can be represented as the following tensor network:
$
  F_n bold(x) = (mat(
  1, 1;
  1, -1
) times.circle I_(n/2)) mat(I_(n/2), 0; 0, D_(n/2)) vec(
  F_(n/2) bold(x)_("odd"),
  F_(n/2) bold(x)_("even")
),
$

#figure(canvas({
  import draw: *
  let n = 4
  ngate((0, 0), n, "x_n", text:[$bold(x)$], gap-y: 0.8, width: 0.7)
  ngate((1.5, -0.4), n - 1, "F_n", text:[$F_(n/2)$], gap-y: 0.8, width: 1)
  ngate((3.5, 0), n, "ID_n", text:[$mat(I_(n/2), 0; 0, D_(n/2))$], gap-y: 0.8, width: 2)
  ngate((5.2, 1.2), 1, "H", text:[$H$], gap-y: 0.8, width: 0.5)
  for i in range(n - 1){
    line("x_n.o" + str(i+1), "F_n.i" + str(i))
    line("F_n.o" + str(i), "ID_n.i" + str(i+1))
    //line("F_n.o" + str(i), (rel: (0.5, 0)))
  }
  line("x_n.o0", "ID_n.i0")
  line("ID_n.o0", "H.i0")
  for i in range(1, n){
    line("ID_n.o" + str(i), (rel: (1.45, 0)))
  }
  line("H.o0", (rel: (0.5, 0)))
}))
where $H = mat(1, 1; 1, -1)$ is the Hadamard matrix (up to a normalization constant).

*Step 2*: Next, we decompose the controlled diagonal matrix $mat(I_(n/2), 0; 0, D_(n/2))$ into a tensor network.
This matrix acts as the identity when bit $q_0 = 0$ (odd index) and applies $D_(n/2)$ when $q_0 = 1$ (even index).
// We define the following control tensor in block matrix form:
// $
//   "ctrl"_1(A) := mat(I, 0; 0, A)
// $
// Only if $q_1 = 1$, the operation $A$ is applied.
// Similarly, we can define $"ctrl"_2(A)$, $"ctrl"_3(A)$, ..., $"ctrl"_n(A)$ if $q_2 = 1$, $q_3 = 1$, ..., $q_n = 1$, the operation $A$ is applied.
// Then we have:
// $
//   mat(I_A times.circle I_B, 0; 0, A times.circle B) = "ctrl"_1(A times.circle I_B) "ctrl"_1(I_A times.circle B)
// $
Observe that
$
D_n = "diag"(1, omega^(n/2)) times.circle "diag"(1, omega, omega^2, ..., omega^(n/2-1)) = "diag"(1, omega^(n/2)) times.circle "diag"(1, omega^(n/4)) times.circle dots times.circle "diag"(1, omega)$. We have
$
  mat(I_(n/2), 0; 0, D_(n/2)) = "ctrl"_0("diag"(1, omega^(n/4))_1) "ctrl"_0("diag"(1, omega^(n/8))_2) ... "ctrl"_0 ("diag"(1, omega)_(log_2 n)),
$
where $"ctrl"_i (A_j)$ means the target operation applied on $A_j$ is applied only if bit $q_i$ is $1$. Here, since the controlled gate is diagonal, it can be represented as a matrix connecting two variables:

#let cphase(x, i, j, k, name: "CP") = {
  import draw: *
  circle((x, i), radius: 0.05, fill: black, stroke: none, name: name + "ctrl1")
  circle((x, j), radius: 0.05, fill: black, stroke: none, name: name + "ctrl2")
  ngate((x, (i+j)/2), 1, name, text:text(8pt)[$M_(#(k+1))$], gap-y: 0.8, width: 0.5)
  line(name + "ctrl1", name+".t")
  line(name + "ctrl2", name+".b")
}

#figure(canvas({
  import draw: *
  let n = 4
  ngate((0, 0), n, "x_n", text:[$bold(x)$], gap-y: 0.8, width: 0.7)
  ngate((1.5, -0.4), n - 1, "F_n", text:[$F_(n/2)$], gap-y: 0.8, width: 1)
  ngate((5.2, 1.2), 1, "H", text:[$H$], gap-y: 0.8, width: 0.5)
  for i in range(n - 1){
    line("x_n.o" + str(i+1), "F_n.i" + str(i))
    //line("F_n.o" + str(i), (rel: (0.5, 0)))
  }
  line("x_n.o0", "H.i0")
  cphase(4.4, 1.2, 0.4, 1)
  cphase(3.6, 1.2, -0.4, 2)
  cphase(2.8, 1.2, -1.2, 3)
  for i in range(n - 1){
    line("F_n.o" + str(i), (rel: (3.95, 0)))
  }
  line("H.o0", (rel: (0.5, 0)))
}))
In this diagram, the gate $M_k = mat(1, 1; 1, e^(i pi \/ 2^(k-1)))$ connects the two qubits involved in the controlled operation, multiplying a phase factor $e^(i pi \/ 2^(k-1))$ when both bits are in state $1$. Recursively decomposing the $F_(n/2)$ tensor yields the complete tensor network:

#figure(canvas({
  import draw: *
  let n = 4
  ngate((-3, 0), n, "x_n", text:[$bold(x)$], gap-y: 0.8, width: 0.7)
  ngate((5.2, 1.2), 1, "H", text:[$H$], gap-y: 0.8, width: 0.5)
  line("x_n.o0", "H.i0")
  line("H.o0", (5.8, 1.2))
  cphase(4.4, 1.2, 0.4, 1, name: "CP11")
  cphase(3.6, 1.2, -0.4, 2, name: "CP12")
  cphase(2.8, 1.2, -1.2, 3, name: "CP13")

  ngate((2.0, 0.4), 1, "H", text:[$H$], gap-y: 0.8, width: 0.5)
  line("H.o0", "CP11ctrl2")
  line("CP11ctrl2", (5.8, 0.4))
  line("x_n.o1", "H.i0")
  cphase(1.2, 0.4, -0.4, 1, name: "CP21")
  cphase(0.4, 0.4, -1.2, 2, name: "CP22")

  ngate((-0.4, -0.4), 1, "H", text:[$H$], gap-y: 0.8, width: 0.5)
  line("H.o0", "CP21ctrl2")
  line("CP21ctrl2", "CP12ctrl2")
  line("CP12ctrl2", (5.8, -0.4))
  line("x_n.o2", "H.i0")
  cphase(-1.2, -0.4, -1.2, 1, name: "CP31")

  ngate((-2.0, -1.2), 1, "H", text:[$H$], gap-y: 0.8, width: 0.5)
  line("H.o0", "CP31ctrl2")
  line("CP31ctrl2", "CP22ctrl2")
  line("CP22ctrl2", "CP13ctrl2")
  line("CP13ctrl2", (5.8, -1.2))
  line("x_n.o3", "H.i0")
}))

Direct evaluation of this tensor network takes $O(n log^2 n)$ operations. However, since the controlled-phase gates are diagonal matrices, adjacent phase operations on the same qubit can be merged, reducing the total complexity to $O(n log n)$ --- matching the classical FFT.

= Parametric Circuit Bases

We now present four circuit topologies for parametric sparse bases. Each topology defines which qubit pairs are connected by gates, while sharing the same gate parameterization (Hadamard on $U(2)$, controlled-phase on $U(1)^4$) and optimization framework.

== QFT Basis

The QFT circuit implements a unitary transform $cal(T)(bold(theta)): CC^(2^m times 2^n) -> CC^(2^m times 2^n)$ parameterized by Hadamard gates $H in U(2)$ and controlled-phase gates $M_k in U(1)^4$. With fixed parameters, this reduces to the classical DFT; making the parameters learnable enables the optimizer to search for a transform with better sparsity properties for a given dataset.

The tensor network structure derived in @bg-tn applies directly: each gate in the recursive QFT decomposition becomes a learnable parameter. The Hadamard gates $H$ live on the unitary manifold $U(2)$, while the controlled-phase gates $M_k = "diag"(1, 1, 1, e^(i phi))$ live on the phase manifold $U(1)^4$. For a 2D image of size $2^m times 2^n$, the QFT basis applies independent QFT circuits on the $m$ row qubits and $n$ column qubits, yielding the separable transform $cal(T) = F_m times.circle F_n$.

== Entangled QFT Basis: XY Correlation

The separable QFT basis processes the $x$ and $y$ dimensions independently. For a square image of size $2^n times 2^n$, the QFT is applied separately on the $n$ row qubits and $n$ column qubits. This independence assumption is suboptimal for natural images, which exhibit strong spatial correlations between rows and columns (e.g., edges at arbitrary angles, diagonal textures).

We propose an _entangled QFT basis_ that introduces controlled-phase gates between x and y qubits after each layer of the QFT circuit. For the square case $m = n$, we use a _one-to-one_ entanglement structure where each x qubit $x_k$ is coupled with the corresponding y qubit $y_k$. The entanglement gate $E_k$ has exactly the same form as the $M$ gate defined in the previous section:
$
  E_k = mat(1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, e^(i phi_k))
$
acting on qubits $(x_(n-k), y_(n-k))$, where $phi_k$ is a learnable phase parameter. Similar to the $M_k$ gate which multiplies a phase factor when both control and target qubits are in state $|1 angle.r$, the entanglement gate $E_k$ multiplies a phase $e^(i phi_k)$ when both $x_(n-k) = 1$ and $y_(n-k) = 1$. The key difference is that while $M_k$ uses the fixed phase $pi\/2^(k-1)$ determined by the QFT structure, $E_k$ uses a learnable phase $phi_k$ that can be optimized to capture image-specific correlations.

The circuit structure for $n = 4$ qubits per dimension is shown below. Each row qubit $x_k$ first passes through its QFT circuit (Hadamard gate $H$ followed by controlled-phase gates $M_j$ with other row qubits), and similarly for column qubits $y_k$. The entanglement gates $E_k$ (shown as boxes connecting x and y qubit lines) are applied after the Hadamard layers, coupling the corresponding row and column qubits:

#let egate(x, i, j, k, name: "E") = {
  import draw: *
  circle((x, i), radius: 0.05, fill: black, stroke: none, name: name + "ctrl1")
  circle((x, j), radius: 0.05, fill: black, stroke: none, name: name + "ctrl2")
  ngate((x, (i+j)/2), 1, name, text:text(8pt)[$E_(#k)$], gap-y: 0.8, width: 0.5)
  line(name + "ctrl1", name+".t")
  line(name + "ctrl2", name+".b")
}

#figure(canvas({
  import draw: *
  let n = 4
  let dy = 0.8
  let ysep = (n + 0.5) * dy
  // Single combined input tensor
  let ytop = dy / 2
  let ybot = -ysep - (n - 1) * dy - dy / 2
  let xc = -3
  let hw = 0.35
  rect((xc - hw, ybot), (xc + hw, ytop), fill: white, name: "img")
  content("img", [$bold(x)$])
  // Dashed separator inside the input block
  line((xc - hw, -(n - 0.25) * dy), (xc + hw, -(n - 0.25) * dy), stroke: (dash: "dashed", paint: gray))

  // Row qubit x_0 (topmost, y=0)
  ngate((7.0, 0), 1, "Hx0", text:[$H$], gap-y: dy, width: 0.5)
  line((xc + hw, 0), "Hx0.i0")
  line("Hx0.o0", (9.4, 0))
  cphase(6.2, 0, -dy, 1, name: "Mx01")
  cphase(5.4, 0, -2 * dy, 2, name: "Mx02")
  cphase(4.6, 0, -3 * dy, 3, name: "Mx03")

  // Row qubit x_1 (y=-dy)
  ngate((3.2, -dy), 1, "Hx1", text:[$H$], gap-y: dy, width: 0.5)
  line("Hx1.o0", "Mx01ctrl2")
  line("Mx01ctrl2", (9.4, -dy))
  line((xc + hw, -dy), "Hx1.i0")
  cphase(2.4, -dy, -2 * dy, 1, name: "Mx11")
  cphase(1.6, -dy, -3 * dy, 2, name: "Mx12")

  // Row qubit x_2 (y=-2dy)
  ngate((0.2, -2 * dy), 1, "Hx2", text:[$H$], gap-y: dy, width: 0.5)
  line("Hx2.o0", "Mx11ctrl2")
  line("Mx11ctrl2", "Mx02ctrl2")
  line("Mx02ctrl2", (9.4, -2 * dy))
  line((xc + hw, -2 * dy), "Hx2.i0")
  cphase(-0.6, -2 * dy, -3 * dy, 1, name: "Mx21")

  // Row qubit x_3 (bottommost row qubit, y=-3dy)
  ngate((-2.0, -3 * dy), 1, "Hx3", text:[$H$], gap-y: dy, width: 0.5)
  line("Hx3.o0", "Mx21ctrl2")
  line("Mx21ctrl2", "Mx12ctrl2")
  line("Mx12ctrl2", "Mx03ctrl2")
  line("Mx03ctrl2", (9.4, -3 * dy))
  line((xc + hw, -3 * dy), "Hx3.i0")

  // Column qubit y_0 (y=-ysep)
  ngate((7.0, -ysep), 1, "Hy0", text:[$H$], gap-y: dy, width: 0.5)
  line((xc + hw, -ysep), "Hy0.i0")
  line("Hy0.o0", (9.4, -ysep))
  cphase(6.2, -ysep, -ysep - dy, 1, name: "My01")
  cphase(5.4, -ysep, -ysep - 2 * dy, 2, name: "My02")
  cphase(4.6, -ysep, -ysep - 3 * dy, 3, name: "My03")

  // Column qubit y_1 (y=-ysep-dy)
  ngate((3.2, -ysep - dy), 1, "Hy1", text:[$H$], gap-y: dy, width: 0.5)
  line("Hy1.o0", "My01ctrl2")
  line("My01ctrl2", (9.4, -ysep - dy))
  line((xc + hw, -ysep - dy), "Hy1.i0")
  cphase(2.4, -ysep - dy, -ysep - 2 * dy, 1, name: "My11")
  cphase(1.6, -ysep - dy, -ysep - 3 * dy, 2, name: "My12")

  // Column qubit y_2 (y=-ysep-2dy)
  ngate((0.2, -ysep - 2 * dy), 1, "Hy2", text:[$H$], gap-y: dy, width: 0.5)
  line("Hy2.o0", "My11ctrl2")
  line("My11ctrl2", "My02ctrl2")
  line("My02ctrl2", (9.4, -ysep - 2 * dy))
  line((xc + hw, -ysep - 2 * dy), "Hy2.i0")
  cphase(-0.6, -ysep - 2 * dy, -ysep - 3 * dy, 1, name: "My21")

  // Column qubit y_3 (y=-ysep-3dy)
  ngate((-2.0, -ysep - 3 * dy), 1, "Hy3", text:[$H$], gap-y: dy, width: 0.5)
  line("Hy3.o0", "My21ctrl2")
  line("My21ctrl2", "My12ctrl2")
  line("My12ctrl2", "My03ctrl2")
  line("My03ctrl2", (9.4, -ysep - 3 * dy))
  line((xc + hw, -ysep - 3 * dy), "Hy3.i0")

  // Qubit labels
  for i in range(n) {
    content((-3.8, -i * dy), [$x_#i$])
    content((9.8, -i * dy), [$x'_#i$])
    content((-3.8, -ysep - i * dy), [$y_#i$])
    content((9.8, -ysep - i * dy), [$y'_#i$])
  }
  // Dashed separator on the circuit
  line((-4.0, -(n - 0.25) * dy), (9.4, -(n - 0.25) * dy), stroke: (dash: "dashed", paint: gray))

  // Entanglement gates E_k connecting x_{n-k} with y_{n-k}
  egate(-1.4, -3 * dy, -ysep - 3 * dy, 4, name: "E4")
  egate(0.8, -2 * dy, -ysep - 2 * dy, 3, name: "E3")
  egate(3.8, -dy, -ysep - dy, 2, name: "E2")
  egate(8.2, 0, -ysep, 1, name: "E1")
}))

Summarizing the gate applications for each qubit:
- *Row qubit $x_k$*: Hadamard gate $H$, followed by controlled-phase gates $M_j$ with qubits $x_0, ..., x_(k-1)$ (standard QFT structure), then entanglement gate $E_(n-k)$ with column qubit $y_k$
- *Column qubit $y_k$*: Hadamard gate $H$, followed by controlled-phase gates $M_j$ with qubits $y_0, ..., y_(k-1)$ (standard QFT structure), then entanglement gate $E_(n-k)$ with row qubit $x_k$

For a square $2^n times 2^n$ image encoded with $n$ row qubits and $n$ column qubits (total $2n$ qubits), we add exactly $n$ entanglement gates ${E_1, E_2, ..., E_n}$, one for each pair of corresponding row/column qubits.

The total transformation becomes:
$
  cal(T)_"entangled" = U_"entangle" dot (F_n times.circle F_n)
$
where $U_"entangle" = product_(k=1)^n E_k$ is the product of all entanglement gates acting on qubit pairs $(x_(n-k), y_(n-k))$, and $F_n$ is the $n$-qubit QFT applied along each spatial dimension.

This approach has several desirable properties:
- It captures diagonal features and cross-dimensional correlations common in natural images.
- The computational complexity remains $O(n log n)$ per spatial dimension (equivalently $O(N log N)$ for image side length $N = 2^n$), matching the standard QFT.
- Only $n$ additional real-valued parameters $phi_k$ are introduced --- one phase per qubit pair, i.e., $O(n)$ overhead.
- Setting all entanglement phases $phi_k = 0$ recovers the standard separable 2D QFT.

== TEBD Basis

Time-Evolving Block Decimation (TEBD) is a tensor network ansatz originally developed for simulating 1D quantum many-body systems. Unlike the QFT's hierarchical all-to-all connectivity, TEBD uses a _ring topology_ of nearest-neighbor controlled-phase gates preceded by Hadamard gates on each qubit. For image processing on a $2^n times 2^n$ grid, the $n$ row qubits and $n$ column qubits each form an independent ring of controlled-phase gates.

#let tebdgate(x, i, j, label, name: "T") = {
  import draw: *
  circle((x, i), radius: 0.05, fill: black, stroke: none, name: name + "c1")
  circle((x, j), radius: 0.05, fill: black, stroke: none, name: name + "c2")
  ngate((x, (i + j) / 2), 1, name, text: text(8pt)[#label], gap-y: 0.8, width: 0.6)
  line(name + "c1", name + ".t")
  line(name + "c2", name + ".b")
}

#figure(canvas({
  import draw: *
  let dy = 0.8
  let n = 4
  let ysep = (n + 0.5) * dy
  // Single combined input tensor for all qubits
  // Use a manually drawn rect to span all qubit lines with a gap
  let ytop = dy / 2
  let ybot = -ysep - (n - 1) * dy - dy / 2
  let xc = -2.5
  let hw = 0.35
  rect((xc - hw, ybot), (xc + hw, ytop), fill: white, name: "img")
  content("img", [$bold(x)$])
  // Dashed separator inside the input block between row and column qubits
  line((xc - hw, -(n - 0.25) * dy), (xc + hw, -(n - 0.25) * dy), stroke: (dash: "dashed", paint: gray))
  // Draw qubit lines and labels
  for i in range(n) {
    line((xc + hw, -i * dy), (9.5, -i * dy), stroke: gray)
    content((-3.3, -i * dy), [$|x_#(i + 1) angle.r$])
    content((9.9, -i * dy), [$x'_#(i + 1)$])
    line((xc + hw, -ysep - i * dy), (9.5, -ysep - i * dy), stroke: gray)
    content((-3.3, -ysep - i * dy), [$|y_#(i + 1) angle.r$])
    content((9.9, -ysep - i * dy), [$y'_#(i + 1)$])
  }
  // Dashed separator between row and column qubits on the circuit
  line((-3.5, -(n - 0.25) * dy), (9.5, -(n - 0.25) * dy), stroke: (dash: "dashed", paint: gray))
  // Hadamard gates for all qubits
  for i in range(n) {
    ngate((-1.0, -i * dy), 1, "Hx" + str(i), text:[$H$], gap-y: dy, width: 0.5)
    ngate((-1.0, -ysep - i * dy), 1, "Hy" + str(i), text:[$H$], gap-y: dy, width: 0.5)
  }
  // Row ring: nearest-neighbor controlled-phase gates (staircase pattern)
  tebdgate(0.8, 0, -dy, [$T_(x 1)$], name: "Tx1")
  tebdgate(2.2, -dy, -2 * dy, [$T_(x 2)$], name: "Tx2")
  tebdgate(3.6, -2 * dy, -3 * dy, [$T_(x 3)$], name: "Tx3")
  // Wrap-around gate closing the row ring
  tebdgate(5.5, 0, -3 * dy, [$T_(x 4)$], name: "Tx4")
  // Column ring: nearest-neighbor controlled-phase gates
  tebdgate(0.8, -ysep, -ysep - dy, [$T_(y 1)$], name: "Ty1")
  tebdgate(2.2, -ysep - dy, -ysep - 2 * dy, [$T_(y 2)$], name: "Ty2")
  tebdgate(3.6, -ysep - 2 * dy, -ysep - 3 * dy, [$T_(y 3)$], name: "Ty3")
  // Wrap-around gate closing the column ring
  tebdgate(5.5, -ysep, -ysep - 3 * dy, [$T_(y 4)$], name: "Ty4")
}), caption: [TEBD circuit for $n = 4$ row and column qubits with ring topology. Hadamard gates $H$ are applied to all qubits, followed by nearest-neighbor controlled-phase gates $T_(x k)$ (row ring) and $T_(y k)$ (column ring). Wrap-around gates $T_(x 4)$ and $T_(y 4)$ close each ring.])

Each two-qubit gate in the TEBD circuit is a controlled-phase gate:
$
  T_k = "diag"(1, 1, 1, e^(i phi_k))
$
with a single learnable phase $phi_k$ per gate. This gate multiplies a phase factor $e^(i phi_k)$ when both qubits are in state $|1 angle.r$, identical in form to the $M_k$ gates in the QFT circuit but with learnable phases instead of fixed ones.

For $n$ row qubits and $n$ column qubits with ring topology, we have $n$ row ring gates and $n$ column ring gates, giving $2n$ total learnable phases. The parameter manifold is:
$
  cal(M)_"TEBD" = product_(k=1)^(2n) U(1)^4
$

Optimization proceeds on this product of phase manifolds using the same Riemannian framework as the QFT basis (@sec:riemannian). The TEBD circuit evaluates in $O(n)$ operations (one $2 times 2$ gate per qubit), making it the cheapest basis to evaluate at inference time.

== MERA-inspired Basis

The Multi-scale Entanglement Renormalization Ansatz (MERA) is a hierarchical tensor network inspired by the renormalization group. The original MERA for quantum many-body systems uses _disentanglers_ (full $U(4)$ unitaries) and _isometries_ (coarse-graining maps on the Stiefel manifold $"St"(2,4)$). However, our application requires a _unitary_ transform $cal(T): CC^(2^n) -> CC^(2^n)$ for image compression --- true coarse-graining would reduce the output dimension, making the transform non-invertible. Instead, we borrow MERA's _hierarchical connectivity pattern_ while parameterizing all gates as controlled-phase gates, consistent with the QFT and TEBD bases. We call the resulting circuit a _MERA-inspired_ basis.

For $n = 2^k$ qubits, the MERA-inspired circuit has $k = log_2 n$ layers. Each layer $l$ has stride $s = 2^(l-1)$ and $n \/ (2s)$ pairs of gates. Within each pair, we apply a _disentangler_ gate followed by an _isometry_ gate, both parameterized as controlled-phase gates $"diag"(1, 1, 1, e^(i phi))$ acting on qubit pairs determined by the MERA connectivity:

- *Disentangler* at pair $p$, layer $l$: acts on qubits $(2 p s + 2,  (2 p s + s + 2) mod n)$
- *Isometry* at pair $p$, layer $l$: acts on qubits $(2 p s + 1,  2 p s + s + 1)$

#let meragate(x, i, j, label, name: "M", gwidth: 0.6) = {
  import draw: *
  circle((x, i), radius: 0.05, fill: black, stroke: none, name: name + "c1")
  circle((x, j), radius: 0.05, fill: black, stroke: none, name: name + "c2")
  ngate((x, (i + j) / 2), 1, name, text: text(8pt)[#label], gap-y: 0.8, width: gwidth)
  line(name + "c1", name + ".t")
  line(name + "c2", name + ".b")
}

#figure(canvas({
  import draw: *
  let n = 8
  let dy = 0.6
  let ysep = (n + 0.5) * dy
  let xend = 12.5
  let gw = 0.6  // gate width (same for D and W)
  // Single combined input tensor
  let ytop = dy / 2
  let ybot = -ysep - (n - 1) * dy - dy / 2
  let xc = -2.5
  let hw = 0.35
  rect((xc - hw, ybot), (xc + hw, ytop), fill: white, name: "img")
  content("img", [$bold(x)$])
  line((xc - hw, -(n - 0.25) * dy), (xc + hw, -(n - 0.25) * dy), stroke: (dash: "dashed", paint: gray))
  // Draw qubit lines and labels
  for i in range(n) {
    line((xc + hw, -i * dy), (xend, -i * dy), stroke: gray)
    content((-3.3, -i * dy), [$x_#(i + 1)$])
    content((xend + 0.5, -i * dy), [$x'_#(i + 1)$])
    line((xc + hw, -ysep - i * dy), (xend, -ysep - i * dy), stroke: gray)
    content((-3.3, -ysep - i * dy), [$y_#(i + 1)$])
    content((xend + 0.5, -ysep - i * dy), [$y'_#(i + 1)$])
  }
  // Dashed separator on the circuit
  line((-3.5, -(n - 0.25) * dy), (xend, -(n - 0.25) * dy), stroke: (dash: "dashed", paint: gray))
  // Hadamard gates for all qubits
  for i in range(n) {
    ngate((-1.2, -i * dy), 1, "Hx" + str(i), text:[$H$], gap-y: dy, width: 0.45)
    ngate((-1.2, -ysep - i * dy), 1, "Hy" + str(i), text:[$H$], gap-y: dy, width: 0.45)
  }
  // === Row MERA (8 qubits, 3 layers) ===
  // Layer 1 (s=1, 4 pairs)
  //   Disentanglers: (2,3), (4,5), (6,7) at x1d; wrap-around (8,1) at x1d2
  let x1d = 1.0
  meragate(x1d, -1 * dy, -2 * dy, [$D$], name: "xD1", gwidth: gw)
  meragate(x1d, -3 * dy, -4 * dy, [$D$], name: "xD2", gwidth: gw)
  meragate(x1d, -5 * dy, -6 * dy, [$D$], name: "xD3", gwidth: gw)
  // Wrap-around D gate (8,1) — placed at its own x to avoid overlap
  let x1d2 = 2.2
  meragate(x1d2, -7 * dy, 0, [$D$], name: "xD4", gwidth: gw)
  //   Isometries: (1,2), (3,4), (5,6), (7,8)
  let x1w = 3.4
  meragate(x1w, 0, -1 * dy, [$W$], name: "xW1", gwidth: gw)
  meragate(x1w, -2 * dy, -3 * dy, [$W$], name: "xW2", gwidth: gw)
  meragate(x1w, -4 * dy, -5 * dy, [$W$], name: "xW3", gwidth: gw)
  meragate(x1w, -6 * dy, -7 * dy, [$W$], name: "xW4", gwidth: gw)
  // Layer 2 (s=2, 2 pairs)
  //   Disentanglers: (2,4), (6,8)
  let x2d = 5.5
  meragate(x2d, -1 * dy, -3 * dy, [$D$], name: "xD5", gwidth: gw)
  meragate(x2d, -5 * dy, -7 * dy, [$D$], name: "xD6", gwidth: gw)
  //   Isometries: (1,3), (5,7)
  let x2w = 7.2
  meragate(x2w, 0, -2 * dy, [$W$], name: "xW5", gwidth: gw)
  meragate(x2w, -4 * dy, -6 * dy, [$W$], name: "xW6", gwidth: gw)
  // Layer 3 (s=4, 1 pair) — compressed
  //   Disentangler: (2,6)
  let x3d = 9.0
  meragate(x3d, -1 * dy, -5 * dy, [$D$], name: "xD7", gwidth: gw)
  //   Isometry: (1,5)
  let x3w = 10.5
  meragate(x3w, 0, -4 * dy, [$W$], name: "xW7", gwidth: gw)
  // === Column MERA (8 qubits, 3 layers) ===
  // Layer 1
  meragate(x1d, -ysep - 1 * dy, -ysep - 2 * dy, [$D$], name: "yD1", gwidth: gw)
  meragate(x1d, -ysep - 3 * dy, -ysep - 4 * dy, [$D$], name: "yD2", gwidth: gw)
  meragate(x1d, -ysep - 5 * dy, -ysep - 6 * dy, [$D$], name: "yD3", gwidth: gw)
  meragate(x1d2, -ysep - 7 * dy, -ysep, [$D$], name: "yD4", gwidth: gw)
  meragate(x1w, -ysep, -ysep - 1 * dy, [$W$], name: "yW1", gwidth: gw)
  meragate(x1w, -ysep - 2 * dy, -ysep - 3 * dy, [$W$], name: "yW2", gwidth: gw)
  meragate(x1w, -ysep - 4 * dy, -ysep - 5 * dy, [$W$], name: "yW3", gwidth: gw)
  meragate(x1w, -ysep - 6 * dy, -ysep - 7 * dy, [$W$], name: "yW4", gwidth: gw)
  // Layer 2
  meragate(x2d, -ysep - 1 * dy, -ysep - 3 * dy, [$D$], name: "yD5", gwidth: gw)
  meragate(x2d, -ysep - 5 * dy, -ysep - 7 * dy, [$D$], name: "yD6", gwidth: gw)
  meragate(x2w, -ysep, -ysep - 2 * dy, [$W$], name: "yW5", gwidth: gw)
  meragate(x2w, -ysep - 4 * dy, -ysep - 6 * dy, [$W$], name: "yW6", gwidth: gw)
  // Layer 3
  meragate(x3d, -ysep - 1 * dy, -ysep - 5 * dy, [$D$], name: "yD7", gwidth: gw)
  meragate(x3w, -ysep, -ysep - 4 * dy, [$W$], name: "yW7", gwidth: gw)
  // Layer labels
  content((2.2, 1.2), text(8pt)[Layer 1])
  content((6.3, 1.2), text(8pt)[Layer 2])
  content((9.7, 1.2), text(8pt)[Layer 3])
  line((0.2, 0.9), (4.2, 0.9), stroke: gray)
  line((4.8, 0.9), (7.8, 0.9), stroke: gray)
  line((8.3, 0.9), (11.2, 0.9), stroke: gray)
}), caption: [MERA-inspired circuit for $n=8$ row and column qubits (16 qubits total). All gates are controlled-phase gates with learnable phases. Disentanglers $D$ and isometries $W$ follow the hierarchical MERA-inspired connectivity: layer 1 (stride 1) connects nearest neighbors, layer 2 (stride 2) connects qubits at distance 2, layer 3 (stride 4) connects distant qubits. Row and column qubits are processed independently.])

Both disentanglers and isometries use identical controlled-phase gate parameterization:
$
  D_k = W_k = "diag"(1, 1, 1, e^(i phi_k))
$
with one learnable phase $phi_k$ per gate. The distinction between "disentangler" and "isometry" is purely in their _connectivity_ (which qubit pairs they act on), not their functional form.

*Why controlled-phase gates instead of full unitaries?*

+ *Unitarity requirement*: Image compression requires an invertible transform $cal(T)$ with $cal(T)^(-1)$. True MERA isometries ($2 times 4$ matrices on $"St"(2,4)$) reduce dimensions and cannot be inverted. We need all gates to be unitary ($2 times 2$) to preserve the output dimension.

+ *Consistency*: All basis types (QFT, Entangled QFT, TEBD) use controlled-phase gates. Using the same parameterization for the MERA-inspired basis ensures a uniform optimization framework, shared manifold operations, and fair comparison between basis types.

+ *Parsimony*: A full $U(4)$ gate has 16 real parameters. For small circuits (e.g., $n = 4$ qubits with 6 gates), this would give 96 parameters --- prone to overfitting when training on small image datasets. A single phase per gate keeps the model lean ($2(n-1)$ real parameters per dimension).

+ *Efficient evaluation*: Controlled-phase gates are diagonal in the computational basis, making them compatible with the einsum tensor contraction framework used for all basis types.

For $n = 2^k$ qubits in one dimension, the parameter count is:
- Layer $l$: $n \/ (2 dot 2^(l-1))$ disentanglers + $n \/ (2 dot 2^(l-1))$ isometries
- Total: $sum_(l=1)^k 2 dot n \/ 2^l = 2(n - 1)$ gates per dimension

For 2D images with $m$ row qubits and $n$ column qubits, the row and column circuits run independently, giving $2(m-1) + 2(n-1)$ total learnable phases. The parameter manifold is:
$
  cal(M)_"MERA" = product_(k=1)^(2(m-1)+2(n-1)) U(1)^4
$

The hierarchical connectivity captures multi-scale features: layer 1 gates act on nearest-neighbor qubits (fine-scale correlations), while deeper layers connect qubits at increasing stride (coarse-scale correlations). This gives $O(log n)$ circuit depth for $n$ qubits, compared to TEBD's $O(n)$ ring depth.

== Comparison with Fixed Bases

#table(
  columns: 3,
  [*Property*], [*Fixed Basis (FFT/DCT)*], [*Parametric QFT*],
  [Adaptivity], [None (fixed transform)], [Data-dependent optimization],
  [Sparsity], [Optimal for periodic/smooth signals], [Learned for specific input],
  [Computation], [$O(N log N)$], [$O(N log N)$ + training cost],
  [Unitarity], [Preserved], [Preserved (manifold constraint)],
)

The learned basis $cal(T)(bold(theta)^*)$ is signal-adaptive: unlike fixed DCT/FFT bases, it can exploit structure specific to the input data (e.g., textures, edges in images).

= Training Objective

Given a parametric basis $cal(T)(bold(theta))$ and a dataset of images, we need a loss function that drives the optimizer toward sparser representations. Different loss functions capture different notions of "good compression" and have different optimization properties.

== Loss Functions

=== L1 Norm Loss (Sparsity Promotion)

The L1 norm loss is defined as:
$cal(L)_(L 1)(bold(theta)) = sum_(i,j) |cal(T)(bold(theta))(bold(x))_(i,j)|$

This promotes *sparsity* by exploiting a key result from compressed sensing theory: minimizing the $ell_1$ norm $||bold(y)||_1$ is the tightest convex relaxation of the $ell_0$ pseudo-norm $||bold(y)||_0 = |{i,j : bold(y)_(i,j) != 0}|$. The optimization thus drives the transform to concentrate signal energy into fewer frequency components.

*Limitation*: The L1 norm is a proxy for sparsity, not a direct measure of reconstruction quality. It is independent of $||bold(x) - cal(T)^(-1)("truncate"(bold(y), k))||_F^2$, so the sparsest transform may not yield the best reconstruction at a given compression ratio $k$.

=== L2 Norm Loss

$cal(L)_(L 2)(bold(theta)) = sum_(i,j) |cal(T)(bold(theta))(bold(x))_(i,j)|^2$

Encourages energy concentration with smoother gradients than L1, but provides weaker sparsity promotion. Useful as a regularizer or when gradient stability is more important than aggressive sparsity.

=== MSE Reconstruction Loss

$cal(L)_"MSE"(bold(theta)) = ||bold(x) - cal(T)(bold(theta))^(-1)("truncate"(cal(T)(bold(theta))(bold(x)), k))||_F^2$

Directly optimizes reconstruction quality after top-$k$ truncation, measuring how well the image can be recovered from its $k$ most important coefficients. This is the loss most aligned with the compression objective, but it requires computing the inverse transform and involves a non-differentiable truncation step (addressed in @sec:topk-grad).

=== Hybrid Loss

$cal(L)_"hybrid"(bold(theta)) = alpha cal(L)_(L 1)(bold(theta)) + beta cal(L)_"MSE"(bold(theta))$

Balances sparsity promotion with reconstruction quality. The weights $alpha$ and $beta$ control the trade-off: larger $alpha$ favors sparser representations, while larger $beta$ favors faithful reconstruction.

== Frequency-Dependent Truncation

Naïve top-$k$ truncation selects coefficients by magnitude alone, treating all frequency components equally. In image compression, however, low-frequency components carry structural information (smooth gradients, large-scale shapes) while high-frequency components encode fine details (edges, textures). To bias retention toward perceptually important components, the truncation uses a frequency-weighted score:

$s_(i,j) = |bold(y)_(i,j)| dot (1 + w_(i,j))$, where $w_(i,j) = 1 - d_(i,j) / (2 d_"max")$

Here $d_(i,j) = sqrt((i - c_i)^2 + (j - c_j)^2)$ is the distance from frequency bin $(i,j)$ to the DC component $(c_i, c_j)$, and $d_"max"$ is the maximum such distance across the grid. The weight $w_(i,j)$ ranges from $1$ (at DC) to $0.5$ (at the Nyquist corner), so low-frequency coefficients receive up to a $2 times$ boost in their retention score relative to the highest frequencies. This ensures that, at equal magnitude, a low-frequency coefficient is preferred over a high-frequency one --- consistent with the human visual system's greater sensitivity to low-frequency content.

== Sparse Basis Learning Objective

Combining the loss functions and truncation rule, the sparse basis learning problem is:

$bold(theta)^* = arg min_(bold(theta) in cal(M)) cal(L)(bold(theta))$

where $cal(M)$ is the parameter manifold (a product of unitary and phase manifolds, as defined in @sec:riemannian) and $cal(L)$ is one of:

+ *L1 Sparsity*: $cal(L)_(L 1) = ||cal(T)(bold(theta))(bold(x))||_1$ --- promotes sparse representations by penalizing the total magnitude of all coefficients.
+ *MSE Reconstruction*: $cal(L)_"MSE" = ||bold(x) - cal(T)^(-1)("topk"(cal(T)(bold(x)), k))||_F^2$ --- directly optimizes reconstruction quality after frequency-weighted top-$k$ truncation.

The optimization is performed via Riemannian gradient methods (@sec:riemannian), which ensure that every iterate remains on the constraint manifold. The result $cal(T)(bold(theta)^*)$ is a signal-adaptive basis: unlike fixed DCT/FFT transforms, it exploits structure specific to the training data.

== Custom Gradient Rule for Top-$k$ Truncation <sec:topk-grad>

The MSE loss $cal(L)_"MSE" = ||bold(x) - cal(T)^(-1)("topk"(cal(T)(bold(x)), k))||_F^2$ involves a top-$k$ truncation that is not differentiable in the classical sense: the set of retained indices changes discontinuously as parameters vary. To make this operation compatible with reverse-mode automatic differentiation (Zygote), we define a custom chain rule (rrule) that provides a well-defined gradient.

*Forward pass*: Given the transformed coefficients $bold(y) = cal(T)(bold(x))$, compute frequency-weighted scores $s_(i,j) = |y_(i,j)| dot (1 + w_(i,j))$ (as defined in the truncation section), select the top-$k$ indices $cal(S) = "argtopk"(s, k)$, and zero out all other entries:
$
  ["topk"(bold(y), k)]_(i,j) = cases(y_(i,j) & "if" (i,j) in cal(S), 0 & "otherwise")
$

*Pullback (reverse pass)*: The gradient flows back only through the retained coefficients. Given an upstream gradient $overline(bold(y))$, the pullback produces:
$
  overline(bold(x))_(i,j) = cases(overline(y)_(i,j) & "if" (i,j) in cal(S), 0 & "otherwise")
$ <eq:topk_pullback>
This is the _straight-through estimator_ applied to truncation: we treat the truncation mask as fixed (non-differentiable) and pass gradients through the selected entries unchanged.

*Why this works*: Although the truncation mask $cal(S)$ depends on $bold(y)$ (and therefore on $bold(theta)$), ignoring this dependency is justified because:
+ The mask changes discretely --- a coefficient either enters or leaves $cal(S)$ --- so the true Jacobian has delta-function contributions at the boundaries that are unusable for gradient-based optimization.
+ The straight-through gradient correctly tells the optimizer: "to reduce the MSE loss, adjust the parameters so that the _retained_ coefficients better approximate the original signal." This is the actionable information.
+ In practice, the mask is relatively stable during training: once a coefficient is large enough to be in the top-$k$, small parameter changes keep it there, so the straight-through approximation is locally exact almost everywhere.

= Riemannian Optimization <sec:riemannian>

The trainable parameters of our quantum circuits --- Hadamard gates $H$, controlled-phase gates $M_k$, and TEBD/MERA two-qubit gates $T_k$ --- live on Riemannian manifolds rather than in Euclidean space. A naïve Euclidean update $H <- H - eta nabla f$ would immediately violate the unitarity constraint $H H^dagger = I$, producing a matrix that no longer represents a valid quantum gate. _Riemannian optimization_ resolves this by performing three operations at each iteration: (1) projecting the Euclidean gradient onto the tangent space of the manifold, (2) computing a descent direction in that tangent space, and (3) mapping the result back to the manifold via a retraction. This guarantees that every iterate is a valid point on the constraint surface, without the need for explicit re-orthogonalization or penalty terms.

== Manifold Structure

The circuit parameters inhabit two distinct manifolds, each with its own geometric structure:

+ *Unitary manifold $U(2)$*: Hadamard-like gates are $2 times 2$ unitary matrices satisfying $U U^dagger = I$. Geometrically, $U(2)$ is a compact Lie group of dimension 4. Its tangent space at a point $U$ consists of all matrices of the form
  $
    T_U U(2) = { U S : S^dagger = -S }
  $
  where $S$ is skew-Hermitian. Intuitively, infinitesimal perturbations of a unitary matrix must be "anti-Hermitian rotations" to preserve the constraint to first order.

+ *Product of unit circles $U(1)^4$*: Controlled-phase gates are diagonal unitary matrices with entries on the unit circle $|z_i| = 1$. The parameter space is a 4-torus $U(1)^4 = (S^1)^4$, with tangent space
  $
    T_z U(1)^4 = { i theta dot.c z : theta in RR^4 }
  $
  corresponding to purely imaginary scalings of each component. Each $z_i$ can only move tangentially along its circle, so the tangent direction at $z_i$ is $i theta_i z_i$ for some $theta_i in RR$.

== Riemannian Gradient

Automatic differentiation (Zygote) produces a Euclidean gradient $nabla_E f$ that lives in the ambient space $CC^(2 times 2)$ or $CC^4$, not on the tangent space of the manifold. To obtain a valid descent direction, we project this gradient onto the tangent space --- yielding the _Riemannian gradient_:

- *For $U(2)$*: $"grad" f(U) = U dot "skew"(U^dagger nabla_E f)$, where $"skew"(A) = (A - A^dagger) \/ 2$ extracts the skew-Hermitian part. The product $U^dagger nabla_E f$ pulls the gradient into the Lie algebra coordinates, and $"skew"$ removes the Hermitian component (which would move off the manifold).
- *For $U(1)^4$*: $"grad" f(z) = i dot "Im"(overline(z) circle.stroked.tiny nabla_E f) circle.stroked.tiny z$, applied element-wise. The term $"Im"(overline(z)_i dot (nabla_E f)_i)$ extracts the tangential component of the gradient along each circle.

#figure(canvas(length: 1.25cm, {
  import draw: *
  let purple-c = rgb("#8e5ea2")
  let red-c = rgb("#e15759")
  let blue-c = rgb("#4e79a7")

  // Manifold M as a curved arc
  bezier((-4.2, -0.5), (0, 2.3), (4.2, -0.5),
         stroke: (thickness: 2pt, paint: purple-c.darken(25%)))
  content((4.55, -0.15), text(12pt, weight: "bold", fill: purple-c.darken(25%))[$cal(M)$])

  // Point x at the peak of the arc (tangent horizontal here)
  let tx = 0.0
  let ty = 1.75
  circle((tx, ty), radius: 0.11, fill: black)
  content((tx - 0.35, ty + 0.3), text(12pt, weight: "bold")[$x$])

  // Tangent plane T_x M as a thin horizontal ribbon
  rect((tx - 2.7, ty - 0.1), (tx + 2.7, ty + 0.1),
       fill: blue-c.lighten(88%),
       stroke: (thickness: 0.7pt, paint: gray.darken(20%), dash: "dashed"))
  content((tx + 3.05, ty + 0.32), text(11pt, fill: gray.darken(25%))[$T_x cal(M)$])

  // Euclidean gradient ∇_E f --- off-manifold (up-right)
  let gx = tx + 1.4
  let gy = ty + 1.6
  line((tx, ty), (gx, gy),
       stroke: (thickness: 1.7pt, paint: red-c.darken(10%)),
       mark: (end: ">", size: 0.3))
  content((gx + 0.4, gy + 0.2),
          text(13pt, weight: "bold", fill: red-c.darken(10%))[$nabla_E f$])

  // Projected grad f --- horizontal component, in the tangent plane
  let gtx = tx + 1.4
  let gty = ty + 0.05
  line((tx, ty), (gtx, gty),
       stroke: (thickness: 1.7pt, paint: purple-c.darken(10%)),
       mark: (end: ">", size: 0.3))
  content((gtx + 0.55, gty - 0.08),
          text(13pt, weight: "bold", fill: purple-c.darken(10%))[$"grad" f$])

  // Dotted perpendicular: the projection itself
  line((gx, gy), (gtx, gty),
       stroke: (thickness: 0.8pt, paint: gray.darken(30%), dash: "dotted"))

  // Right-angle marker at the foot of the perpendicular
  line((gtx - 0.2, gty), (gtx - 0.2, gty + 0.2),
       stroke: (thickness: 0.6pt, paint: gray.darken(40%)))
  line((gtx - 0.2, gty + 0.2), (gtx, gty + 0.2),
       stroke: (thickness: 0.6pt, paint: gray.darken(40%)))

  // Subtle context labels
  content((-2.6, ty + 1.3),
          text(8pt, fill: red-c.darken(20%), style: "italic")[off-manifold])
  content((2.8, ty - 0.45),
          text(8pt, fill: purple-c.darken(20%), style: "italic")[on-tangent])

}), caption: [Riemannian projection: the Euclidean gradient $nabla_E f$ (red) shoots off the manifold $cal(M)$; its orthogonal projection onto the tangent space $T_x cal(M)$ (dashed ribbon) is the Riemannian gradient $"grad" f$ (purple). The dotted line is the projection direction, with the right-angle marker confirming orthogonality. Concrete forms for our two manifolds are given by the formulas above.]) <fig-tangent>

== Retraction

A tangent vector $xi in T_x cal(M)$ indicates a direction of descent, but following it linearly would leave the manifold. A _retraction_ $R_x : T_x cal(M) -> cal(M)$ maps the tangent update back onto the constraint surface:

- *Cayley retraction for $U(2)$*: Let $W = xi dot U^dagger$, projected to the Lie algebra $frak(u)(2)$ via $W <- (W - W^dagger)\/2$. The retraction is
  $
    R_U (alpha xi) = (I - alpha/2 dot W)^(-1) (I + alpha/2 dot W) dot U
  $
  The Cayley map preserves unitarity _exactly_ for any step size $alpha$, because $(I - alpha/2 dot W)^(-1)(I + alpha/2 dot W)$ is unitary whenever $W$ is skew-Hermitian. This is a stronger guarantee than first-order retractions, which only approximate the manifold. The Cayley map also avoids the QR decomposition and its sign-correction step, yielding a simpler implementation.

- *Normalization for $U(1)^4$*: $R_z (alpha xi) = (z + alpha xi) \/ |z + alpha xi|$, applied element-wise. This projects each updated component back onto the unit circle by normalizing its magnitude --- the simplest possible retraction for $U(1)$.

=== Cayley vs QR Retraction

Two widely used retractions for the unitary manifold $U(n)$ are the Cayley map and the QR decomposition. Our implementation uses the Cayley retraction; the comparison below summarizes their trade-offs:

*Cayley retraction*: Given a skew-Hermitian direction $W in frak(u)(n)$,
$
  R^"Cayley"_U (alpha W) = (I - alpha/2 dot W)^(-1) (I + alpha/2 dot W) dot U
$

*QR retraction*: Given $U + alpha xi = Q R$ (thin QR decomposition),
$
  R^"QR"_U (alpha xi) = Q dot "diag"("sgn"(R_(i i)))
$

#table(
  columns: 3,
  [*Property*], [*Cayley*], [*QR*],
  [Cost ($n times n$)], [$O(n^3)$ (matrix inverse)], [$O(n^3)$ (QR factorization)],
  [Exactness], [Exactly unitary for any $alpha$], [Exactly unitary for any $alpha$],
  [Input requirement], [Skew-Hermitian $W in frak(u)(n)$], [Any tangent vector $xi in T_U U(n)$],
  [Numerical stability], [Stable for small $n$], [Numerically robust via Householder],
  [Implementation], [Requires batched inverse], [Requires sign correction],
)

For our application with $2 times 2$ gates, both retractions have $O(1)$ cost per gate, so the choice is driven by algebraic convenience rather than performance:
+ The Cayley map operates directly in the Lie algebra $frak(u)(2)$, making the connection between tangent vectors and group elements algebraically transparent.
+ The re-projection $W <- (W - W^dagger)\/2$ acts as a safety net: even when the direction is not exactly tangent (e.g., after Adam's element-wise scaling), the Cayley map still produces a valid unitary.
+ For $2 times 2$ matrices, the inverse $(I - alpha/2 dot W)^(-1)$ admits a closed-form expression via the adjugate matrix, sidestepping numerical pivoting entirely.

== Riemannian Gradient Descent

The simplest Riemannian optimizer applies the steepest-descent direction in the tangent space, then retracts:
$
  x_(t+1) = R_(x_t) (-eta dot "grad" f(x_t))
$ <eq:rgd>
where $eta > 0$ is the step size and $R$ is the retraction defined above. This is the manifold analogue of Euclidean gradient descent: at each step, the Riemannian gradient gives the direction of steepest decrease on the manifold, and the retraction ensures the iterate remains feasible.

=== Armijo Backtracking Line Search <armijo>

The update rule @eq:rgd requires a step size $eta$ that produces sufficient decrease in the loss. A fixed step size is unreliable: too large and the iterates oscillate or diverge; too small and convergence stalls. The _Armijo backtracking line search_ resolves this by adaptively selecting $eta$ at each iteration.

Starting from an initial step size $alpha_0 > 0$ (e.g., $alpha_0 = 0.01$), the algorithm tests the _sufficient decrease condition_ (Armijo condition):
$
  cal(L)(R_(x_t)(-alpha dot "grad" f(x_t))) <= cal(L)(x_t) - c dot alpha dot ||"grad" f(x_t)||^2
$ <eq:armijo>
where $c in (0, 1)$ is a small constant (typically $c = 10^(-4)$) that sets the bar for "sufficient" decrease. If the condition fails, the step size is contracted:
$
  alpha <- tau dot alpha, quad tau in (0, 1)
$
with $tau = 0.5$ (halving). The process repeats for up to a fixed number of backtracking steps (e.g., 10). The algorithm accepts the first $alpha$ satisfying @eq:armijo, or falls back to the smallest step tried.

*Why Armijo is well-suited to Riemannian optimization*: On a curved manifold, the retraction $R_(x_t)$ maps the tangent space back to the manifold surface. For large step sizes, the manifold "bends away" from the tangent approximation, so a step size that was valid at one point may overshoot at another. The Armijo condition adapts to this curvature automatically: it guarantees that each step makes progress proportional to $alpha ||g||^2$, while the backtracking loop finds the largest acceptable step size, balancing convergence speed against the risk of overshooting.

== Riemannian Adam

Riemannian gradient descent with Armijo line search is reliable but can converge slowly, particularly in loss landscapes with very different curvatures along different parameter directions. We also implement _Riemannian Adam_ (Bécigneul & Ganea, 2019), which extends the classical Adam optimizer to manifolds. Adam maintains per-parameter running averages of the gradient (first moment) and squared gradient (second moment), using their ratio to scale each parameter's update independently. This adaptive scaling implicitly adjusts the effective step size for each gate, often yielding faster convergence than a single global step size --- at the cost of weaker per-step descent guarantees.

Let $g_t = "grad" f(x_t)$ be the Riemannian gradient at step $t$. The algorithm maintains exponentially weighted first and second moment estimates:

$
  m_t &= beta_1 m_(t-1) + (1 - beta_1) g_t \
  v_t &= beta_2 v_(t-1) + (1 - beta_2) |g_t|^2
$ <eq:adam_moments>

where $beta_1, beta_2 in [0, 1)$ are decay rates (typically $beta_1 = 0.9$, $beta_2 = 0.999$). To correct the initialization bias (both moments start at zero), we use the bias-corrected estimates $hat(m)_t = m_t \/ (1 - beta_1^t)$ and $hat(v)_t = v_t \/ (1 - beta_2^t)$. The update direction is:

$
  d_t = hat(m)_t \/ (sqrt(hat(v)_t) + epsilon)
$

where $epsilon approx 10^(-8)$ prevents division by zero. The new iterate is obtained by retraction:

$
  x_(t+1) = R_(x_t) (-eta dot d_t)
$

*Parallel transport of momentum.* A subtlety arises because $m_t$ lives in the tangent space $T_(x_t) cal(M)$, but after the retraction, the new iterate $x_(t+1)$ has a _different_ tangent space $T_(x_(t+1)) cal(M)$. To carry the momentum forward, it must be _parallel transported_ to the new tangent space. We use projection-based transport as a computationally cheap first-order approximation:

$
  Gamma_(x_t -> x_(t+1)) (m_t) = "proj"_(T_(x_(t+1)) cal(M)) (m_t)
$

For $U(2)$, this is $U_(t+1) dot "skew"(U_(t+1)^dagger m_t)$: the old momentum is re-projected into the Lie algebra at the new point. For $U(1)^4$, the tangent projection is re-applied element-wise at the new phase values. This approximation is exact to first order in the step size and introduces negligible error for the small steps typical in practice.

== Batched Einsum and Gradient Computation

Evaluating the forward transform $bold(y) = cal(T)(bold(theta))(bold(x))$ amounts to a tensor contraction: the image $bold(x)$ is reshaped into a rank-$(m+n)$ tensor with binary indices $(q_1, ..., q_(m+n))$, each of dimension 2, and contracted with the circuit gate tensors $H_1, H_2, ..., M_1, M_2, ...$ via an Einstein summation (einsum) code. The einsum formulation has two key advantages: it makes the contraction order explicit and optimizable, and it naturally extends to batched evaluation.

*Single-image forward pass.* The einsum code specifies which tensor indices are contracted (internal circuit wires) and which survive (output qubit indices). The contraction order is pre-optimized using a tree search algorithm (TreeSA) that minimizes the total floating-point operation count. For a QFT on $k$ qubits, the optimized contraction achieves $O(2^k dot k)$ complexity, matching the classical FFT.

*Batched forward pass.* During training, we process $B$ images simultaneously to amortize overhead. A batch label $beta$ is appended to the image input and output indices of the einsum code, and the $B$ images are stacked into a single $(2, 2, ..., 2, B)$ tensor. The einsum then contracts all $B$ images in one call:
$
  bold(Y)_(q'_1, ..., q'_(m+n), beta) = sum_(q_1, ..., q_(m+n)) (product_j G^j_(dots)) bold(X)_(q_1, ..., q_(m+n), beta)
$
The gate tensors $G^j$ are shared across the batch --- they do not carry the $beta$ index. This reduces GPU kernel launch overhead from $O(B)$ to $O(1)$, a significant saving when $B$ is large and each individual contraction is small (as is the case for $2 times 2$ gates).

*Gradient computation.* Zygote's reverse-mode automatic differentiation traces through the einsum call to produce the Euclidean gradient $nabla_E cal(L)$ with respect to each gate tensor. Three design choices ensure correct and stable differentiation:

+ *Tuple vs Vector tangent types*: Zygote represents the gradient of a vector-of-tensors as a tuple of tangent arrays, but the Riemannian optimizer expects a vector. The loss function converts `Vector` inputs to `Tuple` before the einsum call, so that Zygote produces stable, predictable tangent types. After differentiation, the tuple gradient is converted back to a vector for the optimizer.
+ *No mutation in the forward pass*: The einsum contraction is a pure function --- no in-place array modification --- as required by Zygote's source-to-source AD. All intermediate arrays are constructed via allocation (`reshape`, `cat`) rather than mutation (`setindex!`, `.=`).
+ *Conjugation for the inverse transform*: The inverse transform uses conjugated gate tensors $overline(G)^j$ rather than explicit matrix inverses, exploiting unitarity: $U^(-1) = U^dagger = overline(U)^top$. This keeps the inverse pass as cheap as the forward pass and avoids numerical issues from matrix inversion.
