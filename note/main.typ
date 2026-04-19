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

A *sparse basis* is a transform $cal(T)$ for which $bold(y) = cal(T)(bold(x))$ concentrates energy in few components: $||bold(y)||_0 << dim(bold(y))$.

The Fourier basis is not optimal for natural images: it assumes periodic boundaries, and its 2D form treats the $X$ and $Y$ axes as independent, ignoring spatial correlations. The tensor-network form of the FFT, however, exposes parameters (Hadamard gates $H$, controlled-phase gates $M_k$) that can be tuned without changing the $O(n log n)$ cost. This suggests: learn these parameters from data.

We present _parametric tensor network bases_ — data-adaptive unitary transforms parameterized by quantum circuits. Parameters live on Riemannian manifolds ($U(2)$ for Hadamard, $U(1)^4$ for phase gates) and are optimized via Riemannian gradient descent. The classical QFT is one point on this manifold; training moves to a better one.

We cover four topologies (QFT, Entangled QFT, TEBD, MERA), sparsity-promoting losses, and the Riemannian machinery that keeps iterates unitary.

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
The Cooley-Tukey FFT is divide-and-conquer. For $n$ a power of 2, split the matrix into four blocks:
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
where $bold(x)_("odd")$ and $bold(x)_("even")$ are the odd- and even-indexed elements of $bold(x)$. The DFT thus decomposes into two half-size DFTs with a diagonal twiddle $D_(n\/2)$ between them. Diagonal matrices apply in $O(n)$, giving $T(n) = 2T(n\/2) + O(n) = O(n log n)$.

The inverse is $F_n^dagger bold(x)\/n$, since $F_n F_n^dagger = n I$.

== Tensor Network Representation of the FFT <bg-tn>

We reformulate the FFT as a tensor network, exposing the parametric structure used throughout. A vector of size $n = 2^k$ becomes a rank-$k$ tensor with binary indices, via $i = 2^0 q_0 + 2^1 q_1 + dots + 2^(k-1) q_(k-1)$.
#figure(canvas({
  import draw: *
  let n = 4
  ngate((0, 0), n, "F_n", text:[$bold(x)$], gap-y: 0.8, width: 0.7)
  for i in range(n){
    line("F_n.o" + str(i), (rel: (0.5, 0)))
    content((rel: (0.7, 0), to: "F_n.o" + str(i)), [$q_#i$])
  }
}))

We seek a tensor network decomposition of $F_n$:
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

*Step 2*: Decompose $mat(I_(n/2), 0; 0, D_(n/2))$ — identity when $q_0 = 0$, $D_(n/2)$ when $q_0 = 1$ — into a tensor network.
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
where $"ctrl"_i (A_j)$ applies $A_j$ only when bit $q_i = 1$. Since each controlled gate is diagonal, it is drawn as a matrix connecting two qubits:

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
The gate $M_k = mat(1, 1; 1, e^(i pi \/ 2^(k-1)))$ multiplies by $e^(i pi\/2^(k-1))$ when both qubits are in state $1$. Recursing on $F_(n/2)$ yields the complete network:

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

Direct evaluation is $O(n log^2 n)$. Merging adjacent diagonal phase operations on the same qubit recovers the classical $O(n log n)$.

= Parametric Circuit Bases

Four topologies share the same gate parameterization ($H in U(2)$, controlled-phase $in U(1)^4$) and optimization framework; they differ only in which qubit pairs are connected.

== QFT Basis

The QFT basis is the tensor network of @bg-tn with every gate made learnable: Hadamards on $U(2)$ and controlled-phase $M_k = "diag"(1,1,1,e^(i phi))$ on $U(1)^4$. With fixed $M_k$ phases it reduces to the classical DFT. For a $2^m times 2^n$ image, separate QFT circuits act on the $m$ row and $n$ column qubits: $cal(T) = F_m times.circle F_n$.

== Entangled QFT Basis: XY Correlation

Separating $x$ and $y$ ignores cross-dimensional structure in natural images (diagonal edges, oblique textures). We add controlled-phase _entanglement_ gates between corresponding row and column qubits. For $m = n$, pair $x_k$ with $y_k$ via
$
  E_k = mat(1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, e^(i phi_k))
$
acting on $(x_(n-k), y_(n-k))$. $E_k$ has the same form as $M_k$ but with a _learnable_ phase $phi_k$ in place of the fixed $pi\/2^(k-1)$.

Circuit for $n = 4$: each qubit runs its standard QFT (Hadamard then $M_j$ gates), then an $E_k$ couples each row qubit to its column partner:

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

Per-qubit sequence: Hadamard, intra-dimension $M_j$ gates (standard QFT), then $E_(n-k)$ with the partner qubit. The full transform is
$
  cal(T)_"entangled" = U_"entangle" dot (F_n times.circle F_n), quad U_"entangle" = product_(k=1)^n E_k.
$
Cost remains $O(N log N)$; only $n$ real parameters are added; $phi_k = 0$ recovers the separable 2D QFT.

== TEBD Basis

Time-Evolving Block Decimation (TEBD) replaces the QFT's hierarchical all-to-all connectivity with a _ring topology_: each qubit gets a Hadamard, then nearest-neighbor controlled-phase gates around a ring. Row and column qubits form two independent rings.

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

Each two-qubit gate is $T_k = "diag"(1, 1, 1, e^(i phi_k))$ with a learnable $phi_k$ — same form as $M_k$ but with trainable phase. Two rings of $n$ gates give $2n$ phases, with parameter manifold
$
  cal(M)_"TEBD" = product_(k=1)^(2n) U(1)^4.
$
Evaluation is $O(n)$, the cheapest of the four bases.

== MERA-inspired Basis

The original Multi-scale Entanglement Renormalization Ansatz (MERA) uses $U(4)$ disentanglers and Stiefel-manifold isometries. True coarse-graining reduces the output dimension, so it cannot give an invertible transform. We keep MERA's hierarchical connectivity but parameterize every gate as a controlled-phase gate — consistent with the other bases and invertible by construction.

For $n = 2^k$ qubits the circuit has $k = log_2 n$ layers. Layer $l$ has stride $s = 2^(l-1)$ and $n\/(2s)$ pairs; each pair is a _disentangler_ followed by an _isometry_, both controlled-phase gates $"diag"(1,1,1,e^(i phi))$:

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

Both gate types share the form $D_k = W_k = "diag"(1, 1, 1, e^(i phi_k))$; disentangler vs isometry is a distinction of _connectivity_ only.

Controlled-phase (not full $U(4)$) gates are chosen for: invertibility (the transform must be bijective); consistency with the other bases; parsimony ($U(4)$ has 16 parameters vs 1 per gate, avoiding overfitting on small datasets); and diagonality, which plugs cleanly into the shared einsum framework.

Per dimension, $sum_(l=1)^k 2 dot n\/2^l = 2(n-1)$ gates. For 2D images with $m + n$ qubits the parameter manifold is
$
  cal(M)_"MERA" = product_(k=1)^(2(m-1)+2(n-1)) U(1)^4.
$
Layer $l$ has stride $2^(l-1)$, so depth is $O(log n)$ — compared to TEBD's $O(n)$ — and captures correlations across multiple scales.

== Comparison with Fixed Bases

#table(
  columns: 3,
  [*Property*], [*Fixed Basis (FFT/DCT)*], [*Parametric QFT*],
  [Adaptivity], [None (fixed transform)], [Data-dependent optimization],
  [Sparsity], [Optimal for periodic/smooth signals], [Learned for specific input],
  [Computation], [$O(N log N)$], [$O(N log N)$ + training cost],
  [Unitarity], [Preserved], [Preserved (manifold constraint)],
)

The learned basis $cal(T)(bold(theta)^*)$ is signal-adaptive — it exploits structure specific to the training data (textures, edges) rather than the generic assumptions behind DCT/FFT.

= Training Objective

== Loss Functions

=== L1 Norm Loss

$cal(L)_(L 1)(bold(theta)) = sum_(i,j) |cal(T)(bold(theta))(bold(x))_(i,j)|$

$ell_1$ is the tightest convex relaxation of $ell_0$, so minimizing it concentrates energy into few components. _Limitation_: it is a proxy for sparsity, not reconstruction error — the sparsest transform may not reconstruct best at a given $k$.

=== L2 Norm Loss

$cal(L)_(L 2)(bold(theta)) = sum_(i,j) |cal(T)(bold(theta))(bold(x))_(i,j)|^2$

Smoother gradients than L1, weaker sparsity pressure. Useful as a regularizer.

=== MSE Reconstruction Loss

$cal(L)_"MSE"(bold(theta)) = ||bold(x) - cal(T)(bold(theta))^(-1)("truncate"(cal(T)(bold(theta))(bold(x)), k))||_F^2$

Directly optimizes reconstruction from the top-$k$ coefficients. Most aligned with the compression goal, but requires the inverse transform and a non-differentiable truncation (see @sec:topk-grad).

=== Hybrid Loss

$cal(L)_"hybrid"(bold(theta)) = alpha cal(L)_(L 1)(bold(theta)) + beta cal(L)_"MSE"(bold(theta))$

Trades sparsity against reconstruction; larger $alpha$ favors sparsity, larger $beta$ favors fidelity.

== Frequency-Dependent Truncation

Magnitude-only top-$k$ treats all frequencies equally, but the HVS is more sensitive to low frequencies. We score coefficients with a frequency-weighted score:

$s_(i,j) = |bold(y)_(i,j)| dot (1 + w_(i,j))$, with $w_(i,j) = 1 - d_(i,j) / (2 d_"max")$,

where $d_(i,j)$ is the distance from bin $(i,j)$ to the DC center and $d_"max"$ is its max over the grid. Then $w$ ranges from $1$ at DC to $0.5$ at the Nyquist corner — a $2 times$ boost for low-frequency coefficients at equal magnitude.

== Sparse Basis Learning Objective

The training problem is
$
  bold(theta)^* = arg min_(bold(theta) in cal(M)) cal(L)(bold(theta)),
$
with $cal(M)$ the parameter manifold (@sec:riemannian) and $cal(L)$ one of the losses above. Optimization uses Riemannian methods (@sec:riemannian), which keep every iterate on the manifold.

== Custom Gradient Rule for Top-$k$ Truncation <sec:topk-grad>

The retained-index set $cal(S)$ changes discontinuously with $bold(theta)$, so top-$k$ is not classically differentiable. We use a _straight-through_ rrule.

*Forward*: with scores $s_(i,j) = |y_(i,j)|(1 + w_(i,j))$ and $cal(S) = "argtopk"(s, k)$,
$
  ["topk"(bold(y), k)]_(i,j) = cases(y_(i,j) & "if" (i,j) in cal(S), 0 & "otherwise").
$

*Pullback*: treat $cal(S)$ as fixed and pass gradients through retained entries:
$
  overline(bold(x))_(i,j) = cases(overline(y)_(i,j) & "if" (i,j) in cal(S), 0 & "otherwise").
$ <eq:topk_pullback>

The true Jacobian carries delta-function terms at index-switch boundaries — useless for gradient descent. The straight-through estimator instead encodes the actionable signal: "move the retained coefficients closer to $bold(x)$." In training, $cal(S)$ is locally stable, so this approximation is exact almost everywhere.

= Riemannian Optimization <sec:riemannian>

Circuit parameters — Hadamards $H$, controlled-phase gates $M_k$, TEBD/MERA gates $T_k$ — live on manifolds, not in Euclidean space. A Euclidean step $H <- H - eta nabla f$ immediately violates $H H^dagger = I$. Each Riemannian iteration instead (1) projects $nabla_E f$ onto the tangent space, (2) takes a descent step there, and (3) retracts back to the manifold — so every iterate is feasible without penalty terms.

== Manifold Structure

+ *Unitary manifold $U(2)$*: $2 times 2$ matrices with $U U^dagger = I$, a compact Lie group of dimension 4. Tangent space:
  $
    T_U U(2) = { U S : S^dagger = -S },
  $
  i.e. $U$ times a skew-Hermitian matrix — the first-order condition for preserving $U U^dagger = I$.

+ *Phase manifold $U(1)^4$*: diagonal unitaries with $|z_i| = 1$, i.e. the 4-torus $(S^1)^4$. Tangent space:
  $
    T_z U(1)^4 = { i theta dot.c z : theta in RR^4 },
  $
  purely imaginary scalings of each component.

== Riemannian Gradient

Zygote returns a Euclidean gradient $nabla_E f$ in ambient space. Projecting onto the tangent space gives the _Riemannian gradient_:

- $U(2)$: $"grad" f(U) = U dot "skew"(U^dagger nabla_E f)$, with $"skew"(A) = (A - A^dagger)\/2$. $U^dagger nabla_E f$ pulls the gradient into the Lie algebra; $"skew"$ strips the Hermitian component that would leave the manifold.
- $U(1)^4$: element-wise $"grad" f(z) = i dot "Im"(overline(z) circle.stroked.tiny nabla_E f) circle.stroked.tiny z$ — the tangential component along each circle.

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

A tangent step $alpha xi$ leaves the manifold; a _retraction_ $R_x : T_x cal(M) -> cal(M)$ maps it back.

- *Cayley, for $U(2)$*: set $W = xi dot U^dagger$ and project to $frak(u)(2)$ via $W <- (W - W^dagger)\/2$. Then
  $
    R_U (alpha xi) = (I - alpha/2 dot W)^(-1) (I + alpha/2 dot W) dot U.
  $
  Exactly unitary for any $alpha$ whenever $W$ is skew-Hermitian; no QR or sign correction.

- *Normalization, for $U(1)^4$*: element-wise $R_z (alpha xi) = (z + alpha xi)\/|z + alpha xi|$.

=== Cayley vs QR Retraction

We use Cayley. For $U(n)$ the two standard options are:
$
  R^"Cayley"_U (alpha W) = (I - alpha/2 dot W)^(-1) (I + alpha/2 dot W) dot U, quad W in frak(u)(n)
$
$
  R^"QR"_U (alpha xi) = Q dot "diag"("sgn"(R_(i i))), quad U + alpha xi = Q R.
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

For $2 times 2$ gates both are $O(1)$ per gate, so we pick on algebraic grounds: Cayley stays in $frak(u)(2)$; the re-projection $W <- (W - W^dagger)\/2$ absorbs any non-tangent component from Adam's element-wise scaling; and $(I - alpha/2 dot W)^(-1)$ has a closed form via the adjugate — no pivoting.

== Riemannian Gradient Descent

Steepest descent in the tangent space followed by retraction:
$
  x_(t+1) = R_(x_t) (-eta dot "grad" f(x_t)).
$ <eq:rgd>

=== Armijo Backtracking Line Search <armijo>

A fixed $eta$ either oscillates or stalls; we adapt it per step. Starting from $alpha_0 > 0$ (e.g., $0.01$), accept the first $alpha$ satisfying the sufficient-decrease condition
$
  cal(L)(R_(x_t)(-alpha dot "grad" f(x_t))) <= cal(L)(x_t) - c dot alpha dot ||"grad" f(x_t)||^2,
$ <eq:armijo>
with $c = 10^(-4)$. Otherwise contract $alpha <- tau dot alpha$ with $tau = 0.5$, for up to $~10$ steps. This adapts to manifold curvature: the tangent approximation breaks down for large $alpha$, and Armijo bounds each step's progress by $alpha ||g||^2$.

== Riemannian Adam

RGD+Armijo is reliable but slow when curvatures differ across parameters. Riemannian Adam (Bécigneul & Ganea, 2019) scales each parameter's step by the ratio of first and second moments.

With $g_t = "grad" f(x_t)$, the moments are
$
  m_t &= beta_1 m_(t-1) + (1 - beta_1) g_t \
  v_t &= beta_2 v_(t-1) + (1 - beta_2) |g_t|^2
$ <eq:adam_moments>
with $beta_1 = 0.9$, $beta_2 = 0.999$. Bias-corrected $hat(m)_t = m_t\/(1-beta_1^t)$, $hat(v)_t = v_t\/(1-beta_2^t)$ give the direction and update
$
  d_t = hat(m)_t \/ (sqrt(hat(v)_t) + epsilon), quad x_(t+1) = R_(x_t) (-eta dot d_t),
$
with $epsilon approx 10^(-8)$.

*Parallel transport of momentum.* After retraction, $m_t in T_(x_t) cal(M)$ must be transported to $T_(x_(t+1)) cal(M)$. We use projection-based transport
$
  Gamma_(x_t -> x_(t+1)) (m_t) = "proj"_(T_(x_(t+1)) cal(M)) (m_t),
$
i.e. $U_(t+1) dot "skew"(U_(t+1)^dagger m_t)$ for $U(2)$ and the element-wise tangent projection at the new phases for $U(1)^4$. First-order accurate; the error is negligible at typical step sizes.

== Batched Einsum and Gradient Computation

The forward transform is a tensor contraction: reshape $bold(x)$ to a rank-$(m+n)$ tensor with binary indices and contract with the gate tensors via an Einstein-summation (einsum) code. This exposes the contraction order to optimization and extends naturally to batches.

*Single-image forward pass.* The einsum code names contracted (internal) and surviving (output) indices. The contraction order is pre-optimized by TreeSA, minimizing total flops; QFT on $k$ qubits reaches $O(2^k dot k)$.

*Batched forward pass.* Stack $B$ images into $(2,...,2,B)$, append a batch label $beta$ to the image and output indices:
$
  bold(Y)_(q'_1, ..., q'_(m+n), beta) = sum_(q_1, ..., q_(m+n)) (product_j G^j_(dots)) bold(X)_(q_1, ..., q_(m+n), beta).
$
Gate tensors $G^j$ are shared (no $beta$ index), cutting kernel launches from $O(B)$ to $O(1)$ — a large saving on GPU for small $2 times 2$ gates.

*Gradient computation.* Zygote differentiates through the einsum. Three conventions keep AD correct and efficient:

+ *Tuple tangents*: the loss wraps the tensor `Vector` into a `Tuple` before contraction, since Zygote produces stable tangent types on tuples; the result is converted back after.
+ *No mutation in the forward pass*: intermediates are built by `reshape`/`cat`, never `setindex!` or `.=`, as required by Zygote.
+ *Inverse via conjugation*: since $U^(-1) = U^dagger = overline(U)^top$, the inverse transform uses $overline(G)^j$ instead of matrix inversion — same cost as forward, no numerical inversion issues.
