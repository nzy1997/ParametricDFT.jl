#import "@preview/cetz:0.4.2": canvas, draw
#import "@preview/quill:0.7.1": *
#show link: set text(blue)
#set math.equation(numbering: "(1)")
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

= Sparsity in images, a basis better than Fourier basis

== Background knowlege: Cooley-Tukey FFT
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
It indicates that the discrete Fourier transormation in $RR^n$ can be decomposed into two smaller discrete Fourier transormations in $RR^(n/2)$ with a diagonal matrix $D_n$ in between.
Note applying diagonal matrices can be done in $O(n)$ operations, this decomposition leads to the recurrence relation $T(n) = 2T(n/2) + O(n)$, which solves to $O(n log n)$ total operations.

The inverse transformation is given by $F_n^dagger bold(x)\/n$. The DFT matrix is unitary up to a scale factor: $F_n F_n^dagger = n I$.

== Tensor network representation of the Cooley-Tukey FFT
This section requires preliminary knowledge of tensor networks (TODO: add reference).
In tensor network diagram, a vector of size $n=2^k$ can be represented as a tensor with $k$ indices, denoting the basis index $i = 2^0 q_0 + 2^1 q_1 + ... + 2^(k-1) q_(k-1)$.
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
) times.o I_(n/2)) mat(I_(n/2), 0; 0, D_(n/2)) vec(
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
where $H = mat(1, 1; 1, -1)$ is a Hadamard matrix (upto a constant factor).

Step 2: Then, we will decompose the diagonal matrix $mat(I_(n/2), 0; 0, D_(n/2))$ into a tensor network.
This diagonal matrix corresponds to operation: if $q_0$ is $0$ (odd index), the no operation is applied, otherwise (even index) the operation $D_(n/2)$ is applied.
// We define the following control tensor in block matrix form:
// $
//   "ctrl"_1(A) := mat(I, 0; 0, A)
// $
// Only if $q_1 = 1$, the operation $A$ is applied.
// Similarly, we can define $"ctrl"_2(A)$, $"ctrl"_3(A)$, ..., $"ctrl"_n(A)$ if $q_2 = 1$, $q_3 = 1$, ..., $q_n = 1$, the operation $A$ is applied.
// Then we have:
// $
//   mat(I_A times.o I_B, 0; 0, A times.o B) = "ctrl"_1(A times.o I_B) "ctrl"_1(I_A times.o B)
// $
Observe that
$
D_n = "diag"(1, omega^(n/2)) times.o "diag"(1, omega, omega^2, ..., omega^(n/2-1)) = "diag"(1, omega^(n/2)) times.o "diag"(1, omega^(n/4)) times.o dots times.o "diag"(1, omega)$. We have
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
In this diagram, $M_k = mat(1, 1; 1, e^(i pi \/ 2^(k-1)))$ connects the two qubits involved in the controlled operation, which effectively multilies a phase factor $e^(i pi \/ 2^(k-1))$ if two bit are both in state $1$. By recursively decomposing the $F_(n/2)$ tensor, we can obtain the following tensor network.

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

Direct evaluation of this tensor network takes $O(n log^2 n)$ operations. By respecting the fact that the _controlled phase_ operation is a diagonal matrix, we can merge these operations and further reduce the complexity to $O(n log(n))$.

== Entangled Fourier Basis: Adding XY Correlation
In the standard 2D Fourier transform, the $x$ and $y$ coordinates are processed independently. For an image of size $2^n times 2^n$ (i.e., square images with $m = n$), we apply QFT on the $n$ row qubits and separately on the $n$ column qubits. This independence assumption is often suboptimal for natural images where spatial correlations exist between rows and columns.

We propose an _entangled QFT basis_ that introduces controlled-phase gates between x and y qubits after each layer of the QFT circuit. For the square case $m = n$, we use a _one-to-one_ entanglement structure where each x qubit $x_k$ is coupled with the corresponding y qubit $y_k$. This creates correlation between the two spatial dimensions:

#figure(canvas({
  import draw: *
  let n = 8
  // x_n block
  ngate((-3, 0), 4, "x_n", text:[$bold(x)$], gap-y: 0.8, width: 0.7)

  ngate((7.0, 1.2), 1, "Hx1", text:[$H$], gap-y: 0.8, width: 0.5)
  line("x_n.o0", "Hx1.i0")
  line("Hx1.o0", (8.8, 1.2))
  cphase(6.2, 1.2, 0.4, 1, name: "CP11")
  cphase(5.4, 1.2, -0.4, 2, name: "CP12")
  cphase(4.6, 1.2, -1.2, 3, name: "CP13")

  ngate((3.2, 0.4), 1, "Hx2", text:[$H$], gap-y: 0.8, width: 0.5)
  line("Hx2.o0", "CP11ctrl2")
  line("CP11ctrl2", (8.8, 0.4))
  line("x_n.o1", "Hx2.i0")
  cphase(2.4, 0.4, -0.4, 1, name: "CP21")
  cphase(1.6, 0.4, -1.2, 2, name: "CP22")

  ngate((0.2, -0.4), 1, "Hx3", text:[$H$], gap-y: 0.8, width: 0.5)
  line("Hx3.o0", "CP21ctrl2")
  line("CP21ctrl2", "CP12ctrl2")
  line("CP12ctrl2", (8.8, -0.4))
  line("x_n.o2", "Hx3.i0")
  cphase(-0.6, -0.4, -1.2, 1, name: "CP31")

  ngate((-2.0, -1.2), 1, "Hx4", text:[$H$], gap-y: 0.8, width: 0.5)
  line("Hx4.o0", "CP31ctrl2")
  line("CP31ctrl2", "CP22ctrl2")
  line("CP22ctrl2", "CP13ctrl2")
  line("CP13ctrl2", (8.8, -1.2))
  line("x_n.o3", "Hx4.i0")

  // y_n block - same structure as x_n
  ngate((-3, -3.2), 4, "y_n", text:[$bold(y)$], gap-y: 0.8, width: 0.7)

  ngate((7.0, -2.0), 1, "Hy1", text:[$H$], gap-y: 0.8, width: 0.5)
  line("y_n.o0", "Hy1.i0")
  line("Hy1.o0", (8.8, -2.0))
  cphase(6.2, -2.0, -2.8, 1, name: "CPy11")
  cphase(5.4, -2.0, -3.6, 2, name: "CPy12")
  cphase(4.6, -2.0, -4.4, 3, name: "CPy13")

  ngate((3.2, -2.8), 1, "Hy2", text:[$H$], gap-y: 0.8, width: 0.5)
  line("Hy2.o0", "CPy11ctrl2")
  line("CPy11ctrl2", (8.8, -2.8))
  line("y_n.o1", "Hy2.i0")
  cphase(2.4, -2.8, -3.6, 1, name: "CPy21")
  cphase(1.6, -2.8, -4.4, 2, name: "CPy22")

  ngate((0.2, -3.6), 1, "Hy3", text:[$H$], gap-y: 0.8, width: 0.5)
  line("Hy3.o0", "CPy21ctrl2")
  line("CPy21ctrl2", "CPy12ctrl2")
  line("CPy12ctrl2", (8.8, -3.6))
  line("y_n.o2", "Hy3.i0")
  cphase(-0.6, -3.6, -4.4, 1, name: "CPy31")

  ngate((-2.0, -4.4), 1, "Hy4", text:[$H$], gap-y: 0.8, width: 0.5)
  line("Hy4.o0", "CPy31ctrl2")
  line("CPy31ctrl2", "CPy22ctrl2")
  line("CPy22ctrl2", "CPy13ctrl2")
  line("CPy13ctrl2", (8.8, -4.4))
  line("y_n.o3", "Hy4.i0")

  // Entanglement gates - placed right after each H gate
  // E4: connects x3 (y=-1.2) with y3 (y=-4.4) - right after Hx4/Hy4
  circle((-1.4, -1.2), radius: 0.08, fill: black)
  circle((-1.4, -4.4), radius: 0.08, fill: black)
  line((-1.4, -1.2), (-1.4, -4.4))
  content((-1.4, -2.8), [$E_4$], anchor: "west", padding: 0.1)

  // E3: connects x2 (y=-0.4) with y2 (y=-3.6) - right after Hx3/Hy3
  circle((0.8, -0.4), radius: 0.08, fill: black)
  circle((0.8, -3.6), radius: 0.08, fill: black)
  line((0.8, -0.4), (0.8, -3.6))
  content((0.8, -2.0), [$E_3$], anchor: "west", padding: 0.1)

  // E2: connects x1 (y=0.4) with y1 (y=-2.8) - right after Hx2/Hy2
  circle((3.8, 0.4), radius: 0.08, fill: black)
  circle((3.8, -2.8), radius: 0.08, fill: black)
  line((3.8, 0.4), (3.8, -2.8))
  content((3.8, -1.2), [$E_2$], anchor: "west", padding: 0.1)

  // E1: connects x0 (y=1.2) with y0 (y=-2.0) - right after Hx1/Hy1
  circle((7.6, 1.2), radius: 0.08, fill: black)
  circle((7.6, -2.0), radius: 0.08, fill: black)
  line((7.6, 1.2), (7.6, -2.0))
  content((7.6, -0.4), [$E_1$], anchor: "west", padding: 0.1)
}))

The entanglement gates $E_k = "diag"(1, 1, 1, e^(i phi_k))$ are parameterized controlled-phase gates that couple the $k$-th qubit from the x-axis with the $k$-th qubit from the y-axis. For a square $n times n$ qubit system with one-to-one coupling, we add exactly $n$ entanglement gates, one after each Hadamard layer.

The total transformation becomes:
$
  cal(T)_"entangled" = U_"entangle" dot (F_n times.o F_n)
$
where $U_"entangle" = product_(k=1)^n E_k$ is the product of all entanglement gates, and $F_n$ is the $n$-qubit QFT.

Key advantages of this approach:
- Captures diagonal features and cross-dimensional patterns common in natural images
- Maintains $O(n log n)$ computational complexity (same as standard QFT)
- Adds only $O(n)$ additional learnable parameters (one phase per qubit pair)
- Reduces to standard 2D QFT when all entanglement phases $phi_k = 0$

== Learning a better Fourier basis
Observing that in this representation, tensor parameters can be tuned without affecting the computational complexity, e.g. the parameters in $M_k$ and $H$. Can we find a transformation better than the Fourier basis? Or is Fourier basis already optimal for image processing?

Intuitively, the fourier basis is not optimal for image processing, because:
- the fourier basis assumes periodic boundary condition, which is not suitable for image processing.
- the 2d fourier basis assumes the $X$ and $Y$ coordinates are independent, which is not suitable for image processing.

== Tasks
- Create an image dataset $cal(D) = {bold(x)_i}_(i=1)^N$.
- Create a tensor network transformation based on the above QFT circuit, denoted as $cal(T)(bold(theta))$, where $bold(theta)$ is the parameters of the tensor network.
- Variationally optimize the circuit parameters to capture the sparsity of the image. The cost function is
  $
    cal(L)(bold(theta)) = sum_(i=1)^N ||bold(x)_i - cal(T)(bold(theta))^(-1)("truncate"(cal(T)(bold(theta))(bold(x)_i), k))||_2^2
  $
  Here, we can choose a different loss function to capture details in the image, e.g. the edges. For simplicity, we use the $l_1$-norm instead:
  $
    cal(L)(bold(theta)) = sum_(i=1)^N ||cal(T)(bold(theta))(bold(x)_i)||_1
  $
  This loss will encourage the tensor network to output a sparse pattern in the "moment space". It is a standard trick that widely used in _compressed sensing_.
- In the 2D Fourier transformation, the $X$ and $Y$ coordinates are independent. Here we allow $X$ and $Y$ coordinates to correlate with each other in the tensor network basis.
- Add edge detection features.
- Compare the performance with the Fourier basis.
