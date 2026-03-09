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

== Entangled Fourier Basis: XY Correlation
In the standard 2D Fourier transform, the $x$ and $y$ coordinates are processed independently. For an image of size $2^n times 2^n$ (i.e., square images with $m = n$), we apply QFT on the $n$ row qubits and separately on the $n$ column qubits. This independence assumption is often suboptimal for natural images where spatial correlations exist between rows and columns.

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

Key advantages of this approach:
- Captures diagonal features and cross-dimensional patterns common in natural images
- Maintains $O(n log n)$ computational complexity in the number $n$ of qubits per spatial dimension (equivalently $O(N log N)$ for linear image size $N = 2^n$), matching the standard QFT
- Adds exactly $n$ additional real-valued learnable parameters $phi_k$ (one phase per qubit pair), i.e., $O(n)$ in $n$
- Reduces to standard 2D QFT when all entanglement phases $phi_k = 0$

== Alternative Basis: Time Evolving Block Decimation (TEBD)
Time Evolving Block Decimation (TEBD) is a tensor network ansatz originally developed for simulating 1D quantum many-body systems. In our implementation, it employs a _ring topology_ of nearest-neighbor controlled-phase gates preceded by Hadamard gates on each qubit. For image processing on a $2^n times 2^n$ grid, the $n$ row qubits and $n$ column qubits each form an independent ring of controlled-phase gates.

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

For Riemannian optimization, we optimize on this product of phase manifolds using the same gradient descent approach as the QFT basis. Direct evaluation of the tensor network takes $O(n)$ operations (one $2 times 2$ gate application per controlled-phase gate).

== Alternative Basis: MERA-inspired Hierarchical Circuit
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

== Learning a better Fourier basis
Observing that in this representation, tensor parameters can be tuned without affecting the computational complexity, e.g. the parameters in $M_k$ and $H$. Can we find a transformation better than the Fourier basis? Or is Fourier basis already optimal for image processing?

Intuitively, the fourier basis is not optimal for image processing, because:
- the fourier basis assumes periodic boundary condition, which is not suitable for image processing.
- the 2d fourier basis assumes the $X$ and $Y$ coordinates are independent, which is not suitable for image processing.

== Loss Functions for Basis Learning

=== L1 Norm Loss (Sparsity Regularization)

The L1 norm loss is defined as:
$cal(L)_(L 1)(bold(theta)) = sum_(i,j) |cal(T)(bold(theta))(bold(x))_(i,j)|$

This promotes *sparsity*: under compressed sensing theory, minimizing $||bold(y)||_1$ approximates minimizing the $ell_0$ pseudo-norm $||bold(y)||_0 = |{i,j : bold(y)_(i,j) != 0}|$. The optimization drives the transform to concentrate energy in fewer frequency components.

*Limitations*: The L1 norm does not directly optimize reconstruction quality. The loss is independent of $||bold(x) - cal(T)^(-1)("truncate"(bold(y), k))||_F^2$, so the sparsest transform may not yield the best reconstruction.

=== L2 Norm Loss

$cal(L)_(L 2)(bold(theta)) = sum_(i,j) |cal(T)(bold(theta))(bold(x))_(i,j)|^2$

Encourages energy concentration with smoother gradients than L1, but less aggressive sparsity promotion.

=== MSE Reconstruction Loss

$cal(L)_"MSE"(bold(theta)) = ||bold(x) - cal(T)(bold(theta))^(-1)("truncate"(cal(T)(bold(theta))(bold(x)), k))||_F^2$

Directly optimizes reconstruction quality after top-$k$ truncation. Requires computing the inverse transform.

=== Hybrid Loss

$cal(L)_"hybrid"(bold(theta)) = alpha cal(L)_(L 1)(bold(theta)) + beta cal(L)_"MSE"(bold(theta))$

Balances sparsity promotion with reconstruction quality.

== Sparse Basis in Parametric QFT

=== Motivation

A *sparse basis* is a transform basis in which a signal can be represented with few non-zero coefficients. For compression, the goal is to find a basis $cal(T)$ such that $bold(y) = cal(T)(bold(x))$ has most energy concentrated in a small number of components: $||bold(y)||_0 << dim(bold(y))$.

Classical transforms (FFT, DCT) provide fixed bases optimized for specific signal classes. The parametric QFT learns a *data-adaptive sparse basis* by optimizing circuit parameters $bold(theta) in M$.

=== QFT as Sparse Basis Transform

The QFT circuit implements a unitary transform $cal(T)(bold(theta)): CC^(2^m times 2^n) -> CC^(2^m times 2^n)$ parameterized by Hadamard gates $H in U(2)$ and controlled phase gates $M_k in U(1)^4$. The standard QFT (fixed parameters) equals the classical DFT. Making parameters learnable on the unitary manifold $M$ searches for a transform with better sparsity for specific inputs.

=== Frequency-Dependent Truncation

For compression, low-frequency components carry structural information while high-frequency components capture fine details. The truncation uses a weighted score:

$s_(i,j) = |bold(y)_(i,j)| dot (1 + w_(i,j))$, where $w_(i,j) = 1 - d_(i,j) / (2 d_"max")$

Here $d_(i,j) = sqrt((i - c_i)^2 + (j - c_j)^2)$ is the frequency distance from the DC component, and $d_"max"$ is the maximum distance. This weighting favors low-frequency components.

=== Sparse Basis Learning Objective

The optimization finds parameters $bold(theta)^* = arg min_(bold(theta) in M) cal(L)(bold(theta))$ where:

+ *L1 Sparsity*: $cal(L)_(L 1) = ||cal(T)(bold(theta))(bold(x))||_1$ directly promotes sparse representations
+ *MSE Reconstruction*: $cal(L)_"MSE" = ||bold(x) - cal(T)^(-1)("topk"(cal(T)(bold(x)), k))||_F^2$ optimizes for reconstruction quality after truncation

The learned basis $cal(T)(bold(theta)^*)$ is signal-adaptive: unlike fixed DCT/FFT bases, it can exploit structure specific to the input data (e.g., textures, edges in images).

=== Comparison with Fixed Bases

#table(
  columns: 3,
  [*Property*], [*Fixed Basis (FFT/DCT)*], [*Parametric QFT*],
  [Adaptivity], [None (fixed transform)], [Data-dependent optimization],
  [Sparsity], [Optimal for periodic/smooth signals], [Learned for specific input],
  [Computation], [$O(N log N)$], [$O(N log N)$ + training cost],
  [Unitarity], [Preserved], [Preserved (manifold constraint)],
)

== Riemannian Optimization on Unitary Manifolds

The trainable parameters of our quantum circuits—Hadamard gates $H$, controlled-phase gates $M_k$, and TEBD gates $T_k$—live on Riemannian manifolds rather than Euclidean space. Standard gradient-based optimizers cannot be applied directly because Euclidean updates would violate the unitarity constraint. We employ _Riemannian optimization_ that respects the manifold geometry.

=== Manifold Structure

Our circuits involve two types of parameter manifolds:

+ *Unitary manifold $U(2)$*: Hadamard-like gates are $2 times 2$ unitary matrices satisfying $U U^dagger = I$. The tangent space at $U$ is
  $
    T_U U(2) = { U S : S^dagger = -S }
  $
  where $S$ is skew-Hermitian.

+ *Product of unit circles $U(1)^4$*: Controlled-phase gates are $2 times 2$ diagonal matrices with entries on the unit circle $|z_i| = 1$. The tangent space at $z$ is
  $
    T_z U(1)^4 = { i theta dot.c z : theta in RR^4 }
  $
  corresponding to purely imaginary scalings of each component.

=== Riemannian Gradient

Given a Euclidean gradient $nabla_E f$ obtained via automatic differentiation, we project it onto the tangent space to obtain the Riemannian gradient:

- *For $U(2)$*: $"grad" f(U) = U dot "skew"(U^dagger nabla_E f)$, where $"skew"(A) = (A - A^dagger) \/ 2$.
- *For $U(1)^4$*: $"grad" f(z) = i dot "Im"(overline(z) circle.stroked.tiny nabla_E f) circle.stroked.tiny z$, applied element-wise.

=== Retraction

After computing a descent direction $xi in T_x cal(M)$, we map back to the manifold using a _retraction_:

- *QR retraction for $U(2)$*: Given $U + alpha xi = Q R$ (QR decomposition), set $R_U (alpha xi) = Q dot "diag"("sgn"(R_(i i)))$. The sign correction ensures we remain on the correct connected component.
- *Normalization for $U(1)^4$*: $R_z (alpha xi) = (z + alpha xi) \/ |z + alpha xi|$, applied element-wise to project back onto the unit circle.

=== Riemannian Gradient Descent

The basic optimization algorithm iterates:
$
  x_(t+1) = R_(x_t) (-eta dot "grad" f(x_t))
$ <eq:rgd>
where $eta > 0$ is the learning rate and $R$ is the retraction.

=== Riemannian Adam

For faster convergence, we implement the Riemannian Adam optimizer (Bécigneul & Ganea, 2019) that extends classical Adam to manifolds. Let $g_t = "grad" f(x_t)$ be the Riemannian gradient at step $t$. The algorithm maintains per-parameter first and second moment estimates:

$
  m_t &= beta_1 m_(t-1) + (1 - beta_1) g_t \
  v_t &= beta_2 v_(t-1) + (1 - beta_2) |g_t|^2
$ <eq:adam_moments>

with bias-corrected estimates $hat(m)_t = m_t \/ (1 - beta_1^t)$ and $hat(v)_t = v_t \/ (1 - beta_2^t)$. The update direction is:

$
  d_t = hat(m)_t \/ (sqrt(hat(v)_t) + epsilon)
$

and the new iterate is obtained by retraction:

$
  x_(t+1) = R_(x_t) (-eta dot d_t)
$

After retraction, the first moment $m_t$ must be _parallel transported_ from the tangent space at $x_t$ to the tangent space at $x_(t+1)$. We use projection-based transport as a first-order approximation:

$
  Gamma_(x_t -> x_(t+1)) (m_t) = "proj"_(T_(x_(t+1)) cal(M)) (m_t)
$

which re-projects the old momentum onto the new tangent space. For $U(2)$ this is $U_(t+1) dot "skew"(U_(t+1)^dagger m_t)$; for $U(1)^4$ this re-applies the tangent projection at the new point.
