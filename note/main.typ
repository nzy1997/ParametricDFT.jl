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

== Entangled Fourier Basis: XY Correlation
In the standard 2D Fourier transform, the $x$ and $y$ coordinates are processed independently. For an image of size $2^n times 2^n$ (i.e., square images with $m = n$), we apply QFT on the $n$ row qubits and separately on the $n$ column qubits. This independence assumption is often suboptimal for natural images where spatial correlations exist between rows and columns.

We propose an _entangled QFT basis_ that introduces controlled-phase gates between x and y qubits after each layer of the QFT circuit. For the square case $m = n$, we use a _one-to-one_ entanglement structure where each x qubit $x_k$ is coupled with the corresponding y qubit $y_k$. The entanglement gate $E_k$ has exactly the same form as the $M$ gate defined in the previous section:
$
  E_k = mat(1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 1, 0; 0, 0, 0, e^(i phi_k))
$
acting on qubits $(x_(n-k), y_(n-k))$, where $phi_k$ is a learnable phase parameter. Similar to the $M_k$ gate which multiplies a phase factor when both control and target qubits are in state $|1 chevron.r$, the entanglement gate $E_k$ multiplies a phase $e^(i phi_k)$ when both $x_(n-k) = 1$ and $y_(n-k) = 1$. The key difference is that while $M_k$ uses the fixed phase $pi\/2^(k-1)$ determined by the QFT structure, $E_k$ uses a learnable phase $phi_k$ that can be optimized to capture image-specific correlations.

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
  let n = 8
  // x_n block (row qubits x_0, x_1, x_2, x_3 from top to bottom)
  ngate((-3, 0), 4, "x_n", text:[$bold(x)$], gap-y: 0.8, width: 0.7)

  // x_0 qubit line (topmost, y=1.2)
  ngate((7.0, 1.2), 1, "Hx0", text:[$H$], gap-y: 0.8, width: 0.5)
  line("x_n.o0", "Hx0.i0")
  line("Hx0.o0", (9.4, 1.2))
  cphase(6.2, 1.2, 0.4, 1, name: "Mx01")
  cphase(5.4, 1.2, -0.4, 2, name: "Mx02")
  cphase(4.6, 1.2, -1.2, 3, name: "Mx03")

  // x_1 qubit line (y=0.4)
  ngate((3.2, 0.4), 1, "Hx1", text:[$H$], gap-y: 0.8, width: 0.5)
  line("Hx1.o0", "Mx01ctrl2")
  line("Mx01ctrl2", (9.4, 0.4))
  line("x_n.o1", "Hx1.i0")
  cphase(2.4, 0.4, -0.4, 1, name: "Mx11")
  cphase(1.6, 0.4, -1.2, 2, name: "Mx12")

  // x_2 qubit line (y=-0.4)
  ngate((0.2, -0.4), 1, "Hx2", text:[$H$], gap-y: 0.8, width: 0.5)
  line("Hx2.o0", "Mx11ctrl2")
  line("Mx11ctrl2", "Mx02ctrl2")
  line("Mx02ctrl2", (9.4, -0.4))
  line("x_n.o2", "Hx2.i0")
  cphase(-0.6, -0.4, -1.2, 1, name: "Mx21")

  // x_3 qubit line (bottommost row qubit, y=-1.2)
  ngate((-2.0, -1.2), 1, "Hx3", text:[$H$], gap-y: 0.8, width: 0.5)
  line("Hx3.o0", "Mx21ctrl2")
  line("Mx21ctrl2", "Mx12ctrl2")
  line("Mx12ctrl2", "Mx03ctrl2")
  line("Mx03ctrl2", (9.4, -1.2))
  line("x_n.o3", "Hx3.i0")

  // y_n block (column qubits y_0, y_1, y_2, y_3 from top to bottom)
  ngate((-3, -3.2), 4, "y_n", text:[$bold(y)$], gap-y: 0.8, width: 0.7)

  // y_0 qubit line (topmost column qubit, y=-2.0)
  ngate((7.0, -2.0), 1, "Hy0", text:[$H$], gap-y: 0.8, width: 0.5)
  line("y_n.o0", "Hy0.i0")
  line("Hy0.o0", (9.4, -2.0))
  cphase(6.2, -2.0, -2.8, 1, name: "My01")
  cphase(5.4, -2.0, -3.6, 2, name: "My02")
  cphase(4.6, -2.0, -4.4, 3, name: "My03")

  // y_1 qubit line (y=-2.8)
  ngate((3.2, -2.8), 1, "Hy1", text:[$H$], gap-y: 0.8, width: 0.5)
  line("Hy1.o0", "My01ctrl2")
  line("My01ctrl2", (9.4, -2.8))
  line("y_n.o1", "Hy1.i0")
  cphase(2.4, -2.8, -3.6, 1, name: "My11")
  cphase(1.6, -2.8, -4.4, 2, name: "My12")

  // y_2 qubit line (y=-3.6)
  ngate((0.2, -3.6), 1, "Hy2", text:[$H$], gap-y: 0.8, width: 0.5)
  line("Hy2.o0", "My11ctrl2")
  line("My11ctrl2", "My02ctrl2")
  line("My02ctrl2", (9.4, -3.6))
  line("y_n.o2", "Hy2.i0")
  cphase(-0.6, -3.6, -4.4, 1, name: "My21")

  // y_3 qubit line (bottommost column qubit, y=-4.4)
  ngate((-2.0, -4.4), 1, "Hy3", text:[$H$], gap-y: 0.8, width: 0.5)
  line("Hy3.o0", "My21ctrl2")
  line("My21ctrl2", "My12ctrl2")
  line("My12ctrl2", "My03ctrl2")
  line("My03ctrl2", (9.4, -4.4))
  line("y_n.o3", "Hy3.i0")

  // Entanglement gates E_k connecting x_{n-k} with y_{n-k}
  // E_4: connects x_3 (y=-1.2) with y_3 (y=-4.4) - after H on x_3, y_3
  egate(-1.4, -1.2, -4.4, 4, name: "E4")

  // E_3: connects x_2 (y=-0.4) with y_2 (y=-3.6) - after H on x_2, y_2
  egate(0.8, -0.4, -3.6, 3, name: "E3")

  // E_2: connects x_1 (y=0.4) with y_1 (y=-2.8) - after H on x_1, y_1
  egate(3.8, 0.4, -2.8, 2, name: "E2")

  // E_1: connects x_0 (y=1.2) with y_0 (y=-2.0) - after H on x_0, y_0
  egate(8.2, 1.2, -2.0, 1, name: "E1")
}))

Summarizing the gate applications for each qubit:
- *Row qubit $x_k$*: Hadamard gate $H$, followed by controlled-phase gates $M_j$ with qubits $x_0, ..., x_(k-1)$ (standard QFT structure), then entanglement gate $E_(n-k)$ with column qubit $y_k$
- *Column qubit $y_k$*: Hadamard gate $H$, followed by controlled-phase gates $M_j$ with qubits $y_0, ..., y_(k-1)$ (standard QFT structure), then entanglement gate $E_(n-k)$ with row qubit $x_k$

For a square $2^n times 2^n$ image encoded with $n$ row qubits and $n$ column qubits (total $2n$ qubits), we add exactly $n$ entanglement gates ${E_1, E_2, ..., E_n}$, one for each pair of corresponding row/column qubits.

The total transformation becomes:
$
  cal(T)_"entangled" = U_"entangle" dot (F_n times.o F_n)
$
where $U_"entangle" = product_(k=1)^n E_k$ is the product of all entanglement gates acting on qubit pairs $(x_(n-k), y_(n-k))$, and $F_n$ is the $n$-qubit QFT applied along each spatial dimension.

Key advantages of this approach:
- Captures diagonal features and cross-dimensional patterns common in natural images
- Maintains $O(n log n)$ computational complexity in the number $n$ of qubits per spatial dimension (equivalently $O(N log N)$ for linear image size $N = 2^n$), matching the standard QFT
- Adds exactly $n$ additional real-valued learnable parameters $phi_k$ (one phase per qubit pair), i.e., $O(n)$ in $n$
- Reduces to standard 2D QFT when all entanglement phases $phi_k = 0$

== Alternative Basis: Time Evolving Block Decimation (TEBD)
Time Evolving Block Decimation (TEBD) is a tensor network ansatz originally developed for simulating 1D quantum many-body systems. It employs a _brickwork_ pattern of nearest-neighbor two-qubit gates applied in alternating layers.

#figure(canvas({
  import draw: *
  let n = 4
  let dy = 0.8
  // Input tensor
  ngate((-1.5, -(n - 1) * dy / 2), n, "x_n", text:[$bold(x)$], gap-y: dy, width: 0.7)
  // Draw qubit lines from input to output
  for i in range(n){
    line("x_n.o" + str(i), (7.5, -i * dy), stroke: black)
    content((7.9, -i * dy), [$q_#i$])
  }
  // Layer 1: even pairs (q0-q1, q2-q3)
  ngate((0.6, -0.5 * dy), 2, "U1", text:[$U_(z z)$], gap-y: dy, width: 1)
  ngate((0.6, -2.5 * dy), 2, "U3", text:[$U_(z z)$], gap-y: dy, width: 1)
  // Layer 2: odd pairs (q1-q2)
  ngate((2.0, -1.5 * dy), 2, "U2", text:[$U_(z z)$], gap-y: dy, width: 1)
  // Layer 3: Unitary matrix for the trotter decomposition
  ngate((3.4,  0), 2, "U4", text:[$U_(Z)$], gap-y: dy/4, width: 0.7)
  ngate((3.4,  -dy), 2, "U5", text:[$U_(Z)$], gap-y: dy/4, width: 0.7)
  ngate((3.4,  -2*dy), 2, "U6", text:[$U_(Z)$], gap-y: dy/4, width: 0.7)
  ngate((3.4,  -3*dy), 2, "U7", text:[$U_(Z)$], gap-y: dy/4, width: 0.7)
  // Layer 4: Unitary matrix for the trotter decomposition
  ngate((4.8,  0), 2, "U4", text:[$U_(X)$], gap-y: dy/4, width: 0.7)
  ngate((4.8,  -dy), 2, "U5", text:[$U_(X)$], gap-y: dy/4, width: 0.7)
  ngate((4.8,  -2*dy), 2, "U6", text:[$U_(X)$], gap-y: dy/4, width: 0.7)
  ngate((4.8,  -3*dy), 2, "U7", text:[$U_(X)$], gap-y: dy/4, width: 0.7)

}), caption: [TEBD matrix block for Trotter approximation.])

The Trotter-Suzuki decomposition approximates time evolution as:
$
  e^(-i hat(H) eta) approx limits(product)_(i=1)^N e^(-i h_X X_i eta) limits(product)_(i=1)^(N-1) e^(-i J Z_i Z_(i+1) eta) limits(product)_(i=1)^N e^(-i h_Z Z_i eta)
$

*Notation:* $times.o$ = tensor product (independent subsystems); $product$ = matrix multiplication (composition).

=== Exact $U_(Z Z)$ Decomposition of $C P$ Gates

The QFT controlled-phase gate $C P_k = "diag"(1, 1, 1, e^(i phi_k))$ with $phi_k = pi \/ 2^(k-1)$ decomposes exactly as:
$
  C P_k = e^(i phi_k \/ 4) (R_Z^((i))(phi_k \/ 2) times.o R_Z^((j))(phi_k \/ 2)) U_(Z Z)(-phi_k \/ 4)
$ <eq:cp-zz-decomposition>
where $R_Z (theta) = e^(-i theta Z \/ 2)$ and $U_(Z Z) (theta) = e^(-i theta Z_i times.o Z_j)$. The $R_Z times.o R_Z$ is a tensor product; the full expression is their composition with $U_(Z Z)$.

=== Connection to the Diagonal Matrix $D_n$

The twiddle factor matrix has *tensor product* form:
$
  D_n = "diag"(1, omega^(n\/2)) times.o "diag"(1, omega^(n\/4)) times.o dots times.o "diag"(1, omega)
$
In the FFT, $D_n$ appears controlled:
$
  mat(I_(n\/2), 0; 0, D_(n\/2)) = limits(product)_(k) "ctrl"_0 ("diag"(1, omega^(n\/2^(k+1)))_k)
$
Each $"ctrl"_0$ operation is a $C P$ gate, decomposable via @eq:cp-zz-decomposition:
$
  "ctrl"_0 ("diag"(1, e^(i phi))_j) = e^(i phi \/ 4) (R_Z^((0))(phi\/2) times.o R_Z^((j))(phi\/2)) U_(Z Z)^((0,j))(-phi\/4)
$

This gives an *exact* TEBD-style implementation of the QFT using $U_(Z Z)$ gates and $R_Z$ rotations:

#figure(canvas({
  import draw: *
  let n = 4
  let dy = 0.8
  
  // Helper function for U_ZZ gate (two-qubit interaction)
  let uzzgate(x, y1, y2, name) = {
    circle((x, y1), radius: 0.05, fill: black, stroke: none, name: name + "c1")
    circle((x, y2), radius: 0.05, fill: black, stroke: none, name: name + "c2")
    rect((x - 0.3, (y1+y2)/2 - 0.2), (x + 0.3, (y1+y2)/2 + 0.2), fill: white, stroke: black)
    content((x, (y1+y2)/2), text(7pt)[$U_(Z Z)$])
    line((x, y1), (x, (y1+y2)/2 + 0.2), stroke: black)
    line((x, y2), (x, (y1+y2)/2 - 0.2), stroke: black)
  }
  
  // Helper function for R_Z gate (single-qubit)
  let rzgate(x, y, name) = {
    rect((x - 0.35, y - 0.2), (x + 0.35, y + 0.2), fill: white, stroke: black)
    content((x, y), text(7pt)[$R_Z$])
  }
  
  // Input tensor x (same style as reference)
  ngate((0, 0), n, "x_n", text:[$bold(x)$], gap-y: dy, width: 0.7)
  
  // Recursive F_{n/2} block for q1, q2, q3
  ngate((1.5, -dy/2), n - 1, "F_n", text:[$F_(n\/2)$], gap-y: dy, width: 1)
  
  // Connect x to F_{n/2} for lower qubits (q1, q2, q3)
  for i in range(n - 1){
    line("x_n.o" + str(i+1), "F_n.i" + str(i))
  }
  
  // Draw output lines from F_{n/2}
  for i in range(n - 1){
    line("F_n.o" + str(i), (rel: (6.5, 0)))
  }
  
  // q0 line: from x through all gates to the end
  line("x_n.o0", (rel: (9.0, 0)))
  
  // Output labels
  let x_end = 9.5
  content((x_end, 1.2), [$q_0$])
  content((x_end, 0.4), [$q_1$])
  content((x_end, -0.4), [$q_2$])
  content((x_end, -1.2), [$q_3$])
  
  // U_ZZ gate between q0 and q3 (first, closest to F_{n/2})
  uzzgate(2.8, 1.2, -1.2, "uzz03")
  rzgate(3.5, 1.2, "rz03a")
  rzgate(3.5, -1.2, "rz03b")
  
  // U_ZZ gate between q0 and q2
  uzzgate(4.5, 1.2, -0.4, "uzz02")
  rzgate(5.2, 1.2, "rz02a")
  rzgate(5.2, -0.4, "rz02b")
  
  // U_ZZ gate between q0 and q1
  uzzgate(6.2, 1.2, 0.4, "uzz01")
  rzgate(7.2, 1.2, "rz01a")
  rzgate(7.2, 0.4, "rz01b")
  
  // Hadamard on q0 at the very end
  rect((8.0, 1.2 - 0.2), (8.6, 1.2 + 0.2), fill: white, stroke: black)
  content((8.3, 1.2), text(8pt)[$H$])
  
}), caption: [TEBD-style decomposition of one QFT recursion level. Input $bold(x)$ feeds into $F_(n\/2)$ for qubits $q_1, q_2, q_3$. Qubit $q_0$ passes through $U_(Z Z)$ gates (with paired $R_Z$ rotations) connecting to each lower qubit, followed by a final Hadamard gate $H$.])

The remained $F_(n/2)$ could also be decomposed recursively, which is not drawn here explicitly.

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
