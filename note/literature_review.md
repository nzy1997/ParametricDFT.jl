# Literature Review: Learning Optimized Sparse Representations via Parametric Tensor Networks Inspired by the Quantum Fourier Transform

## Abstract

This literature review examines the theoretical foundations and recent advances relevant to learning optimized sparse representations for images using differentiable tensor networks inspired by the Quantum Fourier Transform (QFT) circuit structure. We focus on four key areas: (1) the tensor network representation of FFT/QFT, (2) the theory of sparse representations and compressed sensing, (3) limitations of Fourier bases for images and alternative transforms, and (4) learnable structured transforms and dictionary learning. This review provides the theoretical context for developing parametric transforms that can outperform fixed Fourier bases for specific image domains.

---

## 1. Introduction

The Discrete Fourier Transform (DFT) has been a cornerstone of signal and image processing for decades. However, the standard Fourier basis has fundamental limitations for natural images:

1. **Periodic boundary conditions**: The DFT assumes signals wrap around, creating artificial discontinuities at image boundaries
2. **Separability assumption**: 2D DFT treats X and Y coordinates independently, ignoring directional correlations
3. **Fixed basis**: Cannot adapt to the specific statistics of a given image domain

These limitations motivate the development of **learnable transforms** that maintain the computational efficiency of FFT ($O(N \log N)$) while adapting to capture image-specific structure. The key insight is that the FFT can be represented as a **tensor network** with parameterizable gates, enabling end-to-end optimization of the transform basis.

---

## 2. Tensor Network Representation of FFT and QFT

### 2.1 The Cooley-Tukey FFT as Matrix Factorization

The Fast Fourier Transform [@cooley1965algorithm] achieves $O(N \log N)$ complexity by factorizing the DFT matrix $F_N$ into a product of sparse matrices:

$$F_N = \begin{pmatrix} I_{N/2} & D_{N/2} \\ I_{N/2} & -D_{N/2} \end{pmatrix} \begin{pmatrix} F_{N/2} & 0 \\ 0 & F_{N/2} \end{pmatrix} P_N$$

where $D_{N/2} = \text{diag}(1, \omega, \omega^2, \ldots, \omega^{N/2-1})$ contains the **twiddle factors** and $P_N$ is a permutation matrix. This recursive structure corresponds to a **butterfly network** of $\log_2 N$ layers.

### 2.2 Quantum Fourier Transform Circuit

The Quantum Fourier Transform [@nielsen2010quantum; @coppersmith2002approximate] provides an alternative perspective. The QFT on $n$ qubits (where $N = 2^n$) is implemented as a circuit with:

- **Hadamard gates**: $H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$
- **Controlled phase gates**: $R_k = \begin{pmatrix} 1 & 0 \\ 0 & e^{2\pi i/2^k} \end{pmatrix}$

The circuit has depth $O(n^2)$ with gates, but the diagonal structure of controlled-phase gates allows merging to achieve $O(N \log N)$ classical complexity.

**Key insight**: Both the Hadamard matrices $H$ and phase matrices $M_k = \begin{pmatrix} 1 & 1 \\ 1 & e^{i\pi/2^{k-1}} \end{pmatrix}$ can be treated as **learnable parameters** without changing the computational complexity.

### 2.3 Tensor Network Formulation

Representing a vector of size $N = 2^k$ as a tensor with $k$ indices allows the FFT to be expressed as a **tensor network** [@orus2014practical; @bridgeman2017hand]. Each layer of the butterfly network corresponds to a tensor contraction, and the entire transform is a **Matrix Product Operator (MPO)**.

**Relevant works**:
- Orus (2014) provides a comprehensive introduction to tensor networks [@orus2014practical]
- Bridgeman and Chubb (2017) offer an accessible tutorial on tensor network methods [@bridgeman2017hand]

---

## 3. Sparse Representations and Compressed Sensing

### 3.1 The Sparsity Hypothesis

Natural signals, including images, are believed to have **sparse representations** in appropriate bases. This means that while a signal may require $N$ samples to represent directly, it can be described by only $k \ll N$ significant coefficients in a suitable transform domain.

**The foundational work** by Olshausen and Field (1996) [@olshausen1996emergence] showed that sparse coding of natural images produces basis functions resembling V1 simple cell receptive fields, providing biological evidence for the sparsity principle.

### 3.2 Compressed Sensing Theory

Compressed sensing [@candes2006robust; @donoho2006compressed] provides the mathematical framework for recovering sparse signals from incomplete measurements. The key theoretical results include:

**Theorem (Candès, Romberg, Tao 2006)**: A signal $x \in \mathbb{R}^N$ with $k$-sparse representation in some basis $\Psi$ can be exactly recovered from $m = O(k \log N)$ random measurements via $\ell_1$ minimization:

$$\min_x \|\Psi x\|_1 \quad \text{subject to} \quad Ax = b$$

This justifies the use of **L1 loss** for learning sparse representations:

$$\mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^N \|\mathcal{T}(\boldsymbol{\theta})(\mathbf{x}_i)\|_1$$

where $\mathcal{T}(\boldsymbol{\theta})$ is the parametric transform.

### 3.3 Restricted Isometry Property (RIP)

The **Restricted Isometry Property** [@candes2008restricted] provides conditions for successful sparse recovery. A matrix $A$ satisfies RIP of order $k$ with constant $\delta_k$ if:

$$(1 - \delta_k)\|x\|_2^2 \leq \|Ax\|_2^2 \leq (1 + \delta_k)\|x\|_2^2$$

for all $k$-sparse vectors $x$. This property is relevant for understanding when learned bases will enable good reconstruction.

---

## 4. Limitations of Fourier Bases for Images

### 4.1 Boundary Conditions and Gibbs Phenomenon

The DFT assumes **periodic boundary conditions**, treating the signal as if it wraps around. For natural images, this creates:

- Artificial discontinuities at boundaries
- **Gibbs phenomenon**: oscillatory artifacts near edges
- Reduced sparsity due to edge effects

The **Discrete Cosine Transform (DCT)** [@ahmed1974discrete] partially addresses this by using symmetric extension, which is why DCT is preferred in JPEG compression. However, DCT still assumes separability.

### 4.2 Separability and Directional Information

The 2D DFT/DCT treats X and Y coordinates independently:

$$\hat{f}(u,v) = \sum_{x}\sum_{y} f(x,y) e^{-2\pi i(ux/N + vy/M)}$$

This **separability assumption** fails to capture:
- Oriented edges and textures
- Curvilinear structures
- Non-axis-aligned features

**Alternative transforms** that address directionality include:
- **Wavelets** [@mallat1989theory]: Multi-resolution analysis with localized basis functions
- **Curvelets** [@candes2004new]: Optimal sparse representation of edges
- **Shearlets** [@kutyniok2012shearlets]: Directional representation with optimal sparsity
- **Contourlets** [@do2005contourlet]: Discrete directional multiresolution transform

### 4.3 Data-Adaptive vs. Fixed Bases

Fixed transforms cannot adapt to the specific statistics of an image domain. For example:
- Medical images have different structure than natural photographs
- Texture patterns vary across materials
- Scientific imaging has domain-specific features

This motivates **learning the transform** from data, which is the core goal of the parametric tensor network approach.

---

## 5. Dictionary Learning and Learned Transforms

### 5.1 Classical Dictionary Learning

**Dictionary learning** [@elad2010sparse] seeks to find an overcomplete basis $D$ such that signals $x$ can be represented sparsely:

$$x \approx D\alpha, \quad \|\alpha\|_0 \leq k$$

Key algorithms include:
- **K-SVD** [@aharon2006ksvd]: Iteratively updates dictionary atoms
- **Online Dictionary Learning** [@mairal2009online]: Scalable stochastic optimization
- **Convolutional Sparse Coding** [@zeiler2010deconvolutional]: Shift-invariant dictionaries

**Limitation**: Classical dictionary learning produces **unstructured** dictionaries with $O(N^2)$ computational cost, losing the efficiency of FFT.

### 5.2 Structured Learnable Transforms

Recent work explores **structured transforms** that maintain computational efficiency while being learnable:

**Butterfly Matrices** [@dao2019learning; @dao2020kaleidoscope]: 
- Parameterize FFT-like butterfly networks
- Learn in $O(N \log N)$ operations
- Achieve compression while preserving expressiveness

**Kaleidoscope Matrices** [@dao2020kaleidoscope]:
- Generalize butterfly factorizations
- Can express Hadamard, Fourier, and other structured transforms
- Enable learning of efficient linear transforms

**FNet** [@lee2021fnet]:
- Replaces Transformer self-attention with fixed Fourier transforms
- Demonstrates FFT as an efficient mixing mechanism
- Opens possibilities for learnable Fourier-like layers

### 5.3 Tensor Network Approaches

Tensor networks provide a natural framework for structured, learnable transforms:

**Matrix Product States for Classification** [@stoudenmire2016supervised]:
- MPS achieve <1% error on MNIST with far fewer parameters
- Demonstrate tensor networks can capture image structure
- Provide interpretable, compressed representations

**Projected Entangled Pair States (PEPS)** [@cheng2020supervised]:
- 2D tensor networks matching image topology
- Outperform tree-like tensor networks on image tasks
- Capture spatial correlations naturally

**Tensor LISTA** [@zhao2021tensor]:
- Differentiable sparse coding for tensors
- Achieves linear convergence with theoretical guarantees
- Applicable to multi-dimensional image data

**Tensor-Based Dictionary Learning** [@soltani2015tensor]:
- Tensor factorization for image reconstruction
- Compact representation of repeated features
- Demonstrated on tomographic imaging

---

## 6. Quantum-Inspired Machine Learning

### 6.1 Tensor Networks from Quantum Physics

Tensor networks originated in quantum many-body physics for representing entangled states efficiently [@white1992density; @verstraete2008matrix]. Key structures include:

- **MPS (Matrix Product States)**: 1D chains, efficient for sequential data
- **PEPS (Projected Entangled Pair States)**: 2D grids, natural for images
- **MERA (Multiscale Entanglement Renormalization Ansatz)**: Hierarchical, captures multi-scale features

### 6.2 Quantum Circuits as Tensor Networks

Every quantum circuit can be represented as a tensor network, and vice versa. The QFT circuit structure specifically provides:

- **Logarithmic depth**: $O(\log N)$ layers
- **Controlled entanglement**: Phase gates couple different frequency components
- **Invertibility**: Unitary structure ensures perfect reconstruction

### 6.3 Hybrid Quantum-Classical Approaches

**Variational Quantum Circuits** [@cerezo2021variational] optimize parameterized quantum circuits for machine learning tasks. Classical simulation of these circuits using tensor networks enables:

- Training on classical hardware
- Insights from quantum algorithm design
- Novel architectures inspired by quantum computing

---

## 7. Relevance to the Parametric DFT Project

The project described in `main.typ` synthesizes these ideas:

### 7.1 Core Innovation

The key insight is that the **QFT tensor network structure** can be parameterized:
- Replace fixed Hadamard $H$ with learnable $2 \times 2$ matrices
- Replace fixed phase gates $M_k$ with learnable phase parameters
- Maintain $O(N \log N)$ computational complexity

### 7.2 Advantages Over Standard Approaches

| Approach | Complexity | Adaptive | Boundary | Directionality |
|----------|------------|----------|----------|----------------|
| DFT/FFT | $O(N \log N)$ | ✗ | Periodic | ✗ (separable) |
| DCT | $O(N \log N)$ | ✗ | Symmetric | ✗ (separable) |
| Wavelets | $O(N)$ | ✗ | Various | Limited |
| Dictionary | $O(N^2)$ | ✓ | Any | ✓ |
| **Parametric TN** | $O(N \log N)$ | ✓ | Learnable | ✓ (via coupling) |

### 7.3 Key Design Choices

1. **L1 Loss for Sparsity**: Following compressed sensing theory, the L1 norm encourages sparse representations
2. **X-Y Coupling**: Tensor network structure allows controlled gates between X and Y dimensions, breaking separability
3. **End-to-End Differentiability**: Enables gradient-based optimization of all parameters
4. **Invertibility**: Maintains reconstruction capability through unitary/orthogonal constraints

### 7.4 Connections to Prior Work

The parametric DFT approach relates to:
- **Butterfly matrices** [@dao2019learning]: Similar factorization structure
- **Sparse coding** [@olshausen1996emergence]: Same sparsity objective
- **Tensor networks for ML** [@stoudenmire2016supervised]: Shared computational framework
- **Compressed sensing** [@candes2006robust]: Theoretical foundation for L1 optimization

---

## 8. Open Questions and Future Directions

### 8.1 Theoretical Questions

1. **Convergence guarantees**: Under what conditions does L1 optimization converge to optimal sparse bases?
2. **Generalization**: How does the learned transform generalize across image domains?
3. **RIP preservation**: Do learned transforms maintain restricted isometry properties?

### 8.2 Computational Challenges

1. **Optimization landscape**: Are there local minima that trap gradient descent?
2. **Initialization**: How to initialize parameters for stable training?
3. **Scalability**: Extending to large images and 3D data

### 8.3 Extensions

1. **Edge detection**: Incorporating edge-aware loss functions
2. **Multi-scale**: Hierarchical tensor networks (MERA-like)
3. **Domain-specific**: Medical imaging, scientific visualization, satellite imagery

---

## 9. Conclusion

The parametric tensor network approach to image representation learning offers a principled way to combine:
- The **computational efficiency** of FFT ($O(N \log N)$)
- The **adaptivity** of dictionary learning
- The **theoretical foundations** of compressed sensing
- The **structural insights** from quantum computing

By parameterizing the QFT circuit structure and optimizing for sparsity, this approach can potentially discover image-specific bases that outperform fixed transforms like Fourier and DCT. The rich theoretical connections to tensor networks, sparse coding, and quantum computing provide multiple avenues for further development.

---

## References

See `references.bib` for full bibliography.

### Key References by Topic

**FFT and QFT**:
- Cooley & Tukey (1965) [@cooley1965algorithm]
- Nielsen & Chuang (2010) [@nielsen2010quantum]

**Compressed Sensing**:
- Candès, Romberg, Tao (2006) [@candes2006robust]
- Donoho (2006) [@donoho2006compressed]

**Sparse Coding**:
- Olshausen & Field (1996) [@olshausen1996emergence]
- Elad (2010) [@elad2010sparse]

**Tensor Networks**:
- Orús (2014) [@orus2014practical]
- Stoudenmire & Schwab (2016) [@stoudenmire2016supervised]

**Learnable Transforms**:
- Dao et al. (2019, 2020) [@dao2019learning; @dao2020kaleidoscope]
- Lee-Thorp et al. (2021) [@lee2021fnet]
