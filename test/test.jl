using Yao
using FFTW

A(i, j) = control(i, j=>shift(2π/(1<<(i-j+1))))
B(n, k) = chain(n, j==k ? put(k=>H) : A(j, k) for j in k:n)
qft(n) = chain(B(n, k) for k in 1:n)
qc = qft(4)

tn = yao2einsum(qc)

