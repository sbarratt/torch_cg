# torch_cg

A simple implementation of the preconditioned conjugate gradient (CG)
algorithm in Pytorch.
The algorithm is implemented as a function with the signature:
```
def cg_batch(A_bmm, B, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False)
```
Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.

This function solves a batch of matrix linear systems of the form

    A_i X_i = B_i,  i=1,...,K,

where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
and X_i is the n x m matrix representing the solution for the ith system.
There is also a pytorch Function/layer called ```CG``` that is differentiable.

Installation:
```
$ python setup.py install
```

Run tests (requires some extra packages):
```
$ cd torch_cg
$ python test.py
```

Usage:
```
from torch_cg import CG

# create A_bmm, B (requires grad), M_bmm

# solve AX=B using preconditioner M
X = CG(A_bmm, M_bmm)(B)

# take derivative of sum(X) with respect to B
X.sum().backward()
```
