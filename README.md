# torch_cg

A simple implementation of the preconditioned conjugate gradient (CG)
algorithm in Pytorch.
The algorithm is implemented as a function with the signature:
```
def cg_batch(A_bmm, B, M_bmm=None, X0=None, tol=1e-3, stop='mean', maxiter=None, verbose=False):
    """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.

    This function solves a batch of matrix linear systems of the form

        A_i X_i = B_i,  i=1,...,K,

    where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
    and X_i is the n x m matrix representing the solution for the ith system.

    Args:
        A_bmm: A callable that performs a batch matrix multiply of A and a K x n x m matrix.
        B: A K x n x m matrix representing the right hand sides.
        M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a K x n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        tol: (optional) Tolerance for norm of residual. (default=1e-3)
        stop: (optional) Form of stopping condition.
            'max' for worst case tolerance across all batches and right hand sides,
            or 'mean' for average tolerance. (default='mean')
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
```
There is also a pytorch Function/layer called CG that is differentiable.

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
X = CG(B)(A_bmm, M_bmm)

# take derivative of sum(X) with respect to A and B
X.sum().backward()
```
