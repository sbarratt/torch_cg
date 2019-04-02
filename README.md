# torch_cg

A simple implementation of the preconditioned conjugate gradient (CG)
algorithm in Pytorch, on the CPU.
(Extending the code to the GPU is trivial.)
CG computes the matrix X such that AX=B, where A is a positive definite torch.sparse.FloatTensor and B is a matrix.
In addition, it is differentiable, meaning we can differentiate the output X with respect to A and B.
Under the hood, the CG iterations are implemented in C++, making it ~2x faster than a pure Python implementation,
due to the iterative nature of the algorithm.

Installation:
```
$ python setup.py install
```

Run tests:
```
$ cd torch_cg
$ python test.py
```

Usage:
```
from torch_cg import CG

# create A (torch.sparse.FloatTensor) and B (torch.FloatTensor) that require_grad
# create preconditioner M (torch.sparse.FloatTensor); a good default is M_{ii} = 1 / A_{ii}

# solve AX=B using preconditioner M
X = CG(rtol=1e-5, atol=1e-5)(A, B, M)

# take derivative of sum(X) with respect to A and B
X.sum().backward()
```
