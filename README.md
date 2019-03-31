# torch_cg

A simple implementation of the preconditioned conjugate gradient
algorithm in Pytorch.
Under the hood, it is implemented in C++.

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
# create preconditioner M (torch.sparse.FloatTensor) 

# solve AX=B using preconditioner M
X = CG(rtol=1e-5, atol=1e-5)(A, B, M)

# take derivative of sum(X) with respect to A and B
X.sum().backward()
```
