import networkx as nx
from scipy import sparse
import scipy.sparse.linalg as splinalg
import numpy as np
import torch
from cg_batch import CG
import IPython as ipy
import time

torch.set_default_tensor_type(torch.DoubleTensor)


def sparse_numpy_to_torch(A):
    rows, cols = A.nonzero()
    values = A.data
    indices = np.vstack((rows, cols))
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    return torch.sparse.DoubleTensor(i, v, A.shape)

n = 500
m = 12
K = 32
As = [nx.laplacian_matrix(
    nx.gnm_random_graph(n, 20 * n)) + .1 * sparse.eye(n) for _ in range(K)]
Ms = [sparse.diags(1. / A.diagonal(), format='csc') for A in As]
A_bdiag = sparse.block_diag(As)
M_bdiag = sparse.block_diag(Ms)
Bs = [np.random.randn(n, m) for _ in range(K)]
As_torch = [None] * K
Ms_torch = [None] * K
B_torch = torch.DoubleTensor(K, n, m).requires_grad_()
A_bdiag_torch = sparse_numpy_to_torch(A_bdiag)
M_bdiag_torch = sparse_numpy_to_torch(M_bdiag)

for i in range(K):
    As_torch[i] = sparse_numpy_to_torch(As[i])
    Ms_torch[i] = sparse_numpy_to_torch(Ms[i])
    B_torch[i] = torch.tensor(Bs[i])


def A_bmm(X):
    Y = [(As_torch[i]@X[i]).unsqueeze(0) for i in range(K)]
    return torch.cat(Y, dim=0)


def M_bmm(X):
    Y = [(Ms_torch[i]@X[i]).unsqueeze(0) for i in range(K)]
    return torch.cat(Y, dim=0)


def A_bmm_2(X):
    Y = A_bdiag_torch@(X.view(K * n, m))
    return Y.view(K, n, m)


def M_bmm_2(X):
    Y = M_bdiag_torch@(X.view(K * n, m))
    return Y.view(K, n, m)

print(f"Solving K={K} linear systems that are {n} x {n} with {As[0].nnz} nonzeros and {m} right hand sides.")

cg = CG(A_bmm, M_bmm=M_bmm, rtol=1e-5, atol=1e-5, verbose=True)
X = cg(B_torch)

start = time.perf_counter()
X_np = np.concatenate([np.hstack([splinalg.cg(A, B[:, i], M=M)[0][:, np.newaxis] for i in range(m)])[np.newaxis, :, :]
                       for A, B, M in zip(As, Bs, Ms)], 0)
end = time.perf_counter()
print("Scipy took %.3f seconds" % (end - start))
np.testing.assert_allclose(X_np, X.cpu().data.numpy(), atol=1e-4)
