import torch
from torch_cg import cg
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import cg as cg_np
import numpy as np
import time


def sparse_numpy_to_torch(A):
    A = A.tocoo()
    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

if __name__ == '__main__':
    cuda = torch.cuda.is_available()

    n = 100000
    m = 1000
    Anp = nx.laplacian_matrix(
        nx.gnm_random_graph(n, 5 * n)) + .1 * sparse.eye(n)
    Mnp = sparse.diags(1. / Anp.diagonal())
    bnp = np.random.randn(n, m)
    A = sparse_numpy_to_torch(Anp)
    M = sparse_numpy_to_torch(Mnp)
    B = torch.from_numpy(bnp).float()
    if cuda:
        A = A.cuda()
        M = M.cuda()
        B = B.cuda()

    start = time.time()
    X = cg(A, B, M, tol=1e-5)
    end = time.time()
    print("Pytorch batch:", (end - start) * 1000, "ms")

    start = time.time()
    X_np = np.array([cg_np(Anp, bnp[:, i], M=Mnp, tol=1e-5)[0]
                     for i in range(m)]).T
    end = time.time()
    print("np:", (end - start) * 1000, "ms")
    assert np.allclose(X.cpu().data.numpy(), X_np, atol=1e-2)
