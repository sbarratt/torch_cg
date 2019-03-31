import torch
from torch_cg import CG
import networkx as nx
from scipy import sparse
import scipy.sparse.linalg as splinalg
import numpy as np
import multiprocessing as mp
import time
import IPython as ipy


def sparse_numpy_to_torch(A):
    A = A.tocoo()
    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

if __name__ == '__main__':
    cuda = False #torch.cuda.is_available()
    
    torch.set_default_dtype(torch.double)

    n = 1000
    m = 1
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
    X = CG(rtol=1e-5, atol=1e-5)(A, B, M)
    end = time.time()
    print("Pytorch batch:", (end - start) * 1000, "ms")
    
    p = mp.Pool(mp.cpu_count())
    start = time.time()
    X_np = np.array([splinalg.cg(Anp, B[:,i], M=Mnp, tol=1e-5, atol=1e-5)[0] for i in range(m)]).T
    end = time.time()
    print("np:", (end - start) * 1000, "ms")
    np.testing.assert_allclose(X.cpu().data.numpy(), X_np, atol=1e-5, rtol=1e-5)

    A.requires_grad_()
    B.requires_grad_()

    X = CG(rtol=1e-5)(A, B, M)
    G = torch.randn_like(X)
    s = (X * G).sum()
    s.backward()
    dA = torch.sparse.FloatTensor(A.grad._indices(),
        1e-5 * torch.randn_like(A.grad._values()), A.grad.shape)
    ds = (dA._values() * A.grad._values()).sum()
    X = CG(rtol=1e-5)(A + dA, B, M)
    sp = (X * G).sum()
    print(ds.item(), (s - sp).item())
