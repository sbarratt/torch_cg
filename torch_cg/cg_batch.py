import torch
import IPython as ipy
import time


def cg_batch(A, B, M, X0=None, tol=1e-3, maxiter=None, verbose=False):
    K, n, _ = A.shape
    _, _, m = B.shape

    if X0 is None:
        X0 = torch.zeros_like(B)
    if maxiter is None:
        maxiter = 10 * n

    assert A.shape == (K, n, n)
    assert B.shape == (K, n, m)
    assert M.shape == (K, n, n)
    assert X0.shape == (K, n, m)
    assert tol > 0
    assert isinstance(maxiter, int)

    X_k = X0
    R_k = B - A.bmm(X_k)
    Z_k = M.bmm(R_k)

    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    if verbose:
        print("%03s | %010s %06s" % ("it", "res", "it/s"))

    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        start_iter = time.perf_counter()
        Z_k = M.bmm(R_k)

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            beta = (R_k1 * Z_k1).sum(1) / (R_k2 * Z_k2).sum(1)
            P_k = Z_k1 + beta.unsqueeze(1) * P_k1

        alpha = (R_k1 * Z_k1).sum(1) / (P_k * A.bmm(P_k)).sum(1)
        X_k = X_k1 + alpha.unsqueeze(1) * P_k
        R_k = R_k1 - alpha.unsqueeze(1) * A.bmm(P_k)
        end_iter = time.perf_counter()

        residual = torch.norm(A@X_k - B).item()
        if residual < tol:
            break
        if verbose:
            print("%03d | %8.4e %4.2f" %
                  (k, residual, 1. / (end_iter - start_iter)))
    end = time.perf_counter()

    if verbose:
        print("Terminated in %d steps. Took %.3f ms." %
              (k, (end - start) * 1000))

    return X_k

if __name__ == '__main__':
    import networkx as nx
    from scipy import sparse
    import scipy.sparse.linalg as splinalg
    import numpy as np
    import multiprocessing as mp
    import time
    import IPython as ipy

    def sparse_numpy_to_torch(A):
        rows, cols = A.nonzero()
        values = A.data
        indices = np.vstack((rows, cols))
        i = torch.LongTensor(indices)
        v = torch.DoubleTensor(values)
        shape = A.shape
        return torch.sparse.DoubleTensor(i, v, torch.Size(shape))

    n = 25
    m = 100
    K = 128
    As = [nx.laplacian_matrix(
        nx.gnm_random_graph(n, 5 * n)) + .1 * sparse.eye(n) for _ in range(K)]
    Ms = [sparse.diags(1. / A.diagonal()) for A in As]
    Bs = [np.random.randn(n, m) for _ in range(K)]
    A = torch.DoubleTensor(K, n, n)
    M = torch.DoubleTensor(K, n, n)
    B = torch.DoubleTensor(K, n, m)

    for i in range(K):
        A[i] = torch.from_numpy(As[i].todense())
        M[i] = torch.from_numpy(Ms[i].todense())
        B[i] = torch.from_numpy(Bs[i])

    X = cg_batch(A, B, M, verbose=True)
    print(X.shape)
