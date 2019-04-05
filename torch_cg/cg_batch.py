import torch
import IPython as ipy
import time


def cg_batch(A_bmm, B, M_bmm, X0=None, tol=1e-3, maxiter=None, verbose=False):
    K, n, m = B.shape

    if X0 is None:
        X0 = torch.zeros_like(B)
    if maxiter is None:
        maxiter = 10 * n

    assert B.shape == (K, n, m)
    assert X0.shape == (K, n, m)
    assert tol > 0
    assert isinstance(maxiter, int)

    X_k = X0
    R_k = B - A_bmm(X_k)
    Z_k = M_bmm(R_k)

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
        Z_k = M_bmm(R_k)

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

        alpha = (R_k1 * Z_k1).sum(1) / (P_k * A_bmm(P_k)).sum(1)
        X_k = X_k1 + alpha.unsqueeze(1) * P_k
        R_k = R_k1 - alpha.unsqueeze(1) * A_bmm(P_k)
        end_iter = time.perf_counter()

        residual = torch.norm(A_bmm(X_k) - B).item()
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

    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def sparse_numpy_to_torch(A):
        rows, cols = A.nonzero()
        values = A.data
        indices = np.vstack((rows, cols))
        i = torch.cuda.LongTensor(indices)
        v = torch.cuda.FloatTensor(values)
        return torch.cuda.sparse.FloatTensor(i, v, A.shape)

    n = 5000
    m = 1
    K = 128
    As = [nx.laplacian_matrix(
        nx.gnm_random_graph(n, 10 * n)) + .1 * sparse.eye(n) for _ in range(K)]
    Ms = [sparse.diags(1. / A.diagonal(), format='csc') for A in As]
    A_bdiag = sparse.block_diag(As)
    M_bdiag = sparse.block_diag(Ms)
    Bs = [np.random.randn(n, m) for _ in range(K)]
    As_torch = [None] * K
    Ms_torch = [None] * K
    B_torch = torch.cuda.FloatTensor(K, n, m)
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

    print(f"Solving K={K} linear systems  that are {n} x {n} with {As[0].nnz} nonzeros and {m} right hand sides.")
    X = cg_batch(A_bmm, B_torch, M_bmm, maxiter=10, verbose=True)
    X = cg_batch(A_bmm_2, B_torch, M_bmm_2, maxiter=10, verbose=True)
