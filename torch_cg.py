import torch
import IPython as ipy
import copy

def cg(A, B, M, X0=None, tol=1e-3, maxiter=None, atol=0.):
    n = len(B)
    if maxiter is None:
        maxiter = 10*n
    X = X0 if X0 is not None else torch.mm(M, B)

    norm_B = torch.norm(B)
    residual = lambda X: B - torch.mm(A, X)
    done = lambda X: torch.norm(residual(X)) <= max(tol*norm_B, atol)

    k = 0
    X_k = X
    R_k = residual(X_k)
    optimal = True
    while not done(X_k):
        Z_k = torch.mm(M, R_k)
        k = k + 1
        if k == maxiter:
            optimal = False
            break
        elif k == 1:
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
            beta = (R_k1*Z_k1).sum(dim=0) / (R_k2*Z_k2).sum(dim=0)
            P_k = Z_k1 + beta*P_k1

        alpha = (R_k1*Z_k1).sum(dim=0) / (P_k*torch.mm(A,P_k)).sum(dim=0)
        X_k = X_k1 + alpha * P_k
        R_k = R_k1 - alpha * torch.mm(A, P_k)
    return X_k, {
        "optimal": optimal,
        "|R|": torch.norm(residual(X_k)).item(),
        "niter": k
    }

def sparse_numpy_to_torch(A):
    A = A.tocoo()
    values = A.data
    indices = np.vstack((A.row,A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape
    return torch.sparse.FloatTensor(i,v,torch.Size(shape))

if __name__ == '__main__':
    import networkx as nx
    from scipy import sparse
    from scipy.sparse.linalg import cg as cg_np
    import numpy as np

    n = 10000
    Anp = nx.laplacian_matrix(nx.gnm_random_graph(n,15*n))+.1*sparse.eye(n)
    Mnp = sparse.diags(1./Anp.diagonal())
    bnp = np.random.randn(n, 50)
    A = sparse_numpy_to_torch(Anp)
    M = sparse_numpy_to_torch(Mnp)
    B = torch.from_numpy(bnp).float()

    X, info = cg(A, B, M, tol=1e-5)
    print ((torch.norm(torch.mm(A, X) - B)**2).item())

    res_sq = 0.
    for i in range(50):
        b = bnp[:,i]
        x = cg_np(Anp, b, M=Mnp)[0]
        res_sq += np.linalg.norm(Anp@x-b)**2
    print (res_sq)
