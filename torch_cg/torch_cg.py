import torch
from cg_cpp import cg_cpp


def cg(A, B, M, X0=None, tol=1e-3, maxiter=None, atol=float("inf")):
    n = len(A)
    X0 = X0 if X0 is not None else torch.mm(M, B)
    B_norm = B.norm(2).item()
    if maxiter is None:
        maxiter = 10 * n
    return cg_cpp(A, B, M, X0, min(tol * B_norm, atol), maxiter)
