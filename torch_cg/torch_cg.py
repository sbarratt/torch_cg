import torch
from torch.autograd import Function
import IPython as ipy
from cg_cpp import cg_cpp


class CG(Function):

    def __init__(self, rtol=1e-5, verbose=False, maxiter=None,
                 atol=1e-5):
        self.rtol = rtol
        self.verbose = verbose
        self.maxiter = maxiter
        self.atol = atol

    def forward(self, A, B, M, X0=None):
        n = len(A)
        maxiter = 10 * n if self.maxiter is None else self.maxiter
        with torch.no_grad():
            X0 = X0 if X0 is not None else torch.mm(M, B)
            B_norm = B.norm(2).item()
            tol = max(self.rtol * B_norm, self.atol)
        X = cg_cpp(A, B, M, X0, tol, maxiter)

        self.save_for_backward(A, B, M, X)
        return X

    def backward(self, dX):
        n = len(dX)
        A, B, M, X = self.saved_tensors
        maxiter = 10 * n if self.maxiter is None else self.maxiter
        with torch.no_grad():
            C0 = torch.mm(M, dX)
            tol = min(self.rtol * torch.norm(dX, 2), self.atol)
        C = cg_cpp(A, dX, M, C0, tol, maxiter)
        indices = A._indices()
        shape = A.shape
        rows = indices[0, :]
        cols = indices[1, :]

        values = (C[rows, :] * X[cols, :]).sum(dim=1)
        dA = torch.sparse.FloatTensor(indices, values, shape)
        return dA, C, None, None
