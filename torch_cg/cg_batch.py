import torch
import time


def cg_batch(A_bmm, B, M_bmm=None, X0=None, tol=1e-3, stop='mean', maxiter=None, verbose=False):
    """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.

    This function solves a batch of matrix linear systems of the form

        A_i X_i = B_i,  i=1,...,K,

    where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
    and X_i is the n x m matrix representing the solution for the ith system.

    Args:
        A_bmm: A callable that performs a batch matrix multiply of A and a K x n x m matrix.
        B: A K x n x m matrix representing the right hand sides.
        M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a K x n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        tol: (optional) Tolerance for norm of residual. (default=1e-3)
        stop: (optional) Form of stopping condition.
            'max' for worst case tolerance across all batches and right hand sides,
            or 'mean' for average tolerance. (default='mean')
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
    K, n, m = B.shape

    if M_bmm is None:
        M_bmm = lambda x: x
    if X0 is None:
        X0 = M_bmm(B)
    if maxiter is None:
        maxiter = 5 * n

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

        if stop == 'mean':
            residual = torch.norm(A_bmm(X_k) - B).item() / (K * m)
        else:
            return NotImplemented
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


class CG(torch.autograd.Function):

    def __init__(self, A_bmm, M_bmm=None, tol=1e-3, stop='mean', maxiter=None, verbose=False):
        self.A_bmm = A_bmm
        self.M_bmm = M_bmm
        self.tol = tol
        self.stop = stop
        self.maxiter = maxiter
        self.verbose = verbose

    def forward(self, B, X0=None):
        X = cg_batch(self.A_bmm, B, M_bmm=self.M_bmm, X0=X0, stop=self.stop, tol=self.tol,
                     maxiter=self.maxiter, verbose=self.verbose)
        return X

    def backward(self, dX):
        dB = cg_batch(self.A_bmm, dX, M_bmm=self.M_bmm, X0=X0, stop=self.stop, tol=self.tol,
                      maxiter=self.maxiter, verbose=self.verbose)
        return dB
