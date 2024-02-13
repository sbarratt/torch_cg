from typing import Callable, Tuple, Dict, Any
import torch
import time


def bicgstab_batch(
    A_bmm: Callable,
    B: torch.Tensor,
    # M_bmm: torch.Tensor = None,
    X0: torch.Tensor = None,
    rtol: float = 1e-03,
    atol: float = 0.0,
    maxiter: int = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Solves a batch of PD linear systems using the BiCGSTAB algorithm.
    This function solves a batch of linear systems of the form
        A_i X_i = B_i,  i=1,...,K,
    where A is a n x n positive definite matrix and B_i is a n x m matrix,
    and X_i is the n x m matrix representing the solution for the ith system.

    So this is batched in 2 different ways: it can be batched for m different RHS vectors for a given A matrix,
    or it can be batched for K different A matrices.

    Args:
        A_bmm (Callable): A callable that performs a batch matrix multiply of A and a (K x n x m) matrix.
        B (torch.Tensor): A (K x n x m) matrix representing the right hand sides.
        M_bmm (torch.Tensor, optional): A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a (K x n x m) matrix. Defaults to None.
        X0 (torch.Tensor, optional): Initial guess for X, defaults to M_bmm(B). Defaults to None.
        rtol (float, optional): Relative tolerance for norm of residual. Defaults to 1e-03.
        atol (float, optional): Absolute tolerance for norm of residual. Defaults to 0.0.
        maxiter (int, optional): Maximum number of iterations to perform. Defaults to None.
        verbose (bool, optional): Whether or not to print status messages. Defaults to False.

    Returns:
        Tuple[torch.Tensor, Dict[str, Any]]: _description_
    """

    # Get shape information and assert the input shapes are correct.
    K, n, m = B.shape
    # if M_bmm is None:
    #     M_bmm = lambda x: x
    if X0 is None:
        # X0 = M_bmm(B)
        X0 = torch.zeros_like(B)
    if maxiter is None:
        maxiter = 5 * n

    assert B.shape == (K, n, m)
    assert X0.shape == (K, n, m)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    print("B: ", B, B.shape)

    # Initialize the variables for the BiCGSTAB algorithm. I am using the variable names as given in
    # Matrix Computations by Golub and Van Loan.

    X_k = X0
    R_k = B - A_bmm(X_k)
    R_tilde = R_k.clone()
    P_k = R_k.clone()

    inner_prods = _inner_prod(R_tilde, R_k)
    assert (
        inner_prods != torch.zeros_like(inner_prods)
    ).any(), "Initializing R_tilde failed. May have initialized at the optimum."

    B_norm = torch.norm(B, dim=1)
    stopping_matrix = torch.max(rtol * B_norm, atol * torch.ones_like(B_norm))
    print("Stopping matrix: ", stopping_matrix.shape)

    if verbose:
        print("%03s | %010s %06s" % ("it", "dist", "it/s"))
    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        start_iter = time.perf_counter()

        # Update the variables for the BiCGSTAB algorithm
        Ap_k = A_bmm(P_k)
        print("Ap_k: ", Ap_k, Ap_k.shape)

        # Our objects have shape (K, n, m), so inner products are pointwise multiplication, then
        # summing over the n dimension.

        mu = (_inner_prod(R_tilde, R_k) / _inner_prod(R_tilde, Ap_k)).unsqueeze(1)
        print("Mu: ", mu, mu.shape)
        S_k = R_k - mu * Ap_k

        # Test whether S_k is close enough to zero for an early exit
        S_norm = torch.norm(S_k, dim=1)
        if (S_norm <= stopping_matrix).all():
            optimal = True
            X_k = X_k + mu * P_k
            break

        As_k = A_bmm(S_k)

        omega = _inner_prod(S_k, As_k) / _inner_prod(As_k, As_k)
        print("Omega: ", omega)
        X_k = X_k + mu * P_k + omega * S_k
        R_k_new = S_k - omega * As_k
        tau = mu * _inner_prod(R_tilde, R_k_new) / (omega * _inner_prod(R_tilde, R_k))
        print("Tau: ", tau)
        P_k = R_k_new + tau * (P_k - omega * Ap_k)
        R_k = R_k_new
        end_iter = time.perf_counter()
        print("P_k: ", P_k, P_k.shape)
        print("R_k: ", R_k, R_k.shape)
        print("X_k: ", X_k, X_k.shape)
        print("End of iteration")
        # Calculate the stopping criterion and check if we are done.
        residual_norm = torch.norm(A_bmm(X_k) - B, dim=1)

        if verbose:
            print(
                "%03d | %8.4e %4.2f"
                % (
                    k,
                    torch.max(residual_norm - stopping_matrix),
                    1.0 / (end_iter - start_iter),
                )
            )

        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break

    end = time.perf_counter()
    if verbose:
        if optimal:
            print(
                "Terminated in %d steps (reached maxiter). Took %.3f ms."
                % (k, (end - start) * 1000)
            )
        else:
            print(
                "Terminated in %d steps (optimal). Took %.3f ms."
                % (k, (end - start) * 1000)
            )

    info = {"niter": k, "optimal": optimal}

    return X_k, info


def _inner_prod(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Assumes x and y are of shape (K, n, m) and returns the inner product of x and y
    in the n dimension. The result is of shape (K, m)."""
    return torch.sum(x.conj() * y, dim=1)
