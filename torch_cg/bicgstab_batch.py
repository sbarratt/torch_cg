from typing import Callable, Tuple, Dict, Any
import torch


def bicgstab_batch(
    A_bmm: Callable,
    B: torch.Tensor,
    M_bmm: torch.Tensor = None,
    X0: torch.Tensor = None,
    rtol: float = 1e-03,
    atol: float = 0.0,
    maxiter: int = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Solves a batch of PD linear systems using the BiCGSTAB algorithm.
    This function solves a batch of linear systems of the form
        A X_i = B_i,  i=1,...,K,
    where A is a n x n positive definite matrix and B_i is a n x m matrix,
    and X_i is the n x m matrix representing the solution for the ith system.

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
    if M_bmm is None:
        M_bmm = lambda x: x
    if X0 is None:
        X0 = M_bmm(B)
    if maxiter is None:
        maxiter = 5 * n

    assert B.shape == (K, n, m)
    assert X0.shape == (K, n, m)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    # Initialize the variables for the BiCGSTAB algorithm.
    raise NotImplementedError
