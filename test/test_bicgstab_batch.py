import pytest
import torch
import numpy as np
from scipy.stats import special_ortho_group
from scipy.sparse.linalg import LinearOperator, bicgstab

from torch_cg.bicgstab_batch import bicgstab_batch


def A_test_id(x: torch.Tensor) -> torch.Tensor:
    return x


def A_test_two(x: torch.Tensor) -> torch.Tensor:
    """A looks like
    [1 0 0,
    1 1 0,
    0 0 1]"""
    K, n, m = x.shape

    A = torch.eye(n)
    A[1, 0] = 1.0

    y = torch.zeros_like(x)
    for i in range(K):
        for j in range(m):
            y[i, :, j] = A @ x[i, :, j]

    return y


class Test_bicgstab_batch:
    def test_0(self) -> None:
        """Make sure things run and return the correct shape A = Id"""
        torch.manual_seed(0)
        K = 1
        n = 2
        m = 1
        B = torch.tensor(
            [
                [
                    [1.0],
                    [0.0],
                ]
            ]
        )
        assert B.shape == (K, n, m)
        print(B.shape)

        out, out_info = bicgstab_batch(
            A_test_id,
            B,
            X0=None,
            rtol=1e-5,
            atol=1e-5,
            maxiter=2,
            verbose=False,
        )
        assert out.shape == (K, n, m)
        assert np.logical_not(np.any(np.isnan(out.numpy())))

    def test_1(self) -> None:
        """Make sure things approximately are correct given A = Id"""
        torch.manual_seed(0)

        K = 1
        n = 20
        m = 1
        B = torch.randn(K, n, m)

        out, out_info = bicgstab_batch(
            A_test_id,
            B,
            X0=None,
            rtol=1e-5,
            atol=1e-5,
            maxiter=10,
            verbose=False,
        )
        print(f"Output: ", out)
        print(f"Expected: ", B)
        assert out.shape == (K, n, m)
        assert np.allclose(out.numpy(), B.numpy())

    def test_2(self) -> None:
        """Make sure things run and return the correct shape A != Id and
        K = 5, n = 2, m = 3"""
        torch.manual_seed(0)

        K = 5
        n = 2
        m = 3
        B = torch.zeros(K, n, m)
        for i in range(K):
            for j in range(m):
                B[i, 0, j] = 1.0
        print(B.shape)

        out, out_info = bicgstab_batch(
            A_test_two,
            B,
            X0=None,
            rtol=1e-5,
            atol=1e-5,
            maxiter=n + 1,
            verbose=False,
        )
        assert out.shape == (K, n, m)
        assert np.logical_not(np.any(np.isnan(out.numpy())))

    def test_3(self) -> None:
        """Tests convergence of a large Real system with a random A matrix. A is
        designed to be clearly PD, with eigenvalues ~ Unif[0,1].
        Also, uses a real vector for B"""
        torch.manual_seed(0)
        np.random.seed(0)

        K = 1
        n = 100
        m = 1
        B = torch.randn(K, n, m)
        B = B.to(torch.float64)

        R = special_ortho_group.rvs(n)
        D = np.diag(np.random.uniform(size=n))
        A = R @ D @ R.T
        A_inv = R @ np.diag(1 / np.diag(D)) @ R.T
        A = torch.from_numpy(A)
        A_inv = torch.from_numpy(A_inv)
        AA_inv = A @ A_inv
        print(A)
        print(A_inv)
        print(AA_inv)

        # Because of floating point errors, I have to increase the atol and rtol
        assert np.allclose(AA_inv.numpy(), np.eye(n), atol=1e-5, rtol=1e-5)

        def _A(x: torch.Tensor) -> torch.Tensor:
            y = torch.zeros_like(x)
            for i in range(K):
                for j in range(m):
                    y[i, :, j] = A @ x[i, :, j]
            return y

        expected_out = A_inv @ B
        assert expected_out.shape == (K, n, m)

        out, out_info = bicgstab_batch(
            _A,
            B,
            X0=None,
            rtol=1e-5,
            atol=1e-5,
            maxiter=n + 1,
            verbose=False,
        )

        out_resid = B - _A(out)

        print("Exited after ", out_info["niter"], " iterations")

        # print(f"Output: ", out)
        # print(f"Expected: ", expected_out)
        assert out.shape == (K, n, m)
        assert np.allclose(
            out_resid.numpy(), np.zeros_like(out_resid.numpy()), atol=1e-04, rtol=1e-04
        )

    def test_4(self) -> None:
        """Tests the function with a complex matrix that is clearly invertible.
        We generate the A matrix by drawing a random rotation matrix from SO(n),
        drawing random complex eigenvalues, and then constructing A = R @ D @ R^T.
        Also, uses a complex vector for B.
        """
        torch.manual_seed(0)
        np.random.seed(0)

        K = 1
        n = 100
        m = 1
        B = torch.randn(K, n, m) + 1j * torch.randn(K, n, m)
        B = B.to(torch.complex128)

        R = special_ortho_group.rvs(n)
        D = np.diag(np.random.uniform(size=n) + 1j * np.random.uniform(size=n))
        A = R @ D @ R.T
        A_inv = R @ np.diag(1 / np.diag(D)) @ R.T
        A = torch.from_numpy(A)
        A_inv = torch.from_numpy(A_inv)
        AA_inv = A @ A_inv
        print(A)
        print(A_inv)
        print(AA_inv)

        # Because of floating point errors, I have to increase the atol and rtol
        assert np.allclose(AA_inv.numpy(), np.eye(n), atol=1e-5, rtol=1e-5)

        def _A(x: torch.Tensor) -> torch.Tensor:
            y = torch.zeros_like(x)
            for i in range(K):
                for j in range(m):
                    y[i, :, j] = A @ x[i, :, j]
            return y

        expected_out = A_inv @ B
        assert expected_out.shape == (K, n, m)

        out, out_info = bicgstab_batch(
            _A,
            B,
            X0=None,
            rtol=1e-5,
            atol=1e-5,
            maxiter=n + 1,
            verbose=False,
        )

        print("Exited after ", out_info["niter"], " iterations")

        out_resid = (B - _A(out)).numpy()

        print(f"Output: ", out)
        print(f"Expected: ", expected_out)
        assert out.shape == (K, n, m)
        assert np.allclose(out_resid, np.zeros_like(out_resid), atol=5e-05, rtol=5e-05)

    def test_5(self) -> None:
        """Tests backpropogation."""

        torch.manual_seed(0)
        np.random.seed(0)

        K = 1
        n = 3
        m = 1

        q = torch.tensor([[1.0], [1.0], [1.0]], requires_grad=True)
        B = q * torch.randn(K, n, m)
        assert B.shape == (K, n, m)

        A = torch.diag(q.flatten())
        assert A.shape == (n, n)

        def _A(x: torch.Tensor) -> torch.Tensor:
            y = torch.zeros_like(x)
            for i in range(K):
                for j in range(m):
                    y[i, :, j] = A @ x[i, :, j]
            return y

        # zero out the grads for q, B, A
        q.grad = None
        B.grad = None
        A.grad = None

        out, out_info = bicgstab_batch(
            _A,
            B,
            X0=None,
            rtol=1e-5,
            atol=1e-5,
            maxiter=10,
            verbose=False,
        )
        resid_norm = torch.linalg.norm(B - _A(out))

        print("Residual norm: ", resid_norm.norm, resid_norm.shape)

        resid_norm.backward()

        assert q.grad is not None


if __name__ == "__main__":
    pytest.main()
