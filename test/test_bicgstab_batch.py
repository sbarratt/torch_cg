import pytest
import torch
import numpy as np

from torch_cg.bicgstab_batch import bicgstab_batch


def A_test_id(x: torch.Tensor) -> torch.Tensor:
    return x


def A_test_two(x: torch.Tensor) -> torch.Tensor:
    """A looks like
    [1 0 0,
    1 1 0,
    0 0 1]"""
    K, n, m = x.shape

    A = torch.ones((K, n, n))
    y = x
    for i in range(K):
        for j in range(m):
            y[i, :, j] = y[i, 0, j] + x[i, 0, j]

    return y


class Test_bicgstab_batch:
    def test_0(self) -> None:
        """Make sure things run and return the correct shape A = Id"""
        torch.manual_seed(0)
        K = 1
        n = 3
        m = 1
        B = torch.randn(K, n, m)
        print(B.shape)

        out, out_info = bicgstab_batch(
            A_test_id,
            B,
            X0=None,
            rtol=1e-5,
            atol=1e-5,
            maxiter=3,
            verbose=False,
        )
        assert out.shape == (K, n, m)
        assert np.logical_not(np.any(np.isnan(out.numpy())))

    def test_1(self) -> None:
        """Make sure things run and return the correct shape A = ones"""
        torch.manual_seed(0)

        K = 1
        n = 3
        m = 1
        B = torch.randn(K, n, m)
        print(B.shape)

        out, out_info = bicgstab_batch(
            A_test_two,
            B,
            X0=None,
            rtol=1e-5,
            atol=1e-5,
            maxiter=3,
            verbose=False,
        )
        assert out.shape == (K, n, m)
        assert np.logical_not(np.any(np.isnan(out.numpy())))

    def test_2(self) -> None:
        """Make sure things approximately are correct given A = Id"""
        torch.manual_seed(0)

        K = 5
        n = 3
        m = 4
        B = torch.randn(K, n, m)

        out, out_info = bicgstab_batch(
            A_test_id,
            B,
            X0=None,
            rtol=1e-5,
            atol=1e-5,
            maxiter=3,
            verbose=False,
        )
        assert out.shape == (K, n, m)
        assert np.allclose(out.numpy(), B.numpy())


if __name__ == "__main__":
    pytest.main()
