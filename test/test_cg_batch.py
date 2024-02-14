import pytest
import torch
import numpy as np

from torch_cg.cg_batch import cg_batch, CG


def A_test(x: torch.Tensor) -> torch.Tensor:
    y = 3 * x
    return y


class Test_cg_batch:
    def test_0(self) -> None:
        """Make sure things run and return the correct shape"""

        B = torch.tensor([[[1.0, 0, 0]]])
        K, n, m = B.shape

        out, out_info = cg_batch(
            A_test,
            B,
            M_bmm=None,
            X0=None,
            rtol=1e-5,
            atol=1e-5,
            maxiter=10,
            verbose=False,
        )
        assert out.shape == (K, n, m)
        assert np.logical_not(np.any(np.isnan(out.numpy())))


if __name__ == "__main__":
    pytest.main()
