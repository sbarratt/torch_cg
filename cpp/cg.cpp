#include <torch/extension.h>

at::Tensor cg_cpp(
    at::Tensor A,
    at::Tensor B,
    at::Tensor M,
    at::Tensor X0,
    float tol,
    int maxiter) {

  int k = 0;
  auto X_k = X0;
  auto R_k = B - A.mm(X_k);
  auto Z_k = M.mm(R_k);
  at::Tensor P_k, R_k2, Z_k2, P_k1, R_k1, Z_k1, X_k1;

  while(((R_k.norm(2)).gt(tol)).is_nonzero()) {
    Z_k = M.mm(R_k);
    k = k + 1;
    if (k == maxiter) {
      break;
    }
    else if (k == 1) {
      P_k = Z_k;
      R_k1 = R_k; X_k1 = X_k; Z_k1 = Z_k;
    }
    else {
      R_k2 = R_k1; Z_k2 = Z_k1;
      P_k1 = P_k; R_k1 = R_k; Z_k1 = Z_k; X_k1 = X_k;
      auto beta = (R_k1 * Z_k1).sum(0) / (R_k2 * Z_k2).sum(0);
      P_k = Z_k1 + beta * P_k1;
    }

    auto alpha = (R_k1 * Z_k1).sum(0) / (P_k * A.mm(P_k)).sum(0);
    X_k = X_k1 + alpha * P_k;
    R_k = R_k1 - alpha * A.mm(P_k);
  }

  return X_k;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cg_cpp", &cg_cpp, "CG batch");
}
