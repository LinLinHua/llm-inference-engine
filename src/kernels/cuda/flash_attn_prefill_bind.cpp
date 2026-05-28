/*
src/kernels/cuda/flash_attn_prefill_bind.cpp

Python binding for FlashAttention-2 prefill kernel.
Separated from .cu file per official PyTorch docs:
  torch/extension.h should NOT be parsed by nvcc.
Reference: https://pytorch.org/docs/stable/cpp_extension.html
*/

#include <torch/extension.h>

// Forward declaration (defined in flash_attn_prefill.cu)
at::Tensor fa2_prefill(
    at::Tensor Q,
    at::Tensor K,
    at::Tensor V,
    bool causal
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fa2_prefill", &fa2_prefill,
          "FlashAttention-2 prefill kernel (CUDA, Tensor Core, GQA)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("causal") = true);
}