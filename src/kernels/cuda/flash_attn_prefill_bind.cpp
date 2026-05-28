#include <torch/extension.h>

// forward declaration
torch::Tensor fa2_prefill(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fa2_prefill", &fa2_prefill,
          "FlashAttention-2 prefill kernel (CUDA, Tensor Core, GQA)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("causal") = true);
}