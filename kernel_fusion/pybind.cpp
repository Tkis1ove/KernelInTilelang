#include <pybind11/pybind11.h>
#include <torch/extension.h>

torch::Tensor fusion(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor rel1,
    torch::Tensor rel2,
    torch::Tensor mask
);

PYBIND11_MODULE(fusion_kernel, m) {
    m.def("fusion", &fusion, "Fused attention kernel",
         pybind11::arg("q"),
         pybind11::arg("k"),
         pybind11::arg("v"),
         pybind11::arg("rel1"),
         pybind11::arg("rel2"),
         pybind11::arg("mask"));
}