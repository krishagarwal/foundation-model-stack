// #include <stdio.h>
// #include <stdint.h>
#include <torch/extension.h>
// #include <ATen/ATen.h>

// int fast_had_trans(const torch::Tensor &a, uint32_t had_size)
// {
//     // helloCUDA<<<1, 1>>>(a.data_ptr<Float8>()
//     // cudaDeviceSynchronize();
//     // return 0;
//     printf("hello yeet");
//     return 0;
// }

// PYBIND11_MODULE(example, m) {
//     m.doc() = "pybind11 yeet";

//     m.def("fast_had_trans", &fast_had_trans, "test func", py::arg("a"), py::arg("had_size"));
// }

#include <pybind11/pybind11.h>

int add(int i, int j, const torch::Tensor &a) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}