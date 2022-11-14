#include <torch/extension.h>
#include <stdexcept>
#include <cmath>
#include <vector>


torch::Tensor scatter_triu_indices(int size, int step_size){
    if (step_size < 1){
            throw std::invalid_argument( "step_size needs to be greater or equal 1" );
    }
    if (step_size > size){
            throw std::invalid_argument( "step_size needs to be less or equal size" );
    }
    int p = (size / step_size) + (size % step_size != 0);
    int full_size = ((p * p) - p) / 2 + p;
    auto options = torch::TensorOptions().dtype(torch::kInt32).requires_grad(false);
    std::vector<int> m;
    std::vector<int> l;
    l.reserve(full_size);
    m.reserve(full_size);
    for (int i = 0; i < size; i = i + step_size){
        for (int j = i; j < size; j = j + step_size){
            m.push_back(j);
            l.push_back(i);
        }
    }
    torch::Tensor fm = torch::from_blob(m.data(), {full_size}, options);
    torch::Tensor fl = torch::from_blob(l.data(), {full_size}, options);
    torch::Tensor full_t = torch::stack({fl, fm}, {1});
    return full_t;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_scatter_triu_indices", &scatter_triu_indices, "Creates scattered triu indices");
}