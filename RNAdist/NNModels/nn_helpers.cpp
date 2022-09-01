#include <torch/extension.h>
#include <stdexcept>

torch::Tensor scatter_triu_indices(int size, int step_size){
    if (step_size < 1){
            throw std::invalid_argument( "step_size needs to be greater or equal 1" );
    }
    if (step_size > size){
            throw std::invalid_argument( "step_size needs to be less or equal size" );
    }
    int m = (size / step_size) + (size % step_size != 0);
    int full_size = ((m * m) - m) / 2 + m;
    auto options = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);
    torch::Tensor full_t = torch::empty({full_size, 2}, options);
    int the_idx = 0;
    for (int i = 0; i < size; i = i + step_size){
        torch::Tensor range = torch::arange(i, size, step_size, options);
        int cur_size = range.sizes()[0];
        torch::Tensor indices = torch::arange(the_idx, the_idx + cur_size, options);
        full_t.index_put_({indices, 0}, range);
        full_t.index_put_({indices, 1}, i);
        the_idx = the_idx + cur_size;
    }
    return full_t;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_scatter_triu_indices", &scatter_triu_indices, "Creates scattered triu indices");
}