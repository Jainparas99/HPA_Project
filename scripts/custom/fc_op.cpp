#include <torch/script.h>
#include <ATen/ATen.h> 
#include "fc_kernel.h"

// C++ wrapper that allocates the output tensor and calls your CUDA impl
at::Tensor fc_forward(const at::Tensor& input,
                      const at::Tensor& weight,
                      const at::Tensor& bias) {
  TORCH_CHECK(input.device().is_cuda(),  "input must be CUDA");
  TORCH_CHECK(weight.device().is_cuda(), "weight must be CUDA");
  TORCH_CHECK(bias.device().is_cuda(),   "bias must be CUDA");

  auto N     = input.size(0);
  auto D_in  = input.size(1);
  auto D_out = weight.size(0);

  auto output = at::empty({N, D_out},
                          input.options().dtype(input.dtype()));

  // raw pointers
  const float* in_ptr   = input.data_ptr<float>();
  const float* wt_ptr   = weight.data_ptr<float>();
  const float* b_ptr    = bias.data_ptr<float>();
        float* out_ptr  = output.data_ptr<float>();

  // launch your CUDA kernel
  fc_forward_cuda(in_ptr, wt_ptr, b_ptr, out_ptr,
                  N, D_in, D_out);

  return output;
}

// Register under torch.ops.my_ops.fc
TORCH_LIBRARY(hpc, m) {
  m.def("fc(Tensor input, Tensor weight, Tensor bias) -> Tensor");
  m.impl("fc", TORCH_FN(fc_forward));
}

//C++ implemetation of the gc_layer
