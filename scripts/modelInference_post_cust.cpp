#include <dlfcn.h>
#include <torch/script.h>
#include <iostream>

int main(int argc, char** argv) {
  // 1) Load the custom‐op library
  void* h = dlopen("libhpc.so", RTLD_NOW | RTLD_GLOBAL);
  if (!h) {
    std::cerr << "dlopen(libhpc.so) failed: " << dlerror() << "\n";
    return -1;
  }

  // 2) Load the scripted model
  torch::jit::Module model;
  try {
    model = torch::jit::load("resnet50_custom_fc.pt");
  } catch (const c10::Error& e) {
    std::cerr << "Error loading model: " << e.what() << "\n";
    return -1;
  }

  model.to(at::kCUDA);
  model.eval();

  // 3) Run a dummy batch
  at::Tensor input = torch::randn({1,3,224,224}, torch::kCUDA);
  at::Tensor out   = model.forward({input}).toTensor();

  std::cout << "Output shape: " << out.sizes() << "\n";
  return 0;
}
