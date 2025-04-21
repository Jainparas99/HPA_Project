#include "common.h"
#include <sstream>
#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <cstdlib> 
#include <ctime>  

// Definition of the helper
std::string shapeToString(const c10::IntArrayRef& sizes) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < sizes.size(); ++i) {
        oss << sizes[i];
        if (i + 1 < sizes.size())
            oss << ", ";
    }
    oss << "]";
    return oss.str();
}
torch::Tensor make_random_image_batch_tensor(
    int batch_size,
    int channels,
    int height,
    int width,
    const std::vector<float>& mean,
    const std::vector<float>& stddev,
    torch::Device device) {

    // std::cout << "Creating random image batch tensor with shape: "
    //           << "[" << batch_size << ", " << channels << ", "
    //           << height << ", " << width << "]\n";
    // 1) Seed RNG once
    static bool seeded = false;
    if (!seeded) {
      std::srand(static_cast<unsigned>(std::time(nullptr)));
      seeded = true;
    }
    // 2) Generate & normalize raw host data
    int total = batch_size * channels * height * width;
    std::vector<float> raw(total);
    std::generate(raw.begin(), raw.end(),
                  []() { return static_cast<float>(std::rand()) / RAND_MAX; });
    

    int image_size = channels * height * width;
    for (int b = 0; b < batch_size; ++b) {
      float* base = raw.data() + b * image_size;
      for (int c = 0; c < channels; ++c) {
        float m = mean[c], s = stddev[c];
        float* ptr = base + c * (height * width);
        for (int i = 0; i < height * width; ++i) {
          ptr[i] = (ptr[i] - m) / s;
        }
      }
    }
    std::cout << 4;

    // 3) Wrap as a CPU tensor (from_blob must match host pointer)
    auto cpu_opts = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(torch::kCPU);
    std::cout << 5;
    
    torch::Tensor t_cpu = torch::from_blob(
                             raw.data(),
                             {batch_size, channels, height, width},
                             cpu_opts)
                           .clone();   // clone so tensor owns its data
    std::cout << 6;
    // 4) Transfer to CUDA if requested
    if (device.is_cuda()) {
      // non_blocking can be true if you later do pinned_memory allocations
      t_cpu = t_cpu.to(device, /*non_blocking=*/false);
    }

    return t_cpu;
}