#include <iostream>
#include <vector>
#include <torch/script.h>
#include <ATen/ATen.h>
#include "fc_kernel.h" // Include the header for your CUDA functions
#include <cuda_runtime.h>
#include <chrono>


// Function to calculate Mean Squared Error (MSE) Loss
float mse_loss_cust(const at::Tensor& output_cuda, const at::Tensor& output_cpu) {
  at::Tensor diff = output_cuda.cpu() - output_cpu;
  at::Tensor squared_diff = diff * diff;
  float sum = squared_diff.sum().item<float>();
  float mean = sum / output_cuda.numel();
  return mean;
}

int main() {
  // set the gpu number
  int gpu_index = 17;
  c10::cuda::CUDAGuard device_guard(gpu_index);
  torch::Device device(torch::kCUDA, gpu_index);

  // Define a vector of input sizes to test
  std::vector<std::tuple<int, int, int>> input_sizes = {
    {1, 224, 224},    // Small input
    {4, 2048, 512},    // Medium input
    {16, 4096, 1024},   // Large input
    {32, 8192, 2048},  // Very large input
    {64, 16384, 4096}, // Huge input
    {128, 32768, 8192} // Extremely huge input
  };

  // Loop over the input sizes
  for (const auto& [N, D_in, D_out] : input_sizes) {
    std::cout << "Testing with input size: N=" << N << ", D_in=" << D_in << ", D_out=" << D_out << std::endl;

    // ********************
    // Create inputs on CPU
    // ********************
    // Create random input, weight, and bias tensors on the CPU
    at::Tensor input_cpu = torch::randn({N, D_in}, at::device(at::kCPU).dtype(at::kFloat));
    at::Tensor weight_cpu = torch::randn({D_out, D_in}, at::device(at::kCPU).dtype(at::kFloat));
    at::Tensor bias_cpu = torch::randn({D_out}, at::device(at::kCPU).dtype(at::kFloat));

    // ********************
    // CUDA Version
    // ********************
    // Copy input tensors to GPU
    at::Tensor input_cuda = input_cpu.to(at::device(at::kCUDA));
    at::Tensor weight_cuda = weight_cpu.to(at::device(at::kCUDA));
    at::Tensor bias_cuda = bias_cpu.to(at::device(at::kCUDA));

    // Extract raw pointers
    const float* in_ptr = input_cuda.data_ptr<float>();
    const float* wt_ptr = weight_cuda.data_ptr<float>();
    const float* b_ptr = bias_cuda.data_ptr<float>();

    // Allocate output tensor
    at::Tensor output_cuda = torch::empty({N, D_out}, input_cuda.options());
    float* out_ptr = output_cuda.data_ptr<float>();
    
    // Time the fc_forward_cuda function for 10 iterations
    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);

    cudaEventRecord(start_cuda);
    for (int i = 0; i < 10; ++i) {
        fc_layer_custom_tiled(in_ptr, wt_ptr, b_ptr, out_ptr, N, D_in, D_out);
    }
    cudaEventRecord(stop_cuda);
    cudaEventSynchronize(stop_cuda);

    float milliseconds_cuda = 0;
    cudaEventElapsedTime(&milliseconds_cuda, start_cuda, stop_cuda);
    std::cout << "CUDA Average time for 10 iterations: " << milliseconds_cuda / (10.0*1000) << " s" << std::endl;

    cudaEventDestroy(start_cuda);
    cudaEventDestroy(stop_cuda);

    // Print the output tensor shape
    std::cout << "CUDA Output tensor shape: " << output_cuda.sizes() << std::endl;

    // ********************
    // CPU Version
    // ********************
    // Allocate output tensor for CPU
    at::Tensor output_cpu = torch::zeros({N, D_out}, at::device(at::kCPU).dtype(at::kFloat));

    // Time the CPU version using torch::addmm
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
      output_cpu = torch::addmm(bias_cpu, input_cpu, weight_cpu.transpose(0, 1));
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;
    std::cout << "CPU (torch::addmm) Average time for 10 iterations: "
              << duration_cpu.count() / 10.0 << " s" << std::endl;

    // Print the output tensor shape
    std::cout << "CPU Output tensor shape: " << output_cpu.sizes() << std::endl;

    // Calculate and Print the Loss
    float loss_mse = mse_loss_cust(output_cuda, output_cpu);
    std::cout << "Mean Squared Error (MSE) Loss: " << loss_mse << std::endl;
    std::cout << "Speeedup :"<<   (milliseconds_cuda / (10.0*1000))/(duration_cpu.count() / 10.0)<<"x\n";
    std::cout << "------------------------" << std::endl;
  }
  std::cout << "All tests completed." << std::endl;
  return 0;
}
