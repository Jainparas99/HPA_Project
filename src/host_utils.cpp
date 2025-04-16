#include "host_utils.h"
#include "cuda_kernels.h"
#include "cnpy.h"
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

void placeholder_utils_function() {
    std::cout << " Running CUDA Conv2D test on conv1 weights...\n";

    // --- Load .npy files ---
    auto input_np = cnpy::npy_load("models/input_tensor.npy");
    auto weight_np = cnpy::npy_load("models/conv1_weights.npy");
    auto bias_np = cnpy::npy_load("models/conv1_bias.npy");
    auto expected_np = cnpy::npy_load("models/conv1_output.npy");

    float* input = input_np.data<float>();      // [1, 3, 224, 224]
    float* weight = weight_np.data<float>();    // [64, 3, 3, 3]
    float* bias = bias_np.data<float>();        // [64]
    float* expected = expected_np.data<float>(); // [1, 64, 224, 224]

    // --- Tensor dims for conv1 ---
    int N = 1, C = 3, H = 224, W = 224;     // Input
    int K = 64, R = 3, S = 3;               // Weights
    int P = 224, Q = 224;                   // Output

    size_t input_bytes = N * C * H * W * sizeof(float);
    size_t weight_bytes = K * C * R * S * sizeof(float);
    size_t bias_bytes = K * sizeof(float);
    size_t output_bytes = N * K * P * Q * sizeof(float);

    // --- Allocate device memory ---
    float *d_input, *d_weight, *d_bias, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_weight, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_bias, bias_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));

    // --- Copy inputs to device ---
    CHECK_CUDA(cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, weight, weight_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, bias, bias_bytes, cudaMemcpyHostToDevice));

    // --- Launch kernel ---
    std::cout << "Launching conv2d_naive...\n";
    launch_conv2d_naive(d_input, d_weight, d_bias, d_output,
                        N, C, H, W, K, R, S, P, Q);

    // --- Copy output back ---
    std::vector<float> output(N * K * P * Q);
    CHECK_CUDA(cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

    // --- Compare with expected ---
    std::cout << "Comparing output with expected...\n";
    float max_diff = 0.0f, sum_diff_sq = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        float diff = output[i] - expected[i];
        max_diff = std::max(max_diff, std::abs(diff));
        sum_diff_sq += diff * diff;
    }
    float l2_error = std::sqrt(sum_diff_sq / output.size());

    std::cout << "Comparison complete\n";
    std::cout << "Max absolute difference: " << max_diff << "\n";
    std::cout << "L2 norm error: " << l2_error << "\n";

    // --- Cleanup ---
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}
