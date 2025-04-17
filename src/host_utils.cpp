#include "host_utils.h"
#include "cuda_kernels.h"
#include "cnpy.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }
void run_conv2d_tiled_test(
        float* d_input, float* d_weight, float* d_bias, float* d_output,
        const std::vector<float>& expected,
        int N, int C, int H, int W, int K, int R, int S, int P, int Q);
    
void placeholder_utils_function() {
    std::cout << "\nHPA Project - CUDA VGG16 Convolution Testing\n" << std::endl;

    // Load input, weights, bias, and expected output from npy
    auto input_np = cnpy::npy_load("models/input_tensor.npy");
    auto weight_np = cnpy::npy_load("models/conv1_weights.npy");
    auto bias_np = cnpy::npy_load("models/conv1_bias.npy");
    auto expected_np = cnpy::npy_load("models/expected_output.npy");

    std::vector<float> input = input_np.as_vec<float>();
    std::vector<float> weight = weight_np.as_vec<float>();
    std::vector<float> bias = bias_np.as_vec<float>();
    std::vector<float> expected = expected_np.as_vec<float>();

    // VGG16 conv1 shape (as an example)
    int N = 1, C = 3, H = 224, W = 224;
    int K = 64, R = 3, S = 3;
    int P = 224, Q = 224;  // Assuming padding=1, stride=1

    size_t input_bytes = input.size() * sizeof(float);
    size_t weight_bytes = weight.size() * sizeof(float);
    size_t bias_bytes = bias.size() * sizeof(float);
    size_t output_bytes = expected.size() * sizeof(float);

    float *d_input, *d_weight, *d_bias, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_weight, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_bias, bias_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));

    CHECK_CUDA(cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, weight.data(), weight_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, bias.data(), bias_bytes, cudaMemcpyHostToDevice));

    std::cout << "\n🔧 Running CUDA Conv2D test on conv1 weights...\n";

    // Run tiled kernel and compare with expected output
    run_conv2d_tiled_test(d_input, d_weight, d_bias, d_output,
                          expected, N, C, H, W, K, R, S, P, Q);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

void run_conv2d_tiled_test(
    float* d_input, float* d_weight, float* d_bias, float* d_output,
    const std::vector<float>& expected,
    int N, int C, int H, int W, int K, int R, int S, int P, int Q
) {
    std::cout << "\n🧪 Running conv2d_tiled...\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::cout << "Launching with:\n"
          << "N=" << N << " C=" << C << " H=" << H << " W=" << W << "\n"
          << "K=" << K << " R=" << R << " S=" << S << " P=" << P << " Q=" << Q << "\n"
          << "Output size: " << P*Q*K*N << " elements\n";

    cudaEventRecord(start);
    launch_conv2d_tiled(d_input, d_weight, d_bias, d_output,
                        N, C, H, W, K, R, S, P, Q);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Error: " << cudaGetErrorString(err) << std::endl;
                        }
                        
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;

    int output_size = N * K * P * Q;
    std::vector<float> output(output_size);
    size_t output_bytes = output.size() * sizeof(float);

    cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    float l2_sum = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        float diff = output[i] - expected[i];
        max_diff = std::max(max_diff, std::abs(diff));
        l2_sum += diff * diff;
    }

    float l2_error = std::sqrt(l2_sum);
    float ops = 2.0f * K * C * R * S * P * Q * N;
    float gflops = ops / (seconds * 1e9);

    std::cout << "✅ conv2d_tiled complete\n";
    std::cout << "⏱️  Time: " << milliseconds << " ms\n";
    std::cout << "⚡ GFLOPS: " << gflops << "\n";
    std::cout << "📏 Max abs diff: " << max_diff << "\n";
    std::cout << "📐 L2 error: " << l2_error << "\n\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
