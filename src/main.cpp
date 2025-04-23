// #include <onnxruntime_cxx_api.h>
#include <cuda_kernels.h>
#include <host_utils.h>
#include <cuda_runtime.h>
#include "cnpy.h"
#include <iostream>
#include <cmath>

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

int main() {

    std::cout << "HPA Project - CUDA VGG16 Convolution Testing\n";

    // Placeholder function to confirm CUDA + host code builds
    placeholder_utils_function();
    std::cout << " Running VGG16 Conv Layer Inference...\n";

    // Constants
    const int N = 1;
    const int C = 3;
    const int H = 224;
    const int W = 224;
    const int num_layers = 13;

    // Load input and expected output
    auto input_np = cnpy::npy_load("models/input_tensor.npy");
    auto expected_np = cnpy::npy_load("models/expected_output.npy");

    std::vector<float> input = input_np.as_vec<float>();
    std::vector<float> expected = expected_np.as_vec<float>();

    size_t input_bytes = input.size() * sizeof(float);
    size_t output_bytes = expected.size() * sizeof(float);

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice));

    // Load all 13 conv weights and biases
    float** d_weights;
    float** d_biases;
    load_all_conv_weights_biases(d_weights, d_biases, num_layers);

    // Start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Run all conv layers
    run_vgg16_conv_layers(d_input, d_weights, d_biases, d_output, N, H, W);

    // End timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    if (elapsed_time_ms < 1e-3f) elapsed_time_ms = 1e-3f;

    // Copy output and compare
    std::vector<float> output(expected.size());
    cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost);

    // Accuracy metrics
    float max_diff = 0.0f, l2_error = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        float diff = output[i] - expected[i];
        max_diff = std::max(max_diff, std::abs(diff));
        l2_error += diff * diff;
    }
    l2_error = std::sqrt(l2_error);

    // GFLOPs: Conv ops (approx.) = 2 * K * C * R * S * P * Q * N (summed over all layers)
    long long total_ops = 0;
    const int conv_out_channels[13] = {64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512};
    int in_channels = C;
    for (int i = 0; i < num_layers; ++i) {
        int K = conv_out_channels[i];
        total_ops += (2LL * K * in_channels * 3 * 3 * H * W * N);
        in_channels = K;
    }
    float gflops = total_ops / (elapsed_time_ms * 1e6f);

    // Results
    std::cout << "Inference complete\n";
    std::cout << "Time: " << elapsed_time_ms << " ms\n";
    std::cout << "GFLOPs: " << gflops << "\n";
    std::cout << "Max Abs Diff: " << max_diff << "\n";
    std::cout << "L2 Norm Error: " << l2_error << "\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    for (int i = 0; i < num_layers; ++i) {
        cudaFree(d_weights[i]);
        cudaFree(d_biases[i]);
    }
    delete[] d_weights;
    delete[] d_biases;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}