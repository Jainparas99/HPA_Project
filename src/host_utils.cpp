#include "host_utils.h"
#include "cuda_kernels.h"
#include "cnpy.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

void run_conv2d_naive_test(
    float* d_input, float* d_weight, float* d_bias, float* d_output,
    const std::vector<float>& expected,
    int N, int C, int H, int W, int K, int R, int S, int P, int Q) {

    std::cout << "Running conv2d_naive...\n";

    // Time using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    launch_conv2d_naive(d_input, d_weight, d_bias, d_output,
                        N, C, H, W, K, R, S, P, Q);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    // Copy result back to host
    size_t output_bytes = N * K * P * Q * sizeof(float);
    std::vector<float> output(N * K * P * Q);
    cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost);

    std::cout << "Performing warm-up run for naive kernel...";
    launch_conv2d_naive(d_input, d_weight, d_bias, d_output,
                        N, C, H, W, K, R, S, P, Q);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Compute metrics
    float gflops = (2.0f * K * C * R * S * P * Q * N) / (elapsed_time_ms * 1e6f);
    float max_diff = 0.0f;
    float l2_error = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        float diff = std::abs(output[i] - expected[i]);
        max_diff = std::max(max_diff, diff);
        l2_error += diff * diff;
    }
    l2_error = std::sqrt(l2_error);

    // Print and log
    std::cout << "Execution Time (ms): " << elapsed_time_ms << "\n";
    std::cout << "GFLOPS: " << gflops << "\n";
    std::cout << "Max absolute difference: " << max_diff << "\n";
    std::cout << "L2 norm error: " << l2_error << "\n";

    write_benchmark_to_file("conv2d_naive", elapsed_time_ms, gflops, max_diff, l2_error);
}
        
void run_conv2d_tiled_test(
    float* d_input, float* d_weight, float* d_bias, float* d_output,
    const std::vector<float>& expected,
    int N, int C, int H, int W, int K, int R, int S, int P, int Q
) {

    std::cout << "Performing warm-up run for tiled kernel...";
    launch_conv2d_tiled(d_input, d_weight, d_bias, d_output,
                               N, C, H, W, K, R, S, P, Q);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "Running conv2d_tiled...\n";

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
    float gflops = ops / (seconds * 1e6);

    std::cout << "conv2d_tiled complete\n";
    std::cout << "Time: " << milliseconds << " ms\n";
    std::cout << "GFLOPS: " << gflops << "\n";
    std::cout << "Max abs diff: " << max_diff << "\n";
    std::cout << "L2 norm error: " << l2_error << std::endl;

    // write_benchmark_to_file("conv2d_tiled", milliseconds, gflops, max_diff, l2_error);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


void write_benchmark_to_file(const std::string& kernel_name, float time_ms, float gflops, float max_diff, float l2_error) {
    std::ofstream out("profile/benchmark_results.txt", std::ios::app); // append mode
    if (out.is_open()) {
        out << "Kernel: " << kernel_name << "\n";
        out << "Execution Time (ms): " << time_ms << "\n";
        out << "GFLOPS: " << gflops << "\n";
        out << "Max Absolute Difference: " << max_diff << "\n";
        out << "L2 Norm Error: " << l2_error << "\n";
        out << "-----------------------------\n";
        out.close();
    } else {
        std::cerr << "Unable to write benchmark results.\n";
    }
}


void run_conv2d_tiled_coarsened_test(
    float* d_input, float* d_weight, float* d_bias, float* d_output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int P, int Q,
    const std::vector<float>& expected_output)
{
    std::cout << "\n Running conv2d_tiled_coarsened...\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    launch_conv2d_tiled_coarsened(d_input, d_weight, d_bias, d_output,
                                   N, C, H, W, K, R, S, P, Q);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    std::cout << "Execution Time: " << elapsed_ms << " ms\n";

    std::vector<float> host_output(N * K * P * Q);
    cudaMemcpy(host_output.data(), d_output,
               N * K * P * Q * sizeof(float), cudaMemcpyDeviceToHost);

    float max_diff = 0;
    float l2_sum = 0;
    for (int i = 0; i < host_output.size(); ++i) {
        float diff = host_output[i] - expected_output[i];
        max_diff = std::max(max_diff, std::abs(diff));
        l2_sum += diff * diff;
    }

    float l2_norm = std::sqrt(l2_sum);
    std::cout << "Max absolute difference: " << max_diff << "\n";
    std::cout << "L2 norm error: " << l2_norm << "\n";

    // Optional: log to file
    std::ofstream file("profile/conv2d_tiled_coarsened.txt");
    file << "Kernel: conv2d_tiled_coarsened\n";
    file << "Execution Time: " << elapsed_ms << " ms\n";
    file << "Max Difference: " << max_diff << "\n";
    file << "L2 Norm Error: " << l2_norm << "\n";
    file.close();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
        
        
    
void placeholder_utils_function() {
    std::cout << "\nHPA Project - CUDA VGG16 Convolution Testing\n" << std::endl;

    // Load input, weights, bias, and expected output from npy
    auto input_np = cnpy::npy_load("models/input_tensor.npy");
    auto weight_np = cnpy::npy_load("models/conv1_weights.npy");
    auto bias_np = cnpy::npy_load("models/conv1_bias.npy");
    auto conv1_output_np = cnpy::npy_load("models/conv1_output.npy");
    // auto expected_np = cnpy::npy_load("models/expected_output.npy");

    std::vector<float> input = input_np.as_vec<float>();
    std::vector<float> weight = weight_np.as_vec<float>();
    std::vector<float> bias = bias_np.as_vec<float>();
    std::vector<float> conv1_expected_output = conv1_output_np.as_vec<float>();
    //std::vector<float> expected = expected_np.as_vec<float>();

//     float* input = input_np.data<float>();      // [1, 3, 224, 224]
// -    float* weight = weight_np.data<float>();    // [64, 3, 3, 3]
// -    float* bias = bias_np.data<float>();        // [64]z
// -    float* expected = expected_np.data<float>();

    // VGG16 conv1 shape (as an example)
    int N = 1, C = 3, H = 224, W = 224;
    int K = 64, R = 3, S = 3;
    int P = 224, Q = 224;  // Assuming padding=1, stride=1

    std::cout << "Input size: " << input.size() << std::endl;
    std::cout << "Weight size: " << weight.size() << std::endl;
    std::cout << "Bias size: " << bias.size() << std::endl;
    std::cout << "Expected output size: " << conv1_expected_output.size() << std::endl;

    size_t input_bytes = input.size() * sizeof(float);
    size_t weight_bytes = weight.size() * sizeof(float);
    size_t bias_bytes = bias.size() * sizeof(float);
    size_t output_bytes = conv1_expected_output.size() * sizeof(float);

    float *d_input, *d_weight, *d_bias, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_weight, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_bias, bias_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));

    CHECK_CUDA(cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, weight.data(), weight_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, bias.data(), bias_bytes, cudaMemcpyHostToDevice));

    std::cout << "\nRunning CUDA Conv2D test on conv1 weights...\n";

    // Run tiled kernel and compare with expected output
    run_conv2d_naive_test(d_input, d_weight, d_bias, d_output,
        conv1_expected_output, N, C, H, W, K, R, S, P, Q);
    
    // Run tiled kernel and compare with expected output
    run_conv2d_tiled_test(d_input, d_weight, d_bias, d_output,
                          conv1_expected_output, N, C, H, W, K, R, S, P, Q);

    run_conv2d_tiled_coarsened_test(
        d_input, d_weight, d_bias, d_output,
        N, C, H, W, K, R, S, P, Q, conv1_expected_output
    );

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
}

void load_all_conv_weights_biases(float**& d_weights, float**& d_biases, int num_layers) {
    d_weights = new float*[num_layers];
    d_biases  = new float*[num_layers];

    for (int i = 0; i < num_layers; ++i) {
        std::string w_path = "models/conv" + std::to_string(i) + "_weights.npy";
        std::string b_path = "models/conv" + std::to_string(i) + "_bias.npy";

        // Load host memory
        auto weight_np = cnpy::npy_load(w_path);
        auto bias_np   = cnpy::npy_load(b_path);

        float* h_weight = weight_np.as_vec<float>().data();
        float* h_bias   = bias_np.as_vec<float>().data();

        size_t weight_size = 1;
        for (auto dim : weight_np.shape) weight_size *= dim;

        size_t bias_size = 1;
        for (auto dim : bias_np.shape) bias_size *= dim;


        // size_t weight_size;
        // size_t bias_size; 
        // Allocate device memory
        cudaMalloc(&d_weights[i], weight_size * sizeof(float));
        cudaMalloc(&d_biases[i],  bias_size * sizeof(float));

        // Copy to device
        cudaMemcpy(d_weights[i], h_weight, weight_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_biases[i],  h_bias,   bias_size * sizeof(float), cudaMemcpyHostToDevice);

        std::cout << "Loaded conv" << i << " weights and bias" << std::endl;

        // Optionally free host memory if load_npy allocates dynamically
        // free(h_weight);
        // free(h_bias);
    }
}


void run_vgg16_conv_layers(float* input, float** weights, float** biases, float* output,
                           int N, int H, int W) {
    const int num_layers = 13;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 1;  // VGG-16 uses padding=1 to preserve spatial dimensions

    const int conv_out_channels[num_layers] = {
        64, 64, 128, 128, 256, 256, 256,
        512, 512, 512, 512, 512, 512
    };

    float *buf1, *buf2;
    int in_channels = 3;
    int out_channels;

    // Initial output buffer: [N, 64, H, W] → we’ll keep sizes constant initially
    size_t max_buf_size = N * 512 * H * W * sizeof(float);
    cudaMalloc(&buf1, max_buf_size);
    cudaMalloc(&buf2, max_buf_size);

    float* current_input = input;
    float* current_output = buf1;

    for (int i = 0; i < num_layers; ++i) {
        out_channels = conv_out_channels[i];

        // launch_conv2d_tiled(current_input, weights[i], biases[i], current_output,
        //              N, in_channels, H, W, out_channels, kernel_size, kernel_size, H, W);

        launch_conv2d_tiled_safe(current_input, weights[i], biases[i], current_output,
                    N, in_channels, H, W, out_channels, 3, 3, H, W);

        std::cout << "Conv" << i << ": " << in_channels << "→" << out_channels << ", size: " << H << "x" << W << std::endl;

        // Prepare for next layer
        in_channels = out_channels;
        std::swap(current_input, current_output);
    }

    // Copy final conv output to 'output'
    cudaMemcpy(output, current_input, N * in_channels * H * W * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(buf1);
    cudaFree(buf2);
}
