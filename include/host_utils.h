#pragma once

#include <vector>
#include <string>

void placeholder_utils_function();

void run_conv2d_naive_test(
    float* d_input,
    float* d_weight,
    float* d_bias,
    float* d_output,
    const std::vector<float>& expected,
    int N, int C, int H, int W,
    int K, int R, int S, int P, int Q
);

void run_conv2d_tiled_test(
    float* d_input,
    float* d_weight,
    float* d_bias,
    float* d_output,
    const std::vector<float>& expected,
    int N, int C, int H, int W,
    int K, int R, int S, int P, int Q
);

void run_conv2d_tiled_coarsened_test(
    float* d_input,
    float* d_weight,
    float* d_bias,
    float* d_output,
    const std::vector<float>& expected,
    int N, int C, int H, int W,
    int K, int R, int S, int P, int Q
);

void write_benchmark_to_file(
    const std::string& kernel_name,
    float time_ms,
    float gflops,
    float max_diff,
    float l2_error
);

void load_all_conv_weights_biases(float**& d_weights, float**& d_biases, int num_layers);