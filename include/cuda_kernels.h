#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Error checking macro for CUDA calls
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Add error checking for cuBLAS
#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s at line %d: %d\n", \
            __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

// Team A: Basic convolution function declaration
void launch_conv2d_baseline(float* input, float* weights, float* bias, float* output,
                            int N, int C, int H, int W, int K, int R, int S, int P, int Q);

// Team C: FC layer function using cuBLAS
void fc_layer_cublas(float* input, float* weights, float* bias, float* output,
                     int batch_size, int input_features, int output_features);

// Optimized batched FC layer implementation using cuBLAS
void fc_layer_cublas_batched(float* input, float* weights, float* bias, float* output,
    int batch_size, int input_features, int output_features);

// Fused convolution+ReLU implementation (for Team A coordination)
void launch_conv2d_relu_fused(float* input, float* weights, float* bias, float* output,
                              int N, int C, int H, int W, int K, int R, int S, int P, int Q);

// Naive custom FC layer implementation
void fc_layer_custom_naive(float* input, float* weights, float* bias, float* output,
                           int batch_size, int input_features, int output_features);

// Tiled shared memory FC layer implementation
void fc_layer_custom_tiled(float* input, float* weights, float* bias, float* output,
                           int batch_size, int input_features, int output_features);

// Vectorized FC layer implementation with coalesced memory access
void fc_layer_custom_vectorized(float* input, float* weights, float* bias, float* output,
                                int batch_size, int input_features, int output_features);

#endif // CUDA_KERNELS_H