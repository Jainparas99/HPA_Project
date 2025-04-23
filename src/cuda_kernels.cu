#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

// Basic convolution implementation - will be optimized by Team A
__global__ void conv2d_kernel(float* input, float* weights, float* bias, float* output,
                              int N, int C, int H, int W, int K, int R, int S, int P, int Q) {
    // Get output coordinates
    int n = blockIdx.x;
    int k = blockIdx.y;
    int p = blockIdx.z / Q;
    int q = blockIdx.z % Q;

    // Check boundaries
    if (n >= N || k >= K || p >= P || q >= Q) return;

    // Compute convolution at this position
    float result = 0.0f;

    // For each input channel and filter position
    for (int c = 0; c < C; c++) {
        for (int r = 0; r < R; r++) {
            for (int s = 0; s < S; s++) {
                int h = p + r - R / 2;  // Assuming padding = R/2
                int w = q + s - S / 2;  // Assuming padding = S/2

                if (h >= 0 && h < H && w >= 0 && w < W) {
                    float input_val = input[n * C * H * W + c * H * W + h * W + w];
                    float weight_val = weights[k * C * R * S + c * R * S + r * S + s];
                    result += input_val * weight_val;
                }
            }
        }
    }

    // Add bias
    result += bias[k];

    // Store result
    output[n * K * P * Q + k * P * Q + p * Q + q] = result;
}

// This is a fused convolution+ReLU kernel - will need coordination with Team A
__global__ void conv2d_relu_kernel(float* input, float* weights, float* bias, float* output,
                                   int N, int C, int H, int W, int K, int R, int S, int P, int Q) {
    // Get output coordinates
    int n = blockIdx.x;
    int k = blockIdx.y;
    int p = blockIdx.z / Q;
    int q = blockIdx.z % Q;

    // Check boundaries
    if (n >= N || k >= K || p >= P || q >= Q) return;

    // Compute convolution at this position
    float result = 0.0f;

    // For each input channel and filter position
    for (int c = 0; c < C; c++) {
        for (int r = 0; r < R; r++) {
            for (int s = 0; s < S; s++) {
                int h = p + r - R / 2;  // Assuming padding = R/2
                int w = q + s - S / 2;  // Assuming padding = S/2

                if (h >= 0 && h < H && w >= 0 && w < W) {
                    float input_val = input[n * C * H * W + c * H * W + h * W + w];
                    float weight_val = weights[k * C * R * S + c * R * S + r * S + s];
                    result += input_val * weight_val;
                }
            }
        }
    }

    // Add bias
    result += bias[k];

    // Apply ReLU directly - this is the fusion part
    result = fmaxf(0.0f, result);

    // Store result
    output[n * K * P * Q + k * P * Q + p * Q + q] = result;
}

// Host function to launch fused convolution+ReLU kernel
void launch_conv2d_relu_fused(float* input, float* weights, float* bias, float* output,
                              int N, int C, int H, int W, int K, int R, int S, int P, int Q) {
    // Allocate device memory
    float* d_input, * d_weights, * d_bias, * d_output;

    size_t input_size = N * C * H * W * sizeof(float);
    size_t weights_size = K * C * R * S * sizeof(float);
    size_t bias_size = K * sizeof(float);
    size_t output_size = N * K * P * Q * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input, input_size));
    CHECK_CUDA(cudaMalloc(&d_weights, weights_size));
    CHECK_CUDA(cudaMalloc(&d_bias, bias_size));
    CHECK_CUDA(cudaMalloc(&d_output, output_size));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, weights, weights_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 grid(N, K, P * Q);
    conv2d_relu_kernel<<<grid, 1>>>(d_input, d_weights, d_bias, d_output, N, C, H, W, K, R, S, P, Q);

    // Check for errors
    CHECK_CUDA(cudaGetLastError());

    // Copy results back
    CHECK_CUDA(cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_output));
}

// Kernel to add bias and apply ReLU activation
__global__ void bias_relu_kernel(float* output, float* bias, int batch_size, int output_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * output_features) {
        // Calculate feature index to get correct bias value
        int feature_idx = idx % output_features;
        
        // Add bias
        output[idx] += bias[feature_idx];
        
        // Apply ReLU activation: max(0, x)
        output[idx] = fmaxf(0.0f, output[idx]);
    }
}

// Host function to launch convolution kernel
void launch_conv2d_baseline(float* input, float* weights, float* bias, float* output,
    int N, int C, int H, int W, int K, int R, int S, int P, int Q) {
    // Allocate device memory
    float* d_input, * d_weights, * d_bias, * d_output;

    size_t input_size = N * C * H * W * sizeof(float);
    size_t weights_size = K * C * R * S * sizeof(float);
    size_t bias_size = K * sizeof(float);
    size_t output_size = N * K * P * Q * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_input, input_size));
    CHECK_CUDA(cudaMalloc(&d_weights, weights_size));
    CHECK_CUDA(cudaMalloc(&d_bias, bias_size));
    CHECK_CUDA(cudaMalloc(&d_output, output_size));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, weights, weights_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 grid(N, K, P * Q);
    conv2d_kernel << <grid, 1 >> > (d_input, d_weights, d_bias, d_output, N, C, H, W, K, R, S, P, Q);

    // Check for errors
    CHECK_CUDA(cudaGetLastError());

    // Copy results back
    CHECK_CUDA(cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_output));
}

// Team C: FC layer implementation with cuBLAS
void fc_layer_cublas(float* input, float* weights, float* bias, float* output,
                     int batch_size, int input_features, int output_features) {
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate device memory
    float* d_input, * d_weights, * d_bias, * d_output;

    CHECK_CUDA(cudaMalloc(&d_input, batch_size * input_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weights, output_features * input_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, output_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, batch_size * output_features * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, input, batch_size * input_features * sizeof(float),
        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, weights, output_features * input_features * sizeof(float),
        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, bias, output_features * sizeof(float),
        cudaMemcpyHostToDevice));

    // Perform matrix multiplication: output = input * weights^T
    // Note: cuBLAS uses column-major order by default
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
        CUBLAS_OP_T,         // Transpose weights
        CUBLAS_OP_N,         // No transpose for input
        output_features,      // Rows of output 
        batch_size,           // Columns of output
        input_features,       // Inner dimension
        &alpha,               // Scale factor
        d_weights,            // Weights matrix
        input_features,       // Leading dimension of weights
        d_input,              // Input matrix
        input_features,       // Leading dimension of input
        &beta,                // Scale factor for output
        d_output,             // Output matrix
        output_features);     // Leading dimension of output

    // Add bias and apply ReLU activation using a custom kernel
    dim3 blockSize(256);
    dim3 gridSize((batch_size * output_features + blockSize.x - 1) / blockSize.x);
    bias_relu_kernel<<<gridSize, blockSize>>>(d_output, d_bias, batch_size, output_features);
    
    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());
    
    // Synchronize to ensure completion before copying back
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(output, d_output, batch_size * output_features * sizeof(float),
        cudaMemcpyDeviceToHost));

    // Clean up
    cublasDestroy(handle);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_output));
}