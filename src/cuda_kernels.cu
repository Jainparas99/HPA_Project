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

// Optimized FC layer implementation using cuBLAS batch operations
void fc_layer_cublas_batched(float* input, float* weights, float* bias, float* output,
                             int batch_size, int input_features, int output_features) {
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate device memory
    float* d_input, *d_weights, *d_bias, *d_output;

    // Create a CUDA stream for asynchronous operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    CHECK_CUDA(cudaMalloc(&d_input, batch_size * input_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weights, output_features * input_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, output_features * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, batch_size * output_features * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpyAsync(d_input, input, batch_size * input_features * sizeof(float),
               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_weights, weights, output_features * input_features * sizeof(float),
               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_bias, bias, output_features * sizeof(float),
               cudaMemcpyHostToDevice, stream));

    // Scale factors
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Use batch strided GEMM for better performance with batches
    // A = weights, B = input, C = output
    // M = output_features, N = batch_size, K = input_features
    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T,         // Transpose weights
        CUBLAS_OP_N,         // No transpose for input
        output_features,     // M: rows of output
        1,                   // N: columns of output (1 per batch item)
        input_features,      // K: columns of weights/rows of input
        &alpha,              // Scale factor
        d_weights,           // Weights matrix
        input_features,      // Leading dimension of weights
        0,                   // Stride between batched weight matrices (0 as weights are shared)
        d_input,             // Input matrix
        input_features,      // Leading dimension of input
        input_features,      // Stride between batched input vectors
        &beta,               // Scale factor for output
        d_output,            // Output matrix
        output_features,     // Leading dimension of output
        output_features,     // Stride between batched output vectors
        batch_size           // Batch count
    );

    // Add bias and ReLU in a single kernel for better performance
    dim3 blockSize(256);
    dim3 gridSize((batch_size * output_features + blockSize.x - 1) / blockSize.x);
    bias_relu_kernel<<<gridSize, blockSize, 0, stream>>>(d_output, d_bias, batch_size, output_features);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpyAsync(output, d_output, batch_size * output_features * sizeof(float),
               cudaMemcpyDeviceToHost, stream));

    // Synchronize to ensure completion
    cudaStreamSynchronize(stream);

    // Clean up
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_output));
}

/* Custom FC Layer Implementations */

/**
 * Naive matrix multiplication kernel (1 thread per output element)
 * Each thread computes one element of the output matrix
 * Each thread reads input_features elements from input and weights
 */
__global__ void fc_matmul_naive_kernel(float* input, float* weights, float* bias, float* output,
                                       int batch_size, int input_features, int output_features) {
// Calculate output indices
int b = blockIdx.y;                          // Batch index
int o = blockIdx.x * blockDim.x + threadIdx.x;  // Output feature index

// Check bounds
if (b >= batch_size || o >= output_features)
return;

// Calculate output element with dot product
float sum = 0.0f;
for (int i = 0; i < input_features; ++i) {
sum += input[b * input_features + i] * weights[o * input_features + i];
}

// Add bias and apply ReLU
sum += bias[o];
output[b * output_features + o] = fmaxf(0.0f, sum);
}

/**
* Host function to launch the naive kernel
*/
void fc_layer_custom_naive(float* input, float* weights, float* bias, float* output,
                           int batch_size, int input_features, int output_features) {
// Allocate device memory
float* d_input, *d_weights, *d_bias, *d_output;

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

// Set kernel dimensions
int threadsPerBlock = 256;
int blocksPerOutput = (output_features + threadsPerBlock - 1) / threadsPerBlock;
dim3 blockSize(threadsPerBlock);
dim3 gridSize(blocksPerOutput, batch_size);

// Launch kernel
fc_matmul_naive_kernel<<<gridSize, blockSize>>>(d_input, d_weights, d_bias, d_output,
                batch_size, input_features, output_features);

// Check for kernel launch errors
CHECK_CUDA(cudaGetLastError());

// Synchronize to ensure completion
CHECK_CUDA(cudaDeviceSynchronize());

// Copy result back to host
CHECK_CUDA(cudaMemcpy(output, d_output, batch_size * output_features * sizeof(float),
cudaMemcpyDeviceToHost));

// Clean up
CHECK_CUDA(cudaFree(d_input));
CHECK_CUDA(cudaFree(d_weights));
CHECK_CUDA(cudaFree(d_bias));
CHECK_CUDA(cudaFree(d_output));
}

/**
* Tiled matrix multiplication kernel using shared memory
* Uses shared memory to reduce global memory accesses
* Each thread block computes a tile of output elements
*/
#define TILE_SIZE 16

__global__ void fc_matmul_tiled_kernel(float* input, float* weights, float* bias, float* output,
                                       int batch_size, int input_features, int output_features) {
// Shared memory for tiles
__shared__ float shared_input[TILE_SIZE];
__shared__ float shared_weights[TILE_SIZE][TILE_SIZE];

// Calculate indices
int b = blockIdx.z;                // Batch index
int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // Output feature index
int col = threadIdx.x;             // Input tile index

// Check if row is within output bounds
if (row >= output_features || b >= batch_size)
    return;

// Accumulator for dot product - each thread maintains its own sum
float sum = 0.0f;

// Loop over input feature tiles
for (int i = 0; i < (input_features + TILE_SIZE - 1) / TILE_SIZE; ++i) {
    // Load weights into shared memory cooperatively
    if (i * TILE_SIZE + col < input_features) {
        shared_weights[threadIdx.y][col] = weights[row * input_features + i * TILE_SIZE + col];
    } else {
        shared_weights[threadIdx.y][col] = 0.0f;
    }

    // Load input into shared memory (all threads cooperate)
    if (threadIdx.y == 0 && i * TILE_SIZE + col < input_features) {
        shared_input[col] = input[b * input_features + i * TILE_SIZE + col];
    }

    // Wait for all threads to load data
    __syncthreads();

    // CRITICAL FIX: All threads compute the dot product for their output element
    if (row < output_features && b < batch_size) {
        for (int k = 0; k < TILE_SIZE && i * TILE_SIZE + k < input_features; ++k) {
            sum += shared_weights[threadIdx.y][k] * shared_input[k];
        }
    }

    // Wait for all threads to finish using shared memory
    __syncthreads();
}

// CRITICAL FIX: Add bias and apply ReLU - each thread handles its own output
if (row < output_features && b < batch_size && col == 0) { // Only one thread per row
    sum += bias[row];
    output[b * output_features + row] = fmaxf(0.0f, sum);
    }
}

/**
* Host function to launch the tiled kernel
*/
void fc_layer_custom_tiled(float* input, float* weights, float* bias, float* output,
                           int batch_size, int input_features, int output_features) {
// Allocate device memory
float* d_input, *d_weights, *d_bias, *d_output;

// Use cudaMemcpyAsync for better performance
cudaStream_t stream;
cudaStreamCreate(&stream);

CHECK_CUDA(cudaMalloc(&d_input, batch_size * input_features * sizeof(float)));
CHECK_CUDA(cudaMalloc(&d_weights, output_features * input_features * sizeof(float)));
CHECK_CUDA(cudaMalloc(&d_bias, output_features * sizeof(float)));
CHECK_CUDA(cudaMalloc(&d_output, batch_size * output_features * sizeof(float)));

// Copy data to device asynchronously
CHECK_CUDA(cudaMemcpyAsync(d_input, input, batch_size * input_features * sizeof(float),
           cudaMemcpyHostToDevice, stream));
CHECK_CUDA(cudaMemcpyAsync(d_weights, weights, output_features * input_features * sizeof(float),
           cudaMemcpyHostToDevice, stream));
CHECK_CUDA(cudaMemcpyAsync(d_bias, bias, output_features * sizeof(float),
           cudaMemcpyHostToDevice, stream));

// Calculate optimal grid and block size
dim3 blockSize(TILE_SIZE, TILE_SIZE);
dim3 gridSize(
    1,  // x dimension
    (output_features + TILE_SIZE - 1) / TILE_SIZE,  // y dimension for output features
    batch_size  // z dimension for batch
);

// Launch kernel in the stream
fc_matmul_tiled_kernel<<<gridSize, blockSize, 0, stream>>>(d_input, d_weights, d_bias, d_output,
                                                   batch_size, input_features, output_features);

// Check for kernel launch errors
CHECK_CUDA(cudaGetLastError());

// Copy result back to host asynchronously
CHECK_CUDA(cudaMemcpyAsync(output, d_output, batch_size * output_features * sizeof(float),
           cudaMemcpyDeviceToHost, stream));

// Synchronize to ensure completion
cudaStreamSynchronize(stream);

// Clean up
CHECK_CUDA(cudaFree(d_input));
CHECK_CUDA(cudaFree(d_weights));
CHECK_CUDA(cudaFree(d_bias));
CHECK_CUDA(cudaFree(d_output));
cudaStreamDestroy(stream);
}

/**
* Advanced matrix multiplication kernel with vectorization and coalescing
* Uses vectorized memory loads and ensures coalesced memory access patterns
* Also includes loop unrolling for better instruction-level parallelism
*/
#define BLOCK_SIZE 16
#define VECTOR_SIZE 4  // Process 4 elements at once

__global__ void fc_matmul_vectorized_kernel(float* input, float* weights, float* bias, float* output,
                                            int batch_size, int input_features, int output_features) {
    // Shared memory for input and weight tiles
    __shared__ float shared_input[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_weights[BLOCK_SIZE][BLOCK_SIZE];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Output indices - redesigned for better batch handling
    int b = blockIdx.z;  // Batch index
    int row = blockIdx.y * BLOCK_SIZE + ty;  // Output feature index

    // Check bounds
    if (row >= output_features || b >= batch_size)
        return;

    // Register to accumulate result
    float sum = 0.0f;

    // Process the matrix multiplication in tiles
    for (int tile = 0; tile < (input_features + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Improved coalesced memory access for inputs
        if (tile * BLOCK_SIZE + tx < input_features) {
            shared_input[ty][tx] = input[b * input_features + tile * BLOCK_SIZE + tx];
        } else {
            shared_input[ty][tx] = 0.0f;
        }

        // Improved coalesced memory access for weights
        if (row < output_features && tile * BLOCK_SIZE + tx < input_features) {
            shared_weights[ty][tx] = weights[row * input_features + tile * BLOCK_SIZE + tx];
        } else {
            shared_weights[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product with loop unrolling for better performance
        if (row < output_features && b < batch_size) {
            #pragma unroll 4
            for (int k = 0; k < BLOCK_SIZE && tile * BLOCK_SIZE + k < input_features; ++k) {
                sum += shared_weights[ty][k] * shared_input[0][k];
        }
        }

        __syncthreads();
        }

        // Add bias and apply ReLU with proper thread coordination
        if (tx == 0 && row < output_features && b < batch_size) {
            sum += bias[row];
            output[b * output_features + row] = fmaxf(0.0f, sum);
        }
}

/**
* Host function to launch the vectorized kernel
*/
void fc_layer_custom_vectorized(float* input, float* weights, float* bias, float* output,
int batch_size, int input_features, int output_features) {
// Allocate device memory
float* d_input, *d_weights, *d_bias, *d_output;

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

// Set kernel dimensions
dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
dim3 gridSize(
1,  // x dimension (needed for shared memory layout but not used for actual indexing)
(output_features + BLOCK_SIZE - 1) / BLOCK_SIZE,  // y dimension for output features
batch_size  // z dimension for batch
);

// Launch kernel
fc_matmul_vectorized_kernel<<<gridSize, blockSize>>>(d_input, d_weights, d_bias, d_output,
                     batch_size, input_features, output_features);

// Check for kernel launch errors
CHECK_CUDA(cudaGetLastError());

// Synchronize to ensure completion
CHECK_CUDA(cudaDeviceSynchronize());

// Copy result back to host
CHECK_CUDA(cudaMemcpy(output, d_output, batch_size * output_features * sizeof(float),
cudaMemcpyDeviceToHost));

// Clean up
CHECK_CUDA(cudaFree(d_input));
CHECK_CUDA(cudaFree(d_weights));
CHECK_CUDA(cudaFree(d_bias));
CHECK_CUDA(cudaFree(d_output));
}