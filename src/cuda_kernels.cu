#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include<iostream>
#define CUDA_DEBUG


__global__ void conv2d_naive_kernel(
    const float* input,     // [N, C, H, W]
    const float* weight,    // [K, C, R, S]
    const float* bias,      // [K]
    float* output,          // [N, K, P, Q]
    int N, int C, int H, int W,
    int K, int R, int S,
    int P, int Q            // Output height and width
) {
    int n = blockIdx.z;
    int k = blockIdx.y;
    int h_out = blockIdx.x * blockDim.x + threadIdx.x;
    int w_out = threadIdx.y;

    if (h_out >= P || w_out >= Q) return;

    float acc = bias[k];  // Start with bias

    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int h_in = h_out + r - 1; 
                int w_in = w_out + s - 1;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int input_idx = ((n * C + c) * H + h_in) * W + w_in;
                    int weight_idx = ((k * C + c) * R + r) * S + s;
                    acc += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = ((n * K + k) * P + h_out) * Q + w_out;
    output[output_idx] = acc;
}

// Host function to launch the kernel
void launch_conv2d_naive(
    float* input, float* weight, float* bias, float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int P, int Q) {
    dim3 blockDim(16, 16); // 16x16 output tiles
    dim3 gridDim((P + 15) / 16, K, N);  // One block per output channel per sample

    conv2d_naive_kernel<<<gridDim, blockDim>>>(
        input, weight, bias, output,
        N, C, H, W, K, R, S, P, Q
    );

    cudaDeviceSynchronize();  // Ensure kernel is done before returning
}

#define TILE_WIDTH 16
#define PAD 1

__global__ void conv2d_tiled_single_channel_kernel(
    const float* __restrict__ input,    // [N, C, H, W]
    const float* __restrict__ weight,   // [K, C, R, S]
    const float* __restrict__ bias,     // [K]
    float* output,                      // [N, K, P, Q]
    int N, int C, int H, int W,
    int K, int R, int S, int P, int Q
) {
    extern __shared__ float sh_input[];  // Shape: (TILE+2)^2 per thread block

    const int tile_size = TILE_WIDTH + 2 * PAD;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int block_x = blockIdx.x;  // over P
    const int block_y = blockIdx.y;  // over Q
    const int nk = blockIdx.z;

    const int k = nk % K;
    const int n = nk / K;

    const int h_out = block_x * TILE_WIDTH + tx;
    const int w_out = block_y * TILE_WIDTH + ty;

    if (h_out >= P || w_out >= Q) return;

    float acc = (k < K) ? bias[k] : 0.0f;

    // Loop over input channels, one at a time
    for (int c = 0; c < C; ++c) {
        // Load tile for this channel into shared memory
        for (int i = tx; i < tile_size; i += blockDim.x) {
            for (int j = ty; j < tile_size; j += blockDim.y) {
                int h_in = block_x * TILE_WIDTH + i - PAD;
                int w_in = block_y * TILE_WIDTH + j - PAD;

                float val = 0.0f;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int input_idx = ((n * C + c) * H + h_in) * W + w_in;
                    val = input[input_idx];
                }

                int sh_idx = i * tile_size + j;
                sh_input[sh_idx] = val;
            }
        }

        __syncthreads();  // Wait for all tiles to be loaded

        // Convolve using shared memory
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int sh_r = tx + r;
                int sh_s = ty + s;
                if (sh_r < tile_size && sh_s < tile_size) {
                    int sh_idx = sh_r * tile_size + sh_s;
                    int weight_idx = ((k * C + c) * R + r) * S + s;
                    acc += sh_input[sh_idx] * weight[weight_idx];
                }
            }
        }

        __syncthreads();  // Clean before next channel load
    }

    // Write output
    int out_idx = ((n * K + k) * P + h_out) * Q + w_out;
    output[out_idx] = acc;
}





void launch_conv2d_tiled_single_channel(
    float* d_input, float* d_weight, float* d_bias, float* d_output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int P, int Q
) {
    
    const int tile_size = TILE_WIDTH + 2 * PAD;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((P + TILE_WIDTH - 1) / TILE_WIDTH,
                 (Q + TILE_WIDTH - 1) / TILE_WIDTH,
                 N * K);

    size_t shared_mem_size = sizeof(float) * tile_size * tile_size;

    std::cout << "Launching tiled-single-channel kernel...\n";
    std::cout << "Shared memory size: " << shared_mem_size << " bytes\n";

    conv2d_tiled_single_channel_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        d_input, d_weight, d_bias, d_output,
        N, C, H, W, K, R, S, P, Q
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Error (tiled single-channel): " << cudaGetErrorString(err) << std::endl;
    }

    // CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void conv2d_tiled_coarsened_kernel(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int P, int Q) {

    const int COARSENING = 2;
    // const int TILE_WIDTH = 8;  // Per block

    int n = blockIdx.z;
    int k = blockIdx.y;

    int tile_h = blockIdx.x * TILE_WIDTH;
    int tile_w = threadIdx.y * COARSENING;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int h_out = tile_h + tx * COARSENING;
    int w_out = tile_w;

    float acc[COARSENING][COARSENING] = {0};

    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                for (int i = 0; i < COARSENING; ++i) {
                    for (int j = 0; j < COARSENING; ++j) {
                        int h_in = h_out + i + r - 1;  // padding=1
                        int w_in = w_out + j + s - 1;
                        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                            int input_idx = ((n * C + c) * H + h_in) * W + w_in;
                            int weight_idx = ((k * C + c) * R + r) * S + s;
                            acc[i][j] += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < COARSENING; ++i) {
        for (int j = 0; j < COARSENING; ++j) {
            int h = h_out + i;
            int w = w_out + j;
            if (h < P && w < Q) {
                int out_idx = ((n * K + k) * P + h) * Q + w;
                output[out_idx] = acc[i][j] + bias[k];
            }
        }
    }
}

void launch_conv2d_tiled_coarsened(
    float* input, float* weight, float* bias, float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int P, int Q) {

    const int COARSENING = 2;
    // const int TILE_WIDTH = 8;  // Block size

    dim3 blockDim(TILE_WIDTH / COARSENING, TILE_WIDTH / COARSENING);
    dim3 gridDim((P + TILE_WIDTH - 1) / TILE_WIDTH, K, N);

    conv2d_tiled_coarsened_kernel<<<gridDim, blockDim>>>(
        input, weight, bias, output, N, C, H, W, K, R, S, P, Q);

    cudaDeviceSynchronize();
}
// Safe 
__global__ void conv2d_tiled_safe(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S, int P, int Q
) {
    int n = blockIdx.z;
    int k = blockIdx.y;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int q = threadIdx.y;

    if (p >= P || q >= Q) return;

    float acc = bias[k];

    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int h_in = p + r - 1;
                int w_in = q + s - 1;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    float val = input[((n * C + c) * H + h_in) * W + w_in];
                    float w = weights[((k * C + c) * R + r) * S + s];
                    acc += val * w;
                }
            }
        }
    }

    output[((n * K + k) * P + p) * Q + q] = acc;
}

void launch_conv2d_tiled_safe(
    float* input, float* weight, float* bias, float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int P, int Q) {

    dim3 blockDim(16, 16);
    dim3 gridDim((P + 15) / 16, K, N);
    size_t shmem_size = 0;

    conv2d_tiled_safe<<<gridDim, blockDim, shmem_size>>>(
        input, weight, bias, output, N, C, H, W, K, R, S, P, Q);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
}