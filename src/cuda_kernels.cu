#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include<iostream>

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
                int h_in = h_out + r - 1;  // padding = 1
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

//-------------------------------------------------------------------------------------------------------------------------



#define TILE_WIDTH 16

__global__ void conv2d_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S, int P, int Q
) {
    extern __shared__ float sh_input[];  // Size = C × (TILE+2) × (TILE+2)

    const int tile_size = TILE_WIDTH + 2;

    int n = blockIdx.z;
    int k = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int h_out = blockIdx.x * TILE_WIDTH + tx;
    int w_out = ty;

    if (h_out >= P || w_out >= Q)
        return;

    float acc = bias[k];

    // === DEBUG (optional) ===
    if (blockIdx.x == 0 && tx == 0 && ty == 0)
        printf("⛓️  C=%d TILE=%d SharedMemBytes=%d\n", C, TILE_WIDTH, C * tile_size * tile_size * (int)sizeof(float));

    // Load input tile into shared memory
    for (int c = 0; c < C; ++c) {
        for (int r = tx; r < tile_size; r += blockDim.x) {
            for (int s = ty; s < tile_size; s += blockDim.y) {
                int h_in = blockIdx.x * TILE_WIDTH + r - R / 2;
                int w_in = s - S / 2;

                float val = 0.0f;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int input_idx = ((n * C + c) * H + h_in) * W + w_in;
                    val = input[input_idx];
                }

                int sh_idx = (c * tile_size + r) * tile_size + s;
                sh_input[sh_idx] = val;
            }
        }
    }

    __syncthreads();  // Wait for all tiles to be loaded

    // Perform convolution using shared memory
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int sh_r = tx + r;
                int sh_s = ty + s;

                if (sh_r < tile_size && sh_s < tile_size) {
                    int sh_idx = (c * tile_size + sh_r) * tile_size + sh_s;
                    int weight_idx = ((k * C + c) * R + r) * S + s;

                    acc += sh_input[sh_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = ((n * K + k) * P + h_out) * Q + w_out;
    output[output_idx] = acc;
}



void launch_conv2d_tiled(float* d_input, float* d_weight, float* d_bias, float* d_output,
                         int N, int C, int H, int W,
                         int K, int R, int S, int P, int Q) {
    size_t shared_mem_size = sizeof(float) * C * (TILE_WIDTH + 2) * (TILE_WIDTH + 2);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((P + TILE_WIDTH - 1) / TILE_WIDTH, K, N);                            
    std::cout << "Shared memory size: " << shared_mem_size << " bytes" << std::endl;

    conv2d_tiled_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        d_input, d_weight, d_bias, d_output,
        N, C, H, W, K, R, S, P, Q
    );

    cudaDeviceSynchronize();
}