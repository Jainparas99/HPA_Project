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
    extern __shared__ float sh_input[];

    const int tile_size = TILE_WIDTH + 2;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tile_x = blockIdx.x;
    const int tile_y = blockIdx.y;

    const int h_out = tile_x * TILE_WIDTH + tx;
    const int w_out = tile_y * TILE_WIDTH + ty;

    const int nk = blockIdx.z;
    const int k = nk % K;
    const int n = nk / K;

    if (n >= N || k >= K || h_out >= P || w_out >= Q)
        return;

    float acc = (k < K) ? bias[k] : 0.0f;
    __syncthreads(); 

    int tid = ty * blockDim.x + tx;
    int shared_size = C * tile_size * tile_size;

    for (int i = tid; i < shared_size; i += blockDim.x * blockDim.y)
        sh_input[i] = 0.0f;

    __syncthreads();


    // Load input tile into shared memory
    for (int c = 0; c < C; ++c) {
        for (int i = tx; i < tile_size; i += blockDim.x) {
            for (int j = ty; j < tile_size; j += blockDim.y) {
                if (i < tile_size && j < tile_size) {
                    int h_in = tile_x * TILE_WIDTH + i - R / 2;
                    int w_in = tile_y * TILE_WIDTH + j - S / 2;

                    float val = 0.0f;
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        int input_idx = ((n * C + c) * H + h_in) * W + w_in;
                        if (input_idx < N * C * H * W)
                            val = input[input_idx];
                    }

                    int sh_idx = (c * tile_size + i) * tile_size + j;
                    if (sh_idx < C * tile_size * tile_size)
                        sh_input[sh_idx] = val;
                }
            }
        }
    }

    __syncthreads();

    // Perform convolution using shared memory
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int sh_r = tx + r;
                int sh_s = ty + s;
                if (sh_r < tile_size && sh_s < tile_size) {
                    int sh_idx = (c * tile_size + sh_r) * tile_size + sh_s;
                    int weight_idx = ((k * C + c) * R + r) * S + s;

                    if (sh_idx < C * tile_size * tile_size &&
                        weight_idx < K * C * R * S)
                        acc += sh_input[sh_idx] * weight[weight_idx];
                }
            }
        }
    }

    int out_idx = ((n * K + k) * P + h_out) * Q + w_out;
    if (out_idx < N * K * P * Q)
        output[out_idx] = acc;
}




void launch_conv2d_tiled(float* d_input, float* d_weight, float* d_bias, float* d_output,
    int N, int C, int H, int W,
    int K, int R, int S, int P, int Q) {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim((P + TILE_WIDTH - 1) / TILE_WIDTH,
                     (Q + TILE_WIDTH - 1) / TILE_WIDTH,
                     N * K);
        size_t shared_mem_size = sizeof(float) * C * (TILE_WIDTH + 2) * (TILE_WIDTH + 2);
        

std::cout << "Shared memory size: " << shared_mem_size << " bytes" << std::endl;

conv2d_tiled_kernel<<<gridDim, blockDim, shared_mem_size>>>(
d_input, d_weight, d_bias, d_output,
N, C, H, W, K, R, S, P, Q
);

cudaError_t err = cudaGetLastError();
if (err != cudaSuccess)
std::cerr << "CUDA Launch Error: " << cudaGetErrorString(err) << std::endl;

cudaDeviceSynchronize();
cudaError_t syncErr = cudaGetLastError();
if (syncErr != cudaSuccess)
std::cerr << "CUDA Sync Error: " << cudaGetErrorString(syncErr) << std::endl;
}

