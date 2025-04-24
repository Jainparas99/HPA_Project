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

#define TILE_WIDTH 8
#define KERNEL_SIZE 3

__global__ void conv2d_tiled_kernel(
    const float* __restrict__ input,    // [N, C, H, W]
    const float* __restrict__ weights,  // [K, C, R, S]
    const float* __restrict__ bias,     // [K]
    float* output,                      // [N, K, P, Q]
    int N, int C, int H, int W,
    int K, int R, int S, int P, int Q
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int out_x = blockIdx.x * TILE_WIDTH + tx;
    int out_y = blockIdx.y * TILE_WIDTH + ty;
    int k = blockIdx.z % K;
    int n = blockIdx.z / K;

    const int TILE_PAD = KERNEL_SIZE / 2;
    const int TILE_SIZE = TILE_WIDTH + 2 * TILE_PAD;

    // Use dynamic shared memory
    extern __shared__ float tile[];

    if (out_x >= P || out_y >= Q || n >= N || k >= K) return;

    float acc = bias[k];

    for (int c = 0; c < C; ++c) {
        // Global input location for this thread
        int input_x = out_x - TILE_PAD;
        int input_y = out_y - TILE_PAD;

        // Each thread loads into shared memory
        int shared_x = tx + TILE_PAD;
        int shared_y = ty + TILE_PAD;

        // Load center pixel
        if (input_x >= 0 && input_x < H && input_y >= 0 && input_y < W) {
            tile[shared_y * TILE_SIZE + shared_x] = input[((n * C + c) * H + input_x) * W + input_y];
        } else {
            tile[shared_y * TILE_SIZE + shared_x] = 0.0f;
        }

        // Load halo edges (top, left, right, bottom) as needed
        if (tx < TILE_PAD) {
            // Left
            int ix = input_x - TILE_PAD;
            int iy = input_y;
            tile[shared_y * TILE_SIZE + (shared_x - TILE_PAD)] =
                (ix >= 0 && iy >= 0 && iy < W) ? input[((n * C + c) * H + ix) * W + iy] : 0.0f;
            // Right
            ix = input_x + TILE_WIDTH;
            tile[shared_y * TILE_SIZE + (shared_x + TILE_WIDTH)] =
                (ix < H && iy >= 0 && iy < W) ? input[((n * C + c) * H + ix) * W + iy] : 0.0f;
        }

        if (ty < TILE_PAD) {
            // Top
            int ix = input_x;
            int iy = input_y - TILE_PAD;
            tile[(shared_y - TILE_PAD) * TILE_SIZE + shared_x] =
                (iy >= 0 && ix >= 0 && ix < H) ? input[((n * C + c) * H + ix) * W + iy] : 0.0f;
            // Bottom
            iy = input_y + TILE_WIDTH;
            tile[(shared_y + TILE_WIDTH) * TILE_SIZE + shared_x] =
                (iy < W && ix >= 0 && ix < H) ? input[((n * C + c) * H + ix) * W + iy] : 0.0f;
        }

        __syncthreads();

        // Apply kernel
       for (int r = 0; r < R; ++r) {
        for (int s = 0; s < S; ++s) {
            int tile_r = shared_y + s - TILE_PAD;
            int tile_c = shared_x + r - TILE_PAD;

            if (tile_r >= 0 && tile_r < TILE_SIZE && tile_c >= 0 && tile_c < TILE_SIZE) {
                float val = tile[tile_r * TILE_SIZE + tile_c];
                float w = weights[((k * C + c) * R + r) * S + s];
                acc += val * w;
            }
        }
}

        __syncthreads(); // prepare tile for next channel
    }

    if (out_x < P && out_y < Q) {
        int out_idx = ((n * K + k) * P + out_x) * Q + out_y;
        output[out_idx] = acc;
    }
}

void launch_conv2d_tiled(
    float* input, float* weight, float* bias, float* output,
    int N, int C, int H, int W,
    int K, int R, int S, int P, int Q) {

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((P + TILE_WIDTH - 1) / TILE_WIDTH,
                 (Q + TILE_WIDTH - 1) / TILE_WIDTH,
                 N * K);

    int pad = KERNEL_SIZE / 2;
    int tile_size = TILE_WIDTH + 2 * pad;
    size_t shared_mem_size = sizeof(float) * tile_size * tile_size;

    std::cout << "Launching tiled kernel with shared memory: "
              << shared_mem_size << " bytes\n";

    conv2d_tiled_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        input, weight, bias, output, N, C, H, W, K, R, S, P, Q
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
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
// // Safe 
// __global__ void conv2d_tiled_safe(
//     const float* __restrict__ input,
//     const float* __restrict__ weights,
//     const float* __restrict__ bias,
//     float* output,
//     int N, int C, int H, int W,
//     int K, int R, int S, int P, int Q
// ) {
//     int n = blockIdx.z;
//     int k = blockIdx.y;
//     int p = blockIdx.x * blockDim.x + threadIdx.x;
//     int q = threadIdx.y;

//     if (p >= P || q >= Q) return;

//     float acc = bias[k];

//     for (int c = 0; c < C; ++c) {
//         for (int r = 0; r < R; ++r) {
//             for (int s = 0; s < S; ++s) {
//                 int h_in = p + r - 1;
//                 int w_in = q + s - 1;
//                 if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
//                     float val = input[((n * C + c) * H + h_in) * W + w_in];
//                     float w = weights[((k * C + c) * R + r) * S + s];
//                     acc += val * w;
//                 }
//             }
//         }
//     }

//     output[((n * K + k) * P + p) * Q + q] = acc;
// }

// void launch_conv2d_tiled_safe(
//     float* input, float* weight, float* bias, float* output,
//     int N, int C, int H, int W,
//     int K, int R, int S,
//     int P, int Q) {

//     dim3 blockDim(16, 16);
//     dim3 gridDim((P + 15) / 16, K, N);
//     size_t shmem_size = 0;

//     conv2d_tiled_safe<<<gridDim, blockDim, shmem_size>>>(
//         input, weight, bias, output, N, C, H, W, K, R, S, P, Q);

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
//     }
// }