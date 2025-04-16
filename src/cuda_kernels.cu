#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>

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