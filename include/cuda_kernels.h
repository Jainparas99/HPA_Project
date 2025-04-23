#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H
#pragma once

void launch_conv2d_naive(float* d_input, float* d_weight, float* d_bias, float* d_output,
                         int N, int C, int H, int W,
                         int K, int R, int S, int P, int Q);

#endif

void launch_conv2d_tiled(float* d_input, float* d_weight, float* d_bias, float* d_output,
    int N, int C, int H, int W,
    int K, int R, int S, int P, int Q);

void launch_conv2d_tiled_coarsened(
    float* input, float* weight, float* bias, float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int P, int Q);