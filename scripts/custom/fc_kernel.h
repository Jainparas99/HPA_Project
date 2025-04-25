// scripts/custom/fc_kernel.h
#pragma once

// Raw pointers must match your kernel impl:
void fc_forward_cuda(
  const float* in,
  const float* wt,
  const float* bias,
  float*       out,
  int          N,
  int          D_in,
  int          D_out
);

void fc_layer_custom_tiled(const float* input, const float* weights, const float* bias, float* output,
                            int batch_size, int input_features, int output_features);