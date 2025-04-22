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
