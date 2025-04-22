#include <cuda.h>
#include <cuda_runtime.h>

// A naive CUDA kernel: out[i,j] = sum_k in[i,k] * wt[j,k] + bias[j]
__global__ void fc_forward_kernel(
    const float* __restrict__ in,
    const float* __restrict__ wt,
    const float* __restrict__ bias,
    float*       __restrict__ out,
    int N,       // batch size
    int D_in,    // input dim
    int D_out    // output dim
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * D_out;
  if (idx >= total) return;
  int i = idx / D_out;    // batch index
  int j = idx % D_out;    // output feature index

  const float* in_row = in + i * D_in;
  const float* wt_row = wt + j * D_in;

  float acc = 0.f;
  #pragma unroll
  for (int k = 0; k < D_in; ++k) {
    acc += in_row[k] * wt_row[k];
  }
  out[idx] = acc + bias[j];
}

void fc_forward_cuda(
    const float* in,
    const float* wt,
    const float* bias,
    float*       out,
    int N, int D_in, int D_out
) {
  const int threads = 256;
  const int blocks  = (N*D_out + threads - 1) / threads; // bs*
  fc_forward_kernel<<<blocks,threads>>>(in, wt, bias, out, N, D_in, D_out);
  cudaDeviceSynchronize();
}
