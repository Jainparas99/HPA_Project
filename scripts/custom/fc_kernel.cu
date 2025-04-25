#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#define TILE_SIZE 16
// Macro to check CUDA errors
#define CHECK_CUDA(x)                                                                \
  {                                                                                  \
    cudaError_t err = x;                                                             \
    if (err != cudaSuccess) {                                                         \
      fprintf(stderr,                                                                 \
              "Error: %s in %s at %s:%d\n", cudaGetErrorString(err), __func__, __FILE__, \
              __LINE__);                                                              \
      exit(1);                                                                       \
    }                                                                                \
  }

// A naive CUDA kernel: out[i,j] = sum_k in[i,k] * wt[j,k] + bias[j]
__global__ void fc_forward_kernel(
    const float* __restrict__ in,
    const float* __restrict__ wt,
    const float* __restrict__ bias,
    float* __restrict__ out,
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
    float* out,
    int N, int D_in, int D_out
) {
  const int threads = 256;
  const int blocks  = (N*D_out + threads - 1) / threads; // bs*
  fc_forward_kernel<<<blocks,threads>>>(in, wt, bias, out, N, D_in, D_out);
  cudaDeviceSynchronize();
}

/**
* Tiled matrix multiplication kernel using shared memory
* Uses shared memory to reduce global memory accesses
* Each thread block computes a tile of output elements
*/


__global__ void fc_matmul_tiled_kernel(const float* input, const float* weights, const float* bias, float* output,
                                        int batch_size, int input_features, int output_features) {
  // Shared memory for tiles
  __shared__ float shared_input[TILE_SIZE];
  __shared__ float shared_weights[TILE_SIZE][TILE_SIZE];

  // Calculate indices
  int b = blockIdx.z;                                   // Batch index
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // Output feature index
  int col = threadIdx.x;                                   // Input tile index

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
void fc_layer_custom_tiled(const float* input, const float* weights, const float* bias, float* output,
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
    1,                                                      // x dimension
    (output_features + TILE_SIZE - 1) / TILE_SIZE,  // y dimension for output features
    batch_size                                         // z dimension for batch
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
