#include "host_utils.h"
#include "cuda_kernels.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <utility>
#include <cstring>
#include <memory>

// CPU implementation of FC layer for validation
void fc_layer_cpu(float* input, float* weights, float* bias, float* output,
                  int batch_size, int input_features, int output_features) {
    
    // Compute matrix multiplication (output = input * weights^T)
    for (int n = 0; n < batch_size; n++) {
        for (int o = 0; o < output_features; o++) {
            float sum = 0.0f;
            
            // Matrix multiplication
            for (int i = 0; i < input_features; i++) {
                sum += input[n * input_features + i] * weights[o * input_features + i];
            }
            
            // Add bias
            sum += bias[o];
            
            // Apply ReLU activation
            output[n * output_features + o] = std::max(0.0f, sum);
        }
    }
}

// Utility to generate random data
void generate_random_data(float* data, size_t size, float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    
    for (size_t i = 0; i < size; i++) {
        data[i] = dist(gen);
    }
}

// Check if two arrays are approximately equal
bool arrays_equal(float* arr1, float* arr2, size_t size, float tolerance) {
    for (size_t i = 0; i < size; i++) {
        if (std::abs(arr1[i] - arr2[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << arr1[i] << " vs " << arr2[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Test function for FC layer implementation correctness
void test_fc_layer_correctness(int batch_size, int input_features, int output_features) {
    std::cout << "Testing FC layer correctness..." << std::endl;
    
    // Allocate memory for input, weights, bias, and outputs
    float* input = new float[batch_size * input_features];
    float* weights = new float[output_features * input_features];
    float* bias = new float[output_features];
    float* output_gpu = new float[batch_size * output_features];
    float* output_cpu = new float[batch_size * output_features];
    
    // Initialize with random data
    generate_random_data(input, batch_size * input_features);
    generate_random_data(weights, output_features * input_features);
    generate_random_data(bias, output_features);
    
    // Time CPU implementation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    // Run CPU implementation
    fc_layer_cpu(input, weights, bias, output_cpu, batch_size, input_features, output_features);
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    double cpu_time = cpu_duration.count();
    
    // Calculate CPU GFLOPS
    // Operations: 2 * batch_size * output_features * input_features (multiply-add)
    //             + batch_size * output_features (bias addition)
    //             + batch_size * output_features (ReLU comparison)
    double ops = 2.0 * batch_size * output_features * input_features +
                 2.0 * batch_size * output_features;
    double cpu_gflops = (ops / (cpu_time / 1000.0)) / 1e9;
    
    std::cout << "CPU implementation:" << std::endl;
    std::cout << "  Time: " << cpu_time << " ms" << std::endl;
    std::cout << "  Performance: " << cpu_gflops << " GFLOPS" << std::endl;
    
    // Time GPU implementation with CUDA events for accuracy
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    // Run GPU implementation
    fc_layer_cublas(input, weights, bias, output_gpu, batch_size, input_features, output_features);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);
    float gpu_time = gpu_milliseconds;
    
    // Calculate GPU GFLOPS
    double gpu_gflops = (ops / (gpu_time / 1000.0)) / 1e9;
    
    std::cout << "GPU implementation:" << std::endl;
    std::cout << "  Time: " << gpu_time << " ms" << std::endl;
    std::cout << "  Performance: " << gpu_gflops << " GFLOPS" << std::endl;
    std::cout << "  Speedup (CPU/GPU): " << cpu_time / gpu_time << "x" << std::endl;
    
    // Compare results
    bool equal = arrays_equal(output_cpu, output_gpu, batch_size * output_features);
    
    if (equal) {
        std::cout << "FC layer implementation PASSED correctness test!" << std::endl;
    } else {
        std::cout << "FC layer implementation FAILED correctness test!" << std::endl;
    }
    
    // Clean up
    delete[] input;
    delete[] weights;
    delete[] bias;
    delete[] output_gpu;
    delete[] output_cpu;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Measure performance of FC layer implementation
float measure_fc_layer_performance(int batch_size, int input_features, int output_features, int iterations) {
    std::cout << "Measuring FC layer performance..." << std::endl;
    
    // Allocate memory for input, weights, bias, and output
    float* input = new float[batch_size * input_features];
    float* weights = new float[output_features * input_features];
    float* bias = new float[output_features];
    float* output = new float[batch_size * output_features];
    
    // Initialize with random data
    generate_random_data(input, batch_size * input_features);
    generate_random_data(weights, output_features * input_features);
    generate_random_data(bias, output_features);
    
    // CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm-up run
    fc_layer_cublas(input, weights, bias, output, batch_size, input_features, output_features);
    
    // Timing measurement
    cudaEventRecord(start);
    
    for (int i = 0; i < iterations; i++) {
        fc_layer_cublas(input, weights, bias, output, batch_size, input_features, output_features);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / iterations;
    
    // Calculate GFLOPS
    // For each output element: input_features multiplications and input_features-1 additions
    // Plus 1 bias addition and 1 ReLU comparison per output element
    double operations = batch_size * output_features * (2.0 * input_features + 2.0);
    double gflops = (operations / (avg_time / 1000.0)) / 1e9;
    
    std::cout << "FC Layer (" << batch_size << "x" << input_features 
              << " -> " << batch_size << "x" << output_features << "):" << std::endl;
    std::cout << "  Average time: " << avg_time << " ms over " << iterations << " iterations" << std::endl;
    std::cout << "  Throughput: " << gflops << " GFLOPS" << std::endl;
    
    // Clean up
    delete[] input;
    delete[] weights;
    delete[] bias;
    delete[] output;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return avg_time;
}

/**
 * Calculates and displays the Compute to Global Memory Access Ratio (CGMA) for different
 * FC layer implementations, which helps determine if a kernel is compute-bound or memory-bound.
 * 
 * @param batch_size Number of samples in the batch
 * @param input_features Number of input features (neurons)
 * @param output_features Number of output features (neurons)
 */
void analyze_cgma(int batch_size, int input_features, int output_features) {
    std::cout << "\n=== CGMA Analysis for FC Layer ===" << std::endl;
    std::cout << "Layer size: " << batch_size << "x" << input_features << " -> " 
              << batch_size << "x" << output_features << std::endl;
    
    // Calculate total arithmetic operations
    // Matrix multiply: 2*M*N*K floating-point operations (multiply-add pairs)
    // In our case: M=batch_size, N=output_features, K=input_features
    // Plus bias addition and ReLU: 2 operations per output element
    uint64_t arithmetic_ops = batch_size * output_features * (
        2 * input_features +  // Multiply-add for matrix multiplication
        2                     // Bias addition and ReLU comparison
    );
    
    std::cout << "Total arithmetic operations: " << arithmetic_ops << std::endl;
    
    // 1. FC layer using cuBLAS - Global memory accesses
    // - Read input matrix: batch_size * input_features elements
    // - Read weights matrix: output_features * input_features elements
    // - Read bias vector: output_features elements
    // - Write output matrix: batch_size * output_features elements
    uint64_t cublas_mem_accesses = 
        batch_size * input_features +            // Input matrix read
        output_features * input_features +       // Weight matrix read
        output_features +                        // Bias vector read
        batch_size * output_features;            // Output matrix write
    
    double cublas_cgma = static_cast<double>(arithmetic_ops) / cublas_mem_accesses;
    
    std::cout << "\n1. Standard cuBLAS implementation:" << std::endl;
    std::cout << "  Global memory accesses: " << cublas_mem_accesses << " elements" 
              << " (" << (cublas_mem_accesses * sizeof(float) / 1e6) << " MB)" << std::endl;
    std::cout << "  CGMA ratio: " << cublas_cgma << " operations per memory access" << std::endl;
    
    // 2. Naive custom implementation - Global memory accesses
    // In a naive implementation, each thread independently accesses global memory
    // - Each output element requires reading input_features elements from both input and weights
    uint64_t naive_mem_accesses = batch_size * output_features * (
        input_features +      // Read each input element once per output element
        input_features +      // Read each weight element once per output element
        1 +                   // Read bias once per output element
        1                     // Write output once
    );
    
    double naive_cgma = static_cast<double>(arithmetic_ops) / naive_mem_accesses;
    
    std::cout << "\n2. Naive custom implementation:" << std::endl;
    std::cout << "  Global memory accesses: " << naive_mem_accesses << " elements" 
              << " (" << (naive_mem_accesses * sizeof(float) / 1e6) << " MB)" << std::endl;
    std::cout << "  CGMA ratio: " << naive_cgma << " operations per memory access" << std::endl;
    
    // 3. Tiled implementation with shared memory - Global memory accesses
    // With tiling, each input and weight element is only read once from global memory
    uint64_t tiled_global_mem_accesses = 
        batch_size * input_features +            // Input matrix read once
        output_features * input_features +       // Weight matrix read once
        output_features +                        // Bias vector read once
        batch_size * output_features;            // Output matrix write once
    
    double tiled_cgma = static_cast<double>(arithmetic_ops) / tiled_global_mem_accesses;
    
    std::cout << "\n3. Tiled implementation with shared memory:" << std::endl;
    std::cout << "  Global memory accesses: " << tiled_global_mem_accesses << " elements" 
              << " (" << (tiled_global_mem_accesses * sizeof(float) / 1e6) << " MB)" << std::endl;
    std::cout << "  CGMA ratio: " << tiled_cgma << " operations per memory access" << std::endl;
    
    // Determine if kernel is compute-bound or memory-bound
    // For RTX 3080 (adjust for your GPU):
    int compute_units = 68;     // 68 SMs
    int cores_per_cu = 128;     // ~128 CUDA cores per SM
    double clock_ghz = 1.71;    // 1.71 GHz clock speed
    double peak_flops = compute_units * cores_per_cu * 2 * clock_ghz * 1e9; // 2 FLOPs per FMA
    
    // Memory bandwidth: 760 GB/s for RTX 3080
    double memory_bandwidth = 760e9;  // bytes per second
    
    // Required CGMA to be compute-bound = (peak_flops) / (memory_bandwidth / sizeof(float))
    double required_cgma = peak_flops / (memory_bandwidth / sizeof(float));
    
    std::cout << "\nTheoretical analysis:" << std::endl;
    std::cout << "  Peak FP32 performance: " << (peak_flops / 1e12) << " TFLOPS" << std::endl;
    std::cout << "  Memory bandwidth: " << (memory_bandwidth / 1e9) << " GB/s" << std::endl;
    std::cout << "  Required CGMA to be compute-bound: " << required_cgma << std::endl;
    
    // Determine if each implementation is compute-bound or memory-bound
    std::cout << "\nImplementation characteristics:" << std::endl;
    std::cout << "  Standard cuBLAS: " << (cublas_cgma >= required_cgma ? "compute-bound" : "memory-bound") 
              << std::endl;
    std::cout << "  Naive custom: " << (naive_cgma >= required_cgma ? "compute-bound" : "memory-bound") 
              << std::endl;
    std::cout << "  Tiled shared memory: " << (tiled_cgma >= required_cgma ? "compute-bound" : "memory-bound") 
              << std::endl;
}

/**
 * Displays top-k model predictions from the output tensor.
 * 
 * @param output Pointer to the output tensor data
 * @param batch_size Number of samples in the batch
 * @param num_classes Number of output classes
 * @param k Number of top predictions to display
 */
void display_predictions(float* output, int batch_size, int num_classes, int k) {
    // Load ImageNet class labels if available, otherwise use indices
    std::vector<std::string> class_labels;
    
    // Try to load class labels from file
    std::ifstream label_file("imagenet_classes.txt");
    if (label_file.is_open()) {
        std::string line;
        while (std::getline(label_file, line)) {
            class_labels.push_back(line);
        }
        label_file.close();
    } else {
        // If file not found, use class indices
        for (int i = 0; i < num_classes; i++) {
            class_labels.push_back("Class " + std::to_string(i));
        }
        std::cout << "Note: ImageNet class labels file not found. Using class indices." << std::endl;
    }
    
    // For each item in the batch
    for (int b = 0; b < batch_size; b++) {
        std::cout << "Predictions for batch item " << b << ":" << std::endl;
        
        // Create pairs of (probability, class index)
        std::vector<std::pair<float, int>> probabilities;
        for (int i = 0; i < num_classes; i++) {
            probabilities.push_back({output[b * num_classes + i], i});
        }
        
        // Sort by probability (descending)
        std::sort(probabilities.begin(), probabilities.end(), 
            [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Display top k predictions
        for (int i = 0; i < std::min(k, num_classes); i++) {
            int class_idx = probabilities[i].second;
            float prob = probabilities[i].first;
            std::cout << "  " << class_labels[class_idx] << ": " 
                      << std::fixed << std::setprecision(4) << prob << std::endl;
        }
        std::cout << std::endl;
    }
}

/**
 * Runs the VGG16 model prediction with either CPU or GPU implementation
 * and displays the prediction results and timing information.
 * 
 * @param session ONNX Runtime session object
 * @param use_gpu Whether to use GPU implementation (true) or CPU implementation (false)
 */
void run_model_prediction(Ort::Session& session, bool use_gpu) {
    std::cout << "\n=== Running VGG16 Model Prediction (" 
              << (use_gpu ? "GPU" : "CPU") << ") ===" << std::endl;
    
    // Get input and output names
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    
    // Get input and output shapes
    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    
    int batch_size = static_cast<int>(input_shape[0]);
    int channels = static_cast<int>(input_shape[1]);
    int height = static_cast<int>(input_shape[2]);
    int width = static_cast<int>(input_shape[3]);
    int num_classes = static_cast<int>(output_shape[1]);
    
    std::cout << "Input shape: " << batch_size << "x" << channels << "x" 
              << height << "x" << width << std::endl;
    std::cout << "Output shape: " << batch_size << "x" << num_classes << std::endl;
    
    // Allocate input and output tensors
    std::vector<float> input_tensor_values(batch_size * channels * height * width);
    std::vector<float> output_tensor_values(batch_size * num_classes);
    
    // Generate random data for testing (or load an actual image)
    generate_random_data(input_tensor_values.data(), input_tensor_values.size());
    
    // Create input tensor
    std::vector<int64_t> input_dims = {
        static_cast<int64_t>(batch_size), 
        static_cast<int64_t>(channels),
        static_cast<int64_t>(height), 
        static_cast<int64_t>(width)
    };
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_dims.data(), input_dims.size());
    
    // Names for input and output tensors
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};
    
    // Time the inference
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run inference
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    
    // Get pointer to output tensor data
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    
    // Copy results to output vector
    std::memcpy(output_tensor_values.data(), output_data, 
               output_tensor_values.size() * sizeof(float));
    
    std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
    
    // Display top predictions
    display_predictions(output_tensor_values.data(), batch_size, num_classes);
}

/**
 * Tests and benchmarks custom FC layer implementations against cuBLAS.
 * This function compares performance and accuracy of different implementations.
 *
 * @param batch_size Number of samples in the batch
 * @param input_features Number of input neurons
 * @param output_features Number of output neurons
 */
void test_custom_fc_implementations(int batch_size, int input_features, int output_features) {
    std::cout << "\n=== Testing Custom FC Layer Implementations ===" << std::endl;
    
    // Allocate memory for test data
    float* input = new float[batch_size * input_features];
    float* weights = new float[output_features * input_features];
    float* bias = new float[output_features];
    float* output_cublas = new float[batch_size * output_features];
    float* output_cublas_batched = new float[batch_size * output_features];
    float* output_naive = new float[batch_size * output_features];
    float* output_tiled = new float[batch_size * output_features];
    float* output_vectorized = new float[batch_size * output_features];
    
    // Initialize with random data
    generate_random_data(input, batch_size * input_features);
    generate_random_data(weights, output_features * input_features);
    generate_random_data(bias, output_features);
    
    std::cout << "Input features: " << input_features << ", Output features: " << output_features 
              << ", Batch size: " << batch_size << std::endl;
    
    // CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
    
    // Run cuBLAS implementation for reference
    std::cout << "\n1. cuBLAS implementation:" << std::endl;
    fc_layer_cublas(input, weights, bias, output_cublas, batch_size, input_features, output_features);
    
    // Warm-up run
    for (int i = 0; i < 3; i++) {
        fc_layer_cublas(input, weights, bias, output_cublas, batch_size, input_features, output_features);
    }
    
    // Timed runs
    cudaEventRecord(start);
    int iterations = 10;
    for (int i = 0; i < iterations; i++) {
        fc_layer_cublas(input, weights, bias, output_cublas, batch_size, input_features, output_features);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float cublas_time = milliseconds / iterations;
    
    // Calculate GFLOPS for cuBLAS
    // Operations: 2 * batch_size * output_features * input_features (GEMM)
    //            + batch_size * output_features (bias)
    //            + batch_size * output_features (ReLU)
    double ops = 2.0 * batch_size * output_features * input_features + 
                2.0 * batch_size * output_features;
    double cublas_gflops = (ops / (cublas_time / 1000.0)) / 1e9;
    
    std::cout << "  Average time: " << cublas_time << " ms over " << iterations << " iterations" << std::endl;
    std::cout << "  Performance: " << cublas_gflops << " GFLOPS" << std::endl;

    // Test batched cuBLAS implementation
    std::cout << "\n1b. Batched cuBLAS implementation:" << std::endl;
    fc_layer_cublas_batched(input, weights, bias, output_cublas_batched, batch_size, input_features, output_features);
    
    // Warm-up runs
    for (int i = 0; i < 3; i++) {
        fc_layer_cublas_batched(input, weights, bias, output_cublas_batched, batch_size, input_features, output_features);
    }
    
    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        fc_layer_cublas_batched(input, weights, bias, output_cublas_batched, batch_size, input_features, output_features);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float cublas_batched_time = milliseconds / iterations;
    
    double cublas_batched_gflops = (ops / (cublas_batched_time / 1000.0)) / 1e9;
    
    std::cout << "  Average time: " << cublas_batched_time << " ms over " << iterations << " iterations" << std::endl;
    std::cout << "  Performance: " << cublas_batched_gflops << " GFLOPS" << std::endl;
    std::cout << "  Performance relative to standard cuBLAS: " << (cublas_time / cublas_batched_time * 100.0f) << "%" << std::endl;
    
    // Check correctness
    bool cublas_batched_correct = arrays_equal(output_cublas, output_cublas_batched, batch_size * output_features);
    std::cout << "  Correctness: " << (cublas_batched_correct ? "PASS" : "FAIL") << std::endl;
    
    // Run naive custom implementation
    std::cout << "\n2. Naive custom implementation:" << std::endl;
    fc_layer_custom_naive(input, weights, bias, output_naive, batch_size, input_features, output_features);
    
    // Warm-up run
    for (int i = 0; i < 3; i++) {
        fc_layer_custom_naive(input, weights, bias, output_naive, batch_size, input_features, output_features);
    }
    
    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        fc_layer_custom_naive(input, weights, bias, output_naive, batch_size, input_features, output_features);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float naive_time = milliseconds / iterations;
    double naive_gflops = (ops / (naive_time / 1000.0)) / 1e9;
    
    std::cout << "  Average time: " << naive_time << " ms over " << iterations << " iterations" << std::endl;
    std::cout << "  Performance: " << naive_gflops << " GFLOPS" << std::endl;
    std::cout << "  Performance relative to cuBLAS: " << (cublas_time / naive_time * 100.0f) << "%" << std::endl;
    
    // Check correctness of naive implementation
    bool naive_correct = arrays_equal(output_cublas, output_naive, batch_size * output_features);
    std::cout << "  Correctness: " << (naive_correct ? "PASS" : "FAIL") << std::endl;
    
    // Run tiled custom implementation
    std::cout << "\n3. Tiled custom implementation:" << std::endl;
    fc_layer_custom_tiled(input, weights, bias, output_tiled, batch_size, input_features, output_features);
    
    // Warm-up run
    for (int i = 0; i < 3; i++) {
        fc_layer_custom_tiled(input, weights, bias, output_tiled, batch_size, input_features, output_features);
    }
    
    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        fc_layer_custom_tiled(input, weights, bias, output_tiled, batch_size, input_features, output_features);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float tiled_time = milliseconds / iterations;
    double tiled_gflops = (ops / (tiled_time / 1000.0)) / 1e9;
    
    std::cout << "  Average time: " << tiled_time << " ms over " << iterations << " iterations" << std::endl;
    std::cout << "  Performance: " << tiled_gflops << " GFLOPS" << std::endl;
    std::cout << "  Performance relative to cuBLAS: " << (cublas_time / tiled_time * 100.0f) << "%" << std::endl;
    
    // Check correctness of tiled implementation
    bool tiled_correct = arrays_equal(output_cublas, output_tiled, batch_size * output_features);
    std::cout << "  Correctness: " << (tiled_correct ? "PASS" : "FAIL") << std::endl;
    
    // Run vectorized custom implementation
    std::cout << "\n4. Vectorized custom implementation:" << std::endl;
    fc_layer_custom_vectorized(input, weights, bias, output_vectorized, batch_size, input_features, output_features);
    
    // Warm-up run
    for (int i = 0; i < 3; i++) {
        fc_layer_custom_vectorized(input, weights, bias, output_vectorized, batch_size, input_features, output_features);
    }
    
    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        fc_layer_custom_vectorized(input, weights, bias, output_vectorized, batch_size, input_features, output_features);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float vectorized_time = milliseconds / iterations;
    double vectorized_gflops = (ops / (vectorized_time / 1000.0)) / 1e9;
    
    std::cout << "  Average time: " << vectorized_time << " ms over " << iterations << " iterations" << std::endl;
    std::cout << "  Performance: " << vectorized_gflops << " GFLOPS" << std::endl;
    std::cout << "  Performance relative to cuBLAS: " << (cublas_time / vectorized_time * 100.0f) << "%" << std::endl;
    
    // Check correctness of vectorized implementation
    bool vectorized_correct = arrays_equal(output_cublas, output_vectorized, batch_size * output_features);
    std::cout << "  Correctness: " << (vectorized_correct ? "PASS" : "FAIL") << std::endl;
    
    // Summary table
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << "|--------------------|------------|-----------|-----------------|------------|" << std::endl;
    std::cout << "| Implementation     | Time (ms)  | GFLOPS    | Relative to     | Correct?   |" << std::endl;
    std::cout << "|                    |            |           | cuBLAS (%)      |            |" << std::endl;
    std::cout << "|--------------------|------------|-----------|-----------------|------------|" << std::endl;
    std::cout << "| cuBLAS             | " << std::setw(10) << std::fixed << std::setprecision(2) << cublas_time 
              << " | " << std::setw(9) << std::fixed << std::setprecision(2) << cublas_gflops 
              << " | " << std::setw(15) << "100.00" 
              << " | " << std::setw(10) << "Yes" << " |" << std::endl;
    
    std::cout << "| cuBLAS Batched     | " << std::setw(10) << std::fixed << std::setprecision(2) << cublas_batched_time 
              << " | " << std::setw(9) << std::fixed << std::setprecision(2) << cublas_batched_gflops 
              << " | " << std::setw(15) << std::fixed << std::setprecision(2) << (cublas_time / cublas_batched_time * 100.0f)
              << " | " << std::setw(10) << (cublas_batched_correct ? "Yes" : "No") << " |" << std::endl;
    
    std::cout << "| Naive Custom       | " << std::setw(10) << std::fixed << std::setprecision(2) << naive_time 
              << " | " << std::setw(9) << std::fixed << std::setprecision(2) << naive_gflops 
              << " | " << std::setw(15) << std::fixed << std::setprecision(2) << (cublas_time / naive_time * 100.0f) 
              << " | " << std::setw(10) << (naive_correct ? "Yes" : "No") << " |" << std::endl;
    
    std::cout << "| Tiled Shared Mem   | " << std::setw(10) << std::fixed << std::setprecision(2) << tiled_time 
              << " | " << std::setw(9) << std::fixed << std::setprecision(2) << tiled_gflops 
              << " | " << std::setw(15) << std::fixed << std::setprecision(2) << (cublas_time / tiled_time * 100.0f) 
              << " | " << std::setw(10) << (tiled_correct ? "Yes" : "No") << " |" << std::endl;
    
    std::cout << "| Vectorized         | " << std::setw(10) << std::fixed << std::setprecision(2) << vectorized_time 
              << " | " << std::setw(9) << std::fixed << std::setprecision(2) << vectorized_gflops 
              << " | " << std::setw(15) << std::fixed << std::setprecision(2) << (cublas_time / vectorized_time * 100.0f) 
              << " | " << std::setw(10) << (vectorized_correct ? "Yes" : "No") << " |" << std::endl;
    std::cout << "|--------------------|------------|-----------|-----------------|------------|" << std::endl;
    
    // Clean up
    delete[] input;
    delete[] weights;
    delete[] bias;
    delete[] output_cublas;
    delete[] output_cublas_batched;
    delete[] output_naive;
    delete[] output_tiled;
    delete[] output_vectorized;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}