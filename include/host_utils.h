#ifndef HOST_UTILS_H
#define HOST_UTILS_H

// Include necessary headers
#include <string>
#include <vector>

// Include ONNX Runtime
#include "onnxruntime_cxx_api.h"

// Include CUDA function declarations
#include "cuda_kernels.h"

// CPU implementation of FC layer for validation
void fc_layer_cpu(float* input, float* weights, float* bias, float* output,
                  int batch_size, int input_features, int output_features);

// Utility to generate random data
void generate_random_data(float* data, size_t size, float min = -1.0f, float max = 1.0f);

// Check if two arrays are approximately equal
bool arrays_equal(float* arr1, float* arr2, size_t size, float tolerance = 1e-2);

// Initialize ONNX Runtime and create a session
void* initialize_onnx_session(const wchar_t* model_path);

// Function to extract weights and biases from ONNX model for FC layers
bool extract_fc_parameters(void* session_ptr, const char* layer_name, 
                           float** weights, float** bias, 
                           int* input_features, int* output_features);

// Test and performance measurement functions
void test_fc_layer_correctness(int batch_size, int input_features, int output_features);
float measure_fc_layer_performance(int batch_size, int input_features, int output_features, int iterations);

// CGMA analysis for different FC layer implementations
void analyze_cgma(int batch_size, int input_features, int output_features);

// Model prediction functions
void display_predictions(float* output, int batch_size, int num_classes, int k = 5);
void run_model_prediction(Ort::Session& session, bool use_gpu = true);

// Test custom FC layer implementations
void test_custom_fc_implementations(int batch_size, int input_features, int output_features);

#endif // HOST_UTILS_H