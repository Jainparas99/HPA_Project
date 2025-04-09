#include <onnxruntime_cxx_api.h>
#include <cuda_kernels.h>
#include <host_utils.h>
#include <iostream>

int main() {
    // Initialize ONNX Runtime with CUDA provider
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VGG16HPA");
    Ort::SessionOptions session_options;
    OrtCUDAProviderOptions cuda_options{0}; // Device ID 0 (RTX 3080)
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    
    // Load the model
    Ort::Session session(env, "vgg16.onnx", session_options);
    std::cout << "Model loaded successfully.\n";

    // Example: Run a basic inference (to be expanded)
    // Placeholder for sub-team CUDA integration
    run_sample_inference();

    return 0;
}