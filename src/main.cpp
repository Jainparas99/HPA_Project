#include <iostream>
#include <vector>
#include "cuda_kernels.h"
#include "host_utils.h"

// Include ONNX Runtime
#include "onnxruntime_cxx_api.h"

int main() {
    std::cout << "VGG16 Optimization Project" << std::endl;

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vgg16_test");

    // Configure session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Create session - CPU baseline
    try {
        std::cout << "Loading VGG16 ONNX model..." << std::endl;
        Ort::Session session(env, L"vgg16.onnx", session_options);

        // Print model info
        Ort::AllocatorWithDefaultOptions allocator;
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "Inputs: " << session.GetInputCount() << std::endl;
        std::cout << "Outputs: " << session.GetOutputCount() << std::endl;

        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        std::cout << "Input name: " << input_name.get() << std::endl;
        std::cout << "Output name: " << output_name.get() << std::endl;

        // Test FC layer implementations
        std::cout << "\n=== Testing FC Layer Implementation ===" << std::endl;
        
        // Test correctness for VGG16's three FC layers
        test_fc_layer_correctness(1, 25088, 4096); // FC1
        test_fc_layer_correctness(1, 4096, 4096);  // FC2
        test_fc_layer_correctness(1, 4096, 1000);  // FC3
        
        // Performance measurements (10 iterations for each layer)
        std::cout << "\n=== FC Layer Performance Benchmarks ===" << std::endl;
        measure_fc_layer_performance(1, 25088, 4096, 10); // FC1
        measure_fc_layer_performance(1, 4096, 4096, 10);  // FC2
        measure_fc_layer_performance(1, 4096, 1000, 10);  // FC3
        
        // Batch size scaling test
        std::cout << "\n=== FC Layer Batch Size Scaling ===" << std::endl;
        for (int batch_size : {1, 2, 4, 8, 16}) {
            std::cout << "Batch size: " << batch_size << std::endl;
            measure_fc_layer_performance(batch_size, 25088, 4096, 5);
        }

        // Test custom FC layer implementations
        std::cout << "\n=== Custom FC Layer Implementation Tests ===" << std::endl;
        test_custom_fc_implementations(1, 4096, 4096);     // FC2 size for initial testing (smaller)
        test_custom_fc_implementations(1, 25088, 4096);    // FC1 size (largest layer)
        test_custom_fc_implementations(16, 25088, 4096);   // FC1 with batch=16 to test scaling

        // Add CGMA analysis for the three FC layers in VGG16
        std::cout << "\n=== CGMA Analysis for VGG16 FC Layers ===" << std::endl;
        analyze_cgma(1, 25088, 4096); // FC1 (largest layer)
        analyze_cgma(1, 4096, 4096);  // FC2
        analyze_cgma(1, 4096, 1000);  // FC3
        
        // Run model prediction with CPU and GPU
        run_model_prediction(session, false); // CPU implementation
        run_model_prediction(session, true);  // GPU implementation
    }
    catch (const Ort::Exception& exception) {
        std::cerr << "ONNX Runtime error: " << exception.what() << std::endl;
        
        // Keep terminal open on error
        std::cout << "\nPress Enter key to continue..." << std::endl;
        std::cin.get();
        
        
        return -1;
    }

    std::cout << "Testing completed successfully!" << std::endl;
    
    // Add this code to keep the console window open
    std::cout << "\nPress Enter key to exit..." << std::endl;
    std::cin.get();
    
    
    return 0;
}