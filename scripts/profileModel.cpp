#include <torch/script.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <dlfcn.h>
#include "common.h"

// Helper to check CUDA calls
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                << " code=" << err                                         \
                << " \"" << cudaGetErrorString(err) << "\"\n";             \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

// Profiles each top‑level submodule by timing its forward() via CUDA events.
std::vector<std::pair<std::string, float>>
profileSubmodulesCUDA(torch::jit::Module& module,
                      const at::Tensor&   input,
                      int                 device_id) {
    std::cout << "[DEBUG] Entering profileSubmodulesCUDA" << std::endl;
    // 1) Bind CUDA runtime to the right GPU
    std::cout << "[DEBUG] Setting CUDA device to " << device_id << std::endl;
    CUDA_CHECK(cudaSetDevice(device_id));

    // 2) Create events
    std::cout << "[DEBUG] Creating CUDA events" << std::endl;
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    std::vector<std::pair<std::string,float>> timings;
    at::Tensor inp = input;
    std::cout << "[DEBUG] Starting module.named_children() loop" << std::endl;

    // 3) Iterate by value so child.value is a mutable Module&
    for (auto child : module.named_children()) {
        const auto& name   = child.name;
        auto&       submod = child.value;

        std::cout << "[DEBUG] Profiling layer: " << name << std::endl;
        
        // Warm‑up
        std::cout << "[DEBUG] Warm-up forward pass" << std::endl;
        try {
            submod.forward({inp}).toTensor();
        } catch (const c10::Error& e) {
            std::cerr << "[ERROR] Warm-up forward error in " << name << ": " << e.what() << std::endl;
            continue;
        }
        std::cout << "[DEBUG] Warm-up complete, synchronizing" << std::endl;
        CUDA_CHECK(cudaDeviceSynchronize());

        // Time the kernel
        std::cout << "[DEBUG] Recording start event" << std::endl;
        CUDA_CHECK(cudaEventRecord(start));
        at::Tensor out;
        try {
            out = submod.forward({inp}).toTensor();
        } catch (const c10::Error& e) {
            std::cerr << "[ERROR] Timed forward error in " << name << ": " << e.what() << std::endl;
            break;
        }
        std::cout << "[DEBUG] Recording end event and synchronizing" << std::endl;
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
        std::cout << "[DEBUG] Elapsed time for " << name << ": " << ms << " ms" << std::endl;
        timings.emplace_back(name, ms);

        inp = std::move(out);
        // Flatten after avgpool so fc sees correct shape
        if (name == "avgpool") {
            inp = inp.flatten(1);
            std::cout << "[DEBUG] Flattened tensor after avgpool: " << inp.sizes() << std::endl;
        }
    }

    // Cleanup
    std::cout << "[DEBUG] Destroying CUDA events" << std::endl;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    std::cout << "[DEBUG] Exiting profileSubmodulesCUDA" << std::endl;
    return timings;
}

int main() {
    // 1) Load the custom‐op library
    void* h = dlopen("libhpc.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        std::cerr << "dlopen(libhpc.so) failed: " << dlerror() << "\n";
        return -1;
    }
    std::cout << "[DEBUG] Starting Profiling \n";

    int device_id = 0;
    if (!torch::cuda::is_available()) {
        std::cerr << "[ERROR] CUDA not available\n";
        return 1;
    }

    torch::Device device(torch::kCUDA, device_id);

    // Load model paths from CSV
    std::string csv_path = "config/models.csv";
    auto csv_data = read_csv(csv_path);
    
    // Createa a arr of batch sized to see how well the kernel scales

    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128};


    if (csv_data.empty()) {
        std::cerr << "[ERROR] No models found in CSV.\n";
        return 1;
    }
    for (const auto& row : csv_data) {
        if (row.empty()) continue;
        const std::string& model_path = row[0];
        std::cout << "\n" << std::string(90, '*') << "\n";
        std::cout << "\n[DEBUG] === Profiling: " << model_path << " ===\n";
        std::cout << "\n" << std::string(90, '*') << "\n";

        torch::jit::Module module;
        try {
            std::cout << "[DEBUG] Loading model: " << model_path << std::endl;
            module = torch::jit::load(model_path);
        } catch (const c10::Error& e) {
            std::cerr << "[ERROR] Failed to load model: " << model_path << "\n" << e.what() << std::endl;
            continue;
        }

        module.to(device);
        module.eval();
        for(int batch_size : batch_sizes) {
            std::cout << "[DEBUG] Profiling with batch size: " << batch_size << std::endl;
            at::Tensor x = torch::randn({batch_size, 3, 224, 224}, torch::TensorOptions().device(device));
            auto results = profileSubmodulesCUDA(module, x, device_id);
            std::cout << "Layer‑wise GPU times for " << model_path << " with batch size " << batch_size << ":\n";
            for (auto& [layer, ms] : results) {
                std::cout << "  " << layer << ": " << ms << " ms\n";
            }
        }
        // at::Tensor x = torch::randn({50, 3, 224, 224}, torch::TensorOptions().device(device));

        // auto results = profileSubmodulesCUDA(module, x, device_id);

        // std::cout << "Layer‑wise GPU times for " << model_path << ":\n";
        // for (auto& [layer, ms] : results) {
        //     std::cout << "  " << layer << ": " << ms << " ms\n";
        // }
    }

    std::cout << "\n[DEBUG] Finished profiling all models.\n";
    return 0;
}