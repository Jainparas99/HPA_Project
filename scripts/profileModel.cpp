#include <torch/script.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>

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
    std::cout << "[DEBUG] Starting main" << std::endl;
    // 0) Pick GPU
    int device_id = 0;  // adjust as needed
    std::cout << "[DEBUG] Checking CUDA availability" << std::endl;
    if (!torch::cuda::is_available()) {
        std::cerr << "[ERROR] CUDA not available" << std::endl;
        return 1;
    }

    // 1) Load & move model
    torch::Device device(torch::kCUDA, device_id);
    std::cout << "[DEBUG] Loading model" << std::endl;
    auto module = torch::jit::load("models/pytorch_resnet50.pt");
    std::cout << "[DEBUG] Moving model to device" << std::endl;
    module.to(device);
    module.eval();

    // 2) Build a matching input
    std::cout << "[DEBUG] Creating input tensor" << std::endl;
    at::Tensor x = torch::randn({1,3,224,224}, torch::TensorOptions().device(device));

    // 3) Profile
    std::cout << "[DEBUG] Calling profileSubmodulesCUDA" << std::endl;
    auto results = profileSubmodulesCUDA(module, x, device_id);

    // 4) Print
    std::cout << "Layer‑wise GPU times:" << std::endl;
    for (auto& [layer, ms] : results) {
        std::cout << "  " << layer << ": " << ms << " ms" << std::endl;
    }
    std::cout << "[DEBUG] Finished main" << std::endl;
    return 0;
}
