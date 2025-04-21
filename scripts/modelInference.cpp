#include <torch/script.h>
#include <iostream>
#include <vector>
#include <span> // For std::span
#include <algorithm> // For std::transform
#include <ctime>
#include "logger.h"
#include "common.h"

int main() {
  Logger& log = Logger::instance();
  log.setLevel(LogLevel::INFO);
  const std::string model = "models/pytorch_resnet50.pt";

  log.printAsteriskLine();
  log.LOG_INFO("CPU timing for the model: "+ model);
  log.printAsteriskLine();

  try {
    int use_cuda = 0;
    torch::Device device = torch::kCPU;
    
    if (use_cuda){
      device = torch::Device(torch::kCUDA);
    }


    torch::jit::Module module = torch::jit::load(model);
    module.to(device);
    std::cout << "Loaded ResNet-50 model via torch\n";
    log.LOG_INFO("Loaded ResNet-50 model via torch\n");
    // For testing, Creating a dummy image_data vector
    int batch_size = 3;
    int channels = 3;
    int height = 224;
    int width = 224;

    // --- Normalize the image data (using ImageNet statistics ---
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> stddev = {0.229f, 0.224f, 0.225f};
    std::cout<< "Going to generate the data\n";
    auto input_tensor = make_random_image_batch_tensor(
                    batch_size, channels, height, width, mean, stddev, device);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    unsigned int iters = 1000;
    double total_time = 0.0f;
    torch::Tensor output_tensor;

    for(unsigned int i=0; i<iters;i++){
      std::clock_t start = std::clock();
      output_tensor = module.forward(inputs).toTensor();
      total_time += double(std::clock() - start) / CLOCKS_PER_SEC;
    }
    double avg_inference_time = total_time / iters;
    std::cout <<"Total time "<< avg_inference_time << "s \n";
    log.LOG_INFO("Total time "+ std::to_string(avg_inference_time));
    // Execute the model
    // torch::Tensor output_tensor = module.forward(inputs).toTensor();

    std::cout << "Output Tensor shape: " << output_tensor.sizes() << "\n";
    log.LOG_INFO("Output Tensor shape: " + shapeToString(output_tensor.sizes()));
    // Process the output tensor here


    log.printAsteriskLine();
    log.LOG_INFO("Successfiully completed model infrence");
    log.printAsteriskLine();

    // consta

  } catch (const c10::Error& e) {
    std::cerr << "Error loading the model or during inference:\n";
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 1;
}