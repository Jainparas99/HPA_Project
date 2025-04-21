#pragma once
#include <c10/util/ArrayRef.h>
#include <string>
#include <torch/torch.h>

std::string shapeToString(const c10::IntArrayRef& sizes);

torch::Tensor make_random_image_batch_tensor(
    int batch_size,
    int channels,
    int height,
    int width,
    const std::vector<float>& mean,
    const std::vector<float>& stddev,
    torch::Device device = torch::kCPU
);