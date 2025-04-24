cmake_minimum_required(VERSION 3.18)
project(vgg16_optimization LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture for RTX 3080
set(CMAKE_CUDA_ARCHITECTURES 86)

# Find CUDA and related libraries
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Add local includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external)

# Find or fetch ONNX Runtime
find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime/lib)
if(NOT ONNXRUNTIME_LIB)
    message(STATUS "ONNX Runtime not found locally. Please download it manually:")
    message(STATUS "1. Go to https://github.com/microsoft/onnxruntime/releases")
    message(STATUS "2. Download the Windows x64 GPU package")
    message(STATUS "3. Extract to ${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime")
    message(FATAL_ERROR "ONNX Runtime is required to build this project")
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime/include)

# Add source files
set(SOURCES
    src/main.cpp
    src/host_utils.cpp
    src/cuda_kernels.cu
)

# Create executable
add_executable(hpa_project ${SOURCES})

# Link libraries
target_link_libraries(hpa_project
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${ONNXRUNTIME_LIB}
)


# Copy ONNX model to build directory
add_custom_command(
    TARGET hpa_project POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/models/vgg16.onnx
    $<TARGET_FILE_DIR:hpa_project>/vgg16.onnx
)