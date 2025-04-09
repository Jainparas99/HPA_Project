# Minimum CMake version for CUDA support
cmake_minimum_required(VERSION 3.18)
project(HPA_Project LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include ExternalProject for fetching dependencies
include(ExternalProject)

# Find CUDA (required for .cu files and linking)
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 86) # RTX 3080
else()
    message(FATAL_ERROR "CUDA Toolkit not found. Install it manually.")
endif()

# Find ONNX Runtime or fetch it
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib ${CMAKE_SOURCE_DIR}/external/onnxruntime/lib)
find_path(ONNX_RUNTIME_INCLUDE onnxruntime/core/session/onnxruntime_cxx_api.h 
          HINTS /usr/local/include ${CMAKE_SOURCE_DIR}/external/onnxruntime/include)
if(ONNX_RUNTIME_LIB AND ONNX_RUNTIME_INCLUDE)
    message(STATUS "ONNX Runtime found system-wide: ${ONNX_RUNTIME_LIB}")
else()
    message(STATUS "ONNX Runtime not found. Fetching and building...")
    ExternalProject_Add(
        onnxruntime
        GIT_REPOSITORY https://github.com/microsoft/onnxruntime.git
        GIT_TAG v1.17.1  # Stable release as of April 2025
        PREFIX ${CMAKE_SOURCE_DIR}/external/onnxruntime
        CMAKE_ARGS 
            -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/external/onnxruntime/install
            -Donnxruntime_BUILD_SHARED_LIB=ON
            -Donnxruntime_ENABLE_CUDA=ON
            -DCMAKE_CUDA_COMPILER=${CUDA_NVCC_EXECUTABLE}
        UPDATE_COMMAND ""  # Skip updates after first download
    )
    set(ONNX_RUNTIME_LIB ${CMAKE_SOURCE_DIR}/external/onnxruntime/install/lib/libonnxruntime.so)
    set(ONNX_RUNTIME_INCLUDE ${CMAKE_SOURCE_DIR}/external/onnxruntime/install/include)
    set(ONNX_RUNTIME_DEP onnxruntime) # Dependency for ordering
endif()

# Find cuDNN or provide manual fetch instructions
find_library(CUDNN_LIB cudnn HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_path(CUDNN_INCLUDE cudnn.h HINTS ${CUDA_TOOLKIT_ROOT_DIR}/include)
if(CUDNN_LIB AND CUDNN_INCLUDE)
    message(STATUS "cuDNN found: ${CUDNN_LIB}")
else()
    message(WARNING "cuDNN not found. Auto-fetching not implemented due to NVIDIA login requirement.")
    message(STATUS "Assuming manual placement in external/cudnn...")
    set(CUDNN_LIB ${CMAKE_SOURCE_DIR}/external/cudnn/lib64/libcudnn.so)
    set(CUDNN_INCLUDE ${CMAKE_SOURCE_DIR}/external/cudnn/include)
endif()

# Define the main executable
add_executable(hpa_project 
    src/main.cpp 
    src/cuda_kernels.cu 
    src/host_utils.cpp
)

# Include directories
target_include_directories(hpa_project PRIVATE 
    include 
    ${CUDA_INCLUDE_DIRS} 
    ${ONNX_RUNTIME_INCLUDE} 
    ${CUDNN_INCLUDE}
)

# Link libraries
target_link_libraries(hpa_project PRIVATE 
    ${CUDA_LIBRARIES} 
    ${CUDA_CUBLAS_LIBRARIES}  # For fully connected layers
    ${CUDNN_LIB}              # For convolution
    ${ONNX_RUNTIME_LIB}       # For ONNX inference
)

# Set CUDA-specific properties
set_target_properties(hpa_project PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Ensure fetched dependencies are built first
if(ONNX_RUNTIME_DEP)
    add_dependencies(hpa_project ${ONNX_RUNTIME_DEP})
endif()

# Optional: Add tests
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    add_executable(test_hpa_project 
        tests/test_main.cpp 
        tests/test_conv.cpp
    )
    target_include_directories(test_hpa_project PRIVATE 
        include 
        ${CUDA_INCLUDE_DIRS}
    )
    target_link_libraries(test_hpa_project PRIVATE 
        ${CUDA_LIBRARIES}
    )
endif()

# Custom target to copy the ONNX model to build directory
add_custom_command(TARGET hpa_project POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/models/vgg16.onnx ${CMAKE_BINARY_DIR}/vgg16.onnx
    COMMENT "Copying VGG-16 ONNX model to build directory"
)