# Top-level configuration
cmake_minimum_required(VERSION 3.18)
project(HPA_Project LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable/disable optional modules
option(BUILD_TESTS "Build unit tests" OFF)
option(USE_ONNX_RUNTIME "Enable ONNX Runtime support in C++" OFF)

# CUDA Setup
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 86) # For RTX 3080
else()
    message(FATAL_ERROR "CUDA Toolkit not found. Install it manually.")
endif()

# cuDNN Setup
find_library(CUDNN_LIB cudnn HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_path(CUDNN_INCLUDE cudnn.h HINTS ${CUDA_TOOLKIT_ROOT_DIR}/include)

if(CUDNN_LIB AND CUDNN_INCLUDE)
    message(STATUS "cuDNN found: ${CUDNN_LIB}")
else()
    message(WARNING "cuDNN not found in CUDA path. Falling back to external/cudnn...")
    set(CUDNN_LIB ${CMAKE_SOURCE_DIR}/external/cudnn/lib64/libcudnn.so)
    set(CUDNN_INCLUDE ${CMAKE_SOURCE_DIR}/external/cudnn/include)
endif()

# ONNX Runtime Support (optional)
if(USE_ONNX_RUNTIME)
    include(ExternalProject)
    find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib ${CMAKE_SOURCE_DIR}/external/onnxruntime/lib)
    find_path(ONNX_RUNTIME_INCLUDE onnxruntime/core/session/onnxruntime_cxx_api.h 
              HINTS /usr/local/include ${CMAKE_SOURCE_DIR}/external/onnxruntime/include)

    if(ONNX_RUNTIME_LIB AND ONNX_RUNTIME_INCLUDE)
        message(STATUS "ONNX Runtime found system-wide.")
    else()
        message(STATUS "ONNX Runtime not found. Attempting to build from source (experimental).")
        ExternalProject_Add(
            onnxruntime
            GIT_REPOSITORY https://github.com/microsoft/onnxruntime.git
            GIT_TAG v1.17.1
            PREFIX ${CMAKE_SOURCE_DIR}/external/onnxruntime
            CMAKE_ARGS 
                -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/external/onnxruntime/install
                -Donnxruntime_BUILD_SHARED_LIB=ON
                -Donnxruntime_ENABLE_CUDA=ON
                -DCMAKE_CUDA_COMPILER=${CUDA_NVCC_EXECUTABLE}
            UPDATE_COMMAND ""
        )
        set(ONNX_RUNTIME_LIB ${CMAKE_SOURCE_DIR}/external/onnxruntime/install/lib/libonnxruntime.so)
        set(ONNX_RUNTIME_INCLUDE ${CMAKE_SOURCE_DIR}/external/onnxruntime/install/include)
        set(ONNX_RUNTIME_DEP onnxruntime)
    endif()
endif()

# Source files
set(SOURCE_FILES
    src/main.cpp
    src/cuda_kernels.cu
    src/host_utils.cpp   
)

# Executable definition
add_executable(hpa_project 
    ${SOURCE_FILES}
    src/main.cpp 
    src/cuda_kernels.cu 
    src/host_utils.cpp
    external/cnpy/cnpy.cpp     
    )

# Include paths
target_include_directories(hpa_project PRIVATE
    include
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_INCLUDE}
    external/cnpy
)

# Optional ONNX includes
if(USE_ONNX_RUNTIME)
    target_include_directories(hpa_project PRIVATE ${ONNX_RUNTIME_INCLUDE})
endif()

# Linked libraries
target_link_libraries(hpa_project PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDNN_LIB}
)

if(USE_ONNX_RUNTIME)
    target_link_libraries(hpa_project PRIVATE ${ONNX_RUNTIME_LIB})
endif()

# CUDA-specific flags
set_target_properties(hpa_project PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Build order dependency if ONNX is enabled
if(USE_ONNX_RUNTIME AND ONNX_RUNTIME_DEP)
    add_dependencies(hpa_project ${ONNX_RUNTIME_DEP})
endif()

# Copy ONNX model after build
add_custom_command(TARGET hpa_project POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/models/vgg16.onnx ${CMAKE_BINARY_DIR}/vgg16.onnx
    COMMENT "Copying VGG-16 ONNX model to build directory"
)

# Optional: Build tests
if(BUILD_TESTS)
    add_executable(test_hpa_project
        tests/test_main.cpp
        tests/test_conv.cpp
    )
    target_include_directories(test_hpa_project PRIVATE include ${CUDA_INCLUDE_DIRS})
    target_link_libraries(test_hpa_project PRIVATE ${CUDA_LIBRARIES})
endif()