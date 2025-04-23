cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(pytorch_inference)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/libtorch")

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# ------------------------------------------------------------------
# Custom FC extension (scripts/custom)
# ------------------------------------------------------------------
add_library(hpc SHARED
    scripts/custom/fc_op.cpp
    scripts/custom/fc_kernel.cu
)
target_include_directories(hpc PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/scripts/custom
)
target_link_libraries(hpc PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)
set_property(TARGET hpc PROPERTY CXX_STANDARD 17)

# ------------------------------------------------------------------
# Standard inference executable
# ------------------------------------------------------------------
add_executable(inference
    scripts/modelInference.cpp
    scripts/utils.cpp
)
target_include_directories(inference PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(inference PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    hpc
)

# ------------------------------------------------------------------
# Profiling executable
# ------------------------------------------------------------------
add_executable(profiling
    scripts/profileModel.cpp
    scripts/utils.cpp
)
target_include_directories(profiling PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(profiling PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    hpc
)

# ------------------------------------------------------------------
# CPP-only inference (no utils.cpp)
# ------------------------------------------------------------------
add_executable(inference_cpp
    scripts/modelInference.cpp
    scripts/utils.cpp
)
target_include_directories(inference_cpp PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(inference_cpp PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    hpc
)
set_property(TARGET inference_cpp PROPERTY CXX_STANDARD 17)

# ------------------------------------------------------------------
# Post-custom inference executable
# ------------------------------------------------------------------
add_executable(inference_post_cust
    scripts/modelInference_post_cust.cpp
    scripts/utils.cpp
)
target_include_directories(inference_post_cust PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(inference_post_cust PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    hpc
)
set_property(TARGET inference_post_cust PROPERTY CXX_STANDARD 17)
