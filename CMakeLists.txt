cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(pytorch_inference)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED) 


# Infernece 
add_executable(inference 
    scripts/modelInference.cpp 
    scripts/utils.cpp)

target_include_directories(inference
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(inference
    PRIVATE
        ${TORCH_LIBRARIES}
)

# Profiling
add_executable(profiling 
    scripts/profileModel.cpp
    scripts/profiler.cpp
    scripts/utils.cpp)

target_include_directories(profiling
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(profiling
    PRIVATE
        ${TORCH_LIBRARIES}
)

target_link_libraries(profiling PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_CUDART_LIBRARY} 
)

# target_link_libraries(inference "${TORCH_LIBRARIES}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")