# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/stu15/s15/pj2196/HPA/final/HPA_Project/external/onnxruntime/src/onnxruntime")
  file(MAKE_DIRECTORY "/home/stu15/s15/pj2196/HPA/final/HPA_Project/external/onnxruntime/src/onnxruntime")
endif()
file(MAKE_DIRECTORY
  "/home/stu15/s15/pj2196/HPA/final/HPA_Project/external/onnxruntime/src/onnxruntime-build"
  "/home/stu15/s15/pj2196/HPA/final/HPA_Project/external/onnxruntime"
  "/home/stu15/s15/pj2196/HPA/final/HPA_Project/external/onnxruntime/tmp"
  "/home/stu15/s15/pj2196/HPA/final/HPA_Project/external/onnxruntime/src/onnxruntime-stamp"
  "/home/stu15/s15/pj2196/HPA/final/HPA_Project/external/onnxruntime/src"
  "/home/stu15/s15/pj2196/HPA/final/HPA_Project/external/onnxruntime/src/onnxruntime-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/stu15/s15/pj2196/HPA/final/HPA_Project/external/onnxruntime/src/onnxruntime-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/stu15/s15/pj2196/HPA/final/HPA_Project/external/onnxruntime/src/onnxruntime-stamp${cfgdir}") # cfgdir has leading slash
endif()
