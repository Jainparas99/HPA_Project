#!/bin/bash
# scripts/build_project.sh

BUILD_DIR=build
TESTS=${1:-OFF}  # Default to OFF unless "ON" is passed

echo "Building HPA Project (Tests: $TESTS)..."

# Create and enter build directory
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Run CMake with Ninja
cmake -GNinja -DBUILD_TESTS=$TESTS ..

# Build
ninja

echo "Build complete! Executable: $BUILD_DIR/hpa_project"