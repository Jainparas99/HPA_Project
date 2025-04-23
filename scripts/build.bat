@echo off
echo Building VGG16 Optimization Project

mkdir build 2>nul
cd build

echo Configuring CMake...
cmake -G "Visual Studio 17 2022" -A x64 ..

echo Building project...
cmake --build . --config Release

echo Build completed!
cd ..