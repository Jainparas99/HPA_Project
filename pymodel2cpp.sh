#!/usr/bin/env bash
# set -e

# -----------------------------------------------------------------------------
# run_pipeline.sh
# -----------------------------------------------------------------------------
echo "===== 1. Build Python extension ====="
python setup.py build_ext --inplace

echo
echo "===== 2. Script ResNet50 with custom FC ====="
# point to the correct script name
python scripts/script_resent.py

echo
echo "===== 3. Configure & build C++ ====="
BUILD_DIR=build
LIBTORCH_DIR="${PWD}/external/libtorch"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" ..
make -j
cd ..

echo
echo "===== 4. Export LD_LIBRARY_PATH ====="
# C++ LibTorch libs
export LIBTORCH_LIB="$LIBTORCH_DIR/lib"
# Python torch libs
export PYTORCH_LIB=$(python -c "import os, torch; print(os.path.dirname(torch._C.__file__))")

export LD_LIBRARY_PATH="$PWD/$BUILD_DIR:$LIBTORCH_LIB:$PYTORCH_LIB:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH:"
echo "  C++ build dir:  $PWD/$BUILD_DIR"
echo "  LibTorch libs:  $LIBTORCH_LIB"
echo "  Python torch:   $PYTORCH_LIB"

echo
echo "===== 5. Run executables ====="
echo "--> C++ Inference"
"$BUILD_DIR"/inference

echo "--> C++ Profiling"
"$BUILD_DIR"/profiling

echo "--> C++ Post‑custom FC inference"
"$BUILD_DIR"/inference_post_cust

echo
echo "===== All done! ====="
