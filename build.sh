#!/bin/sh
set -eux

PRJ_ROOT="$( cd "$(dirname "$0")" ; pwd -P)"
BUILD_ROOT=$PRJ_ROOT/build
PYTORCH_ROOT="${PYTORCH_ROOT:-$PRJ_ROOT/pytorch}"

# With "python setup.py develop", C++ headers/libraries are "installed" inside the source tree!
PYTORCH_BUILD=$PYTORCH_ROOT/build
PYTORCH_INSTALL=$PYTORCH_ROOT/torch

install_dependencies() {
  # Follow PyTorch local build instruction: https://github.com/pytorch/pytorch#from-source
  echo "Install conda dependencies..."
  conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
}

checkout_pytorch() {
  if [ ! -d "$PYTORCH_ROOT" ]; then
    echo "PyTorch src folder doesn't exist: $PYTORCH_ROOT. Downloading..."
    echo "You can use existing PyTorch src by 'export PYTORCH_ROOT=<path>'"
    mkdir -p "$PYTORCH_ROOT"
    git clone --recursive https://github.com/pytorch/pytorch "$PYTORCH_ROOT"
  else
    echo "Using PyTorch source code at: $PYTORCH_ROOT"
  fi
}

build_pytorch() {
  echo "Building PyTorch..."
  echo "!!! You might need run `python setup.py clean` if the last build failed."

  cd $PYTORCH_ROOT
  CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} \
    BUILD_CAFFE2_OPS=OFF \
    BUILD_BINARY=OFF \
    BUILD_TEST=OFF \
    USE_DISTRIBUTED=OFF \
    python setup.py develop

  echo "PyTorch local build folder: $PYTORCH_BUILD"
  echo "Use headers/libraries at: $PYTORCH_INSTALL"
}

build_demo() {
  # Build demo project
  rm -rf $BUILD_ROOT && mkdir $BUILD_ROOT && cd $BUILD_ROOT

  cmake .. \
  -DCMAKE_PREFIX_PATH=$PYTORCH_INSTALL \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=install

  make

  echo "Build succeeded!"
}

# You can build PyTorch manually and skip the following steps.
#install_dependencies
checkout_pytorch
build_pytorch

# Build the demo using the PyTorch C++ headers/libraries at $PYTORCH_ROOT/torch.
build_demo
