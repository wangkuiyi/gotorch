#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd $DIR

OS=$(uname | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
CXX="g++"
LIBTORCH_DIR=""
GLIBCXX_USE_CXX11_ABI="1"
LOAD="force_load"
LIB_SUFFIX="so"
INSTALL_NAME=""
CUDA_FLAGS=""
DEPS=""

function build_linux_no_cuda() {
    CXX="clang++"
    LIBTORCH_DIR="linux/libtorch"
    GLIBCXX_USE_CXX11_ABI="0"
    if [[ ! -d "$DIR/$LIBTORCH_DIR" ]]; then
        curl -LsO 'https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.6.0%2Bcpu.zip'
        unzip -qq -o libtorch-shared-with-deps-1.6.0%2Bcpu.zip -d linux
    fi
}

if [[ "$OS" == "linux" ]]; then
    if [[ "$ARCH" =~ arm* ]]; then
        echo "Building for Raspbian ...";
        CXX="g++"
        LIBTORCH_DIR="rpi/libtorch"
        if [[ ! -d "$DIR/$LIBTORCH_DIR" ]]; then
            curl -LsO 'https://github.com/ljk53/pytorch-rpi/raw/master/libtorch-rpi-cxx11-abi-shared-1.6.0.zip';
            unzip -qq -o libtorch-rpi-cxx11-abi-shared-1.6.0.zip -d rpi
        fi
    elif $(whereis cuda | cut -f 2 -d ' ')/bin/nvcc --version > /dev/null; then
        CXX="clang++"
        NVCC=$(whereis cuda | cut -f 2 -d ' ')/bin/nvcc
        CUDA_VERSION=$("$NVCC" --version | grep release | grep -Eo "[0-9]+.[0-9]+" | head -1)
        CUDA_FLAGS="$CUDA_FLAGS -DWITH_CUDA -I /usr/local/cuda/include"
        if [[ "$CUDA_VERSION" == "10.1" ]]; then
            echo "Building for Linux with CUDA 10.1";
            LIBTORCH_DIR="linux-cuda101/libtorch"
            if [[ ! -d "$DIR/$LIBTORCH_DIR" ]]; then
                curl -Lso libtorch-cxx11-1.6.0-linux-cuda101.zip 'https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcu101.zip'
                unzip -qq -o libtorch-cxx11-1.6.0-linux-cuda101.zip -d linux-cuda101
            fi
        elif [[ "$CUDA_VERSION" == "10.2" ]]; then
            echo "Building for Linux with CUDA 10.2";
            LIBTORCH_DIR="linux-cuda102/libtorch"
            if [[ ! -d "$DIR/$LIBTORCH_DIR" ]]; then
                curl -Lso libtorch-cxx11-1.6.0-linux-cuda102.zip 'https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.6.0.zip'
                unzip -qq -o libtorch-cxx11-1.6.0-linux-cuda102.zip -d linux-cuda102
            fi
    else
        echo "Unknown CUDA version: $CUDA_VERSION"
        build_linux_no_cuda
        fi
    else
        echo "Building for Linux without CUDA ...";
        build_linux_no_cuda
    fi
elif [[ "$OS" == "darwin" ]]; then
    echo "Building for macOS ...";
    CXX="clang++"
    LIBTORCH_DIR="macos/libtorch"
    LIB_SUFFIX="dylib"
    INSTALL_NAME="-install_name @rpath/\$@"
    LOAD="all_load"
    DEPS=`pkg-config --cflags --libs libuv`
    if [[ ! -d "$DIR/$LIBTORCH_DIR" ]]; then
        curl -LsO https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.6.0.zip
        unzip -qq -o libtorch-macos-1.6.0.zip -d macos
    fi
fi

rm -f libtorch
ln -s ${LIBTORCH_DIR} libtorch

set -o xtrace
make CXX="$CXX" \
     LIB_SUFFIX="$LIB_SUFFIX" \
     INSTALL_NAME="$INSTALL_NAME" \
     LIBTORCH_DIR="$LIBTORCH_DIR" \
     GLIBCXX_USE_CXX11_ABI="$GLIBCXX_USE_CXX11_ABI" \
     LOAD="$LOAD" \
     CUDA_FLAGS="$CUDA_FLAGS" \
     DEPS="$DEPS" \
     -f Makefile -j `getconf _NPROCESSORS_ONLN`
popd
