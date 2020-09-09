#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd $DIR > /dev/null

CXX="g++"
LIBTORCH_DIR=""
GLIBCXX_USE_CXX11_ABI="1"
LOAD="force_load"
LIB_SUFFIX="so"
INSTALL_NAME=""
XLA_LIBS=""

function build_macos() {
    echo "Building for macOS ...";
    LIBTORCH_DIR="macos/libtorch"
    LIB_SUFFIX="dylib"
    INSTALL_NAME="-install_name @rpath/\$@"
    LOAD="all_load"
    if [[ ! -d "$DIR/$LIBTORCH_DIR" ]]; then
        curl -LsO "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.6.0.zip"
        unzip -qq -o libtorch-macos-1.6.0.zip -d macos
    fi
}

function build_raspbian() {
    echo "Building for Raspbian ...";
    LIBTORCH_DIR="rpi/libtorch"
    if [[ ! -d "$DIR/$LIBTORCH_DIR" ]]; then
        curl -LsO "https://github.com/ljk53/pytorch-rpi/raw/master/libtorch-rpi-cxx11-abi-shared-1.6.0.zip"
        unzip -qq -o libtorch-rpi-cxx11-abi-shared-1.6.0.zip -d rpi
    fi
}

function build_cuda() {
    CUDA_VERSION=$("$NVCC" --version | grep release | grep -Eo "[0-9]+.[0-9]+" | head -1)

    if [[ "$CUDA_VERSION" == "10.1" ]]; then
        CUDA_VERSION="101"
    elif [[ "$CUDA_VERSION" == "10.2" ]]; then
        CUDA_VERSION="102"
    else
        echo "Not supported CUDA version. $CUDA_VERSION"
        return -1
    fi

    echo "Building for Linux with CUDA 10.1";
    CXX="clang++"

    LIBTORCH_DIR="linux-cuda$CUDA_VERSION/libtorch"
    if [[ ! -d "$DIR/$LIBTORCH_DIR" ]]; then
        curl -Lso libtorch-cxx11-1.6.0-linux-cuda$CUDA_VERSION.zip 'https://download.pytorch.org/libtorch/cu$CUDA_VERSION/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcu$CUDA_VERSION.zip'
        unzip -qq -o libtorch-cxx11-1.6.0-linux-cuda$CUDA_VERSION.zip -d linux-cuda$CUDA_VERSION
    fi
}

function build_linux(){
    echo "Building for Linux without CUDA ...";
    LIBTORCH_DIR="linux/libtorch"
    GLIBCXX_USE_CXX11_ABI="0"
    if [[ ! -d "$DIR/$LIBTORCH_DIR" ]]; then
        curl -LsO 'https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.6.0%2Bcpu.zip'
        unzip -qq -o libtorch-shared-with-deps-1.6.0%2Bcpu.zip -d linux
    fi

    if [[ ! -d "$DIR/torch_xla" ]]; then
        curl -LsO "https://github.com/wangkuiyi/torch_xla_prebuilt/raw/master/torch_xla-1.6.0.tar.bz2"
        tar xjf torch_xla-1.6.0.tar.bz2
    fi

    if [[ -d "$DIR/torch_xla" ]]; then
        echo "Found XLA library. Use it."
        XLA_LIBS="-Ltorch_xla/lib -lptxla -lxla_computation_client"
    fi
}


OS=$(uname | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
NVCC=$(whereis cuda | cut -f 2 -d ' ')"/bin/nvcc"

if [[ "$OS" == "darwin" ]]; then
    build_macos
elif [[ "$OS" == "linux" ]]; then
    if [[ "$ARCH" =~ arm* ]]; then
        build_raspbian
    elif "$NVCC" --version > /dev/null; then
        if ! build_cuda; then
            build_linux
        fi
    else
        build_linux
    fi
fi


make CXX="$CXX" \
     LIB_SUFFIX="$LIB_SUFFIX" \
     INSTALL_NAME="$INSTALL_NAME" \
     LIBTORCH_DIR="$LIBTORCH_DIR" \
     GLIBCXX_USE_CXX11_ABI="$GLIBCXX_USE_CXX11_ABI" \
     LOAD="$LOAD" \
     XLA_LIBS="$XLA_LIBS" \
     -f Makefile;

popd > /dev/null
