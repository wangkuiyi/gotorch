#!/bin/bash

pushd $(dirname $0)

OS=$(uname | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

if [[ "$OS" == "linux" ]]; then
    if [[ "$ARCH" == "arm" ]]; then
        echo "Building for Raspbian ...";
        make -f Makefile.rpi;
    elif $(whereis cuda | cut -f 2 -d ' ')/bin/nvcc --version > /dev/null; then
        echo "Building for Linux with CUDA ...";
        make -f Makefile.linux-gpu;
    else
        echo "Building for Linux without CUDA ...";
        make -f Makefile.linux
    fi
elif [[ "$OS" == "darwin" ]]; then
    echo "Building for macOS ...";
    make -f Makefile
fi

popd
