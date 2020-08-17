#!/bin/bash

pushd $(dirname $0)

OS=$(uname | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

if [[ "$OS" == "linux" ]]; then
    if [[ "$ARCH" == "arm" ]]; then
        echo "Building for Raspbian ...";
        make -f Makefile.rpi;
    elif $(whereis cuda | cut -f 2 -d ' ')/bin/nvcc --version > /dev/null; then
	CUDA_VERSION=`nvcc --version | grep release | grep -Eo "[0-9]+.[0-9]+" | head -1`
	if [[ "$CUDA_VERSION" == "10.1" ]]; then
		echo "Building for Linux with CUDA 10.1";
		make -f Makefile.linux-cuda101;
	elif [[ "$CUDA_VERSION" == "10.2" ]]; then
		echo "Building for Linux with CUDA 10.2";
		make -f Makefile.linux-cuda102;
	fi
    else
        echo "Building for Linux without CUDA ...";
        make -f Makefile.linux;
    fi
elif [[ "$OS" == "darwin" ]]; then
    echo "Building for macOS ...";
    make -f Makefile;
fi

popd
