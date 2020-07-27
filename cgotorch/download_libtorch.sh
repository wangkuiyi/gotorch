#!/bin/bash

if [[ $(uname) == "Darwin" ]]; then
    D=https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.5.1.zip
elif [[ $(uname) == "Linux" ]]; then
    D=https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcpu.zip
fi

curl -Lso libtorch.zip $D
