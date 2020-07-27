#!/bin/bash

if [[ $(uname) == "Darwin" ]]; then
    D=https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.5.1.zip
elif [[ $(uname) == "Linux" ]]; then
    D=https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcpu.zip
fi

# https://docs.travis-ci.com/user/languages/minimal-and-generic/#generic tells
# that the generic langauge of Travis CI has curl installed.
curl -Lso libtorch.zip $D
