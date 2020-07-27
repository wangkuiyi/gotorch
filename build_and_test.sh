#!/bin/bash

set -e

pushd cgotorch
make
popd

CGO_LDFLAGS="-L$TORCHPATH/lib" go run 01-backward.go
