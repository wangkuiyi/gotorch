#!/bin/bash

set -ex

go get golang.org/x/lint/golint
sudo cp "$GOPATH"/bin/* /usr/local/bin/
