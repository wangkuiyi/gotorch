#!/bin/bash

set -e

go version
go get golang.org/x/lint/golint

sudo cp "$GOPATH"/bin/* /usr/local/bin/
