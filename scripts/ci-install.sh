#!/bin/bash

set -e

go env -w GO111MODULE=on
go get \
   golang.org/x/lint/golint

echo "GOPATH"
ls -lt "GOPATH"/bin

cp "$GOPATH"/bin/* /usr/local/bin/