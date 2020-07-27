#!/bin/bash

set -ex

go env -w GO111MODULE=on
go get \
   golang.org/x/lint/golint

echo "$GOPATH"
echo "$GOROOT"

ls -lt "$GOPATH"/bin

cp "$GOPATH"/bin/* /usr/local/bin/