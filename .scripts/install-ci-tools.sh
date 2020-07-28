#!/bin/bash

set -e

V=$(go version | cut -f 3 -d ' ' | sed 's/go//')
if [[ "$V" < "1.13" ]]; then
    curl -Lso go.tar.gz https://golang.org/dl/go1.14.6.linux-amd64.tar.gz
    sudo tar -C /usr/local -xzf go.tar.gz
    export PATH=/usr/local/go/bin:$PATH
fi

go version
go get golang.org/x/lint/golint

cp "$GOPATH"/bin/* /usr/local/bin/
