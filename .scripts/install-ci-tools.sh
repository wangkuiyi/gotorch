#!/bin/bash

set -e

EV="1.14.6"
V=$(go version | cut -f 3 -d ' ' | sed 's/go//')
if [[ "$V" < "EV" ]]; then
    O=$(uname | tr [:upper:] [:lower:])
    curl -Lso go.tar.gz https://golang.org/dl/go"$EV"."$O"-amd64.tar.gz
    sudo tar -C /usr/local -xzf go.tar.gz
    sudo mv /usr/local/go/bin/* /usr/local/bin/
fi

go version
go get golang.org/x/lint/golint

cp "$GOPATH"/bin/* /usr/local/bin/
