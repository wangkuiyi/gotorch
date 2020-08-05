#!/bin/bash
set -e
pre-commit run -a
if [ $ARCH == "Linux"]; then
  (cd cgotorch; make)
else
  (cd cgotorch; make -f Makefile.linux)
fi

go install ./...
go test -v ./...