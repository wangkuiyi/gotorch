FROM ubuntu:bionic

RUN apt-get -qq update && apt-get -qq install -y curl unzip make git clang clang-format cppcheck python3-pip
RUN python3 -m pip install -qq pre-commit cpplint
RUN curl -Ls https://golang.org/dl/go1.14.6.linux-amd64.tar.gz | tar -C /usr/local -xz
RUN GOPATH=/go /usr/local/go/bin/go get golang.org/x/lint/golint
RUN GOPATH=/go /usr/local/go/bin/go get github.com/stretchr/testify
RUN cp /go/bin/* /usr/local/bin/

RUN echo 'export GOPATH=/go' >> /root/.bashrc
RUN echo 'export PATH=/usr/local/go/bin:/go/bin:$PATH' >> /root/.bashrc
