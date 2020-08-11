FROM ubuntu:bionic

RUN apt-get -qq update && apt-get -qq install -y curl unzip make git clang clang-format cppcheck python3-pip
RUN python3 -m pip install -qq torch==1.6.0 torchvision
RUN curl -Ls https://golang.org/dl/go1.14.6.linux-amd64.tar.gz | tar -C /usr/local -xz

RUN echo 'export GOPATH=/go' >> /root/.bashrc
RUN echo 'export PATH=/usr/local/go/bin:/go/bin:$PATH' >> /root/.bashrc
RUN echo 'export LD_LIBRARY_PATH=$GOPATH/src/github.com/wangkuiyi/gotorch/cgotorch/libtorch/lib:$LD_LIBRARY_PATH' >> /root/.bashrc
