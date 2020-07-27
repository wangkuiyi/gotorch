FROM ubuntu:18.04

RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN apt-get -qq update
RUN apt-get install -y -qq unzip curl build-essential clang wget

ARG GO_MIRROR_URL=http://mirrors.ustc.edu.cn/golang
ENV GOPATH /root/go
ENV PATH /usr/local/go/bin:$GOPATH/bin:$PATH
RUN curl ${GO_MIRROR_URL}/go1.14.6.linux-amd64.tar.gz | tar -C /usr/local -xzf -

RUN wget -q https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcpu.zip
RUN unzip -qq -o libtorch-cxx11-abi-shared-with-deps-1.5.1+cpu.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-1.5.1+cpu.zip 
ENV TORCHPATH /libtorch
ENV LD_LIBRARY_PATH $TORCHPATH/lib:$LD_LIBRARY_PATH
RUN ldconfig
