FROM circleci/golang:1.15

RUN sudo apt-get -qq update && sudo apt-get -qq install -y curl unzip make git clang clang-format cppcheck python3-pip
RUN sudo python3 -m pip install -qq pre-commit cpplint
RUN go get golang.org/x/lint/golint
RUN go get github.com/stretchr/testify
RUN sudo cp $GOPATH/bin/* /usr/local/bin/

ENV PATH=/usr/local/go/bin:$GOPATH/bin:$PATH
ENV LD_LIBRARY_PATH=/go/src/github.com/wangkuiyi/gotorch/cgotorch/libtorch/lib:$LD_LIBRARY_PATH
