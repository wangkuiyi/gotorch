# This Dockerfile generates the image used by CircleCI.  To update the image,
# run the following two commands:
#    docker build -t cxwangyi/gotorch .
#    docker push cxwangyi/gotorch
# The Docker image name cxwangyi/gotorch is the one used in cofig.yml.
FROM circleci/golang:1.15

RUN sudo apt-get -qq update && sudo apt-get -qq install -y curl unzip make git clang clang-format cppcheck python3-pip yamllint ruby-dev
RUN sudo python3 -m pip install -qq pre-commit cpplint
RUN sudo gem install mdl
RUN go get golang.org/x/lint/golint
RUN sudo cp $GOPATH/bin/* /usr/local/bin/
# install gocv
RUN go get -u -d gocv.io/x/gocv
RUN cd $GOPATH/src/gocv.io/x/gocv && make install
