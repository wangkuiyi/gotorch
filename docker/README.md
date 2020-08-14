# The DevBox Docker Image

The [Dockerfile](/docker/Dockerfile) in this directory defined
the devbox Docker images of GoTorch, this Docker image includes
some building toolkit of Go, Python and C++.

## Building the Dev Docker Image

To build the dev Docker image running on CPU, please run the following command:

``` bash
$ docker build -t gotorch:dev .
```

To build the dev Docker image running on GPU device, you can
specify a [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/tags) Docker images
as the base image, which includes [CUDA](https://developer.nvidia.com/cuda-zone)
toolkit. Please run the following command with `--build-arg` argument:

``` bash
$ docker build --build-arg BASE_IMAGE=nvidia/cuda:10.2-runtime-ubuntu18.04 -t gotorch:dev-gpu .
```
