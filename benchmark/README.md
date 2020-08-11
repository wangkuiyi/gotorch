# Benchmark on Different Front Language of Torch

## Prepare the Benchmarking Environment

Build a Docker image as the benchmark environment:

``` bash
docker build -t gotorch:benchmark .
```

Start a Docker container to setup your benchmark environment, you can use `--cpus` argument to limit
the CPU cores:

``` bash
docker run --cpus=2 --rm -it -v $PWD/..:/go/src/github.com/wangkuiyi/gotorch -w /go/src/github.com/wangkuiyi/gotorch gotorch:benchmark bash
```

## Compare the Throughput

Throughput is a measure of the performance of a Deep Learning engine. The Torch community provides
C++ and Python as the front language, this project tried to provide Go+ as the front language of Torch.
This section would compare the throughput on these front language.

| Front language | Throughput (1*CPUs)|Throughput (2*CPUs)|Throughput (4*CPUs)
| -- | -- | -- |
| C++(LibTorch) || 3999 (samples/sec)| |
| Go(GoTorch) ||4697 (samples/sec)||
| Python(PyTorch) || 3728 (samples/sec)||
