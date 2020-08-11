# Benchmark on Different Front Language of Torch

The following table compared the execution time of MNIST training on different Torch front language.

| Front Language | Execution Time |
| -- | -- |
| C++(LibTorch) | |
| Go(GoTorch) | 321.89s user 604.86s system 642% cpu 2:24.13 total|
| Python(PyTorch) | |

You can run the command the reproduce this experiment on your host:

PyTorch:

``` bash
$ time python ./python/mnist.py
The Average Throughput: 3319 samples/sec
python3 python/mnist.py  533.10s user 9.24s system 573% cpu 1:34.55 total
```

GoTorch:

``` bash
$ time go run ./go
The average throughout: 7833 samples/sec
go run ./go  120.85s user 241.99s system 842% cpu 43.058 total

```

LibTorch:

``` bash
$ cd cpp
$ make && time ./mnist
The average throughtput: 2422 samples/sec
./mnist  126.53s user 1.71s system 562% cpu 22.802 total
```
