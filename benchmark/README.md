# Benchmarking: GoTorch v.s. PyTorch

To run the GoTorch version of MNIST training with 5 epochs, type the following
command:

```bash
time go test -v -run TrainMNIST
```

To run the Python version of MNIST training with the same parameters, network
structure, and dataset, type the following command:

```bash
time python benchmark/mnist.py
```

To run the Go version of MNIST training with the same parameters above, type
the following command:

```bash
make -C benchmark && time ./benchmark/mnist
```

All programs use only CPU but no GPU.

Because the GoTorch version uses libtorch 1.6.0, please make sure you have
PyTorch 1.6.0 installed for the PyTorch version.  Also, both GoTorch and PyTorch
versions use the data loader from torchvision.

```bash
python3 -m pip install torch==1.6.0 torchvision
```

On a Late 2018 MacBook Pro runninng macOS 10.15.5, the throughput of the GoTorch
program is about **2 times** of the PyTorch version and **0.5 times** of the
LibTorch version.  The measure of throughput
doesn't include model instantiation, data preparation, but only the train loop.

Consider the total running time, the GoTorch version takes **34 seconds**, but
the PyTorch version takes about **71 seconds**, LibTorch version takes only
**14 seconds**.

A typical run outputs the following:

``` bash
~/go/src/github.com/wangkuiyi/gotorch $ /usr/bin/time python benchmark/mnist.py
The throughput: 4236.326723046286 samples/sec
       71.80 real       417.61 user         4.68 sys

~/go/src/github.com/wangkuiyi/gotorch $ /usr/bin/time go test -run TrainMNIST
2020/08/12 10:52:31 Throughput: 9282.386087 samples/sec
PASS
ok  	github.com/wangkuiyi/gotorch	33.026s
       34.27 real        97.30 user       263.18 sys

~/go/src/github.com/wangkuiyi/gotorch $ make -C benchmark && /usr/bin/time ./benchmark/mnist
The throughput: 21426.107422 samples/sec
       14.37 real        83.01 user         0.64 sys
```
