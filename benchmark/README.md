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
cd benchmark && make && time ./mnist
```

Both programs use only CPU but no GPU.

Because the GoTorch version uses libtorch 1.6.0, please make sure you have
PyTorch 1.6.0 installed for the PyTorch version.  Also, both GoTorch and PyTorch
versions use the data loader from torchvision.

```bash
python3 -m pip install torch==1.6.0 torchvision
```

On a Late 2018 MacBook Pro runninng macOS 10.15.5, the throughtput of the GoTorch
program is about **4 times** of the PyTorch version.  The measure of throughput
doesn't include model instantiation, data preparation, but only the train loop.

Consider the total running time, the GoTorch version takes **31 seconds**, but
the PyTorch version takes about **1 minute**.

A typical run outputs the following:

``` bash
~/go/src/github.com/wangkuiyi/gotorch $ time python benchmark/mnist.py
The throughput: 5106.188462476132 samples/sec
python benchmark/mnist.py  344.71s user 4.87s system 585% cpu 59.740 total

~/go/src/github.com/wangkuiyi/gotorch $ time go test -run TrainMNIST
2020/08/12 09:59:18 Throughput: 10631.552987 samples/sec
PASS
ok  	github.com/wangkuiyi/gotorch	29.664s
go test -run TrainMNIST  83.48s user 230.13s system 1006% cpu 31.171 total

~/go/src/github.com/wangkuiyi/gotorch $ make -C benchmark && time ./benchmark/mnist
The throughput: 20303.140625 samples/sec
./mnist  86.11s user 0.89s system 557% cpu 15.611 total
```
