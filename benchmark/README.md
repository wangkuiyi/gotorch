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

Both programs use only CPU but no GPU.

Because the GoTorch version uses libtorch 1.6.0, please make sure you have
PyTorch 1.6.0 installed for the PyTorch version.  Also, both GoTorch and PyTorch
versions use the data loader from torchvision.

```bash
python3 -m pip install torch==1.6.0 torchvision
```

On a Late 2014 iMac runninng macOS 10.15.5, the throughtput of the GoTorch
program is about **3 times** of the PyTorch version.  The measure of throughput
doesn't include model instantiaation, data preparation, but only the train loop.

Consider the tatal running time, the GoTorch version takes **43 seconds**, but
the PyTorch version takes about **4 minutes**.

A typical run outputs the following:

```
yi@WangYis-iMac:~/go/src/github.com/wangkuiyi/gotorch (benchmark)*$ time python benchmark/mnist.py
The throughput: 4177.628869040359 samples/sec

real    1m22.022s
user    3m59.016s
sys 0m11.934s

yi@WangYis-iMac:~/go/src/github.com/wangkuiyi/gotorch (benchmark)*$ time go test -run TrainMNIST
2020/08/11 09:40:21 Throughput: 13129.192544 samples/sec
PASS
ok      github.com/wangkuiyi/gotorch    23.755s

real    0m24.871s
user    0m43.235s
sys 0m37.307s
```
