# Benchmarking: GoTorch v.s. PyTorch v.s. C++ libtorch

## Benchmarks

To run the GoTorch version of MNIST training for five epochs, type the following
command.  You need to follow the development
[document](https://github.com/wangkuiyi/gotorch/blob/develop/CONTRIBUTING.md)
to install the required software, including the Go compiler.

```bash
/usr/bin/time go test -v -run TrainMNIST
```

To run the PyTorch version with the same parameters, network
structure, and dataset, type the following command.  You need to install
PyTorch with the command `python3 -m pip install torch==1.6.0 torchvision`.

```bash
/usr/bin/time python benchmark/mnist.py
```

To run the C++ version that calls the C++ core of PyTorch, a.k.a., `libtorch`,
type the following command:

```bash
make -C benchmark && /usr/bin/time ./benchmark/mnist
```

All the above programs use only CPU but no GPU.  We will add GPU support in the
next
week.

All the above programs directly or indirectly use libtorch 1.6.0.

## Measures

We compare the three benchmark programs by **throughput** -- the number of
images that the training loop consumes per second.  By doing so, we exclude
the time to compile the program, download and preload the training data,
and model instantiation and initialization.

On a Late 2018 MacBook Pro running macOS 10.15.5, a typical run of the three
programs output their throughput.

```bash
$ /usr/bin/time python benchmark/mnist.py
The throughput: 4236.326723046286 samples/sec
       71.80 real       417.61 user         4.68 sys

$ /usr/bin/time go test -run TrainMNIST
2020/08/12 10:52:31 Throughput: 9282.386087 samples/sec
PASS
ok      github.com/wangkuiyi/gotorch    33.026s
       34.27 real        97.30 user       263.18 sys

$ make -C benchmark && /usr/bin/time ./benchmark/mnist
The throughput: 21426.107422 samples/sec
       14.37 real        83.01 user         0.64 sys
```

## Observations

We have the following observations on the throughput:

1. The throughput of the GoTorch version is about **2 times** of the PyTorch
   version.
1. The throughput of the C++ version is about **2 times** of the GoTorch
   version.

We have the following observations on the total run time:

1. The GoTorch version takes **34 seconds**,
1. The PyTorch version takes about **71 seconds**.
1. The C++ version that calls libtorch directly takes only **14 seconds**.

We have the following observation on the ease of programming.

1. The GoTorch version has almost the same number of lines of code as the
   PyTorch version.
1. The C++ version is lengthy and challenging to read or write.  Thanks to our
   friends from Facebook PyTorch team, without their help, we cannot write the
   C++ version and well-benchmark it.

## Analysis

**Why is the GoTorch version runs that much faster than the PyTorch version?**

Diving into details, we found that the libtorch installed from the
`pip install` command and used by the PyTorch version doesn't allow MKL and
OpenMP to use all CPU cores.

We verified this fact in the additional
[trial](https://github.com/wangkuiyi/gotorch/pull/105#issuecomment-672336636),
where we force both the GoTorch and PyTorch version to use only one CPU core,
they have similar throughput -- in particular, the GoTorch version is 22\%
faster than the PyTorch version.

**Why is the C++ version runs that much faster than the Go version?**

The reasons include:

1. A C++ function call usually takes a few ns; however, it takes about 15μm for
   a Go function to call a C/C++ function.
1. The C++ program frees tensors out-of-scope using reference count maintained
   by `std::shared_ptr` or similar smart pointers.  However, Go's GC mechanism
   run a mark-and-sweep algorithm to detect and free unused tensors after
   each iteration.

## pprof

Two runs of the C++ version of MNIST training example on iMac 2015 without GPU.

```
$ ./mnist
The throughput: 14907.086914 samples/sec
```

```
$ ./mnist
The throughput: 15456.287109 samples/sec
```

Two runs of the GoTorch version on the same computer achieves throughtput very
close to the C++ counterpart.

```
$ go test -cpuprofile cpu.prof -memprofile mem.prof -v -run TrainMLPUsingMNIST
=== RUN   ExampleTrainMLPUsingMNIST
2020/08/19 14:04:44 No CUDA found; CPU only
2020/08/19 14:04:48 Epoch: 0, Loss: 0.1280
2020/08/19 14:04:53 Epoch: 1, Loss: 0.0659
2020/08/19 14:04:53 Throughput: 13678.129358 samples/sec
--- PASS: ExampleTrainMLPUsingMNIST (9.02s)
PASS
ok      github.com/wangkuiyi/gotorch    9.466s
```

```
$ go test -cpuprofile cpu.prof -memprofile mem.prof -v -run TrainMLPUsingMNIST
=== RUN   ExampleTrainMLPUsingMNIST
2020/08/19 14:05:05 No CUDA found; CPU only
2020/08/19 14:05:10 Epoch: 0, Loss: 0.1280
2020/08/19 14:05:14 Epoch: 1, Loss: 0.0659
2020/08/19 14:05:14 Throughput: 14197.120025 samples/sec
```

Run the following command to convert `cpu.prof` into `profile001.pdf`.

```bash
go tool pprof -pdf cpu.prof
```

The two biggest block in the PDF files are

1. `runtime cgocall`: 5.84s (21.11\%)
1. `runtime_ExternalCode -> unknwon`: 19.38s (70.07\%)

It is reasonable as the external C++ code in libtorch takes most of the running
time, and it's known that Cgo call takes more time than Go-call-Go or C-call-C.

The PyTorch version using the official pip package runs much slower than the
GoTorch and C++ versions.

```
$ time python mnist.py
The throughput: 4692.161879128401 samples/sec
```
