# `gotorch`

[![TravisCI](https://travis-ci.com/wangkuiyi/gotorch.svg?branch=develop)](https://travis-ci.com/wangkuiyi/gotorch)
[![codecov](https://codecov.io/gh/wangkuiyi/gotorch/branch/develop/graph/badge.svg)](https://codecov.io/gh/wangkuiyi/gotorch)
[![CircleCI](https://circleci.com/gh/wangkuiyi/gotorch.svg?style=shield)](https://circleci.com/gh/wangkuiyi/gotorch)
[![GoDoc](https://img.shields.io/badge/godoc-reference-teal.svg)](https://pkg.go.dev/mod/github.com/wangkuiyi/gotorch)

GoTorch reimplements PyTorch high-level APIs, including modules and functionals,
in idomatic Go.  Thus enables deep learning programming in Go and Go+.

This project is in its very early stage.

## Efficiency

Developiong in Go is as efficiently as in Python.  You might want to compare the
DCGAN training program in
[GoTorch](https://github.com/wangkuiyi/gotorch/blob/develop/example/dcgan/dcgan.go)
and in
[PyTorch](https://github.com/pytorch/examples/blob/4b119d735b802453479d739bf823f3f7d8d5d422/dcgan/main.py#L113-L273).
Both programs use similar API and have program structure.  The GoTorch version is
even slightly shorter than the PyTorch version.

Go+ has syntax similar to Python.  The Go+ compiler translates Go+ programs into
Go source programs.

## Benefits

### Higher runtime efficiency

The benchmark of GoTorch programs shows about 20% performance improvement than
their PyTorch counterpart.

### Unifying trainng and prediction in the same language

Currently, we program the training system in Python and the online prediction
service in C++. Hnece the overhead of generating TensorFlow or ONNX graphs for
loading by the prediction service.  However, the most accepted form of
computation is not any type of graph, but programs!

### Unifying the data processing for training and prediction

Because training and prediction programs are in different langauges -- Python
and C++, we have to write the data processing code twice -- in Python and C++.
For example, to process an image using OpenCV, we need to wrap the OpenCV
function into a TensorFlow operator for prediction and write the Python wrapper
of the C++ operaotr for calling by the training program.

### Unifying the way to develop various machine learning paradigms

Please be aware that we can only split the batching learning of supervised
models into two stages -- training and prediction, and write them as two systems
and in difference languages. There are way more forms of machine learning --
online learning, adversarial learning, reinforcement learning, imitation
learning -- none of them could be splitted into two stages.

### Unifying the edge and the cloud

To protect user privacy, we want to run federated learning on smart phones.  To
make self-driving cars learn from their drivers, we want to run imitation
learning on the cars.  Python focuses on wasting your valuable battery capacity.
We need to program in comppiled langauges.

## The Tech Stack

GoTorch work with the following open-source communities to form Go+Torch.

- the [Go+](https://github.com/goplus/gop) community,
- the PyTorch community, and
- the TenosrFlow XLA ecosystem.

The following figure reveals the technical relationship.

```text
Go+ applications   # users write DL applicaitons in Go+,
     │             # whose syntax is as concise as Python
 [Go+ compiler]
     ↓
Go source code -→ GoTorch -→ libtorch -→ pytorch/xla -→ XLA ops
     │
 [Go compiler]
     ↓
executable binary  # x86_64, ARM, CUDA, TPU
                   # Linux, macOS, Android, iOS
```

## Documentation

- [Build and Test](CONTRIBUTING.md)

- Examples

  - [MNIST trainining](./mnist_test.go)

- [Design Docs](./doc/design.md)
