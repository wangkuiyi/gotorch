# GoTorch

[![TravisCI](https://travis-ci.com/wangkuiyi/gotorch.svg?branch=develop)](https://travis-ci.com/wangkuiyi/gotorch)
[![codecov](https://codecov.io/gh/wangkuiyi/gotorch/branch/develop/graph/badge.svg)](https://codecov.io/gh/wangkuiyi/gotorch)
[![CircleCI](https://circleci.com/gh/wangkuiyi/gotorch.svg?style=shield)](https://circleci.com/gh/wangkuiyi/gotorch)
[![GoDoc](https://img.shields.io/badge/godoc-reference-teal.svg)](https://pkg.go.dev/mod/github.com/wangkuiyi/gotorch)

GoTorch reimplements PyTorch high-level APIs, including modules and functionals,
in idiomatic Go.  Thus enables deep learning programming in Go and Go+.

This project is in its very early stage.

## Efficiency

Developing in Go is as efficiently as in Python.  The DCGAN training programs in
[GoTorch](https://github.com/wangkuiyi/gotorch/blob/develop/example/dcgan/dcgan.go)
and
[PyTorch](https://github.com/pytorch/examples/blob/4b119d735b802453479d739bf823f3f7d8d5d422/dcgan/main.py#L113-L273)
call similar APIs, have similar program structure, and have a similar number of
lines.

Go+ has a syntax similar to Python.  The Go+ compiler translates Go+ programs
into Go source programs.

## Benefits

1. Higher runtime efficiency

   Go programs run as efficiently as C++.

1. Training and prediction in the same language

   Programm the training system in Python and the online prediction in C++?  No,
   all in Go/Go+.  No need for TensorFlow graphs or PyTorch tracing.

1. Same data processing code for training and prediction

   Wrap OpenCV functions as TensorFlow operators in C++ for prediction and
   Python for training?  No, do it once, in Go.

1. A variety of machine learning paradigms

   GoTorch supports online learning, adversarial learning, reinforcement
   learning, and imitation learning -- those we cannot separate into training
   and prediction.

1. Same program for edge and cloud

   GoTorch programs compile and run on phones and self-driving cars as they do
   on servers and desktops.

## The Tech Stack

GoTorch works with the following open-source communities to form Go+Torch.

- the [Go+](https://github.com/goplus/gop) community,
- the PyTorch community, and
- the TenosrFlow XLA ecosystem.

The following figure reveals the stack of technologies.

```text
Go+ applications   # users write DL applications in Go+,
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
