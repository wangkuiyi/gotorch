# `gotorch`

[![Travis CI](https://travis-ci.com/wangkuiyi/gotorch.svg?branch=develop)](https://travis-ci.com/wangkuiyi/gotorch)
[![codecov](https://codecov.io/gh/wangkuiyi/gotorch/branch/develop/graph/badge.svg)](https://codecov.io/gh/wangkuiyi/gotorch)
[![CircleCI](https://circleci.com/gh/wangkuiyi/gotorch.svg?style=shield)](https://circleci.com/gh/wangkuiyi/gotorch)
[![GoDoc](https://img.shields.io/badge/godoc-reference-teal.svg)](https://pkg.go.dev/mod/github.com/wangkuiyi/gotorch)

GoTorch is a Go-idiomatic PyTorch, including PyTorch modules and functionals
rewritten in Go.

## From GoTorch to Go+Torch

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
Go source code ━(calls)→ GoTorch ━(calls)→ libtorch ━(links)→ pytorch/xla ━(calls)→ XLA ops
     │
 [Go compiler]
     ↓
executable binary  # x86_64, ARM, CUDA, TPU
                   # Linux, macOS, Android, iOS
```

## GoTorch Documents

- [Build and Test](CONTRIBUTING.md)

- Examples

  - [MNIST trainining](./mnist_test.go)

- [Design Docs](./doc/design.md)
