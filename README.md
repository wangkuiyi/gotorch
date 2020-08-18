# `gotorch`

[![Travis CI](https://travis-ci.com/wangkuiyi/gotorch.svg?branch=develop)](https://travis-ci.com/wangkuiyi/gotorch)

A Go-idiomatic binding of PyTorch, to be called by [Go+](https://github.com/goplus/gop) programs.

### Build and Test

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md).

### Examples

Please refer to examples in `*_test.go` files.

### Design Docs

Please refer to documents in [`./doc`](./doc).

### Architecture

```
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
