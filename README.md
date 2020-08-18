# `gotorch`

[![Travis CI](https://travis-ci.com/wangkuiyi/gotorch.svg?branch=develop)](https://travis-ci.com/wangkuiyi/gotorch)

GoTorch is a Go-idiomatic PyTorch, including PyTorch modules and functionals
rewritten in Go.  A complete story about GoTorch involves

- the [Go+](https://github.com/goplus/gop) community,
- the PyTorch community, and
- the TenosrFlow XLA ecosystem.


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

- [Build and Test](CONTRIBUTING.md)

- Examples

  - [MNIST trainining](./mnist_test.go)

- [Design Docs](./doc/design.md)
