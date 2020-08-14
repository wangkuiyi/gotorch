# Releases

## Release 2020-08

The initial release of GoTorch.

- Invented synchronous GC of tensors in Go.  See `SetTensorFinalizer`, `GC`, `FinishGC` in `tensor.go`.
- Enabled module definition in Go, using Go's interface mechanism.
- Bound some native functions and functionals, and ported some torch.nn modules, sufficient to port the MNIST training and DCGAN examples.
- Support Linux/x86\_64 with and without CUDA GPU, macOS x86\_64, and 32-bit Raspbian.
- The initial benchmark using MNIST training shows that the GoTorch version is 22\% faster than its counterpart based on a custom-built fully-optimized version of PyTorch.
- If you install PyTorch using `pip install`, you will see the GoTorch version is 200~300\% faster than the PyTorch counterpart.
- Verified that the GoTorch version of MNIST training works the same as the C++ version directly calls libtorch.
