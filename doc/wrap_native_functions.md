# Wrap PyTorch Native Functions

At the very core of PyTorch is a C++ library libtorch, which contains about 1600
basic deep learning operations known as native functions, most of which operate
tensors.

In PyTorch, Python functions in package `torch.nn.functional` and modules
classes in `torch.nn` call native functions via
[`pybind`](https://github.com/pybind/pybind11).  In GoTorch, the corresponding
packages `gotorch/nn/functional` and `gotorch/nn` call Go wrappers of these
native functions.

This tutorial explains how to wrap native functions using
[Cgo](https://blog.golang.org/cgo).  We will go over three layers of the
wrapping:

1. the PyTorch native functions defined in C++ and released with libtorch,
1. the C wrapper functions of native functions callable by Go wrappers via Cgo,
   and
1. the Go wrapper functions callable by higher-level Go APIs in `gotorch/nn` and
   `gotorch/nn/functional`.

## Native Functions and C++ Tensor

The PyTorch build system generates the C++ source code of native functions from
the YAML file
[`native_functions.yaml`](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml).

In this tutorial, let us try to wrap the following function `mm`, which is short
for matrix multiplication and appear in `native_functions.yaml` as the
following.

```yaml
- func: mm(Tensor self, Tensor mat2) -> Tensor
  use_c10_dispatcher: full
  variants: function, method
  dispatch:
    CPU: mm_cpu
    CUDA: mm_cuda
    SparseCPU, SparseCUDA: _sparse_mm
```

This YAML segment tells that `mm` has three implementations: `mm_cpu`,
`mm_cuda`, and `_sparse_mm`.  The definition of `mm` calls the dispatcher
defined in `c10`, a sub-package of libtorch, to look up and call the
best-matching implementation.

The `variants` field tells that `mm` has two forms: a C++ global function and
the method of C++ class `at::Tensor`, where `at` stands for ATen, which is
another sub-package that defines the fundamental data type `Tensor`.

This tutorial covers the wrapping of the form of global function.  Readers are
welcome to take the wrapping of the method form as an exercise.

The PyTorch build system generates declarations of global function forms in
`ATen/Functions.h`.  Download and unzip the official release of libtorch zip
[archives](https://pytorch.org/get-started/locally/), you will find the file as
`libtorch/include/ATen/Functions.h`.  The declaration of the native function
`mm` is as the following.

```cpp
namespace at {
CAFFE2_API Tensor mm(const Tensor & self, const Tensor & mat2);
}  // namespace at
```

The method `at::Tensor::mm` is in `libtorch/include/ATen/core/TensorBody.h`.

```cpp
namespace at {
class CAFFE2_API Tensor {
  Tensor mm(const Tensor & mat2) const;
 protected:
  c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> impl_;
};
}  // namespace at
```

The only data member of class `at::Tensor` is `impl_`, a smart pointer pointing
to an object that implements the details.

## C Wrappers and C Tensor

The native functions are in C++, and Go can only call C but not C++ functions,
so we need to write a C wrapper for each native function.

There are additional reasons for C wrappers:

1. If a native function returns a `Tensor`, its C wrapper creates a reference
   object on the heap that points to the underlying tensor, so it will not be
   free up by C++ smart pointers so that the Go code can use it.
1. Encapsulate the C++ class `at::Tensor` by a C type that can be used by Go
   code.
1. The native functions might throw C++ exceptions.  The C wrappers convert
   exceptions into C strings, which, in turn, converted by the Go wrapper into
   Go panics.

With Cgo, Go programs can refer to C symbols with their names prefixed by `C.`.
For example, the following Go function `MyExit` calls the C standard function
`exit` declared in `stdlib.h` as `C.exit`.

```go
// #include <stdlib.h>
import "C"
func MyExit(x int) {
    C.exit(x)
}
```

We put all C wrappers of native functions in the subdirectory `cgotorch`.  In
`cgotorch/cgotorch.h`, we can see the wrapper of `at::Tensor` and `at::mm`.

```c
extern "C" {
typedef at::Tensor* Tensor;
const char *MM(Tensor a, Tensor b, Tensor *result);
}
```

C does not have classes, but C has pointers, so we define C type `Tensor` as a
pointer to `at::Tensor`.  Go programs can use C pointers as of type
`unsafe.Pointer`.

The C wrapper `MM` returns a string serialization of the possible C++ exception
thrown by libtorch if there is any, or `nullptr`.

The implementation of the C wrapper `MM` is in C++ and in `cgotorch/torch.cc`.

```cpp
const char *MM(Tensor a, Tensor b, Tensor *result) {
  try {
    at::Tensor c = at::mm(*a, *b);
    *result = new at::Tensor(c);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}
```

It runs the following steps, which are in most other wrappers too.

1. It calls the native function `mm` and creates the result tensor `c` on the
   stack.
1. It allocates a heap object `*result` and **"copies"** `c` to `*result`.  This
   step is necessary because the return from `MM` will destruct `c`.  This step
   is highly efficient as it doesn't actually copy the content of `c`, because
   `at::Tensor` contains only a smart pointer to the underlying tensor content.
1. It returns the string-serialization of the exception if there is any, or
   `nullptr`.

## Go Wrappers and Go Tensor

In package `gotorch`, we define the Go wrappers of native functions, which
operates the Go type `Tensor` defined in `tensor.go`.

```go
// Tensor wrappers a pointer to C.Tensor
type Tensor struct {
    T *unsafe.Pointer
}
```

At first glance, this definition looks too much more complicated than the C type
Tensor, which is a pointer.

The literal translation of the C type Tensor into Go is

```go
type Tensor unsafe.Pointer
```

The Go type `unsafe.Pointer` represents any C pointer type.

However, this is not enough because we need to attach to each `Tensor` a
[*finalizer*](https://golang.org/pkg/runtime/#SetFinalizer) to free the
underlying `at::Tensor` object when a Go tensor is out-of-use.  Only Go pointers
but not C pointers could have finalizers attached, so we add a `*`, indicating a
Go pointer, to the above definition, this makes it a Go pointer to a C pointer.

```go
type Tensor *unsafe.Pointer
```

This pointer-to-pointer is still not enough as we need to attach methods of
`at::Tensor`, like `MM`, `Add`, and `To`, to the Go type.  Again, Go types
with pointer [base type](https://golang.org/ref/spec#Method_declarations)
cannot have methods, so we wrap the above pointer-to-pointer in a struct.

Given the Go type `Tensor`, the Go wrapper `MM` is as follows.

```go
package gotorch

func MM(a, b Tensor) Tensor {
    var t C.Tensor
    MustNil(unsafe.Pointer(C.MM(C.Tensor(*a.T), C.Tensor(*b.T), &t)))
    SetTensorFinalizer((*unsafe.Pointer)(&t))
    return Tensor{(*unsafe.Pointer)(&t)}
}
```

It runs the following steps:

1. It passes the address of `t` of type `C.Tensor` to the C wrapper, `C.MM`.
   Because `C.Tensor` is a pointer to `at::Tensor`, the line `*result = new
   at::Tensor(c)` in the C wrapper makes `t` pointing to the newly allocated
   tensor on the heap.
1. It calls `MustNil` to check the C string returned from the C wrapper.
   `MustNil` panics if the string is not `nullptr`.
1. It calls `gotorch.SetTensorFinalizer` to attach the finalizer to the tensor
   returned by `C.MM`.
1. It returns a value of the Go type `Tensor` that encapsulates `t`.
