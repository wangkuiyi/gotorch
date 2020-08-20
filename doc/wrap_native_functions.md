# Wrap PyTorch Native Functions

At the very core of PyTorch is a C++ library libtorch, which contains about 1600
basic deep learning operations known as native functions, most of which operate
tensors.

In PyTorch, Python functions in package `torch.nn.functional` and modules
classes in `torch.nn` call native functions via pybind. In GoTorch, the
corresponding packages `gotorch/nn/functional` and `gotorch/nn` call them via
Cgo.

This tutorial explains how to expose native functions via Cgo to Go.

First of all, the list of native functions is in
[native_functions.yaml](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml).

In this tutorial, let us try to wrap the following function mm, which shorts for
matrix multiplication.

```yaml
- func: mm(Tensor self, Tensor mat2) -> Tensor
  use_c10_dispatcher: full
  variants: function, method
  dispatch:
    CPU: mm_cpu
    CUDA: mm_cuda
    SparseCPU, SparseCUDA: _sparse_mm
```

This small YAML segment tells that mm has three implementations:
mm_cpu,mm_cuda,_sparse_mm. When a PyTorch runs and calls mm, the definition of
mm calls the dispatcher to lookup and call the best-matching implementation.
The dispatcher is in `c10`, a sub-package of libtorch.

The variants field tells that mm has two forms: a global function and the method
of class aten::Tensor, where aten is another subpackage of libtorch that defines
the fundamental data type tensor.

To wrap mm, we don’t care about its implementations but its two forms.

The global functions are in `ATen/Functions.h`, which is not in the PyTorch
Github repo, because it is generated at build time from native_functions.yaml.
If you download the pre-built libtorch releases from PyTorch official Web
[site](https://pytorch.org/get-started/locally/) and unzip it, you will find the
file in directory `libtorch/include/ATen/Functions.h`.

```c++
CAFFE2_API Tensor mm(const Tensor & self, const Tensor & mat2);
```

The method `Tensor::mm` is in `TenosrBody.h`, which, again is generated from
native_functions.yaml. You can find it in
`libtorch/include/ATen/core/TensorBody.h`.

```c++
class CAFFE2_API Tensor {
  ...
  Tensor mm(const Tensor & mat2) const;
  ...
};
```

Go programs can call C functions, via [Cgo](https://golang.org/cmd/cgo/).  For
example, to call the standard C function exit with a parameter 0, we can write a
Go code line.

```go
C.exit(0)
```

Here is a Go function MyExit that calls the C function exit.

```go
func MyExit(x int) {
    C.exit(x)
}
```

The Go function MM that calls the native function mm is as follows, in tensor.go.

```go
// MM multiplies each element of the input two tensors
func MM(a, b Tensor) Tensor {
	var t C.Tensor
	MustNil(unsafe.Pointer(C.MM(C.Tensor(*a.T), C.Tensor(*b.T), &t)))
	SetTensorFinalizer((*unsafe.Pointer)(&t))
	return Tensor{(*unsafe.Pointer)(&t)}
}
```

It takes two parameters a and b; both are of type Tensor.

```go
// Tensor wrappers a pointer to C.Tensor
type Tensor struct {
	T *unsafe.Pointer
}
```

The Go type Tensor is indeed a pointer, which points to a C++ aten::Tenosr
object. Go allows representing a C pointer by `unsafe.Pointer`.  We define `T`
as `*unsafe.Pointer`, which is a Go pointer to a C pointer.  We need this
pointer-to-pointer because only a Go pointer could have a finalizer attached to
it.  And we need to attach a finalizer to each tensor object so that when the
tensor is out-of-use, the finalizer can free the underlying C++ object.

Also, we wrap this pointer-to-pointer in a struct, because Go doesn’t allow us
to attach methods to a pointer type; however, the type Tensor needs methods like
MM.

With Cgo, all Go identifiers with prefix “C.” are C identifiers.  In this
example, `C.MM` is the C function defined in `cgotorch/cgotorch.h`.

```c
const char *MM(Tensor a, Tensor b, Tensor *result);
```

You might be wondering why the return type is a string.  It is because all
libtorch functions might throw C++ exceptions.  If it does, we want the C
function MM to catch whatever thrown and convert the reason into a string, which
is then converted by `MustNil`, called in the Go function `MM`, as shown in the
above code snippet, into a Go panic.

The C function MM has a C++ implementation in `cgotorch/torch.cc`.

```c++
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

This C++ implementation is the one called by the Go version of MM under the name
`C.MM`.  You might wonder why not Go MM calls the native function `mm` directly
via `C.mm`, but `C.MM` in `cgotorch/torch.cc`.  The only reason is we need to
define `C.MM` to capture possible C++ exceptions thrown by the native function
`mm`.

As a summarization, to wrap up the global native function mm, we need to

Define `C.MM` in `cgotorch/torch.cc` to call mm and capture possible C++
exceptions.  Add `C.MM`’s C signature into `cgotorch/cgotorch.h`.  Rebuild
cgotorch by running `cgotorch/build.sh`.  This command generates
`cgotorch/libcgotorch.so` depending on your CPU type and OS.  Add the Go
function MM in `tensor.go` to call `C.MM`, convert the possible C++ exception
into a Go panic, and attach a finalizer to the tensor returned by `C.MM`.

We might ask how if the `aten::Tensor::mm`? We have not yet wrap it. Let us
leave it as homework. A hint is that Go can only call C functions but not C++
functions. So, we need to wrap `aten::Tensor::mm` into a C function and call it
from the Go method `Tensor.MM`.
