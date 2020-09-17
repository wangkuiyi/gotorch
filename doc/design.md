# GoTorch

This document explains the motivations and critical design challenges of
GoTorch.

## GoTorch and Go+

GoTorch includes a Go binding of the C++ core of PyTorch, known as libtorch.
There are many language bindings of libtorch, including Rust and
Haskell. However, according to our survey, most Python users don’t feel
programming in Rust, Haskell, or Julia, is more efficient than in Python.  So,
language binding does not make much sense alone.

The complete story of GoTorch includes Go+, a language whose syntax is as
concise as Python, but its compiler generates Go programs. Programming deep
learning systems in Go+ is hopefully as efficiently as in Python, and Go+
translates the work into Go source code, which compiles into native code running
on servers and mobile devices, including phones, pads, and self-driving cars.

In addition to the Go binding of libtorch, GoTorch also includes the other two
layers of functionalities of PyTorch provided in Python -- `torch.nn.functional`,
and `torch.nn`.

## Layers of Functionalities

Generally, PyTorch provides three layers of APIs, not all of which are in
libtorch.

1. The finest-grained layer is in libtorch -- about 1600 native functions, each
   is a fundamental operation in mathematics or its corresponding gradient
   operation.  Each native function has CPU and GPU implementations. By linking
   libtorch with [XLA](https://github.com/pytorch/xla), we get an additional
   implementation for Google TPU.

1. A higher-level abstraction is in the Python package `torch.nn.functional`,
   which provides functions defined in Python and calls native functions in
   C/C++.

1. The highest layer is modules; each is a Python class with a method forward
   defining the forward computation process and data members that can store
   states.

## Tensors and Garbage Collection

libtorch includes the C++ definition of the fundamental data type `at::Tensor`
and native functions that operate it.

The key design feature of the tensor is automatic garbage collection (GC). In
C++, the class `at::Tensor` contains only one data member, a strong reference
count-based smart pointer, `c10::intrusive_ptr`, which works like
`std::shared_ptr`, to the real tensor object.  This smart pointer performs
reference count-based GC, which frees a tensor once its reference count gets
zero.  Compared to Go and Java’s GC, which runs the mark-and-sweep algorithm,
reference count reacts instantly but cannot handle the case of
cyclic-dependency.

PyTorch programmers access `at::Tensor` from the Python binding. Python’s GC
uses strong reference count-based algorithm like `std::shared_ptr`, which cannot
handle cyclic-dependencies.  Therefore, Python runs mark-and-sweep from time to
time to free cyclic-dependencies.

Go provides an asynchronous API, `runtime.GC()`, to trigger GC and returns
immediately without waiting for the completion of GC.  If all tensors are in CPU
memory, this mechanism works; however, in deep learning, we would prefer to host
tensors in GPU memory, which is a precious resource. We prefer to free tensors
immediately when they are out-of-use so that the next iteration can create new
tensors in GPU.

### Synchronize Go’s GC

We started with inventing new GC mechanisms in the library, including adding a
global reference count table. However, after trying several strategies, we
noticed that we could customize Go’s GC for the tensor type in GoTorch
specifically to make it synchronous, or able to wait till the completion of GC
before returning.

The basic idea behind the design is the categorization of tensors by different
purposes in deep learning:

1. model parameters -- created before, updated during, and freed after the train
   loop,
1. buffers -- with lifespan similar to model parameters but used to BatchNorm to
   keep statistics of input data, and
1. intermediate results -- including those generated during the forward and
   backward pass in each step of the train loop.

The customized Go GC mechanism doesn’t handle the first two categories, which is
the topic of the next section, Porting Modules.

To handle intermediate results, GoTorch users need to call `gotorch.GC()` at the
beginning of each train loop step.  The first job of `gotorch.GC()` is to mark
that all tensors generated since then, which are considered intermediate
results, are subject to the customized GC.  After the train loop, users are
supposed to call `gotorch.FinishGC()` to unset the mark.

With the mark, each of the subsequent tensor generations, like a call to
`gotorch.RandN` or `gotorch.MM`, increments a waiting group and attaches a Go
finalizer to the created tensor.  Go’s GC will call this finalizer when it frees
a tensor, and the finalizer will close the underlying `at::Tensor` object and
unset the waiting group.

Then, `gotorch.GC` calls `runtime.GC()` and waits the waiting group be
completely unset. The call to `runtime.GC()` trigger’s Go’s mark-and-sweep
algorithm.  The waiting stops after the algorithm frees all tenors created since
the marking operation. Usually, the waiting takes less than one millisecond
(ms).

### A Typical Train Loop

Therefore, a typical train loop in GoTorch looks like the following:

```go
for iter := 0; iter < kIter; iter++ {
    gotorch.GC()
    mb := loadMinibatch()
    cost := forward(mb, model)
    cost.Backward()
    model.Update()
}
gotorch.FinishGC()
```

Some GoTorch APIs, including the data loader’s Scan method, implicitly calls
`gotorch.GC()`.

## Error Handling

The C++ code in libtorch might throw exceptions, which, we want GoTorch to catch
and convert into Go’s panics.  The Cgo code in the cgotorch directory calls
libtorch functions, catches C++ exceptions, and returns a C string.  If there
were no exceptions, the C string is NULL.

Here is an example that wraps `torch::randn` in libtorch.

```c++
const char *RandN(int64_t *size, int64_t length, int64_t require_grad,
                  Tensor *result) {
  try {
    at::Tensor t =
        torch::randn(torch::IntArrayRef(size, length),
                     at::TensorOptions().requires_grad(require_grad));
    *result = new at::Tensor(t);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}
```

We see that `torch::randn` defined in libtorch returns an `at::Tensor`. However,
the Cgo wrapper `RandN` returns a C string representing the possible C++
exception.  The last parameter of `RandN` is the returned tensor.

The Go function `gotorch.RandN` calls the wrapper `RandN` and converts the
returned C string, if not null, into a Go panic by calling `MustNil`.

```go
func RandN(shape []int64, requiresGrad bool) Tensor {
    rg := 0
    if requiresGrad {
        rg = 1
    }
    var t C.Tensor
    MustNil(unsafe.Pointer(C.RandN((*C.int64_t)(unsafe.Pointer(&shape[0])),
        C.int64_t(len(shape)), C.int64_t(rg), &t)))
    SetTensorFinalizer((*unsafe.Pointer)(&t))
    return Tensor{(*unsafe.Pointer)(&t)}
}
```

## Porting Functionals

The Python package torch.nn.functional provides functions defined as compounds
of native functions. Generally, each Python function corresponds to a C++
function in namespace `torch::nn::functional`. So we need to expose the C++
function via Cgo, and then define a Go function that calls the Cgo function.

Let’s take `torch.nn.functional.linear` as an example. The first step is to
expose the C++ function via Cgo. To do that, we define a C/C++ function in
`gotorch/cgotorch/functionals.cc` as the wrapper of the C++ function in
libtorch.

```c++
const char *Linear(Tensor input, Tensor weight, Tensor bias, Tensor *result) {
  try {
    auto out = torch::linear(*input, *weight, (bias ? *bias : torch::Tensor()));
    *result = new at::Tensor(out);
    return nullptr;
  } catch (const std::exception &e) {
    return exception_str(e.what());
  }
}
```

The next step is to define a Go function in `gotorch/nn/functional/*.go` to call
the above wrapper via Cgo.

```go
func Linear(input, weight, bias torch.Tensor) torch.Tensor {
    var t C.Tensor
    var cBias C.Tensor
    if bias.T != nil {
        cBias = C.Tensor(*bias.T)
    }
    torch.MustNil(unsafe.Pointer(C.Linear(
        C.Tensor(*input.T),
        C.Tensor(*weight.T), cBias, &t)))
    torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
    return torch.Tensor{(*unsafe.Pointer)(&t)}
}
```

Now, we can use the function ported from `torch.nn.functional` to
`gotorch/nn/functional` in our application programs.

```go
import torch "github.com/wangkuiyi/gotorch"
import F "github.com/wangkuiyi/gotorch/nn/functional"

input := torch.RandN([]int64{32, 100}, false)
weight := torch.RandN([]int64{100, 10}, true)
out := F.Linear(input, weight, torch.Tensor{})
```

## Porting Modules

A module represents part of the forward computation as a functional. The
difference is that a module, represented by a Python or C++ class, could have
data members, whereas functionals cannot.

PyTorch provides two module-definition frameworks, one in Python and the other
in C++. These two frameworks are independent of each other -- the Python one
doesn't call the C++ one.

Using either framework, users define a module by deriving from the base class
`Module`.  And users need to mark a data member if it is any one of the three
kinds: (1) parameters, (2) buffers, and (3) sub-modules.  The marking is known
as the state type registration.

### State Type Registration

To explain type registration, let us start by defining the fully-connected
linear module using the C++ framework.  The example calls `register_parameter`
to denote that `weight` and `bias` are parameters.

```c++
void LinearImpl::reset() {
  weight = register_parameter("weight",
    torch::empty({options.out_features(), options.in_features()}));
  if (options.bias()) {
    bias = register_parameter("bias", torch::empty(options.out_features()));
  } else {
    bias = register_parameter("bias", {}, /*requires_grad=*/false);
  }
  reset_parameters();
}
```

The registration is necessary.  For example, the `Module::get_parameters` method
traverses all data members registered as parameters.  And `Module::to(device)`
moves all parameters and buffers, as well those in sub-modules recursively, to
the specified device.

The Python module-definition framework doesn't require users to call
`register_{parameter|buffer|module}` explicitly as it utilizes the
`__setattr__` method and other customization capabilities of Python to call
these functions automatically.  However, such convenience doesn't always work.
In the following example of defining the fully-connected module in Python, we
see that users would have to call `register_parameter` if a parameter is
optional.

```python
def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
```

The above two examples explain what a module-definition framework needs to do.

### Registeration in GoTorch

Go doesn't have the concept of class-hierarchy. What is close to class
derivation is [embedded fields](https://golang.org/ref/spec#Struct_types).  Just
like deriving a module class from the base class `Module` in PyTorch, GoTorch
users defining a struct type that has an anonymous field of type `Module`.

Unlike PyTorch, which relies on the registration mechanism to denote types of
data members, GoTorch uses Go's field tag and reflection.  If a Tensor-typed
field has the tag `gotorch:"buffer"`, it is a buffer; otherwise, it is a
parameter.  Any field of type `gotorch/nn.Module` is considered a sub-module.

Here is an example of `BatchNorm2d`, which has parameters `Weight` and `Bias`,
as well as buffers of `RunningMean` and `RunningVar`.

```go
// BatchNorm2dModule torch.nn.BatchNorm2d
type BatchNorm2dModule struct {
        Module
        NumFeatures       int64
        Eps               float64
        Momentum          float64
        Affine            bool
        TrackRunningStats bool
        Weight            torch.Tensor
        Bias              torch.Tensor
        RunningMean       torch.Tensor `gotorch:"buffer"`
        RunningVar        torch.Tensor `gotorch:"buffer"`
}
```

Following PyTorch's naming convention, we name each GoTorch module’s newer
function by the module name. For example, `BatchNorm2d` is the function that
instantiates a module type, and the module type has the name
`BatchNorm2dModule`.

```go
// BatchNorm2d creates a `BatchNorm2dModule` instance
func BatchNorm2d(numFeatures int64, eps, momentum float64,
        affine, trackRunningStats bool) *BatchNorm2dModule {
        b := &BatchNorm2dModule{
                Module:            Module{isTraining: true},
                NumFeatures:       numFeatures,
                Eps:               eps,
                Momentum:          momentum,
                Affine:            affine,
                TrackRunningStats: trackRunningStats,
        }
        if b.Affine {
                b.Weight = torch.Empty([]int64{numFeatures}, true)
                b.Bias = torch.Empty([]int64{numFeatures}, true)
        }
        if b.TrackRunningStats {
                b.RunningMean = torch.Empty([]int64{numFeatures}, false)
                b.RunningVar = torch.Empty([]int64{numFeatures}, false)
        }
        b.resetParameters()
        b.Init(b)
        return b
}
```

Each module must have the `Forward` method, just like in PyTorch. The GoTorch
framework allows users to attach Forward methods of any signatures, as long as
the name is "Forward".  For example, most Forward methods take a tensor-typed
parameter and return a tensor. However, some other modules can have various
numbers of parameters and return values in different types. Go's reflection
enables GoTorch's container modules like `nn.Sequential` to call Forward methods
correctly.
