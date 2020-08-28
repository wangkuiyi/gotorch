# Develop Functionals And Modules For GoTorch

This tutorial demonstrates how to develop modules and functionals for GoTorch.
[Wrap PyTorch Native Functions](wrap_native_functions.md) explains how to wrap
PyTorch native functions for GoTorch.  Functionals are high-level functions
built on native functions.  Modules are structs that encapsulate multiple native
functions or functionals.  Modules and functionals are the main interface to
GoTorch end users.

## Define GoTorch Functionals

In general, functionals in libtorch are in the form of C++ global functions:
exactly the same as native functions.  We can follow the same steps of wrapping
native functions to wrap libtorch functionals as described in
[Wrap PyTorch Native Functions](wrap_native_functions.md).

On the other hand, functionals can also be implemented in pure Go by calling the
wrapped native functions. Let's take the activation `ReLU6` as an example.

ReLU6 is an activation commonly used in deep convolutional neural networks. It
comes up fairly often in mobile machine learning cases because it's ready for
fixed-point inference, which is highly efficient in both space and time.

In PyTorch, the `ReLU6` implementation in Python looks like:

```Python
def relu6(input, inplace=False):
    # type: (Tensor, bool) -> Tensor
    if inplace:
        return torch._C._nn.hardtanh_(input, 0., 6.)
    return torch._C._nn.hardtanh(input, 0., 6.)

```

As you can see, the `relu6` functional just wraps the libtorch C++ function
`hardtanh`.  This is a common paradigm in PyTorch functionals.

In GoTorch, we can write the `ReLU6` activation function in a similar manner:

```go
func ReLU6(input torch.Tensor, inplace bool) {
    if inplace {
        return torch.HardtanhI(input, 0, 6);
    }
    return torch.Hardtanh(input, 0, 6);
}
```

## Define GoTorch Modules

PyTorch requires modules to subclass the `torch.nn.Module` base class or its
subclasses.  Similarly, the PyTorch C++ frontend also requires modules to derive
from the base class `torch::nn::Cloneable<>`, which itself derives from
`torch::nn::Module`.

Go doesn't support subclassing, as a result, GoTorch uses struct embedding and
the `reflect` package to achieve a similar user experience.

A GoTorch module should be defined as a struct, and the `torch.Module` struct
should be embedded in a GoTorch module **by value**.  In addition, a GoTorch
module has to call an `Init` method in its constructor function.

Most modules have lots of code, so let's take a simple module `Linear` as an
example to demonstrate how to define a GoTorch module.  The following code
demonstrates the implementations of the `Linear` module in Python and Go,
respectively.

### Implement `Linear` In Python

At first, let's look back on how to define a module in Python.  The
implementation of the `Linear` module in Python looks like:

```python
class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    # The learned weight.
    weight: Tensor

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    # Transforms the `input` tensor by multiplying with the `weight` and
    # optionally adding the `bias`, if `with_bias` is true.
    def forward(self, input: Tensor) -> Tensor:
        return functional.linear(input, self.weight, self.bias)
```

As we can see, the Python `Linear` module requires the module author to do the
following steps:

1. Define a class that derives from the base class `Module`.
1. Register parameters (call `register_parameter` or `Parameter`) or buffers
   (call `register_buffer`) in the constructor. Buffers are `Tensor`s that don't
   need gradients.  The `Linear` module doesn't need buffers.
1. Define a `forward` method in the class that does the actual computation of
   `Linear`.

### Implement `Linear` In GoTorch

Like PyTorch, GoTorch defined a base struct `torch.Module` to facilitate module
definition.

```go
package gotorch

type LinearModule struct {
    Module
    InFeatures  int64
    OutFeatures int64
    // The learned weight.
    Weight      torch.Tensor `gotorch:param`
    // The learned bias.  If `withBias` is false, this tensor is undefined.
    Bias        torch.Tensor `gotorch:param`
}

func Linear(in, out int64, withBias bool) *LinearModule {
    l := &LinearModule{
        Module:      Module{isTraining: true},
        InFeatures:  in,
        OutFeatures: out,
    }
    l.Weight = torch.Empty([]int64{out, in}, true)
    if withBias {
        l.Bias = torch.Empty([]int64{out}, true)
    }
    initializer.KaimingUniform(
        &l.Weight, math.Sqrt(5.0), "fan_in", "leaky_relu")
    if l.Bias.T != nil {
        fanIn, _ := initializer.CalculateFanInAndFanOut(l.Weight)
        bound := 1.0 / math.Sqrt(float64(fanIn))
        initializer.Uniform(&l.Bias, -bound, bound)
    }
    l.Init(l)
    return l
}

// Forward transforms the `input` tensor by multiplying with the `weight` and
// optionally adding the `bias`, if `with_bias` is true in the options.
func (l *LinearModule) Forward(x torch.Tensor) torch.Tensor {
    return F.Linear(x, l.Weight, l.Bias)
}
```

In the above `Linear` definition, we do the following steps:

1. Define a struct `LinearModule` that embeds `torch.Module` by value.  All the
   fields of type `torch.Tensor` or `torch.Module` in the struct should be
   exported.
1. Define a constructor function `Linear` that initializes the `LinearModule`
   object.  The constructor has to call `Init` on the newly constructed object
   and pass a pointer to the object itself to the `Init` method. This is
   like Python's requirement to call `super().__init__()` in the constructor of
   a derived class.
1. Define a `Forward` method that does the actual computation.

Compared to the Python versions, GoTorch has an advantage: A module author
doesn't have to call `register_module`, `register_buffer`, or `Parameter`
anymore. Instead, GoTorch uses the tags `gotorch:param` and `gotorch:buffer` to
mark whether a `Tensor` is a parameter or a buffer.  `gotorch:param` is the
default and can be omitted.

#### NOTE

1. *As in the C++ and Python frontend, GoTorch doesn't require the signature of
   the `Forward` method of modules. A module author of GoTorch has the
   flexibility to define her `Forward` method with any type and any number of
   parameters and return values. This is very useful for containers like
   `Sequential`.*
1. *The examples in this section omitted some boilerplate code: `#include`s,
   `import`s, and methods that prettily print tensors.  Readers can add the
   omitted code for practice.*

## Summary

In this tutorial, we learned how to define functionals and modules in GoTorch.

1. To define functionals, we either wrap the corresponding C++ functionals or
   write new functions in pure Go.
1. To create a GoTorch module, we define a struct that embeds `torch.Module` by
   value, call `Init` in its constructor function, and define a `Forward`
   method.
