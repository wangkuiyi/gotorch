# 开发 GoTorch 算子（Functional）和模块（Module）

本教程将向你介绍如何开发 GoTorch 的算子和模块。在[包装 PyTorch 本地函数](wrap_native_functions_cn.md)
中，我们介绍了如何将 Pytorch 中的本地函数包装成为 GoTorch 中的函数。算子是建立在本地函数基础上
的更高层的 `Tensor` 操作。而模块是多个本地函数和算子的封装。模块和算子是面向 GoTorch 开发者
的主要编程接口。

## 定义 GoTorch 算子

总体来说，`libtorch` 中的算子是以 C++ 全局函数的形式存在的，这一点上和本地函数一样。因此，我们
可以采用类似[包装 PyTorch 本地函数](wrap_native_functions_cn.md)的思路来进行算子的包装。
而另一方面，算子也可以完全通过 Go 语言调用本地函数的 Go Wrapper 来实现（纯 Go）。我们将以 `ReLU6` 为例来说明。

`ReLU6` 是深度卷积神经网络中常用的一种激活函数。由于它的定点数推理接口在时间和空间上都非常高效，
因此它经常在移动设备上被使用。

在 PyTorch 中，`ReLU6` 的实现如下面的代码。如你所见，`relu6` 函数只是简单的包装了 `libtorch`
中的 `hardtanh` 函数。这也是 `PyTorch` 中包装算子的一种常见模式。

```Python
def relu6(input, inplace=False):
    # type: (Tensor, bool) -> Tensor
    if inplace:
        return torch._C._nn.hardtanh_(input, 0., 6.)
    return torch._C._nn.hardtanh(input, 0., 6.)

```

在 GoTorch 中，我们通过类似的方式来实现 `ReLU6` 算子：

```go
func ReLU6(input torch.Tensor, inplace bool) {
    if inplace {
        return torch.HardtanhI(input, 0, 6);
    }
    return torch.Hardtanh(input, 0, 6);
}
```

## 定义 GoTorch 中的模块

PyTorch 要求模块继承自 `torch.nn.Module` 类或者它的子类。类似的，PyTorch 的 C++ 实现中
模块都继承自 `torch::nn::Cloneable<>`，而这个类本身是由 `torch::nn::Module` 派生出来的。

Go 语言并不支持继承，因此，GoTorch 采用了结构嵌入（struct embedding）+ 反射的方式来达到
类似的效果，从而保证了于 PyTorch 类似的用户体验。

具体来讲，一个 `GoTorch` 模块应该被定义为一个结构（struct），这个结构中嵌入了一个
`torch.Module` 子结构（注意，嵌入的是**值**类型，而并非指针）。`torch.Module` 中定义了一个
`Init` 方法，它用反射的方式初始化一些必要的信息。自定义 GoTorch 的模块在构造时需要显式调用 `Init` 方法。

大部分的模块都包含大量的代码，为简单起见，我们先以一个非常简单的 `Linear` 作为示例来展示如何自定义一个
`GoTorch` 模块。下面的代码分别展示了在 Python 和 Go 中 `Linear` 是如何定义的，我们可以对比来看。

### Python 中 `Linear` 的定义

首先，让我们回顾一下 Python 中如何定义模块：

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

我们注意到，定义 `Linear` 模块需要以下步骤：

1. 定义一个继承自 `Module` 的类。
1. 在构造函数中通过 `register_parameter` 注册参数（parameters），
    通过 `register_buffer` 注册 `buffer`。 这里，`Linear` 模块并不需要 `buffer` 。
1. 在类中定义一个 `forward` 函数，用来实现 `Linear` 模块的功能，对输入 `Tensor` 进行线性变换。

### 在 GoTorch 中实现 `Linear` 模块

类似于 PyTorch，我们在 GoTorch 中定一个了一个基础结构 `torch.Module` 来简化模块的定义。然后定义
了 `LinearModule` 结构来“继承” `torch.Module`。

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

上述 `Linear` 模块的定义中包含了以下步骤：

1. 定一个了一个 `LinearModule` 结构，其中嵌入了 `torch.Module`。需要注意的是，
    在自定义模块中，所有 `torch.Tensor` 和 `torch.Module` 类型的变量都必须
    是[导出的](https://golang.org/ref/spec#Exported_identifiers)。
1. 为 `LinearModule` 模块定义一个“构造”函数，为这个函数取一个合理的名字（这里我们命名为
    `Linear`）。在这个函数中我们创建了 `LinearModule` 对象，并且将其指针传递到 `Init`（
    注意这里实际上是在调用嵌入对象 `torch.Module` 的 `Init` 方法）函数中，这类似于 Python
    中调用 `super().__init__()` 。
1. 定义 `Forward` 函数，实现线性变换功能。

相比于 Python 版本，GoTorch 有一个优势：自定义模块的时候无需调用 `register_module` 和
`register_buffer`。在 GoTorch 中我们通过 `gotorch:param`、`gotorch:buffer` 等标签
（[tags](https://golang.org/ref/spec#Struct_types)) 来标注字段是一个 parameter 还是
一个 buffer。默认情况下字段会被当做 parameter 处理，此时不需要标签。

#### 注意

1. *如 C++ 和 Python 实现一样，GoTorch 并不对 `Forward` 函数的标签(signature)有任何特殊要求。
    实现者可以根据需要为 `Forward` 函数指定任意的参数类型，数量和返回值类型。这使得函数定义有足够
    的灵活性。这对于定义 `Sequential` 之类的模块来说是非常有用的。*
1. *上述示例中我们省略了一些常规的代码，如 `#include`，`import` 等，读者可以根据需要进行补充*

## 总结

在本教程中，我们学习了如何在 GoTorch 中定义算子和模块。

1. 为了定义一个算子，我们可以通过包装 C++ 算子或者完全通过 Go 来实现。
1. 为了定义一个模块，我们需要定义一个结构，嵌入`torch.Module` 对象，同时在构造该模块时调用
    `Init` 函数，最后在 `Forward` 函数中实现该模块的具体逻辑。
