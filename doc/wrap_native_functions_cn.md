# 包装 PyTorch 本地函数

PyTorch 的核心逻辑在一个叫 libtorch 的库中，这个库是用 C++ 编写的，包含了
大约1600个深度学习算子，这些算子中的大部分是用来对张量（Tensor） 进行操作的，
我们常称之为本地函数。

在 PyTorch 中，`torch.nn.functional` 包中的函数和 `torch.nn` 包中的模块 (`Moduel`) 和类会通过
[`pybind`](https://github.com/pybind/pybind11) (一个 Python 调用 C++ 的函数接口生成工具）
来调用本地函数。而在 GoTorch 中，我们有两个对应的包  `gotorch/nn/functional` 和 `gotorch/nn`，
他们通过 Go 中的包装方法（利用CGo实现的）来调用本地函数。

本文将讲解如何通过 [Cgo](https://blog.golang.org/cgo) 来进行本地函数包装，我们将从三个层次上来
介绍包装逻辑，他们是：

1. PyTorch 中用 C++ 编写的本地函数，这些函数最终被打包到 libtroch 中，这部分是由 PyTorch 社区实现的。
1. 用 C 语言编写的 C++ 库的包装函数（CGo 只支持 Go 和 C 函数的链接，因此需要包一下 C++ 库），下文称 C Wrapper。
1. Go 编写的调用 CWrapper 的函数，这些函数是 libtorch 中函数的在 Go 语言中的映射，下文称 Go Wrapper。这些 Go
    Wrapper 将被 `gotorch/nn`、`gotorch/nn/functional` 等包中的更高层的 API 调用。

## 本地函数和 C++ Tensor

PyTorch 的构建工具通过一个 YAML 文件
[`native_functions.yaml`](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml)
来生成本地函数的源码。在本教程中，我们会尝试包装 `mm` 函数，其功能是实现矩阵乘法，在 `native_functions.yaml` 中它被定义为：

```yaml
- func: mm(Tensor self, Tensor mat2) -> Tensor
  use_c10_dispatcher: full
  variants: function, method
  dispatch:
    CPU: mm_cpu
    CUDA: mm_cuda
    SparseCPU, SparseCUDA: _sparse_mm
```

这段 YAML 代码告诉我们 `mm` 有三个实现，分别是 `mm_cpu`、`mm_cuda`、和
`_sparse_mm`。该定义通过 `c10` (libtorch 的中一个包)中的一个分发器来寻找一个最匹配的实现。
`variants` 字段告诉我们 `mm` 有两种实现形式，一种是 C++ 全局函数，另外一种是作为 C++ 类
`at::Tensor` 的成员函数（`at` 是 ATen 的缩写，它是 `libtorch` 中的另一个子库，其中定义了最为核心的类型：`Tensor`)。本文中只介绍
C++ 全局函数的包装方法，读者可以自行尝试如何包装类成员。

PyTorch 构建系统利用将上述配置文件生成各种函数的定义并将其保存在 `Functions.h` 文件中。
我们可以从[这里](https://pytorch.org/get-started/locally/)下载 libtorch 的发布包，
然后在 `libtorch/include/ATen/` 目录中找到上述文件。我们会发现 `mm` 函数的定义如下：

```cpp
namespace at {
CAFFE2_API Tensor mm(const Tensor & self, const Tensor & mat2);
}  // namespace at
```

作为类成员函数的定义在  `libtorch/include/ATen/core/TensorBody.h` 文件中，其内容为：

```cpp
namespace at {
class CAFFE2_API Tensor {
  Tensor mm(const Tensor & mat2) const;
 protected:
  c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> impl_;
};
}  // namespace at
```

我们注意到 `at::Tensor` 类只有一个数据成员：`impl_`，这是一个智能指针（C++ smart pointer），指向了一个实现了具体逻辑的对象。

## C Wrapper 和 C Tensor

上文提到，本地函数是由 C++ 编写的，而 Go 只能直接调用 C 函数，因此我们需要写一些 C Wrapper
来包装这些 C++ 本地函数。还有一些其他的原因导致我们需要 C Wrapper：

1. 如果本地函数返回一个 `Tensor`，那么我们需要在 C Wrappr 中创建一个堆对象来指向该 `Tensor`，否则它将会在函数调用结束
    时被释放掉（通过智能指针机制），这样我们在 Go 代码中就无法访问返回的 `Tensor` 了。
1. 需要通过 C Wrapper 来表示 `at::Tensor` 类型，从而在 Go 代码中可以对其进行操作。
1. 本地代码可能抛出异常，我们通过 C Wrapper 来将异常信息转换为 C 字符串并传递到 Go 端，从而实现 C++ 异常到 Go panic 的映射。

在 Cgo 中，Go 程序员可以通过加 "C." 前缀的方式来访问 C 中的各种符号，如变量和函数等。例如，下面的 Go 代码中 `MyExit` 函数通过名称
`C.exit` 来调用定义在 `stdlib.h` 中的 C 标准库函数 `exit`。

```go
// #include <stdlib.h>
import "C"
func MyExit(x int) {
    C.exit(x)
}
```

在我们的代码库中，所有的 C Wrapper 函数都放在 `cgotorch` 目录下。 特别地，在 `cgotorch/cgotorch.h` 文件中，你可以找到
 `at::Tensor` 和 `at::mm` 等类型的 wrapper。C 语言中没有类的概念，所以我们定义了指向 `at::Tensor` 的指针来表示
 C++ 中的 Tensor，如下所示：

```c
extern "C" {
typedef at::Tensor* Tensor;
const char *MM(Tensor a, Tensor b, Tensor *result);
}
```

再回到矩阵乘法的 C Wrapper 实现，我们注意到 `MM` 返回了一个字符串，它用来表示 C++ 中抛出的异常信息，当没有异常时，
它将返回 `nullptr`。 函数 `MM` 是用 C++ 来实现的，具体代码在 `cgotorch/torch.cc` 文件中。虽然具体实现是 C++，
但注意到头文件中声明了 `extern "C"`，因此它仍然可以被 Cgo 进行编译和链接。

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

这段代码包含了一个典型的 C Wrapper 的基本结构：

1. 它首先调用本地函数 `mm` 然后将结果保存在变量 c 中，此时变量 c 在栈中。
1. 然后在堆上分配了一个对象 `*result` 并将 `c` 中的内容拷贝过来。这一步是必要的，否则当函数返回时
    c 将被析构，而其中包含的智能指针析构时将释放底层的 `Tensor` 数据。虽然进行了复制，但对效率影响是
    很小的，因为 `c` 中实际上只包含了一个指针字段，拷贝的过程不是复制了整个 `Tensor`，而只是复制了这个指针。
1. 最后将 C++ 中可能抛出的异常序列化成字符串，然后返回；如果无异常，则返回 `nullptr`。

## Go Wrapper 和 Go Tensor

我们在 `tensor.go` 中定义了 `struct Tensor`（后文称 `Go Tensor`) 类型作为
`C Tensor` 的对应表示。另外，在 `gotorch` 包中， 我们定义了许多本地函数的
`Go Wrapper`，这些函数用于操作 `Go Tensor`。

```go
// Tensor wrappers a pointer to C.Tensor
type Tensor struct {
    T *unsafe.Pointer
}
```

注意到 `C Tensor` 只是一个指针，形式非常简洁，而 `Go Tensor` 的定义初看之下则显得非常复杂，下面我们简单
解释下这样做的必要性。如果直接从字面上将 `C Tensor` 类型转换到 Go 中，我们可以用下面的形式来表示，

```go
type Tensor unsafe.Pointer
```

其中 C 指针对应表示为 Go 中的 `unsafe.Pointer`。然而这样并不能很好的工作，
因为我们还需要在适当的时机调用 `C Wrapper` 释放掉这些 `Tensor`。
在 Go 中实现自定义对象销毁操作的方式是将其绑定到一个
[*finalizer*](https://golang.org/pkg/runtime/#SetFinalizer) 上。
然而，只有 Go 指针能绑定 `finalizer`。因此，我们在 `unsafe.Pointer` 前面加了一个
`*`，将其转换成为了一个指向 C 指针的Go 指针。

```go
type Tensor *unsafe.Pointer
```

上面的指向 C 指针的 Go 指针仍然是不够的，因为我们需要为 `Go Tensor` 定义各种方法，例如：
`MM`，`Add` 和 `To` 等等。而 Go 中 [base type](https://golang.org/ref/spec#Method_declarations)
为指针的类型是无法定义方法的。 因此，我们将上述指针类型包在了一个结构中，
从而变成了上文中较为复杂的结构。

在定义好 Go 中的 `Tensor` 表示之后，Go Wrapper `MM` 就比较容易实现了，如下：

```go
package gotorch

func MM(a, b Tensor) Tensor {
    var t C.Tensor
    MustNil(unsafe.Pointer(C.MM(C.Tensor(*a.T), C.Tensor(*b.T), &t)))
    SetTensorFinalizer((*unsafe.Pointer)(&t))
    return Tensor{(*unsafe.Pointer)(&t)}
}
```

其中包含如下步骤：

1. 我们声明了 `C Tensor` 变量 `t` 用于保存矩阵乘法的结果，然后将其地址传递给 C 函数`C.MM`。
    C 代码中将会填充 t 的具体值，使其最终指向堆内存上分别的结果矩阵。
1. 代码中调用了 `MustNil` 来检测 C Wrapper 中是否抛出了异常，如果异常则会调用 Go panic。
1. 接下来调用了 `gotorch.SetTensorFinalizer` 来将 `C.MM` 的返回结果绑定到一个 finalizer 上，
    从而能够在必要的时候销毁堆内存中的 `Tensor`。
1. 最后，代码利用 `t` 来构造 `Go Tensor` 对象中并返回。