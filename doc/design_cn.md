# GoTorch

本文档介绍GoTorch项目的设计动机和关键设计挑战。

## GoTorch和Go+

GoTorch使用Go语言封装了libtorch。libtorch是PyTorch的C++核心，它有很多基于其它语言，如Rust和Haskell的封装。从我们的调研看来，大部分Python用户认为用Rust、Haskell或Julia编程不如用Python编程高效。因此，如果只是多一门Go语言来封装PyTorch，没有太大意义。

实际上，GoTorch的完整设计将Go+纳入考虑：Go+在语法上和Python同样简洁，因此使用Go+编写深度学习系统有望和使用Python同样高效，而且Go+编译器将Go+程序转换为Go程序，后者通过Go编译器产生原生代码，而不是和Python一样产生字节码。这些原生代码可以直接在服务器或手机、平板电脑、自动驾驶汽车等移动设备上高效运行，而不需要依赖VM或解释器。

除了用Go封装libtorch之外，GoTorch还包括另外两层API：`torch.nn.functional`和`torch.nn`。PyTorch也为Python用户提供了这两层API。

## 功能分层

总的来说，PyTorch提供了三层API：

1. 最细粒度的一层是libtorch中的原生函数(native functions)，约有1600个。原生函数或者是一个数学上的基本运算，或者是相应的梯度计算。
   每个原生函数都有GPU和CPU两种实现，libtorch可以
   和[XLA](https://github.com/pytorch/xla)链接到一起，从而获得Google TPU上的原生函数实现。

1. `torch.nn.functional`这个Python包提供了更高一级的抽象，这个包用纯Python封装了C++原生函数，更符合Python用户的使用习惯。

1. Module是最高一层的API，PyTorch的module是Python class，该class的`forward`方法定义了模型前向计算的过程。

## Tensor和垃圾回收

libtorch定义了基本数据类型`at::Tensor`的C++实现，以及在其上进行各种运算的原生函数。

Tensor的关键特性是自动垃圾回收(GC)。在C++中，`at::Tensor`这个class只包含了一个类型为`c10::intrusive_ptr<TensorImpl>`的成员变量，`c10::intrusive_ptr`是基于引用计数的智能指针，原理和`std::shared_ptr`相同，唯一区别是`c10::intrusive_ptr`是个侵入式数据结构，因此访问起来会更高效。`TensorImpl`则是真正的tensor对象。`c10::intrusive_ptr`基于引用计数来回收tensor对象，和Go、Java等语言基于标记-清扫算法的自动垃圾回收机制相比，该智能指针能够立即回收生命周期结束的资源，但处理环状依赖时会对程序设计提出更高要求。

PyTorch通过Python封装的函数或对象来访问`at::Tensor`。Python的GC使用类似`std::shared_ptr`的引用计数机制来回收内存，但不能处理环状依赖。因此Python不定时运行标记-清扫算法来处理环状依赖。

Go标准库提供了`runtime.GC()`这个API来手动触发GC运行，`runtime.GC()`在某种程度上是异步的，因为finalizer在单独的goroutine中运行。如果所有的tensor都在内存中，手动触发GC也能够回收内存，但手动触发GC无法保障回收时机，而在深度学习任务中，tensors多数存储在显存中，显存是紧缺资源，我们需要立即回收生命周期结束的tensor所占用的显存，以便下一次迭代可以在显存中创建新的tensor。

### 同步垃圾回收

为了能够及时回收tensor，最初我们的想法是，在GoTorch中添加一些新的GC机制，比如说增加一个全局的引用计数表。但在尝试几种策略后，我们注意到，实际上是可以通过定制Go的GC来同步回收GoTorch的tensor变量的，我们可以使GC在所有环节完成后才返回。

该设计背后的基本思路是，根据tensor在深度学习中的不同用途将其分类：

1. 模型*参数* —— 在train loop之前创建，在train loop中更新，在train loop结束后回收。
1. 模型*buffer* —— 生存周期和模型参数完全相同，但不需要计算梯度。常用于模型内部的数据统计，如batch normalization。
1. 中间结果 —— 在train loop每个迭代中前向计算和后向传播生成的tensor。

我们为GoTorch定制的GC机制不需要处理前两种类型的tensor，因为它们的生存周期较长，*module移植*一节会介绍这一主题。

为了处理上述中间结果，GoTorch用户需要在train loop的每个迭代开始前调用`gotorch.GC()`。
`gotorch.GC()`干的第一件事情是标记在其调用后所有的tensor为中间结果，因此其资源回收统统由GC机制接管。
在train loop结束后，用户需要调用`gotorch.FinishGC()`来清除标记。

该标记设置之后，后续每个产生tensor的操作，如调用`gotorhc.RandN`或`gotorch.MM`，都会在一个`WaitGroup`中记录下来，同时绑定一个finalizer到所创建的tensor上。Go的GC在监测到某个tensor已经*不可达*时，将会在独立的goroutine中调用该finalizer，而这个finalizer将会`delete`底层的`at::Tensor`对象，并清除`WaitGroup`中的记录。

接下来，`gotorch.GC()`调用`runtime.GC()`，并等待`WaitGroup`记录的所有tensor都已回收。对`runtime.GC()`的调用会立刻触发Go的标记-清扫算法。而对`WaitGroup`的等待会在GoTorch标记后生成的所有tensor都被回收后结束。一般来说，这个等待过程低于一毫秒(ms)。

### 典型的Train Loop

综上, GoTorch中一个典型的train loop的代码类似于:

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

部分GoTorch的API, 如data loader的Scan方法，会隐式调用`gotorch.GC()`，因此大部分情况下用户不需要自己调用`gotorch.GC()`。

## 错误处理

libtorch中的C++代码可能会抛出异常，我们希望GoTorch捕获这些异常并将其转换为Go的panic。在`cgotorch`这个子目录下的Cgo代码调用libtorch函数并捕获C++异常，同时返回一个表示错误原因的C字符串。如果没有异常发生，该C字符串为NULL。

下面举个封装libtorch的`torch::randn`函数的例子。

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

可以看到，libtorch的`torch::randn`函数返回一个`at::Tensor`。而Cgo封装的`RandN`则返回一个存有(可能的)异常原因的C字符串，`RandN`的最后一个参数则是待返回的tensor。

Go函数`gotorch.RandN`调用上述的C++函数`RandN`，并将其返回的C字符串传递给Go函数`MustNil`，如果该C字符串不为空，则MustNil将panic。

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

## 移植functional

`torch.nn.functional`这个Python包定义了一系列原生函数的组合函数。一般来说，每个Python的functional对应C++命名空间`torch::nn::functional`下的一个函数。所以我们可以通过Cgo来公开这些C++函数，然后再定义Go函数来调用它们。

以`torch.nn.functional.linear`为例。首先是通过Cgo公开该C++函数给Go，我们在`gotorch/cgotorch/functionals.cc`文件中定义一个C++函数来封装libtorch中的`linear`函数。

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

接下来，我们在`gotorch/nn/functional/functional.go`中调用上述Cgo函数。

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

再接下来，我们就可以在自己的应用程序中使用从`torch.nn.functional`移植到`gotorch/nn/functional`下的函数了。

```go
import torch "github.com/wangkuiyi/gotorch"
import F "github.com/wangkuiyi/gotorch/nn/functional"

input := torch.RandN([]int64{32, 100}, false)
weight := torch.RandN([]int64{100, 10}, true)
out := F.Linear(input, weight, torch.Tensor{})
```

## 移植module

和functional类似，module表示一个前向计算。不同之处是，module是一个C++或者Python的class，可以有成员变量，而functional不行。

PyTorch官方提供了基于Python和C++定义module的frontend，C++和Python的frontend相互独立 ——
Python frontend并非通过调用C++ frontend来实现。

在使用Python/C++ frontend时，用户通过定义基类`Module`的派生类来实现自己的module。在派生类中，用户还需要将成员变量标记为*参数*、*buffer*、*子module*这三类之一。我们称这个标记过程为*状态类型注册*。

### 状态类型注册

我们用C++定义一个线性模型来解释类型注册的过程，可以看到，下面这个例子调用`register_parameter`来标明`weight`和`bias`是模型*参数*。

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

该注册过程是必须的。比如`Module::get_parameters`方法会遍历所有注册为*参数*的成员变量，并将*参数*集合返回给调用者，又如`Module::to(device)`把所有参数、buffer，以及子module的参数、buffer，递归地移动到指定的设备中。如果不作注册，这类方法将无法实现。

PyTorch的Python frontend不强制要求用户调用`register{parameter|buffer|module}`，
而是利用`__setattr__`这个方法和其它Python动态特性来自动注册。但有些情况下用户还是需要手动注册。同样以线性模型为例，在下面的代码中，可以看到在*参数*为可选项的情况下，用户仍然需要自己调用`register_parameter`。

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

以上两个代码片段介绍了用于定义module的框架所需要支持的主要功能。

### GoTorch的module注册过程

Go语言不支持class层次结构。和class派生接近的语言设施称为[内嵌struct](https://golang.org/ref/spec#Struct_types)。
正如PyTorch中从基类`Module`派生新的module class一样，GoTorch用户通过定义内嵌了`Module`类型匿名成员的struct来定义module。

不同于PyTorch的注册机制，GoTorch利用Go的字段tag和反射机制来标记成员变量。如果某个tensor类型字段的tag是`gotorch:"buffer"`，它就是*buffer*；否则就是*参数*。所有类型为`gotorch/nn.Module`的字段都被认为是子module，从而自动注册。

以下以`BatchNorm2d`为例来演示GoTorch的注册过程，`BatchNorm2d`的*参数*包括`Weight`和`Bias`，*buffer*包括`RunningMean`和`RunningVar`。

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

为了和PyTorch的命名风格保持一致，在GoTorch中用module的名称命名其构造函数，而module对应的struct则加以*Module*后缀，如上例的`BatchNorm2dModule`。

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

和PyTorch类似，GoTorch中每个module都必须提供`Forward`方法。GoTorch不限制module中`Forward`方法的签名，只要方法名是`Forward`即可。例如，大部分module的`Forward`方法接受一个tensor类型的参数，并返回一个tensor，但在部分module中，其`Forward`方法可能需要一个以上的参数，而返回值类型也不是tensor。Go的反射机制使GoTorch中诸如`nn.Sequential`此类的module容器能够正确地调用其中元素的`Forward`方法。
