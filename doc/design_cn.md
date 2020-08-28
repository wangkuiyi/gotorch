# GoTorch

本文档介绍GoTorch项目的动机和关键设计挑战。

## GoTorch和Go+

GoTorch使用Go语言封装了libtorch。libtorch是PyTorch的C++核心代码，它有很多基于其它语言的封装，如Rust和Haskell。但从我们的调研看来，大部分Python用户认为用Rust、Haskell或Julia编程不如用Python编程高效。因此，如果只是多一门语言来封装GoTorch，没有太大意义。

实际上，GoTorch的完整设计将Go+纳入考虑：Go+在语法上和Python同样简洁，因此使用Go+编写深度学习系统有望和使用Python编写同样高效，而且Go+编译器将Go+程序转换为Go程序，后者通过Go编译器产生原生代码，而不是和Python一样产生字节码。这些原生代码可以直接在服务器或手机、平板电脑、自动驾驶汽车等移动设备上高效运行，而不需要依赖VM或解释器。

除了用Go封装libtorch之外，GoTorch还包括另外两层API：`torch.nn.functional`和`torch.nn`。PyTorch也为Python用户提供了这两层API。

## 功能分层

总的来说，PyTorch提供了三层API：

1. 最细粒度的一层是libtorch中的原生函数(native functions)，约有1600个。原生函数或者是一个数学上的基本运算，或者是相应的梯度计算。每个原生函数都有GPU和CPU两种实现，libtorch可以和[XLA](github.com/pytorch/xla)链接到一起，从而获得Google TPU上的原生函数实现。

1. `torch.nn.functional`这个Python包提供了更高一级的抽象，这个包用纯Python封装了C++原生函数，更符合Python用户的使用习惯。

1. Module是最高一层的API，PyTorch的module是Python class，该class的`forward`方法定义了模型前向计算的过程。

## Tensor和垃圾回收

libtorch定义了基本数据类型`at::Tensor`的C++实现，以及在其上进行各种运算的原生函数。

Tensor的关键特性是自动垃圾回收(GC)。在C++中，`at::Tensor`这个class只包含了一个类型为`c10::intrusive_ptr<TensorImpl>`的数据成员，`c10::intrusive_ptr`是基于引用计数的智能指针，原理和`std::shared_ptr`相同，唯一区别是`c10::intrusive_ptr`是个嵌入结构，因此访问起来会更快。`TensorImpl`则是真正的tensor对象。`c10::intrusive_ptr`基于引用计数来回收tensor对象，和Go、Java等语言基于标记-清扫算法的自动垃圾回收机制相比，该智能指针能够立即回收生命周期结束的资源，但处理环状依赖时对程序设计提出更高要求。

PyTorch通过Python封装的函数或对象来访问`at::Tensor`。Python的GC使用类似`std::shared_ptr`的引用计数机制来回收内存，但不能处理环状依赖。因此Python不定时运行标记-清扫算法来处理环状依赖。

Go标准库提供了`runtime.GC()`这个API来手动触发GC运行，`runtime.GC()`在某种程度上是异步的，因为finalizer在单独的goroutine中运行。如果所有的tensor都在内存中，手动触发GC是能够回收内存；但手动触发GC无法保障立即回收，而在深度学习任务中，tensors多数存储在显存中，显存是紧缺资源，我们需要立即回收生命周期结束的tensor所占用的显存，以便下一次迭代可以在显存中创建新的tensor。

### 同步垃圾回收

为了能够及时回收tensor，最初我们的想法是，在GoTorch中添加一些新的GC机制，比如说增加一个全局的引用计数表。但在尝试几种策略后，我们注意到，实际上是可以通过定制Go的GC来同步回收GoTorch的tensor类型变量的，我们可以使GC在所有环节完成后才返回。

该设计背后的基本思路是，根据tensor在深度学习中的不同用途将其分类：

1. 模型参数 —— 在train loop之前创建，在train loop中更新，在train loop结束后回收。
1. 模型buffer —— 生存周期和模型参数完全相同，但不需要计算梯度。常用于模型内部的数据统计，如batch normalization。
1. 中间结果 —— 在train loop的每个迭代中前向计算和后向传播过程中生成的tensor。

我们为GoTorch定制的GC机制不需要处理前两种类型的tensor，因为它们的生存周期较长，*module移植*一节会介绍这一主题。

为了处理上述中间结果，GoTorch用户需要在train loop的每个迭代开始前调用`gotorch.GC()`。`gotorch.GC()`干的第一件事情是标记在其调用后所有的tensor为中间结果，因此其资源回收统统由GC机制接管。在train loop结束后，用户需要调用`gotorch.FinishGC()`来取消标记。

该标记设置之后，后续每个产生tensor的操作，如调用`gotorhc.RandN`或`gotorch.MM`，都会在一个`WaitGroup`中记录下来，同时绑定一个finalizer到所创建的tensor上。Go的GC在监测到某个tensor已经不可达的时候，将会在独立的goroutine中调用该finalizer，而这个finalizer将会`delete`底层的`at::Tensor`对象，并销毁`WaitGroup`中的记录。

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

# 错误处理

libtorch中的C++代码可能会抛出异常，而我们希望GoTorch捕获之并转换为Go的panic。在cgotorch这个子目录下的Cgo代码调用libtorch函数，捕获C++异常，并返回一个C字符串(表示错误原因)。如果没有异常抛出，该C字符串为NULL。

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

`torch.nn.functional`这个Python包
