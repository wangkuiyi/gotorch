# 打乱压缩包中图像文件的顺序

## 压缩包作为容器

在计算机视觉模型训练中，我们通常需要打开许多图像文件。一种常见的错误做法是将每个图片都保存成单独
的文件。这种情况下，每次打开文件的都可能会导致（机械磁盘中的）磁头进行寻道，随着读取文件次数的增
加，磁盘损坏的概率会大大增加。

正确的做法是将所有的图像文件打包到一个很大的“容器”文件中。一种理想的容器文件格式便是tar压缩包，
这种文件通常以 `.tar.gz` 作为后缀名。`.gz` 表示它是将 `.tar` 容器文件用 `GZip` 压缩算法
压制而成的。 `GZip` 是一种流式算法，从 `Gzip` 流中顺序读取文件就像从普通字节流中读取数据
一样简单。`.tar` 部分则指明了容器的格式，它是一系列文件和目录的顺序归档，其中每一个对象都包含
一个描述头部和紧随其后的文件内容。特别地，目录只包含描述部分而没有文件内容。这种结构允许我们以
顺序方式去读取包含大量图像文件的 `.tar` 或 `.tar.gz` 包，而无需频繁移动磁头去寻找文件。

## 注意

在 macOS 上应该使用 gnu-tar 代替 bsdtar。

## 打乱顺序（Shuffling）

在深度学习中，保证输入模型的每一批数据（minibatch）中包含不同的标签是至关重要的。这个特性称为
**打乱**。未打乱的数据常常导致模型无法收敛。

然而，通常的数据集中包含的都是未打乱的数据。这是因为人们常常通过 `tar czf` 命令来生成压缩包。比如，
通过以下命令来生成 `mnist_png` 的压缩包  `mnist_png.tar.gz`。我们可以从这个 GitHub
[repo](https://github.com/myleott/mnist_png) 上下载该数据集。

```bash
tar czf mnist_png.tar.gz mnist_png
```

我们可以通过  `tar tvf` 来查看压缩包中所包含的图像文件的名称。

```text
$ tar tvf mnist_png.tar.gz  | head -n 10
drwxr-x---  0 myleott myleott     0 Dec 10  2015 mnist_png/
drwxr-x---  0 myleott myleott     0 Dec 10  2015 mnist_png/testing/
drwxr-x---  0 myleott myleott     0 Dec 10  2015 mnist_png/testing/2/
-rw-r-----  0 myleott myleott   272 Dec 10  2015 mnist_png/testing/2/995.png
-rw-r-----  0 myleott myleott   261 Dec 10  2015 mnist_png/testing/2/8450.png
-rw-r-----  0 myleott myleott   282 Dec 10  2015 mnist_png/testing/2/5629.png
-rw-r-----  0 myleott myleott   280 Dec 10  2015 mnist_png/testing/2/9480.png
-rw-r-----  0 myleott myleott   248 Dec 10  2015 mnist_png/testing/2/2058.png
-rw-r-----  0 myleott myleott   259 Dec 10  2015 mnist_png/testing/2/7762.png
-rw-r-----  0 myleott myleott   243 Dec 10  2015 mnist_png/testing/2/9847.png
```

上述结果表明，`tar czf` 命令将递归地归档目录，它将相同目录的文件放在一起。而大部分以tar包
的方式发布的图像数据集（包括著名的 `ImageNet` 数据集），通常都是以目录名称来表示图像标签
的。这就导致了以 `tar czf` 命令生成的包没有乱序的属性。

## 打乱 tar 包中文件顺序

我们希望生成的tar包中连续的图像文件具有不同的标签（也就是他们来自不同的原始目录）。我们编写了
两个程序来通过先分再合的策略（divide-and-merge）来实现这个目标，这两个程序是分别是：
`tarball_divide` 和 `tarball_merge`。 我们可以通过以下命令来安装它们：

```bash
go get github.com/wangkuiyi/gotorch/tool/...
```

运行上述命令后，我们可以在 `$GOPATH/bin` 中找到这两个工具的二进制文件。

我们将执行以下步骤来将未打乱的  `mnist_png.tar.gz` 拆分到两个打乱的 tar 包
（`mnist_png_training_shuffled.tar.gz` 和 `mnist_png_testing_shuffled.tar.gz`）中。

1. 首先将 `mnist_png.tar.gz` 拆分到 `mnist_png_training.tar.gz` 和
   `mnist_png_testing.tar.gz` 两个包中，它们都是未打乱状态。

   ```bash
   tar xzf mnist_png.tar.gz  # 生成两个子目录 mnist_png/training 和 mnist_png/testing
   cd mnist_png
   tar czf mnist_png_training.tar.gz training
   tar czf mnist_png_testing.tar.gz testing
   ```

1. 将 `mnist_png_training.tar.gz` 拆分成十个更小的 tar 包，每个包中包含一个子目录中的
   图片，这将产生 `[0-9].tar.gz` 等十个文件。

   ```bash
   tarball_divide mnist_png_training.tar.gz
   ```

1. 将这些文件合并到一个打乱的 tar 包 `mnist_png_training_shuffled.tar.gz` 中。

   ```bash
   tarball_merge -out=mnist_png_training_shuffled.tar.gz [0-9].tar.gz
   ```

   我们可以使用 `file` 命令来检查生成的 tar 包是否有效，若文件校验和错误，它将打印出错误信息。

   ```bash
   file mnist_png_training_shuffled.tar.gz
   ```

   我们可以通过以下命令来计算 tar 包中的文件数量。

   ```bash
   tar tvf mnist_png_training.tar.gz | grep \.png$ | wc -l
   tar tvf mnist_png_training_shuffled.tar.gz | grep \.png$ | wc -l
   ```

   上述两种方式都应该打印出 60000。

1. 最后我们清理中间过程生成的文件。

   ```bash
   rm [0-9].tar.gz
   tarball_divide mnist_png_testing.tar.gz
   tarball_merge -out=mnist_png_testing_shuffled.tar.gz [0-9].tar.gz
   rm [0-9].tar.gz
   tar tvf mnist_png_testing_shuffled.tar.gz | grep \.png$ | wc -l
   tar tvf mnist_png_testing.tar.gz | grep \.png$ | wc -l
   ```

   上述最后两个命令都应该打印出 10000。
