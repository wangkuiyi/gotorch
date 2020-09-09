# Train the ResNet50 Model Using the ImageNet Dataset

This sample program trains the ResNet50 model using the ImageNet dataset.

## The ImageNet Dataset

Before downloading the training and validation tarballs from the [official
site](http://www.image-net.org/challenges/LSVRC/2010/downloads), please register
on the [ImageNet website](http://www.image-net.org/) to get the permission.

The training data tarball `ILSVRC2010_images_train.tar` contains tarballs.  To
recursively untar all of them into a directory, say, `/tmp/train`, please run
the following commands.

``` bash
cd /tmp
tar xf ~/Downloads/ILSVRC2010_images_train.tar
cd train
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; \
  tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
```

Similarly, we recursively untar the validation dataset
`ILSVRC2010_images_val.tar` to `/tmp/valid`.

``` bash
mkdir /tmp/valid
cd /tmp/valid
tar xf ILSVRC2010_images_val.tar
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
./valprep.sh
```

As explained in the [data shuffling document](../../doc/shuffle_tarball.md), we
recreate a tarball of images in random order as the training dataset.  The
following command creates two tarballs, `train_shuffle.tgz` and
`validation.tgz`.

``` bash
cd /tmp
find ./train | sort -R | tar zxf train_shuffle.tgz -T -
find ./valid | tar zxf validation.tgz -T -
```

## Build and Run the Sample

Please follow
[CONTRIBUTING.md](https://github.com/wangkuiyi/gotorch/blob/develop/CONTRIBUTING.md)
to build the sample program.

```bash
git clone https://github.com/wangkuiyi/gotorch
cd gotorch
go generate ./...  # Build cgolibtorch
go install ./...   # Build all sample programs
```

The above commands compile the binary file `$GOPATH/bin/resnet`.  Please run it
using the following command.

``` bash
nohup resnet -data $IMAGENET_HOME/train_shuffle.tgz -test \
$IMAGENET_HOME/validation.tgz 2>&1 > resnet_train.log &
```

## Benchmark with PyTorch

To compare the performance of GoTorch with PyTorch, please run PyTorch
counterpart [resnet.py](/example/resnet/resnet.py).

On a Linux server with an NVIDIA P100 GPU, the following log messages show
GoTorch and PyTorch samples' throughput.

The throughput of the GoTorch version is about 80 samples/secs.

``` bash
$go run example/resnet/resnet.go -data $IMAGENET_HOME/train_shuffle.tgz -test $IMAGENET_HOME/validation.tgz
2020/09/09 03:02:16 CUDA is valid
2020/09/09 03:20:44 building label vocabulary done.
2020/09/09 03:20:52 Train Epoch: 0, Iteration: 10, ... throughput: 75.588993 samples/sec
2020/09/09 03:20:56 Train Epoch: 0, Iteration: 20, ... throughput: 81.110672 samples/sec
2020/09/09 03:21:00 Train Epoch: 0, Iteration: 30, ... throughput: 78.099667 samples/sec
2020/09/09 03:21:04 Train Epoch: 0, Iteration: 40, ... throughput: 83.487386 samples/sec
```

The throughput of the PyTorch version is about 23 samples/secs.

``` bash
$python example/resnet/resnet.py -data $IMAGENET_HOME/train -test $IMAGENET_HOME/
epoch: 0, batch: 10, loss: 9.453032, ... throughput: 21.507074 samples/sec
epoch: 0, batch: 20, loss: 10.184307, ... throughput: 24.095710 samples/sec
epoch: 0, batch: 30, loss: 7.609190, ... throughput: 23.886803 samples/sec
epoch: 0, batch: 40, loss: 7.043211, ... throughput: 24.335507 samples/sec
```
