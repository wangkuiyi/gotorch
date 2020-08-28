# DCGAN Example

## Download CIFAR10 Dataset

We first download cifar10 dataset to somewhere, say `$DATAROOT`.

```bash
cd $DATAROOT
wget https://www.cs.toronto.edu/\~kriz/cifar-10-binary.tar.gz
tar zxvf cifar-10-binary.tar.gz
```

## Build

We first build the gotorch project. For how to build and test, please refer to [CONTRIBUTING.md](https://github.com/wangkuiyi/gotorch/blob/develop/CONTRIBUTING.md).

Then, we run the following command in the root directory to install gotorch project.

```bash
go install ./...
```

The `dcgan` binary will be installed at `$GOPATH/bin` directory.

## Run

Then, we execute the compiled binary in current directory to train the model:

```bash
$GOPATH/bin/dcgan -dataroot=$DATAROOT
[0/100][0] D_Loss: 1.501880 G_Loss: 3.284637
[0/100][1] D_Loss: 1.180851 G_Loss: 3.828784
[0/100][2] D_Loss: 0.901259 G_Loss: 4.031792
[0/100][3] D_Loss: 0.868884 G_Loss: 4.756028
[0/100][4] D_Loss: 0.832174 G_Loss: 5.523646
[0/100][5] D_Loss: 0.762144 G_Loss: 5.824744
[0/100][6] D_Loss: 0.488496 G_Loss: 6.646035
[0/100][7] D_Loss: 0.381518 G_Loss: 6.663010
[0/100][8] D_Loss: 0.626897 G_Loss: 9.005035
[0/100][9] D_Loss: 0.393866 G_Loss: 6.695397
```

The training program periodically generates image samples and saves to pickle files.
We use the `pickle_to_png.py` to transform the saved pickle files into png format.

```bash
python pickle_to_png.py
```
