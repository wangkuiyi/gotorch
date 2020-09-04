# DCGAN Demo

In this demo,we train a DCGAN on a faces dataset.
The following are the generated fake faces by GoTorch(the left one) and PyTorch.

![dcgan](http://cdn.sqlflow.tech/dcgan-20200904.gif)

Training loss of GoTorch:

![gotorch-dcgan-loss](gotorch-dcgan-loss.png)

Training loss of PyTorch:

![pytorch-dcgan-loss](pytorch-dcgan-loss.png)

## Download CelebA Dataset

We download [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
to a newly created directory, say `$DATAROOT`.

Then, we shuffle the dataset and transform it to new file in tgz format, `train.tgz`.

```bash
cd $DATAROOT
unzip img_align_celeba.zip
find . | sort -R | tar czvf train.tgz -T -
```

## Build

We first build the gotorch project. For how to build and test,
please refer to [CONTRIBUTING.md](https://github.com/wangkuiyi/gotorch/blob/develop/CONTRIBUTING.md).

Then, we run the following command in the root directory to install gotorch project.

```bash
go install ./...
```

The `dcgan` binary will be installed at `$GOPATH/bin` directory.

## Run

Then, we execute the compiled binary in current directory to train the model:

```bash
$GOPATH/bin/dcgan -dataroot=$DATAROOT/train.tgz
```

The training program periodically generates image samples and saves to pickle files.
We provide a script to transform the saved pickle files into png format.

```bash
python visualize_pickle.py --load_gotorch=1 --save_image=1
```

And the script could also generate an animation to visualize
the training progress of generated fake images.
`ffmpeg` is needed to save the animation to mp4 format.

```bash
python visualize_pickle.py --load_gotorch=1 --save_video=1
```

To see the training loss curve:

```bash
python plot_loss.py --load_gotorch=1
```
