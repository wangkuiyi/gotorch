# DCGAN Example

## Download CelebA Dataset

We download [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to a newly created directory, say `$DATAROOT`.

Then, we shuffle the dataset and transform it to new file in tgz format, `train.tgz`.

```bash
cd $DATAROOT
unzip img_align_celeba.zip
find . | sort -R | tar czvf train.tgz -T -
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
$GOPATH/bin/dcgan -dataroot=$DATAROOT/train.tgz
[0/10][0] D_Loss: 1.384754 G_Loss: 3.605948
[0/10][1] D_Loss: 1.666008 G_Loss: 5.663126
[0/10][2] D_Loss: 0.599109 G_Loss: 6.945201
[0/10][3] D_Loss: 0.215684 G_Loss: 6.853242
[0/10][4] D_Loss: 0.599603 G_Loss: 5.772169
[0/10][5] D_Loss: 0.427933 G_Loss: 7.049569
[0/10][6] D_Loss: 0.703852 G_Loss: 7.072733
[0/10][7] D_Loss: 0.365140 G_Loss: 7.365425
[0/10][8] D_Loss: 0.193228 G_Loss: 7.702262
[0/10][9] D_Loss: 0.344985 G_Loss: 7.280960
```

The training program periodically generates image samples and saves to pickle files.
We provide a script to transform the saved pickle files into png format.

```bash
python visualize_pickle.py --save_image=1
```

Here are some generated images after 10 epoches training:

![example1](1.png) ![example2](2.png) ![example3](3.png)

And the script could also generate an animation to visualize
the training progress of generated fake images.

```bash
python visualize_pickle.py --save_video=1
```
