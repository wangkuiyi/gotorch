# DCGAN Example

We first build the gotorch project. For how to build and test, please refer to CONTRIBUTING.md

Then, we run the following command in the root directory to install gotorch project.

```bash
go install ./...
```

The `dcgan` binary will be installed at `$GOPATH/bin` directory.

Then, we execute the compiled binary in current directory to train the model:

```bash
$GOPATH/bin/dcgan
[0/30][0] D_Loss: 1.511980 G_Loss: 1.214779
[0/30][1] D_Loss: 0.991748 G_Loss: 1.575888
[0/30][2] D_Loss: 0.774483 G_Loss: 2.076495
[0/30][3] D_Loss: 0.647692 G_Loss: 2.421216
[0/30][4] D_Loss: 0.577828 G_Loss: 2.674094
[0/30][5] D_Loss: 0.558098 G_Loss: 2.808377
[0/30][6] D_Loss: 0.561936 G_Loss: 2.944061
[0/30][7] D_Loss: 0.513400 G_Loss: 3.169870
[0/30][8] D_Loss: 0.470114 G_Loss: 3.544255
[0/30][9] D_Loss: 0.457290 G_Loss: 3.877758
```

The training program periodically generates image samples and saves to pickle files.
We use the `pickle_to_png.py` to transform the saved pickle files into png format.

```bash
python pickle_to_png.py
```
