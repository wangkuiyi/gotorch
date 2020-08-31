# Shuffle Images in a Dataset Tarball File

## Tarball as Containers

In the training of a computer vision model, we need to open a large number of images from the training dataset.  A common wrong solution is to save each image as a separate file.  In this case, each call to `fopen` would cause the hard disk tracking operation, which hurts the surface of the disk storage and ruins your hard drive.

The correct solution is to have all images in a single large "container" file.  An ideal container file format is tarball, which, usually has the extension name `.tar.gz` or `.tgz`.  The `.gz` extension means compressed the `.tar` container file using the GZip algorithm.  Becuase the Gzip algorithm is a streamming algorithm, reading sequencially from a Gzipped byte stream is as conveniently as reading from the byte stream directly.  The `.tar` extension indicates the container format, which contains a sequence of files or directories, each as a descriptive header followed by file content.  The directories don't have content, but only headers.  This format allows us to read sequencially from `.tar` or `.tar.gz` file for image files without frequently hard disk tracking operations.

## Shuffling

It is critical in deep learning to make sure that each minibatch, or a short sequence of images read from a tarball, contains images beloning to different labels.  This property is known as shuffling.  Unshuffled data sequence prevents the algorithm from converging.

However, most tarballs of images contains unshuffled data.  People often create tarballs using the `tar czf` command.   For example, to archive a directory `mnist_png` into a tarball `mnist_png.tar.gz`, we run the following command.

```bash
tar czf mnist_png.tar.gz mnist_png
```

You can actually download this tarball from https://github.com/myleott/mnist_png.

Then, we can run the `tar tvf` command to read its content sequentailly and prints the file names.

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

We see that the `tar czf` command archive folder by folder, so images in the same folder form a subsequence.  In the published MNIST and ImageNet datastes, the lower-level of directory, or the base directory, names the label.  This behavior of `tar czf` generates tarballs that breaks the shuffling property.

## Shuffled Tarballs

We want to generate a tarball containing consecutive images having different labels, or, from different base directory.  Here are two programs that achive the goal using a divide-and-merge strategy -- `tarball_divide` and `tarball_merge`.

To get them, you need the Go compiler and run the following commands.

```bash
go get github.com/wangkuiyi/gotorch/tools/...
```

You can find the executable files in `$GOPATH/bin`.

To split the unshuffled `mnist_png.tar.gz` into two shuffled tarballs: `mnist_png_training_shuffled.tar.gz` and `mnist_png_testing_shuffled.tar.gz`, let us take the following steps.

1. Split `mnist_png.tar.gz` into `mnist_png_training.tar.gz` and `mnist_png_testing.tar.gz`, which are unshuffled.

   ```bash
   tar xzf mnist_png.tar.gz  # Generates two directories mnist_png/training and mnist_png/testing
   cd mnist_png
   tar czf mnist_png_training.tar.gz training
   tar czf mnist_png_testing.tar.gz testing
   ```
   
1. Divide `mnist_png_training.tar.gz` into ten small tarballs, each contains images in a base directory.

   ```bash
   tarball_divide mnist_png_training.tar.gz
   ```
   
   This generates `[0-9].tar.gz` files.
   
1. Merge these files into a new shuffled tarball `mnist_png_training_shuffled.tar.gz`.

   ```bash
   tarball_merge -out=mnist_png_training_shuffled.tar.gz [0-9].tar.gz
   ```
   
   To check the generated tarball is valid, using the `file` command, which prints errors if the checksum is wrong.
   
   ```bash
   file mnist_png_training_shuffled.tar.gz
   ```
   
   To count PNG images in these tarballs, run the `tar tvf` command.
   
   ```bash
   tar tvf mnist_png_training.tar.gz | grep \.png$ | wc -l
   tar tvf mnist_png_training_shuffled.tar.gz | grep \.png$ | wc -l
   ```
   
   Both commands should print 60000.
   
1. Clear the intermediate files and divide-and-merge the testing dataset.

   ```bash
   rm [0-9].tar.gz
   tarball_divide mnist_png_testing.tar.gz
   tarball_merge -out=mnist_png_testing_shuffled.tar.gz [0-9].tar.gz
   tar tvf mnist_png_testing_shuffled.tar.gz | grep \.png$ | wc -l
   tar tvf mnist_png_testing.tar.gz | grep \.png$ | wc -l
   ```

   The last two commands should both print 10000.
