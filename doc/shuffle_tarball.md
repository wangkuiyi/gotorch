# Shuffle Images in a Tarball File

## Tarball as Containers

In the training of a computer vision model, we need to open many images from the
training dataset.  A typical wrong solution is to save each image as a separate
file.  In this case, each opening operation causes some mechanical movements in
the hard disk, and hundreds of epochs often damage the disk surface and ruin
hard drives.

The correct solution is to have all images in a single large "container" file.
An ideal container file format is the tarball, which usually has the extension
name `.tar.gz`.  The `.gz` extension means compressing the `.tar` container file
using the GZip algorithm.  Because Gzip is a streaming algorithm, reading
sequentially from a Gzipped byte stream is as convenient as reading from the
byte stream.  The `.tar` extension names the container format, consisting of a
sequence of files or directories, each with a descriptive header followed by the
file content.  A directory does not have content, but only headers.  This format
allows us to read sequentially from `.tar` or `.tar.gz` file for image files
without causing frequent mechanical movements in hard drives.

## Caution

We must use GNU tar instead of bsdtar on macOS.

## Shuffling

It is critical in deep learning to ensure that each minibatch or consecutive
images read from a tarball belong to different labels.  This property is known
as **shuffling**.  Unshuffled data sequence prevents the deep learning training
algorithm from converging.

However, most tarballs contain unshuffled data because people often use the `tar
czf` command to create tarballs.  For example, to archive a directory
`mnist_png` into `mnist_png.tar.gz`, we run the following command.

```bash
tar czf mnist_png.tar.gz mnist_png
```

We can download this tarball from this GitHub [repo](https://github.com/myleott/mnist_png).

We can use the `tar tvf` command to read the content sequentially and print the
image file names.

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

The result reveals that the `tar czf` command archives folder recursively, so it
puts images in the same folder together.  However, most image datasets published
as tarballs, including the well-known ImageNet dataset, use the base directory
to name the label of images in it.  This behavior of `tar czf` generates
tarballs that break the shuffling property.

## Shuffled Tarballs

We want to generate a tarball where consecutive images have different labels or
different base directories.  We wrote two programs to achieve the goal using a
divide-and-merge strategy -- `tarball_divide` and `tarball_merge`.

To install them, we need the Go compiler and run the following commands.

```bash
go get github.com/wangkuiyi/gotorch/tool/...
```

We can then find the executable files in `$GOPATH/bin`.

To split the unshuffled `mnist_png.tar.gz` into two shuffled tarballs:
`mnist_png_training_shuffled.tar.gz` and `mnist_png_testing_shuffled.tar.gz`,
let us take the following steps.

1. Split `mnist_png.tar.gz` into `mnist_png_training.tar.gz` and
   `mnist_png_testing.tar.gz`, which are unshuffled.

   ```bash
   tar xzf mnist_png.tar.gz  # Generates two directories mnist_png/training and mnist_png/testing
   cd mnist_png
   tar czf mnist_png_training.tar.gz training
   tar czf mnist_png_testing.tar.gz testing
   ```

1. Divide `mnist_png_training.tar.gz` into ten small tarballs; each contains
   images in a base directory.

   ```bash
   tarball_divide mnist_png_training.tar.gz
   ```

   This generates `[0-9].tar.gz` files.

1. Merge these files into a new shuffled tarball
   `mnist_png_training_shuffled.tar.gz`.

   ```bash
   tarball_merge -out=mnist_png_training_shuffled.tar.gz [0-9].tar.gz
   ```

   To check the generated tarball is valid, using the `file` command, which
   prints errors if the checksum is wrong.

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
