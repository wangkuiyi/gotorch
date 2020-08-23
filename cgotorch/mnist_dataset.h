/* Copyright 2020, GoTorch Authors */
#ifndef CGOTORCH_MNIST_DATASET_H_
#define CGOTORCH_MNIST_DATASET_H_

typedef struct {
  MNIST p;
  Normalize normalize;
  double mean, stddev;
} MNISTDataset;

const char *CreateMNISTDataset(const char *data_root, MNISTDataset *dataset);
void MNISTDataset_Close(Dataset d);

// Set parameters of the normalize transform in dataset
void MNISTDataset_Normalize(MNISTDataset *dataset, double mean, double stddev);

typedef void *MNISTLoader;
typedef void *MNISTIterator;

MNISTLoader CreateMNISTLoader(Dataset dataset, int64_t batchsize);
void MNISTLoader_Close(MNISTLoader loader);

MNISTIterator MNISTLoader_Begin(MNISTLoader loader);
void MNISTIterator_Batch(MNISTIterator iter, Tensor *data, Tensor *target);
bool MNISTIterator_Next(MNISTIterator iter, MNISTLoader loader);

#endif  // CGOTORCH_MNIST_DATASET_H_
