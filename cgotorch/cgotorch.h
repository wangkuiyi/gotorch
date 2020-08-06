/* Copyright 2020, GoTorch Authors */
#ifndef CGOTORCH_CGOTORCH_H_
#define CGOTORCH_CGOTORCH_H_
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *Tensor;
typedef void *Optimizer;
Tensor RandN(int rows, int cols, int require_grad);
char *MM(Tensor a, Tensor b, Tensor *result);
Tensor Sum(Tensor a);
Tensor Conv2d(Tensor input, Tensor weight, Tensor bias, int64_t *stride_data,
              int64_t stride_len, int64_t *padding_data, int64_t padding_len,
              int64_t *dilation_data, int64_t dilation_len, int64_t groups);

const char *Tensor_String(Tensor a);
void Tensor_Backward(Tensor a);
Tensor Tensor_Grad(Tensor a);
void Tensor_Print(Tensor a);
void Tensor_Close(Tensor a);
void FreeString(const char *s);

Optimizer SGD(double learning_rate, double momentum, double dampening,
              double weight_decay, int nesterov);
Optimizer Adam(double learning_rate, double beta1, double beta2,
               double weight_decay);

void Optimizer_ZeroGrad(Optimizer opt);
void Optimizer_Step(Optimizer opt);
void Optimizer_AddParameters(Optimizer opt, Tensor *tensors, int length);
void Optimizer_Close(Optimizer opt);

// transform APIs
typedef void *Transform;
Transform Normalize(double mean, double stddev);
Transform Stack();

// dataset APIs
typedef void *Dataset;
Dataset MNIST(const char *data_root);
void MNIST_Close(Dataset d);

// Add transform on dataset
void Dataset_Normalize(Dataset dataset, Transform transform);
void Dataset_Stack(Dataset dataset, Transform transform);

typedef void *Iterator;
typedef void *DataLoader;

void Loader_Close(DataLoader loader);
Iterator Loader_Begin(DataLoader loader);
void Loader_Data(Iterator iter, Tensor data[]);
bool Loader_Next(DataLoader loader, Iterator iter);
DataLoader MakeDataLoader(Dataset dataset, int batchsize);

#ifdef __cplusplus
}
#endif

#endif  // CGOTORCH_CGOTORCH_H_
