/* Copyright 2020, GoTorch Authors */
#ifndef CGOTORCH_CGOTORCH_H_
#define CGOTORCH_CGOTORCH_H_
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *Tensor;
typedef void *Optimizer;
Tensor RandN(int rows, int cols, int require_grad);
Tensor MM(Tensor a, Tensor b);
Tensor Sum(Tensor a);

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

void ZeroGrad(Optimizer opt);
void Step(Optimizer opt);
void Optimizer_AddParameters(Optimizer opt, Tensor *tensors, int length);
void Optimizer_Close(Optimizer opt);

// transform APIs
typedef void *Transform;
Transform Normalize(double mean, double stddev);
Transform Stack();

// dataset APIs
typedef void *Dataset;
Dataset MNIST(const char *data_root);

// Add transform on dataset
void Dataset_Normalize(Dataset dataset, Transform transform);
void Dataset_Stack(Dataset dataset, Transform transform);

typedef void *Iterator;
typedef void *DataLoader;

typedef struct Data {
  Tensor Data;
  Tensor Target;
} Data;

Iterator Loader_Begin(DataLoader loader);
Data Loader_Data(Iterator iter);
bool Loader_Next(DataLoader loader, Iterator iter);
DataLoader MakeDataLoader(Dataset dataset, int batchsize);

#ifdef __cplusplus
}
#endif

#endif  // CGOTORCH_CGOTORCH_H_
