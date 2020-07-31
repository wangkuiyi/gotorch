#ifndef __C_TORCH_H_
#define __C_TORCH_H_
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

void ZeroGrad(Optimizer opt);
void Step(Optimizer opt);
void AddParameters(Optimizer opt, Tensor *tensors, int length);
void Optimizer_Close(Optimizer opt);

// transform APIs
typedef void *Transform;
Transform Normalize(double mean, double stddev);
Transform Stack();

// dataset APIs
typedef void *Dataset;
Dataset MNIST(const char *data_root);

// Add transform on dataset
void AddNormalize(Dataset dataset, Transform transform);
void AddStack(Dataset dataset, Transform transform);

// dataloader APIs
typedef void *DataLoader;
typedef void *Iterator;
DataLoader DataLoaderWithSequenceSampler(Dataset dataset, int batchsize);
void CloseDataLoader(DataLoader);

Iterator Begin(DataLoader loader);
void Next(Iterator iter);
Tensor *Batch(Iterator iter);
bool IsEOF(DataLoader loader, Iterator iter);

#ifdef __cplusplus
}
#endif

#endif //__C_TORCH_H_
