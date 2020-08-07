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
// torch.randn
const char *RandN(int64_t *size, int64_t length, int64_t require_grad,
                  Tensor *result);
// torch.empty
const char *Empty(int64_t *size, int64_t length, int64_t require_grad,
                  Tensor *result);
// torch.nn.init.zero_
const char *Zeros_(Tensor input, Tensor *result);
// torch.nn.init.uniform_
const char *Uniform_(Tensor input, Tensor *result);
// torch.nn.init.kaiming_uniform_
const char *KaimingUniform_(Tensor input, double a, const char *fan_mod,
                            const char *non_linearity, Tensor *result);

const char *MM(Tensor a, Tensor b, Tensor *result);
const char *Sum(Tensor a, Tensor *result);
const char *Conv2d(Tensor input, Tensor weight, Tensor bias,
                   int64_t *stride_data, int64_t stride_len,
                   int64_t *padding_data, int64_t padding_len,
                   int64_t *dilation_data, int64_t dilation_len, int64_t groups,
                   Tensor *result);
const char *Relu(Tensor a, Tensor *result);
const char *LeakyRelu(Tensor a, double negative_slope, Tensor *result);
const char *Tanh(Tensor a, Tensor *result);
const char *Sigmoid(Tensor a, Tensor *result);
const char *ConvTranspose2d(Tensor input, Tensor weight, Tensor bias,
                            int64_t *stride_data, int64_t stride_len,
                            int64_t *padding_data, int64_t padding_len,
                            int64_t *output_padding_data,
                            int64_t output_padding_len, int64_t groups,
                            int64_t *dilation_data, int64_t dilation_len,
                            Tensor *result);

const char *Tensor_String(Tensor a);
void Tensor_Backward(Tensor a);
Tensor Tensor_Grad(Tensor a);
void Tensor_Print(Tensor a);
void Tensor_Close(Tensor a);
void FreeString(const char *s);

Optimizer SGD(double learning_rate, double momentum, double dampening,
              double weight_decay, int64_t nesterov);
Optimizer Adam(double learning_rate, double beta1, double beta2,
               double weight_decay);

void Optimizer_ZeroGrad(Optimizer opt);
void Optimizer_Step(Optimizer opt);
void Optimizer_AddParameters(Optimizer opt, Tensor *tensors, int64_t length);
void Optimizer_Close(Optimizer opt);

// transform APIs
typedef void *Transform;
Transform Normalize(double mean, double stddev);
Transform Stack();

// dataset APIs
typedef void *Dataset;
const char *MNIST(const char *data_root, Dataset *dataset);
void MNIST_Close(Dataset d);

// Add transform on dataset
void Dataset_Normalize(Dataset dataset, Transform transform);
void Dataset_Stack(Dataset dataset, Transform transform);

typedef void *Iterator;
typedef void *DataLoader;

void Loader_Close(DataLoader loader);
Iterator Loader_Begin(DataLoader loader);
void Iterator_Batch(Iterator iter, Tensor *data, Tensor *target);
bool Loader_Next(DataLoader loader, Iterator iter);
DataLoader MakeDataLoader(Dataset dataset, int64_t batchsize);

#ifdef __cplusplus
}
#endif

#endif  // CGOTORCH_CGOTORCH_H_
