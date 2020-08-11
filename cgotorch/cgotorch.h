/* Copyright 2020, GoTorch Authors */
#ifndef CGOTORCH_CGOTORCH_H_
#define CGOTORCH_CGOTORCH_H_
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
typedef at::Tensor *Tensor;
typedef torch::optim::Optimizer *Optimizer;
typedef torch::data::datasets::MNIST *MNIST;
typedef torch::data::transforms::Normalize<> *Normalize;
#else
typedef void *Tensor;
typedef void *Optimizer;
typedef void *MNIST;
typedef void *Normalize;
#endif

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

const char *exception_str(const std::exception &e);

////////////////////////////////////////////////////////////////////////////////
// Tensor construction and operations, torch
////////////////////////////////////////////////////////////////////////////////

// torch.randn
const char *RandN(int64_t *size, int64_t length, int64_t require_grad,
                  Tensor *result);
// torch.empty
const char *Empty(int64_t *size, int64_t length, int64_t require_grad,
                  Tensor *result);

const char *MM(Tensor a, Tensor b, Tensor *result);
const char *Sum(Tensor a, Tensor *result);
const char *Relu(Tensor a, Tensor *result);
const char *LeakyRelu(Tensor a, double negative_slope, Tensor *result);
const char *Tanh(Tensor a, Tensor *result);
const char *Sigmoid(Tensor a, Tensor *result);
const char *View(Tensor a, Tensor *result, int64_t *size, int64_t size_len);
const char *LogSoftmax(Tensor a, int64_t dim, Tensor *result);

const char *Tensor_String(Tensor a);
void Tensor_Print(Tensor a);
void Tensor_Close(Tensor a);
void FreeString(const char *s);

// Backward, Gradient
void Tensor_Backward(Tensor a);
Tensor Tensor_Grad(Tensor a);

////////////////////////////////////////////////////////////////////////////////
// torch.nn.init
////////////////////////////////////////////////////////////////////////////////

// torch.nn.init.zero_
const char *Zeros_(Tensor *tensor);
// torch.nn.init.uniform_
const char *Uniform_(Tensor *tensor, double low, double high);
// torch.nn.init.kaiming_uniform_
const char *KaimingUniform_(double a, const char *fan_mod,
                            const char *non_linearity, Tensor *tensor);
const char *CalculateFanInAndFanOut(Tensor tensor, int64_t *fan_in,
                                    int64_t *fan_out);

////////////////////////////////////////////////////////////////////////////////
// torch.nn.functional
////////////////////////////////////////////////////////////////////////////////

const char *BatchNorm(Tensor input, Tensor weight, Tensor bias,
                      Tensor running_mean, Tensor running_var, int8_t training,
                      double momentum, double eps, Tensor *result);

const char *Conv2d(Tensor input, Tensor weight, Tensor bias,
                   int64_t *stride_data, int64_t stride_len,
                   int64_t *padding_data, int64_t padding_len,
                   int64_t *dilation_data, int64_t dilation_len, int64_t groups,
                   Tensor *result);

const char *ConvTranspose2d(Tensor input, Tensor weight, Tensor bias,
                            int64_t *stride_data, int64_t stride_len,
                            int64_t *padding_data, int64_t padding_len,
                            int64_t *output_padding_data,
                            int64_t output_padding_len, int64_t groups,
                            int64_t *dilation_data, int64_t dilation_len,
                            Tensor *result);

const char *NllLoss(Tensor input, Tensor target, Tensor weight,
                    int64_t ignore_index, const char *reduction,
                    Tensor *result);

////////////////////////////////////////////////////////////////////////////////
// Optimizer torch.optim
////////////////////////////////////////////////////////////////////////////////

Optimizer SGD(double learning_rate, double momentum, double dampening,
              double weight_decay, int64_t nesterov);
Optimizer Adam(double learning_rate, double beta1, double beta2,
               double weight_decay);

void Optimizer_ZeroGrad(Optimizer opt);
void Optimizer_Step(Optimizer opt);
void Optimizer_AddParameters(Optimizer opt, Tensor *tensors, int64_t length);
void Optimizer_Close(Optimizer opt);

////////////////////////////////////////////////////////////////////////////////
//  Dataset, DataLoader, and Iterator torch.utils.data
////////////////////////////////////////////////////////////////////////////////

typedef struct DatasetMNIST {
  MNIST p;
  Normalize normalize;
  double mean, stddev;
} Dataset;

const char *Dataset_MNIST(const char *data_root, Dataset *dataset);
void MNIST_Close(Dataset d);

// cache normalize transform on dataset
void Dataset_Normalize(Dataset *dataset, double mean, double stddev);
// void Dataset_Stack(Dataset* dataset, Transform transform);

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
