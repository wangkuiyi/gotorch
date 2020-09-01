/* Copyright 2020, GoTorch Authors */
#ifndef CGOTORCH_CGOTORCH_H_
#define CGOTORCH_CGOTORCH_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
#include <vector>
extern "C" {
typedef at::Tensor *Tensor;
typedef torch::optim::Optimizer *Optimizer;
typedef torch::data::datasets::MNIST *MNIST;
typedef torch::data::transforms::Normalize<> *Normalize;
typedef torch::Device *Device;
typedef std::vector<char> *ByteBuffer;  // NOLINT
#else
typedef void *Tensor;
typedef void *Optimizer;
typedef void *MNIST;
typedef void *Normalize;
typedef void *Device;
typedef void *ByteBuffer;
#endif

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

const char *exception_str(const char *e);

////////////////////////////////////////////////////////////////////////////////
// Tensor construction and operations, torch
////////////////////////////////////////////////////////////////////////////////

// torch.randn
const char *RandN(int64_t *size, int64_t length, int64_t require_grad,
                  Tensor *result);
// torch.rand
const char *Rand(int64_t *size, int64_t length, int64_t require_grad,
                 Tensor *result);
// torch.empty
const char *Empty(int64_t *size, int64_t length, int64_t require_grad,
                  Tensor *result);

const char *Equal(Tensor a, Tensor b, int64_t *result);

const char *MM(Tensor a, Tensor b, Tensor *result);
const char *Sum(Tensor a, Tensor *result);
const char *SumByDim(Tensor a, int64_t dim, int8_t keepDim, Tensor *result);
const char *Relu(Tensor a, Tensor *result);
const char *LeakyRelu(Tensor a, double negative_slope, Tensor *result);
const char *Tanh(Tensor a, Tensor *result);
const char *Sigmoid(Tensor a, Tensor *result);
const char *Add(Tensor a, Tensor other, float alpha, Tensor *result);
const char *Add_(Tensor a, Tensor other, float alpha, Tensor *result);
const char *Sub(Tensor a, Tensor other, float alpha, Tensor *result);
const char *Sub_(Tensor a, Tensor other, float alpha, Tensor *result);
const char *Mul(Tensor a, Tensor other, Tensor *result);
const char *Mul_(Tensor a, Tensor other, Tensor *result);
const char *Div(Tensor a, Tensor other, Tensor *result);
const char *Div_(Tensor a, Tensor other, Tensor *result);
const char *Permute(Tensor a, int64_t *dims, int64_t dims_size, Tensor *result);
const char *AllClose(Tensor a, Tensor b, int64_t *result);
const char *Flatten(Tensor a, int64_t startDim, int64_t endDim, Tensor *result);
const char *TopK(Tensor a, int64_t k, int64_t dim, int8_t largest,
                 int8_t sorted, Tensor *values, Tensor *indices);
const char *Transpose(Tensor a, int64_t dim0, int64_t dim1, Tensor *result);
const char *ExpandAs(Tensor a, Tensor other, Tensor *result);
const char *Eq(Tensor a, Tensor other, Tensor *result);
const char *IndexSelect(Tensor a, int64_t dim, Tensor index, Tensor *result);
const char *View(Tensor a, Tensor *result, int64_t *size, int64_t size_len);
const char *LogSoftmax(Tensor a, int64_t dim, Tensor *result);
const char *Squeeze(Tensor a, Tensor *result);
const char *SqueezeWithDim(Tensor a, int64_t dim, Tensor *result);
const char *Argmin(Tensor a, int64_t *dim, int8_t keepdim, Tensor *result);
const char *Argmax(Tensor a, int64_t *dim, int8_t keepdim, Tensor *result);
// TODO(qijun) only support float
const char *Item(Tensor a, float *result);
const char *ItemInt64(Tensor a, int64_t *result);
const char *ItemFloat64(Tensor a, double *result);
const char *Mean(Tensor a, Tensor *result);
const char *Stack(Tensor *tensors, int64_t tensors_size, int64_t dim,
                  Tensor *result);

const char *Tensor_Detach(Tensor a, Tensor *result);
const char *Tensor_String(Tensor a);
void Tensor_Print(Tensor a);
void Tensor_Close(Tensor a);
void FreeString(const char *s);
const char *Tensor_Save(Tensor tensor, const char *path);
const char *Tensor_Load(const char *path, Tensor *result);
const char *Tensor_Dim(Tensor tensor, int64_t *dim);
const char *Tensor_Shape(Tensor tensor, int64_t *dims);
const char *Tensor_Dtype(Tensor tensor, int8_t *dtype);
const char *Tensor_SetData(Tensor self, Tensor new_data);
const char *Tensor_FromBlob(void *data, int8_t dtype, int64_t *sizes_data,
                            int64_t sizes_data_len, Tensor *result);
// Backward, Gradient
void Tensor_Backward(Tensor a);
Tensor Tensor_Grad(Tensor a);

////////////////////////////////////////////////////////////////////////////////
// Pickle encode/decode Tensors
////////////////////////////////////////////////////////////////////////////////

const char *Tensor_Encode(Tensor, ByteBuffer *);

void *ByteBuffer_Data(ByteBuffer);
int64_t ByteBuffer_Size(ByteBuffer);
void ByteBuffer_Free(ByteBuffer);

const char *Tensor_Decode(void *addr, int64_t size, Tensor *);

////////////////////////////////////////////////////////////////////////////////
// torch.nn.init
////////////////////////////////////////////////////////////////////////////////

// torch.nn.init.zeros_
const char *Zeros_(Tensor *tensor);
// torch.nn.init.ones_
const char *Ones_(Tensor *tensor);
// torch.nn.init.uniform_
const char *Uniform_(Tensor *tensor, double low, double high);
// torch.nn.init.normal_
const char *Normal_(Tensor *tensor, double mean, double std);
// torch.nn.init.kaiming_uniform_
const char *KaimingUniform_(double a, const char *fan_mod,
                            const char *non_linearity, Tensor *tensor);
const char *CalculateFanInAndFanOut(Tensor tensor, int64_t *fan_in,
                                    int64_t *fan_out);

void ManualSeed(int64_t seed);
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

const char *BinaryCrossEntropy(Tensor input, Tensor target, Tensor weight,
                               const char *reduction, Tensor *result);

const char *CrossEntropy(Tensor input, Tensor target, Tensor weight,
                         int64_t ignore_index, const char *reduction,
                         Tensor *result);

const char *NllLoss(Tensor input, Tensor target, Tensor weight,
                    int64_t ignore_index, const char *reduction,
                    Tensor *result);

const char *FRelu(Tensor input, int8_t inplace, Tensor *result);
const char *Linear(Tensor input, Tensor weight, Tensor bias, Tensor *result);

const char *MaxPool2d(Tensor input, int64_t *kernel_data, int64_t kernel_len,
                      int64_t *stride_data, int64_t stride_len,
                      int64_t *padding_data, int64_t padding_len,
                      int64_t *dilation_data, int64_t dilation_len,
                      int8_t ceil_mode, Tensor *result);

const char *AdaptiveAvgPool2d(Tensor input, int64_t *output_size_data,
                              int64_t output_size_len, Tensor *result);
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
void Optimizer_SetLR(Optimizer opt, double learning_rate);
void Optimizer_Close(Optimizer opt);

////////////////////////////////////////////////////////////////////////////////
// Device
////////////////////////////////////////////////////////////////////////////////

const char *Torch_Device(const char *device_type, Device *device);
bool IsCUDAAvailable();

const char *Tensor_To(Tensor input, Device device, int8_t dtype,
                      Tensor *output);
const char *Tensor_CastTo(Tensor input, int8_t dtype, Tensor *output);
const char *Tensor_CopyTo(Tensor input, Device device, Tensor *output);

////////////////////////////////////////////////////////////////////////////////
//  Dataset, DataLoader, and Iterator torch.utils.data
////////////////////////////////////////////////////////////////////////////////

typedef struct {
  MNIST p;
  Normalize normalize;
  double mean, stddev;
} MNISTDataset;

const char *CreateMNISTDataset(const char *data_root, MNISTDataset *dataset);
void MNISTDataset_Close(MNISTDataset d);

// Set parameters of the normalize transform in dataset
void MNISTDataset_Normalize(MNISTDataset *dataset, double *mean,
                            int64_t mean_len, double *stddev,
                            int64_t stddev_len);

typedef void *MNISTLoader;
typedef void *MNISTIterator;

MNISTLoader CreateMNISTLoader(MNISTDataset dataset, int64_t batchsize);
void MNISTLoader_Close(MNISTLoader loader);

MNISTIterator MNISTLoader_Begin(MNISTLoader loader);
void MNISTIterator_Batch(MNISTIterator iter, Tensor *data, Tensor *target);
bool MNISTIterator_Next(MNISTIterator iter, MNISTLoader loader);
bool MNISTIterator_IsEnd(MNISTIterator iter, MNISTLoader loader);
void MNISTIterator_Close(MNISTIterator iter);

#ifdef __cplusplus
}
#endif

#endif  // CGOTORCH_CGOTORCH_H_
