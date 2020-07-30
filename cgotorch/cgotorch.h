#ifndef __C_TORCH_H_
#define __C_TORCH_H_


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
typedef void *CTransform;
CTransform CNormalize(double mean, double stddev);
CTransform CStack();

// dataset APIs
typedef void *CDataset;
CDataset CMnist(const char *data_root);
void DatasetMap(CDataset dataset, CTransform transform);
void AddNormalize(CDataset dataset, CTransform transform);
void AddStack(CDataset dataset, CTransform transform);
#ifdef __cplusplus
}
#endif

#endif //__C_TORCH_H_
